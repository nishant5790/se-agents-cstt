"""FastAPI server exposing the agent team as a simple demo API + chat UI.

Endpoints
---------
GET  /                      -> the single-page chat UI (static/index.html)
GET  /api/status            -> knowledge-base state (built / building / topics)
POST /api/build            -> (re)build the knowledge base in the background
POST /api/chat             -> ask a question grounded in the extracted content
POST /api/generate         -> author a PPTX deck from a natural-language request
GET  /api/decks            -> list generated decks
GET  /api/download/{name}  -> download a generated .pptx

The heavy work (extraction + understanding) runs once in a background thread so
the UI stays responsive; deck authoring is fast and runs inline.
"""
from __future__ import annotations

import re
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm
from .ckm import CKM
from .orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parent
INPUTS = Path(__import__("os").getenv("INPUTS_DIR", str(ROOT / "inputs")))
OUTPUTS = Path(__import__("os").getenv("OUTPUTS_DIR", str(ROOT / "outputs")))
TEMPLATE = ROOT / "templates" / "brand.pptx"
STATIC = ROOT / "static"

app = FastAPI(title="Agent Team — Learning Material Studio")

# ----- shared runtime state -----
_lock = threading.Lock()
_state: dict = {"built": False, "building": False, "error": None,
                "topics": [], "blocks": 0, "sources": []}
_orch: Orchestrator | None = None


def _orchestrator() -> Orchestrator:
    global _orch
    if _orch is None:
        INPUTS.mkdir(parents=True, exist_ok=True)
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        _orch = Orchestrator(INPUTS, OUTPUTS, TEMPLATE if TEMPLATE.exists() else None)
    return _orch


def _build_worker() -> None:
    try:
        orch = _orchestrator()
        orch.build_knowledge()
        ckm: CKM = orch.bb.get("ckm")
        with _lock:
            _state.update(built=True, building=False, error=None,
                          topics=orch.bb.get("topics", []),
                          blocks=len(ckm.blocks) if ckm else 0,
                          sources=ckm.sources if ckm else [])
    except Exception as exc:  # surface build failures to the UI
        with _lock:
            _state.update(built=False, building=False, error=str(exc))


# ----- request models -----
class ChatIn(BaseModel):
    message: str


class GenerateIn(BaseModel):
    request: str


# ----- API -----
@app.get("/api/status")
def status() -> dict:
    with _lock:
        return dict(_state, llm=llm.available())


@app.post("/api/build")
def build() -> dict:
    with _lock:
        if _state["building"]:
            return {"ok": True, "building": True}
        _state.update(building=True, built=False, error=None)
    threading.Thread(target=_build_worker, daemon=True).start()
    return {"ok": True, "building": True}


SUPPORTED_SUFFIXES = {
    ".xlsx", ".xls", ".pdf", ".txt", ".md",
    ".mp4", ".mov", ".mkv", ".avi", ".m4v", ".wav", ".mp3", ".m4a",
}


@app.get("/api/sources")
def list_sources() -> dict:
    INPUTS.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(INPUTS.iterdir())
             if f.is_file() and f.suffix.lower() in SUPPORTED_SUFFIXES]
    return {"sources": [{"name": f.name, "size": f.stat().st_size} for f in files]}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    INPUTS.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for uf in files:
        name = Path(uf.filename or "").name
        if not name or Path(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            skipped.append(name or "(unnamed)")
            continue
        target = INPUTS / name
        with target.open("wb") as out:
            while chunk := await uf.read(1024 * 1024):
                out.write(chunk)
        saved.append(name)
    # New sources invalidate the current knowledge base.
    if saved:
        with _lock:
            _state.update(built=False, topics=[], blocks=0, sources=[])
    return {"ok": True, "saved": saved, "skipped": skipped}


@app.delete("/api/sources/{name}")
def delete_source(name: str) -> dict:
    target = INPUTS / Path(name).name
    if not target.exists() or target.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise HTTPException(404, "Source not found")
    target.unlink()
    with _lock:
        _state.update(built=False, topics=[], blocks=0, sources=[])
    return {"ok": True, "deleted": target.name}


def _search_blocks(query: str, limit: int = 6) -> list:
    orch = _orchestrator()
    ckm: CKM = orch.bb.get("ckm")
    if not ckm:
        return []
    terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if len(w) > 2}
    scored = []
    for b in ckm.blocks:
        hay = f"{b.title} {b.text}".lower()
        score = sum(hay.count(t) for t in terms)
        if score:
            scored.append((score, b))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in scored[:limit]]


@app.post("/api/chat")
def chat(body: ChatIn) -> dict:
    msg = body.message.strip()
    if not msg:
        raise HTTPException(400, "Empty message")
    with _lock:
        built = _state["built"]
        topics = _state["topics"]
    if not built:
        return {"answer": "The knowledge base isn't built yet. Click "
                          "**Build knowledge base** in the sidebar first.",
                "sources": []}

    hits = _search_blocks(msg)
    context = "\n".join(f"- {b.title}: {b.text[:240]}" for b in hits)
    answer = llm.chat_text(
        system="You are a helpful learning assistant. Answer the user's question "
               "using ONLY the provided source snippets. If they don't cover it, "
               "say so briefly. Keep answers concise and practical.",
        user=f"QUESTION: {msg}\n\nSOURCE SNIPPETS:\n{context or '(none)'}",
    )
    if not answer:  # offline fallback
        if hits:
            answer = ("Here's what the sources mention:\n\n" +
                      "\n".join(f"- {b.title}: {b.text[:180]}" for b in hits))
        else:
            answer = ("I couldn't find that in the extracted content. "
                      f"Known topics include: {', '.join(topics[:12]) or '(none)'}.")
    return {"answer": answer,
            "sources": [{"title": b.title, "source": b.source} for b in hits]}


@app.post("/api/generate")
def generate(body: GenerateIn) -> dict:
    req = body.request.strip()
    if not req:
        raise HTTPException(400, "Empty request")
    with _lock:
        if not _state["built"]:
            raise HTTPException(409, "Knowledge base not built yet.")
    orch = _orchestrator()
    path = Path(orch.author(req))
    plan = orch.bb.get("deck_plan")
    slides = [{"title": s.title, "bullets": s.bullets,
               "image": bool(getattr(s, "image", None))} for s in plan.slides]
    return {"ok": True, "file": path.name, "title": plan.deck_title,
            "slides": slides}


@app.get("/api/decks")
def decks() -> dict:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    files = sorted(OUTPUTS.glob("DECK--*.pptx"), key=lambda p: p.stat().st_mtime,
                   reverse=True)
    return {"decks": [{"file": f.name, "size": f.stat().st_size} for f in files]}


@app.get("/api/download/{name}")
def download(name: str) -> FileResponse:
    safe = Path(name).name  # prevent path traversal
    target = OUTPUTS / safe
    if not target.exists() or target.suffix != ".pptx":
        raise HTTPException(404, "Deck not found")
    return FileResponse(
        target,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=safe,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
