"""pdf source tool: one ContentBlock per page (text + largest embedded image)."""
from __future__ import annotations

from pathlib import Path

from agent_team.core.ckm import ContentBlock

from ._base import OnBlock, _emit, _slug, assets_dir


def extract_pdf(path: Path, on_block: OnBlock = None) -> list[ContentBlock]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    blocks: list[ContentBlock] = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        image_ref = _save_pdf_page_image(path, page, page_no)
        if not text and not image_ref:
            continue
        _emit(blocks, ContentBlock(
            id=_slug(path.stem, "p", page_no),
            source=path.name,
            modality="text",
            title=f"{path.stem} — page {page_no}",
            text=text,
            image_ref=image_ref,
            metadata={"page": page_no},
        ), on_block)
    return blocks


def _save_pdf_page_image(pdf: Path, page, page_no: int) -> str | None:
    """Save the largest embedded image on a PDF page; return its path or None."""
    try:
        images = list(getattr(page, "images", []))
    except Exception:
        return None
    if not images:
        return None
    best = max(images, key=lambda im: len(getattr(im, "data", b"") or b""))
    if not getattr(best, "data", None):
        return None
    out_dir = assets_dir()
    ext = Path(getattr(best, "name", "img.png")).suffix or ".png"
    out = out_dir / f"{_slug(pdf.stem, 'p', page_no)}{ext}"
    try:
        out.write_bytes(best.data)
    except Exception:
        return None
    return str(out)
