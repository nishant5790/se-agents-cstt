"""Azure-based audio/video extraction for the ExtractionAgent.

Drop-in replacement for the offline Vosk path in ``extractors.extract_media``.
Uses only the Azure services we have access to:

  * Azure OpenAI **Whisper**            -> transcript segments with timestamps
  * Azure OpenAI **GPT-4o vision**      -> one-line caption + on-screen text per frame
  * Azure **Document Intelligence**     -> precise OCR for text-heavy frames (optional)
  * **ffmpeg** (local)                  -> demux audio + grab frames (mechanical only)

Output is a list of ``ContentBlock`` identical in shape to ``extract_media`` so the
rest of the pipeline (CKM, agents) is unaffected. Caching mirrors the offline path:
results are written to ``<file>.transcript.json`` and reused while the source is
unchanged.

Wire it up by registering the media extensions against ``extract_media_azure`` in
``extractors.EXTRACTORS`` (or import and patch the dict at startup).

Environment variables
----------------------
    AZURE_OPENAI_ENDPOINT              (required)
    AZURE_OPENAI_API_KEY              (required)
    AZURE_OPENAI_API_VERSION          (default: 2024-10-21)
    AZURE_OPENAI_WHISPER_DEPLOYMENT   (default: whisper)
    AZURE_OPENAI_DEPLOYMENT           (default: gpt-4o, used for vision captions)
    AZURE_DOCINTEL_ENDPOINT           (optional, enables slide OCR)
    AZURE_DOCINTEL_KEY                (optional, enables slide OCR)

    AZURE_MEDIA_CHUNK_SECS            (default: 600)   audio chunk length for Whisper
    AZURE_MEDIA_CAPTION_FRAMES        (default: 1)     1 = caption frames, 0 = skip
    AZURE_MEDIA_OCR_FRAMES            (default: 0)     1 = run Doc Intelligence OCR

Local testing (no Azure required)
---------------------------------
    MEDIA_TRANSCRIBE_BACKEND         (default: azure) 'local' uses openai-whisper on CPU
    LOCAL_WHISPER_MODEL              (default: base)  tiny|base|small|medium|large-v3

With MEDIA_TRANSCRIBE_BACKEND=local the transcript is produced entirely offline so
you can validate the transcript + frame pipeline before configuring Azure. The model
weights download from the Azure CDN (not HuggingFace). Frame captioning/OCR still
requires Azure and is skipped automatically when unavailable.
"""
from __future__ import annotations

import base64
import json as _json
import os
import subprocess
import tempfile
from pathlib import Path

from .ckm import ContentBlock
from . import extractors
from .extractors import OnBlock, _emit, _grab_frames, _slug


# ---------------------------------------------------------------- config helpers

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _openai_client():
    """Build an AzureOpenAI client, reusing llm.py's TLS-trust shim if present."""
    try:
        from .llm import _ensure_tls_trust  # honour corporate proxy certs

        _ensure_tls_trust()
    except Exception:
        pass

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not (endpoint and api_key):
        raise RuntimeError(
            "Azure OpenAI not configured (set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY)"
        )
    from openai import AzureOpenAI

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )


# ---------------------------------------------------------------- transcription

def _transcribe_whisper(path: Path, chunk_secs: int) -> list[dict]:
    """Demux to mono mp3, split into <= chunk_secs pieces, transcribe each with
    Whisper, and return time-ordered segments with absolute timestamps."""
    client = _openai_client()
    deploy = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT", "whisper")
    ffmpeg = _ffmpeg()
    segments: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        pattern = Path(tmp) / "chunk_%03d.mp3"
        # Compress hard (mono, 16 kHz, 64 kbps) to stay under Whisper's 25 MB limit.
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "64k",
             "-f", "segment", "-segment_time", str(chunk_secs), str(pattern)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for idx, chunk in enumerate(sorted(Path(tmp).glob("chunk_*.mp3"))):
            offset = idx * chunk_secs
            with chunk.open("rb") as fh:
                resp = client.audio.transcriptions.create(
                    model=deploy, file=fh, response_format="verbose_json",
                )
            for seg in _iter_segments(resp):
                text = (seg.get("text") or "").strip()
                if not text:
                    continue
                segments.append({
                    "start": float(seg.get("start", 0.0)) + offset,
                    "end": float(seg.get("end", 0.0)) + offset,
                    "text": text,
                })
    segments.sort(key=lambda s: s["start"])
    return segments


def _transcribe_local(path: Path) -> list[dict]:
    """Offline transcription with OpenAI Whisper (CPU). For local testing only.

    Uses the ``openai-whisper`` package, whose model weights download from the
    Azure CDN (``openaipublic.azureedge.net``) rather than HuggingFace — handy on
    networks that block HF. Demuxes to 16 kHz mono wav with ffmpeg and splits the
    audio into fixed-length chunks (default 30 s, ``LOCAL_WHISPER_CHUNK_SECS``)
    that are transcribed one at a time. Chunking keeps peak memory low and gives
    incremental progress instead of loading the whole video at once. Each chunk's
    Whisper timestamps are offset by the chunk's start so the returned segments
    carry absolute timestamps, matching ``_transcribe_whisper``.
    """
    try:
        import whisper  # openai-whisper
    except ImportError as exc:  # pragma: no cover - guidance only
        raise RuntimeError(
            "openai-whisper not installed. Run: pip install openai-whisper"
        ) from exc

    import wave

    import numpy as np

    # Honour corporate proxy certs for the model download (Azure CDN).
    try:
        from .llm import _ensure_tls_trust

        _ensure_tls_trust()
    except Exception:
        pass

    ffmpeg = _ffmpeg()
    model_size = os.getenv("LOCAL_WHISPER_MODEL", "base")
    chunk_secs = _env_int("LOCAL_WHISPER_CHUNK_SECS", 30)
    segments: list[dict] = []

    model = whisper.load_model(model_size)

    with tempfile.TemporaryDirectory() as tmp:
        pattern = Path(tmp) / "chunk_%05d.wav"
        # Demux to 16 kHz mono and split into chunk_secs pieces in one ffmpeg pass.
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
             "-f", "segment", "-segment_time", str(chunk_secs), str(pattern)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        chunks = sorted(Path(tmp).glob("chunk_*.wav"))
        total = len(chunks)
        offset = 0.0
        for idx, chunk in enumerate(chunks):
            with wave.open(str(chunk), "rb") as wf:
                n_frames = wf.getnframes()
                rate = wf.getframerate() or 16000
                raw = wf.readframes(n_frames)
            duration = n_frames / float(rate)
            audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

            print(f"  transcribing chunk {idx + 1}/{total} "
                  f"[{offset:.1f}s - {offset + duration:.1f}s]")
            result = model.transcribe(audio)
            for s in result.get("segments", []):
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                segments.append({
                    "start": float(s.get("start", 0.0)) + offset,
                    "end": float(s.get("end", 0.0)) + offset,
                    "text": text,
                })
            offset += duration
    segments.sort(key=lambda s: s["start"])
    return segments


def _iter_segments(resp) -> list[dict]:
    """Normalise the verbose_json response into a list of plain dicts.

    The SDK may return a pydantic-ish object or a dict depending on version, and
    each segment may itself be an object or a dict."""
    segs = getattr(resp, "segments", None)
    if segs is None and isinstance(resp, dict):
        segs = resp.get("segments")
    out: list[dict] = []
    for s in (segs or []):
        if isinstance(s, dict):
            out.append(s)
        else:
            out.append({
                "start": getattr(s, "start", 0.0),
                "end": getattr(s, "end", 0.0),
                "text": getattr(s, "text", ""),
            })
    return out


# ---------------------------------------------------------------- frame -> text

def _caption_frame(client, img_path: Path) -> str:
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    resp = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        temperature=0.2,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": (
                "Describe this video frame in one sentence and transcribe any "
                "visible on-screen text verbatim."
            )},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    )
    return (resp.choices[0].message.content or "").strip()


def _ocr_frame(img_path: Path) -> str:
    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
    if not (endpoint and key):
        return ""
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    di = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with img_path.open("rb") as f:
        poller = di.begin_analyze_document(
            "prebuilt-read", body=f, content_type="application/octet-stream",
        )
    return (poller.result().content or "").strip()


def _enrich_frames(blocks: list[ContentBlock]) -> None:
    """For every block that has an image_ref, add a caption and/or OCR text and
    fold it into the block's text so visuals become searchable."""
    caption_on = _env_flag("AZURE_MEDIA_CAPTION_FRAMES", True)
    ocr_on = _env_flag("AZURE_MEDIA_OCR_FRAMES", True)
    if not (caption_on or ocr_on):
        return
    framed = [b for b in blocks if b.image_ref and Path(b.image_ref).exists()]
    if not framed:
        return

    client = None
    if caption_on:
        try:
            client = _openai_client()
        except Exception:
            # Azure not configured (e.g. local transcription test) — skip captions.
            caption_on = False
    if not (caption_on or ocr_on):
        return
    for b in framed:
        img = Path(b.image_ref)
        extras: list[str] = []
        if caption_on:
            try:
                cap = _caption_frame(client, img)
            except Exception:
                cap = ""
            if cap:
                b.metadata["caption"] = cap
                extras.append(cap)
        if ocr_on:
            try:
                ocr = _ocr_frame(img)
            except Exception:
                ocr = ""
            if ocr:
                b.metadata["ocr"] = ocr
                extras.append(ocr)
        if extras:
            b.text = f"{b.text}\n[visual] " + " ".join(extras) if b.text else "[visual] " + " ".join(extras)
            b.text = b.text.strip()
            # print(f" ===== enrich frame : {b.metadata} ====\n")


# ---------------------------------------------------------------- main entry

def extract_media_azure(path: Path, on_block: OnBlock = None) -> list[ContentBlock]:
    """Azure transcript + frame extraction. Drop-in for ``extract_media``."""
    cache = path.with_suffix(path.suffix + ".transcript.json")

    # Reuse a cached result if the source is unchanged (Azure calls cost money).
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        raw = _json.loads(cache.read_text(encoding="utf-8"))
        blocks = [ContentBlock(**b) for b in raw]
        if not any(b.image_ref for b in blocks):
            _grab_frames(path, blocks)
            _enrich_frames(blocks)
            cache.write_text(
                _json.dumps([b.model_dump() for b in blocks], ensure_ascii=False),
                encoding="utf-8",
            )
        if on_block is not None:
            for b in blocks:
                try:
                    on_block(b)
                except Exception:
                    pass
        return blocks

    backend = os.getenv("MEDIA_TRANSCRIBE_BACKEND", "azure").strip().lower()
    if backend == "local":
        segments = _transcribe_local(path)
    else:
        chunk_secs = _env_int("AZURE_MEDIA_CHUNK_SECS", 600)
        segments = _transcribe_whisper(path, chunk_secs)

    blocks: list[ContentBlock] = []
    for seg in segments:
        start = round(float(seg["start"]), 2)
        _emit(blocks, ContentBlock(
            id=_slug(path.stem, "t", int(start * 100)),
            source=path.name,
            modality="transcript",
            title=f"{path.stem} @ {start:.0f}s",
            text=seg["text"],
            timestamp=start,
        ), on_block)

    # Grab a frame per sampled segment (no-op for pure audio), then describe them.
    _grab_frames(path, blocks)
    _enrich_frames(blocks)

    cache.write_text(
        _json.dumps([b.model_dump() for b in blocks], ensure_ascii=False),
        encoding="utf-8",
    )
    return blocks


def register(replace: bool = True) -> None:
    """Point the media extensions at the Azure extractor in ``EXTRACTORS``.

    Call once at startup (e.g. in run.py) when Azure credentials are present::

        from agent_team import azure_media_extractor
        azure_media_extractor.register()
    """
    media_exts = {".mp4", ".mov", ".mkv", ".avi", ".wav", ".m4a"}
    for ext in media_exts:
        if replace or ext not in extractors.EXTRACTORS:
            extractors.EXTRACTORS[ext] = extract_media_azure
