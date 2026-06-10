"""Shared media helpers for the audio and video tools.

Mechanical bits (ffmpeg demux / frame grab / duration), transcription
(Azure Whisper or local openai-whisper), scene-change frame de-duplication, and
frame understanding (GPT-4o caption + optional Document Intelligence OCR).

Azure calls are wrapped in :func:`_with_retry` so transient failures back off and
retry instead of killing a long extraction run.
"""
from __future__ import annotations

import base64
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Callable, TypeVar

from agent_team.core import llm
from agent_team.core.config import settings
from agent_team.core.logging import get_logger

log = get_logger("tools.media")

T = TypeVar("T")

_RETRYABLE_MARKERS = ("timeout", "temporarily", "rate limit", "429", "503", "502", "connection")


def _with_retry(fn: Callable[[], T], *, what: str, attempts: int = 3, base_delay: float = 2.0) -> T:
    """Run *fn*, retrying transient failures with exponential backoff."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - we re-raise after retries
            last = exc
            msg = str(exc).lower()
            transient = any(m in msg for m in _RETRYABLE_MARKERS)
            if attempt >= attempts or not transient:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            log.warning("%s failed (attempt %d/%d): %s — retrying in %.0fs",
                        what, attempt, attempts, exc, delay)
            time.sleep(delay)
    assert last is not None
    raise last


# ----------------------------------------------------------------- ffmpeg utils

def _resolve_media_tool(name: str) -> str:
    import shutil

    found = shutil.which(name)
    if found:
        return found
    try:
        import imageio_ffmpeg

        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if name == "ffmpeg":
            return str(ffmpeg_path)
        probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        ffprobe_path = ffmpeg_path.with_name(probe_name)
        if ffprobe_path.exists():
            return str(ffprobe_path)
    except Exception:
        pass
    raise FileNotFoundError(f"{name} not found. Install ffmpeg or add it to PATH.")


def ffmpeg() -> str:
    return _resolve_media_tool("ffmpeg")


def get_video_duration(video_path: str | Path) -> float:
    video_path = str(video_path)
    try:
        cmd = [
            _resolve_media_tool("ffprobe"), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        ffmpeg_cmd = [_resolve_media_tool("ffmpeg"), "-i", video_path]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        output = (result.stderr or "") + "\n" + (result.stdout or "")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
        if not match:
            raise RuntimeError("Unable to determine video duration from ffmpeg output")
        h, m, s = int(match.group(1)), int(match.group(2)), float(match.group(3))
        return h * 3600 + m * 60 + s


def extract_audio_chunk(video_path: str, start: float, duration: float, output_path: str) -> None:
    subprocess.run(
        [ffmpeg(), "-y", "-ss", str(start), "-i", video_path, "-t", str(duration),
         "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path],
        check=True, capture_output=True,
    )


def extract_frame(video_path: str, timestamp: float, output_path: str) -> str:
    subprocess.run(
        [ffmpeg(), "-y", "-ss", str(timestamp), "-i", video_path,
         "-frames:v", "1", "-q:v", "2", output_path],
        check=True, capture_output=True,
    )
    return output_path


# ----------------------------------------------------------------- scene dedup

def frames_are_similar(frame1: str, frame2: str, threshold: float = 0.95) -> bool:
    try:
        import numpy as np
        from PIL import Image

        img1 = np.array(Image.open(frame1).resize((320, 180)).convert("RGB"), dtype=np.float32)
        img2 = np.array(Image.open(frame2).resize((320, 180)).convert("RGB"), dtype=np.float32)
        diff = np.abs(img1 - img2).mean()
        similarity = 1.0 - (diff / 255.0)
        return similarity >= threshold
    except Exception:
        return False


# ----------------------------------------------------------------- transcription

def transcribe(path: Path) -> list[dict]:
    """Return time-ordered segments ``[{start, end, text}]`` using the configured
    backend (Azure Whisper, or local openai-whisper as a fallback/offline mode)."""
    cfg = settings()
    if cfg.transcribe_azure:
        try:
            return _transcribe_azure(path, cfg.azure_media_chunk_secs)
        except Exception as exc:  # noqa: BLE001
            log.warning("Azure Whisper failed (%s); falling back to local Whisper.", exc)
    return _transcribe_local(path)


def _transcribe_azure(path: Path, chunk_secs: int) -> list[dict]:
    import tempfile

    pair = llm.azure_openai_client()
    if pair is None:
        raise RuntimeError("Azure OpenAI not configured for Whisper transcription")
    client, _ = pair
    deploy = settings().azure_whisper_deployment
    ff = ffmpeg()
    segments: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp:
        pattern = Path(tmp) / "chunk_%03d.mp3"
        subprocess.run(
            [ff, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
             "-c:a", "libmp3lame", "-b:a", "64k",
             "-f", "segment", "-segment_time", str(chunk_secs), str(pattern)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for idx, chunk in enumerate(sorted(Path(tmp).glob("chunk_*.mp3"))):
            offset = idx * chunk_secs

            def _call(_chunk=chunk):
                with _chunk.open("rb") as fh:
                    return client.audio.transcriptions.create(
                        model=deploy, file=fh, response_format="verbose_json",
                    )

            resp = _with_retry(_call, what=f"Whisper chunk {idx}")
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
    try:
        import whisper  # openai-whisper
    except ImportError as exc:  # pragma: no cover - guidance only
        raise RuntimeError(
            "openai-whisper not installed. Run: pip install openai-whisper"
        ) from exc

    import tempfile
    import wave

    import numpy as np

    try:
        llm._ensure_tls_trust()  # model weights download from the Azure CDN
    except Exception:
        pass

    cfg = settings()
    ff = ffmpeg()
    chunk_secs = cfg.local_whisper_chunk_secs
    segments: list[dict] = []
    model = whisper.load_model(cfg.local_whisper_model)

    with tempfile.TemporaryDirectory() as tmp:
        pattern = Path(tmp) / "chunk_%05d.wav"
        subprocess.run(
            [ff, "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", "16000",
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
            log.info("transcribing chunk %d/%d [%.1fs - %.1fs]",
                     idx + 1, total, offset, offset + duration)
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
    """Normalise a verbose_json response into a list of ``{start,end,text}`` dicts."""
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


# ----------------------------------------------------------------- frame -> text

def encode_image_base64(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def caption_frame(client, model: str, img_path: Path, transcript_context: str = "") -> str:
    """GPT-4o one-line description + verbatim on-screen text for a frame."""
    b64 = encode_image_base64(img_path)
    user_text = "Describe this video frame in one sentence and transcribe any visible on-screen text verbatim."
    if transcript_context:
        user_text = (
            f"Describe this video frame concisely (slides, dashboards, UI, diagrams, "
            f"on-screen text). Speaker is saying: '{transcript_context[:300]}'"
        )

    def _call():
        return client.chat.completions.create(
            model=model,
            temperature=0.2,
            max_tokens=200,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
            ]}],
        )

    resp = _with_retry(_call, what="GPT-4o caption")
    return (resp.choices[0].message.content or "").strip()


def ocr_frame(img_path: Path) -> str:
    """Document Intelligence OCR for a frame; '' if DI is not configured."""
    cfg = settings()
    if not cfg.docintel_configured:
        return ""
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential

    di = DocumentIntelligenceClient(
        endpoint=cfg.docintel_endpoint, credential=AzureKeyCredential(cfg.docintel_key))

    def _call():
        with img_path.open("rb") as f:
            poller = di.begin_analyze_document(
                "prebuilt-read", body=f, content_type="application/octet-stream")
        return poller.result()

    result = _with_retry(_call, what="DocIntel OCR")
    return (result.content or "").strip()
