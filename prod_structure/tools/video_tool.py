"""video source tool — multi-frame analysis.

For each time chunk of a video this produces a single :class:`ContentBlock`:

* ``text``      — the chunk transcript, with each kept frame's visual description
                  folded in so visuals are searchable.
* ``timestamp`` — the chunk's start time (seconds).
* ``image_ref`` — the first kept frame (primary visual for slides).
* ``frames``    — every kept frame as a :class:`Frame` (timestamp, image path,
                  GPT-4o visual description), de-duplicated by scene change.

Transcription uses the configured backend (Azure Whisper / local); frame
descriptions use Azure GPT-4o vision when configured. Results are cached next to
the source as ``<file>.transcript.json`` and reused while the source is unchanged.
"""
from __future__ import annotations

import json as _json
import math
import os
import shutil
import tempfile
from pathlib import Path

from agent_team.core import llm
from agent_team.core.ckm import ContentBlock, Frame
from agent_team.core.config import settings
from agent_team.core.logging import get_logger

from . import media_common
from ._base import OnBlock, _emit, _slug, assets_dir

log = get_logger("tools.video")


def extract_video(path: Path, on_block: OnBlock = None) -> list[ContentBlock]:
    cache = path.with_suffix(path.suffix + ".transcript.json")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        blocks = [ContentBlock(**b) for b in _json.loads(cache.read_text(encoding="utf-8"))]
        _replay(blocks, on_block)
        return blocks

    cfg = settings()
    video_path = str(Path(path).resolve())
    video_name = Path(path).stem
    slug = _slug(video_name)

    total_duration = media_common.get_video_duration(video_path)
    chunk_secs = cfg.video_chunk_secs
    num_chunks = max(1, math.ceil(total_duration / chunk_secs))

    # Whole-file transcription once; segments are grouped into chunks by time.
    segments = media_common.transcribe(Path(path))

    # Vision client (optional — frames still saved without descriptions).
    analyze = cfg.video_analyze_frames
    vision = llm.azure_openai_client() if analyze else None
    if analyze and vision is None:
        log.info("vision disabled (Azure OpenAI not configured) — saving frames without descriptions")
    vision_client, vision_model = vision if vision else (None, None)

    log.info("video=%s duration=%.1fs chunks=%d interval=%.1fs vision=%s",
             Path(path).name, total_duration, num_chunks, cfg.video_frame_interval,
             bool(vision_client))

    out_dir = assets_dir()
    blocks: list[ContentBlock] = []
    prev_frame: str | None = None

    with tempfile.TemporaryDirectory() as temp_dir:
        for idx in range(num_chunks):
            chunk_start = idx * chunk_secs
            chunk_end = min(chunk_start + chunk_secs, total_duration)
            transcript = _chunk_transcript(segments, chunk_start, chunk_end)

            unique, prev_frame = _collect_unique_frames(
                video_path, chunk_start, chunk_end, temp_dir, prev_frame,
                cfg.video_frame_interval, cfg.video_similarity_threshold)

            timestamp = round(chunk_start, 1)
            block_id = f"{slug}__t__{int(timestamp * 100)}"
            frames = _describe_frames(
                unique, transcript, block_id, out_dir,
                vision_client, vision_model, cfg.ocr_frames)

            text = _fold_visuals(transcript, frames)
            _emit(blocks, ContentBlock(
                id=block_id,
                source=path.name,
                modality="transcript",
                title=f"{video_name} @ {int(timestamp)}s",
                text=text,
                timestamp=timestamp,
                image_ref=frames[0].image_ref if frames else None,
                frames=frames,
                metadata={
                    "chunk_duration": round(chunk_end - chunk_start, 2),
                    "frames_kept": len(frames),
                },
            ), on_block)

    cache.write_text(
        _json.dumps([b.model_dump() for b in blocks], ensure_ascii=False),
        encoding="utf-8",
    )
    return blocks


# ----------------------------------------------------------------- helpers

def _chunk_transcript(segments: list[dict], start: float, end: float) -> str:
    parts = [s["text"].strip() for s in segments
             if s.get("text") and start <= float(s.get("start", 0.0)) < end]
    return " ".join(p for p in parts if p).strip()


def _collect_unique_frames(video_path, chunk_start, chunk_end, temp_dir, prev_frame,
                           interval, threshold):
    """Extract candidate frames across the chunk, keep only scene-unique ones."""
    candidates: list[tuple[float, str]] = []
    ts = chunk_start
    while ts < chunk_end:
        frame_path = os.path.join(temp_dir, f"cand_{ts:.1f}.jpg")
        try:
            media_common.extract_frame(video_path, ts, frame_path)
            candidates.append((ts, frame_path))
        except Exception as exc:  # noqa: BLE001 - skip unreadable timestamps
            log.warning("frame grab failed @ %.1fs: %s", ts, exc)
        ts += interval

    unique: list[tuple[float, str]] = []
    last_kept = prev_frame
    for ts, fpath in candidates:
        if last_kept is None or not media_common.frames_are_similar(last_kept, fpath, threshold):
            unique.append((ts, fpath))
            last_kept = fpath
        else:
            try:
                os.remove(fpath)
            except OSError:
                pass
    return unique, last_kept


def _describe_frames(unique, transcript, block_id, out_dir, vision_client,
                     vision_model, ocr_on) -> list[Frame]:
    """Move kept frames into the assets dir and attach a visual description."""
    frames: list[Frame] = []
    for ts, temp_path in unique:
        final = Path(out_dir) / f"{block_id}__f__{int(ts * 100)}.jpg"
        try:
            # shutil.move (not os.replace) so the frame survives a move across
            # filesystems — e.g. Docker's /tmp temp dir to a mounted assets volume.
            shutil.move(temp_path, str(final))
        except OSError as exc:
            final = Path(temp_path)  # fall back to the temp path if move fails
            log.warning("frame move failed @ %.1fs (%s); keeping temp path", ts, exc)

        description = None
        extras: list[str] = []
        if vision_client is not None:
            try:
                cap = media_common.caption_frame(vision_client, vision_model, final, transcript)
            except Exception as exc: 
                cap = ""
                log.warning("caption failed @ %.1fs: %s", ts, exc)
            if cap:
                extras.append(cap)
        if ocr_on:
            try:
                ocr = media_common.ocr_frame(final)
            except Exception as exc: 
                ocr = ""
                log.warning("ocr failed @ %.1fs: %s", ts, exc)
            if ocr:
                extras.append(ocr)
        if extras:
            description = " ".join(extras).strip()

        frames.append(Frame(
            timestamp=round(ts, 2),
            image_ref=str(final),
            visual_description=description,
        ))
    return frames


def _fold_visuals(transcript: str, frames: list[Frame]) -> str:
    """Append each frame's description to the transcript so visuals are searchable."""
    lines = [transcript] if transcript else []
    for f in frames:
        if f.visual_description:
            lines.append(f"[visual] {f.visual_description}")
    return "\n".join(lines).strip()


def _replay(blocks: list[ContentBlock], on_block: OnBlock) -> None:
    if on_block is None:
        return
    for b in blocks:
        try:
            on_block(b)
        except Exception:
            pass
