"""audio source tool: transcript-only ContentBlocks for pure audio files.

Transcribes via the configured backend (Azure Whisper or local) and emits one
block per transcript segment. Results are cached next to the source as
``<file>.transcript.json`` and reused while the source is unchanged.
"""
from __future__ import annotations

import json as _json
from pathlib import Path

from agent_team.core.ckm import ContentBlock
from agent_team.core.logging import get_logger

from . import media_common
from ._base import OnBlock, _emit, _slug

log = get_logger("tools.audio")


def extract_audio(path: Path, on_block: OnBlock = None) -> list[ContentBlock]:
    cache = path.with_suffix(path.suffix + ".transcript.json")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        blocks = [ContentBlock(**b) for b in _json.loads(cache.read_text(encoding="utf-8"))]
        _replay(blocks, on_block)
        return blocks

    segments = media_common.transcribe(path)
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

    cache.write_text(
        _json.dumps([b.model_dump() for b in blocks], ensure_ascii=False),
        encoding="utf-8",
    )
    return blocks


def _replay(blocks: list[ContentBlock], on_block: OnBlock) -> None:
    if on_block is None:
        return
    for b in blocks:
        try:
            on_block(b)
        except Exception:
            pass
