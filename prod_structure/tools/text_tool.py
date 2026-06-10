"""text / markdown source tool: one ContentBlock per heading/paragraph group."""
from __future__ import annotations

import re
from pathlib import Path

from agent_team.core.ckm import ContentBlock

from ._base import OnBlock, _emit, _slug


def extract_text(path: Path, on_block: OnBlock = None) -> list[ContentBlock]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks: list[ContentBlock] = []
    chunks = re.split(r"\n(?=#{1,6}\s)", content)  # split on markdown headings
    for i, chunk in enumerate(chunks):
        chunk = chunk.strip()
        if not chunk:
            continue
        first = chunk.splitlines()[0].lstrip("# ").strip()
        _emit(blocks, ContentBlock(
            id=_slug(path.stem, i),
            source=path.name,
            modality="heading" if chunk.startswith("#") else "text",
            title=first[:80],
            text=chunk,
            metadata={},
        ), on_block)
    return blocks
