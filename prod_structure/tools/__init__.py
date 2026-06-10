"""Source extraction tools.

Each tool turns one source format into a list of :class:`ContentBlock`. Tools
are registered against file extensions in :data:`SOURCE_TOOLS`; :func:`extract_file`
dispatches a path to the right tool.

Shared building blocks (``_emit``, ``_slug``, ``ASSETS_DIR``, ``OnBlock``) live
here so every tool reports blocks the same way and writes visuals to one place.

Adding a format = write a ``extract_*`` function in its own module and register
it below. Heavy/optional deps are imported lazily inside each tool so a missing
dependency only breaks its own format, not the whole team.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from agent_team.core.ckm import ContentBlock

from ._base import (
    OnBlock,
    _emit,
    _slug,
    assets_dir,
    set_assets_dir,
)

# Tool registry, populated at import time below.
SOURCE_TOOLS: dict[str, Callable[..., list[ContentBlock]]] = {}


def register(extensions: list[str], fn: Callable[..., list[ContentBlock]]) -> None:
    for ext in extensions:
        SOURCE_TOOLS[ext.lower()] = fn


def supported_suffixes() -> set[str]:
    return set(SOURCE_TOOLS)


def extract_file(path: Path, on_block: OnBlock = None) -> list[ContentBlock]:
    fn = SOURCE_TOOLS.get(path.suffix.lower())
    if not fn:
        return []
    return fn(path, on_block)


# ---- wire up the built-in tools ----
from .xlsx_tool import extract_xlsx  # noqa: E402
from .pdf_tool import extract_pdf  # noqa: E402
from .text_tool import extract_text  # noqa: E402
from .audio_tool import extract_audio  # noqa: E402
from .video_tool import extract_video  # noqa: E402

register([".xlsx", ".xls"], extract_xlsx)
register([".pdf"], extract_pdf)
register([".txt", ".md"], extract_text)
register([".wav", ".m4a", ".mp3"], extract_audio)
register([".mp4", ".mov", ".mkv", ".avi", ".m4v"], extract_video)

__all__ = [
    "OnBlock",
    "SOURCE_TOOLS",
    "_emit",
    "_slug",
    "assets_dir",
    "set_assets_dir",
    "extract_file",
    "register",
    "supported_suffixes",
    "extract_xlsx",
    "extract_pdf",
    "extract_text",
    "extract_audio",
    "extract_video",
]
