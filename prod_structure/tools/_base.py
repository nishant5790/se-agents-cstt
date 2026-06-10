"""Shared building blocks for source tools.

Kept separate from the package ``__init__`` so individual tool modules can import
these helpers without triggering a circular import through the registry.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, Optional

from agent_team.core.ckm import ContentBlock

# Callback signature used by all tools to report each block as it is produced.
OnBlock = Optional[Callable[[ContentBlock], None]]

# Where extracted visuals (pdf figures, video frames) are written. The
# ExtractionAgent points this at <outputs>/assets before running.
ASSETS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "assets"


def _emit(blocks: list[ContentBlock], block: ContentBlock, on_block: OnBlock) -> None:
    blocks.append(block)
    if on_block is not None:
        try:
            on_block(block)
        except Exception:
            pass


def _slug(*parts: object) -> str:
    raw = "__".join(str(p) for p in parts)
    return re.sub(r"[^a-zA-Z0-9_]+", "-", raw).strip("-").lower()


def set_assets_dir(path: Path) -> None:
    """Repoint where tools write visuals (called once per run by the agent)."""
    global ASSETS_DIR
    ASSETS_DIR = Path(path)


def assets_dir() -> Path:
    """Current assets directory (module-level ``ASSETS_DIR``), created on demand.

    Reads the attribute off this module dynamically so the ExtractionAgent can
    repoint ``ASSETS_DIR`` per run and every tool follows.
    """
    from . import _base

    _base.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    return _base.ASSETS_DIR
