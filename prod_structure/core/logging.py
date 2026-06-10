"""Structured logging for the agent team.

A single helper, :func:`get_logger`, returns a module/agent-scoped logger that
writes a consistent, timestamped line to stderr. Level is controlled by the
``LOG_LEVEL`` environment variable (default ``INFO``). Agents and tools use this
instead of bare ``print`` so output is greppable and severity-aware in a
container.
"""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def _configure_root() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger("agent_team")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``agent_team`` for *name*."""
    _configure_root()
    return logging.getLogger(f"agent_team.{name}")
