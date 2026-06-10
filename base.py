"""Tiny agent runtime: a shared Blackboard + a minimal Agent base class.

Design goals: modular, simple, no framework. Every agent reads/writes the same
Blackboard so stages are composable and re-runnable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Blackboard:
    """Shared state passed between agents. Persisted to outputs/ as JSON."""

    workdir: Path
    data: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def save(self, name: str, payload: Any) -> Path:
        out = self.workdir / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return out


class Agent:
    """Base class. Each agent has a name and a run(bb) method."""

    name: str = "agent"

    def log(self, msg: str) -> None:
        print(f"  [{self.name}] {msg}")

    def run(self, bb: Blackboard) -> Blackboard:  # pragma: no cover - interface
        raise NotImplementedError
