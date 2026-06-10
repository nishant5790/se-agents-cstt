from __future__ import annotations

from pathlib import Path

from _helpers import INPUTS_DIR, show_blocks, trace_block
from agent_team.extractors import extract_text


def main() -> None:
    path = INPUTS_DIR / "sample_training.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    blocks = extract_text(Path(path), on_block=trace_block)
    show_blocks("text extractor", blocks)


if __name__ == "__main__":
    main()