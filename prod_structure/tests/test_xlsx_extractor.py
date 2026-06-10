from __future__ import annotations

from pathlib import Path

from _helpers import INPUTS_DIR, show_blocks, trace_block
from agent_team.tools.xlsx_tool import extract_xlsx


def main() -> None:
    path = INPUTS_DIR / "INPUT--All 2.3 E2E Test Scripts.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    try:
        blocks = extract_xlsx(Path(path), on_block=trace_block)
    except ModuleNotFoundError as exc:
        print(f"xlsx extractor skipped: {exc}")
        print("Install the optional openpyxl dependency to run this script.")
        return

    show_blocks("xlsx extractor", blocks)


if __name__ == "__main__":
    main()