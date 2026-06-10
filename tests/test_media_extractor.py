from __future__ import annotations

from pathlib import Path

from _helpers import INPUTS_DIR, show_blocks, trace_block
from agent_team.extractors import extract_media


def main() -> None:
    path = INPUTS_DIR / "INPUT--UAT Kickoff & Demo-20250331_103150-Meeting Recording.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    try:
        blocks = extract_media(Path(path), on_block=trace_block)
    except ModuleNotFoundError as exc:
        print(f"media extractor skipped: {exc}")
        print("Install the optional vosk dependency and set VOSK_MODEL_PATH if needed.")
        return
    except Exception as exc:
        print(f"media extractor failed: {exc}")
        print("Set VOSK_MODEL_PATH if the offline model is not already available.")
        return

    show_blocks("media extractor", blocks)


if __name__ == "__main__":
    main()