from __future__ import annotations

from pathlib import Path

from _helpers import INPUTS_DIR, show_blocks, trace_block
from agent_team.tools.video_tool import extract_video


def main() -> None:
    path = INPUTS_DIR / "INPUT--UAT Kickoff & Demo-20250331_103150-Meeting Recording.mp4"
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    try:
        blocks = extract_video(Path(path), on_block=trace_block)
    except ModuleNotFoundError as exc:
        print(f"video tool skipped: {exc}")
        print("Install ffmpeg/openai-whisper (local backend) or configure Azure Whisper.")
        return
    except Exception as exc:
        print(f"video tool failed: {exc}")
        print("Set MEDIA_TRANSCRIBE_BACKEND=local for an offline run, or configure Azure.")
        return

    show_blocks("video tool", blocks)
    framed = sum(1 for b in blocks if b.frames)
    print(f"\nblocks with frames: {framed}/{len(blocks)}")


if __name__ == "__main__":
    main()