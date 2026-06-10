"""Quick local test of the media transcription pipeline (faster-whisper).

Run from the parent of the agent_team package:
    python -m agent_team.scripts_test_local_whisper
"""
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # load .env if it exists, but don't fail if it doesn't

# os.environ.setdefault("MEDIA_TRANSCRIBE_BACKEND", "local")
# os.environ.setdefault("LOCAL_WHISPER_MODEL", "tiny")  # fast for a smoke test

from agent_team.azure_media_extractor import _transcribe_local , extract_media_azure

SAMPLE = Path(__file__).resolve().parent / "inputs" / \
    "INPUT--UAT Kickoff & Demo-20250331_103150-Meeting Recording.mp4"


def _on_block(b, _file=SAMPLE.name, _i=0, _t=0):
    preview = " ".join((b.text or "").split())[:80]
    visual = " [img]" if b.image_ref else ""

    print(f"    + {b.modality:<11} {b.id} — {b.title or preview}{visual}")

def main() -> None:
    print(f"Transcribing: {SAMPLE.name}")
    print(f"Model: {os.environ['LOCAL_WHISPER_MODEL']} (CPU)\n")
    # segments = _transcribe_local(SAMPLE)
    output = extract_media_azure(SAMPLE, on_block=_on_block)  # also saves a JSON copy of the segments

    # print(f"Total segments: {len(segments)}\n")
    # for s in segments[:8]:
    #     print(f"  [{s['start']:6.1f} - {s['end']:6.1f}]  {s['text'][:80]}")
    # if len(segments) > 8:
    #     print(f"  ... (+{len(segments) - 8} more)")


if __name__ == "__main__":
    main()
