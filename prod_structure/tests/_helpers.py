from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "agent_team"
INPUTS_DIR = PACKAGE_ROOT / "inputs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def show_blocks(label: str, blocks) -> None:
    print(f"\n{label}: {len(blocks)} block(s)")
    for index, block in enumerate(blocks, start=1):
        print(f"\n[{index}] id={block.id}")
        print(f"source={block.source}")
        print(f"modality={block.modality}")
        print(f"title={block.title}")
        print(f"timestamp={block.timestamp}")
        print(f"image_ref={block.image_ref}")
        frames = getattr(block, "frames", []) or []
        if frames:
            print(f"frames={len(frames)}")
            for f in frames:
                desc = (f.visual_description or "")[:80]
                print(f"  frame @ {f.timestamp}s {f.image_ref} :: {desc}")
        print(f"metadata={json.dumps(block.metadata, ensure_ascii=False, indent=2)}")
        print("text:")
        print(block.text)


def trace_block(block) -> None:
    print(
        f"emitted: id={block.id} modality={block.modality} title={block.title!r} "
        f"timestamp={block.timestamp} image_ref={block.image_ref}"
    )