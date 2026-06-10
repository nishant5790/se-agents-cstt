"""Agent 1 — Source Content Extraction.

Scans an inputs directory, dispatches each file to the right extractor, and
assembles a single Canonical Knowledge Model (CKM). On-the-fly: any mix of
xlsx / pdf / video / text in the same folder.
"""
from __future__ import annotations

from pathlib import Path

from .base import Agent, Blackboard
from .ckm import CKM
from .extractors import EXTRACTORS, extract_file
from . import extractors


class ExtractionAgent(Agent):
    name = "extraction"

    def __init__(self, inputs_dir: Path):
        self.inputs_dir = inputs_dir

    def run(self, bb: Blackboard) -> Blackboard:
        # extracted visuals go under the run's outputs/assets folder
        extractors.ASSETS_DIR = bb.workdir / "assets"
        ckm = CKM()
        files = sorted(p for p in self.inputs_dir.iterdir() if p.is_file())
        total = len(files)
        self.log(f"scanning {self.inputs_dir} — {total} file(s) found")
        for idx, path in enumerate(files, start=1):
            if path.suffix.lower() not in EXTRACTORS:
                self.log(f"[{idx}/{total}] skip (unsupported): {path.name}")
                continue
            size_kb = path.stat().st_size / 1024
            self.log(f"[{idx}/{total}] processing {path.name} ({size_kb:.1f} KB)…")

            def _on_block(b, _file=path.name, _i=idx, _t=total):
                preview = " ".join((b.text or "").split())[:80]
                visual = " [img]" if b.image_ref else ""
                self.log(f"    + {b.modality:<11} {b.id} — {b.title or preview}{visual}")

            try:
                blocks = extract_file(path, on_block=_on_block)
            except Exception as exc:  # one bad file shouldn't kill the run
                self.log(f"[{idx}/{total}] FAILED {path.name}: {exc}")
                continue
            if not blocks:
                self.log(f"[{idx}/{total}] {path.name}: no blocks extracted")
                continue
            ckm.sources.append(path.name)
            ckm.blocks.extend(blocks)
            imgs = sum(1 for b in blocks if b.image_ref)
            self.log(f"[{idx}/{total}] {path.name}: {len(blocks)} blocks ({imgs} with visuals)")

        bb.set("ckm", ckm)
        bb.save("ckm.json", ckm)
        total_imgs = sum(1 for b in ckm.blocks if b.image_ref)
        self.log(f"CKM built: {len(ckm.blocks)} blocks from {len(ckm.sources)} sources "
                 f"({total_imgs} visuals)")
        return bb
