"""CLI entry point for the agent team.

Examples:
  python -m agent_team.run --inputs ./inputs --request "beginner deck on opportunity creation"
  python -m agent_team.run                      # builds knowledge, then interactive prompt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .agents.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Agent team: sources -> CKM -> graph -> PPTX")
    ap.add_argument("--inputs", type=Path, default=ROOT / "agent_team" / "inputs",
                    help="folder of source files (xlsx/pdf/video/text)")
    ap.add_argument("--out", type=Path, default=ROOT / "agent_team" / "outputs")
    ap.add_argument("--template", type=Path, default=ROOT / "agent_team" / "templates" / "brand.pptx")
    ap.add_argument("--request", type=str, default=None,
                    help="natural-language deck request; omit for interactive mode")
    args = ap.parse_args()

    args.inputs.mkdir(parents=True, exist_ok=True)
    orch = Orchestrator(args.inputs, args.out, args.template)
    orch.build_knowledge()

    if args.request:
        path = orch.author(args.request)
        print(f"\nDeck ready: {path}")
        return

    print("\nKnowledge base ready. Available topics:",
          ", ".join(orch.bb.get("topics", [])) or "(none)")
    print("Type a deck request (or 'quit'):")
    while True:
        try:
            req = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not req or req.lower() in {"quit", "exit"}:
            break
        path = orch.author(req)
        print(f"Deck ready: {path}")


if __name__ == "__main__":
    main()
