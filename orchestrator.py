"""Orchestrator — manages the agent team end to end.

Phase A (build knowledge):   ExtractionAgent -> UnderstandingAgent
Phase B (author material):   AnalysisAgent -> PptAgent   (repeatable per request)

The knowledge base (CKM + graph) is built once; you can then ask for many decks.
"""
from __future__ import annotations

from pathlib import Path

from .analysis_agent import AnalysisAgent
from .base import Agent, Blackboard
from .extraction_agent import ExtractionAgent
from .ppt_agent import PptAgent
from .understanding_agent import UnderstandingAgent


class Orchestrator(Agent):
    name = "orchestrator"

    def __init__(self, inputs_dir: Path, out_dir: Path, template: Path | None = None):
        self.inputs_dir = inputs_dir
        self.out_dir = out_dir
        self.bb = Blackboard(workdir=out_dir)
        self.extraction = ExtractionAgent(inputs_dir)
        self.understanding = UnderstandingAgent()
        self.analysis = AnalysisAgent()
        self.ppt = PptAgent(out_dir, template)

    def build_knowledge(self) -> Blackboard:
        """Phase A — run once to (re)build CKM + knowledge graph from inputs."""
        self.log("Phase A: building knowledge base")
        self.extraction.run(self.bb)
        self.understanding.run(self.bb)
        return self.bb

    def author(self, request: str) -> str:
        """Phase B — generate one deck for a natural-language request."""
        self.log(f"Phase B: authoring deck for request: {request!r}")
        self.bb.set("request", request)
        self.analysis.run(self.bb)
        self.ppt.run(self.bb)
        return self.bb.get("pptx_path")

    def run(self, bb: Blackboard | None = None) -> Blackboard:
        self.build_knowledge()
        self.author(self.bb.get("request", "Create an overview training deck."))
        return self.bb
