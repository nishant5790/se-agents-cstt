"""Agent team — modular agents over a shared Blackboard.

Phase A (build knowledge):   ExtractionAgent -> UnderstandingAgent
Phase B (author material):   AnalysisAgent -> PptAgent
"""
from __future__ import annotations

from .analysis_agent import AnalysisAgent, DeckPlan, SlidePlan
from .extraction_agent import ExtractionAgent
from .orchestrator import Orchestrator
from .ppt_agent import PptAgent
from .understanding_agent import UnderstandingAgent

__all__ = [
    "AnalysisAgent",
    "DeckPlan",
    "SlidePlan",
    "ExtractionAgent",
    "Orchestrator",
    "PptAgent",
    "UnderstandingAgent",
]
