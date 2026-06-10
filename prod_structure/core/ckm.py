"""Canonical Knowledge Model (CKM) and Knowledge Graph schemas.

The CKM is a format-agnostic representation: whatever the source (xlsx, pdf,
video, text), extraction normalises it into a flat list of `ContentBlock`s.
Every downstream agent works only against the CKM / graph — never raw files.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Frame(BaseModel):
    """A single described keyframe extracted from a video segment."""

    timestamp: float                  # seconds into the source media
    image_ref: str                    # path to the saved frame image
    visual_description: str | None = None  # GPT-4o caption / on-screen text


class ContentBlock(BaseModel):
    """One atomic, source-agnostic piece of knowledge."""

    id: str
    source: str                       # originating filename
    modality: str                     # text | table_row | transcript | heading | step
    title: str = ""
    text: str = ""
    timestamp: float | None = None    # seconds, for video/transcript
    image_ref: str | None = None      # primary visual (first frame / pdf figure)
    frames: list[Frame] = Field(default_factory=list)  # all kept frames (video)
    metadata: dict = Field(default_factory=dict)


class CKM(BaseModel):
    """The Canonical Knowledge Model: all content blocks from all sources."""

    sources: list[str] = Field(default_factory=list)
    blocks: list[ContentBlock] = Field(default_factory=list)


# ---------- Knowledge Graph ----------

class Node(BaseModel):
    id: str
    type: str                         # topic | concept | entity | step | source
    label: str
    properties: dict = Field(default_factory=dict)


class Edge(BaseModel):
    source: str                       # node id
    target: str                       # node id
    relation: str                     # mentions | part_of | precedes | relates_to


class KnowledgeGraph(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def neighbors(self, node_id: str) -> list[str]:
        out = [e.target for e in self.edges if e.source == node_id]
        out += [e.source for e in self.edges if e.target == node_id]
        return sorted(set(out))

    def nodes_of_type(self, type_: str) -> list[Node]:
        return [n for n in self.nodes if n.type == type_]
