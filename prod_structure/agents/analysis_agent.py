"""Agent 3 — Data Analysis. Turns a user request + the knowledge graph into a
concrete deck plan (a `DeckPlan`): which topic, which blocks, what slides.

It is "interactive": `analyze(request)` takes the user's natural-language ask
(e.g. "make a beginner deck on opportunity creation") and selects the relevant
slice of the graph. The orchestrator can call it repeatedly for a chat loop.
"""
from __future__ import annotations

import re

from agent_team.core import llm
from agent_team.core.base import Agent, Blackboard
from agent_team.core.ckm import CKM, KnowledgeGraph
from pydantic import BaseModel, Field


class SlidePlan(BaseModel):
    title: str
    bullets: list[str] = Field(default_factory=list)
    notes: str = ""
    image: str | None = None          # path to a visual to place on the slide


class DeckPlan(BaseModel):
    deck_title: str
    topic: str
    audience: str = "general"
    slides: list[SlidePlan] = Field(default_factory=list)


class AnalysisAgent(Agent):
    name = "analysis"

    def run(self, bb: Blackboard) -> Blackboard:
        request = bb.get("request", "Create an overview training deck.")
        plan = self.analyze(bb, request)
        bb.set("deck_plan", plan)
        bb.save("deck_plan.json", plan)
        self.log(f"plan '{plan.deck_title}' — {len(plan.slides)} slides on topic '{plan.topic}'")
        return bb

    # --- core ---
    def analyze(self, bb: Blackboard, request: str) -> DeckPlan:
        ckm: CKM = bb.get("ckm")
        graph: KnowledgeGraph = bb.get("graph")
        topics: list[str] = bb.get("topics", [])
        topic = self._pick_topic(request, topics)
        blocks = self._blocks_for_topic(ckm, graph, topic)
        audience = self._audience(request)

        plan = self._plan_with_llm(request, topic, audience, blocks)
        if plan is None:
            plan = self._plan_deterministic(topic, audience, blocks)
        self._attach_images(plan, blocks, ckm)
        return plan

    def _pick_topic(self, request: str, topics: list[str]) -> str:
        req = request.lower()
        for t in topics:
            if t.lower() in req:
                return t
        # token overlap
        rtokens = set(re.findall(r"[a-z0-9]+", req))
        best, score = (topics[0] if topics else "General"), 0
        for t in topics:
            s = len(rtokens & set(re.findall(r"[a-z0-9]+", t.lower())))
            if s > score:
                best, score = t, s
        return best

    def _audience(self, request: str) -> str:
        r = request.lower()
        for level in ("beginner", "intermediate", "advanced", "executive"):
            if level in r:
                return level
        return "general"

    def _blocks_for_topic(self, ckm: CKM, graph: KnowledgeGraph, topic: str):
        tid = f"topic::{re.sub(r'[^a-z0-9]+', '-', topic.lower()).strip('-')}"
        block_ids = {e.source.replace("blk::", "")
                     for e in graph.edges if e.target == tid and e.relation == "part_of"}
        blocks = [b for b in ckm.blocks if b.id in block_ids]
        return blocks or ckm.blocks[:20]

    _STOPWORDS = {
        "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
        "is", "are", "be", "this", "that", "it", "as", "by", "at", "from",
        "you", "your", "we", "our", "will", "can", "how", "what", "when",
        "into", "step", "steps", "slide", "guide", "create", "creating",
    }

    @classmethod
    def _tokens(cls, text: str) -> set[str]:
        return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                if len(w) > 2 and w not in cls._STOPWORDS}

    def _attach_images(self, plan: "DeckPlan", blocks, ckm: CKM | None = None) -> None:
        """Attach a visual to each content slide whose transcript/source text is
        most relevant to that slide's content (title + bullets + notes), so the
        image logically matches the slide. Each image is used at most once.
        Prefers the topic's own visuals, falling back to any CKM visual."""
        import os

        def candidates(pool):
            seen: set[str] = set()
            out: list[tuple[str, set[str]]] = []
            for b in pool:
                ref = getattr(b, "image_ref", None)
                if ref and ref not in seen and os.path.exists(ref):
                    seen.add(ref)
                    toks = self._tokens(f"{b.title} {b.text}")
                    out.append((ref, toks))
            return out

        cands = candidates(blocks)
        if not cands and ckm is not None:
            cands = candidates(ckm.blocks)
        if not cands:
            return

        used: set[str] = set()
        # Pass 1: best semantic match per slide (greedy, unique images).
        for slide in plan.slides[1:]:  # leave the cover slide image-free
            slide_toks = self._tokens(
                f"{slide.title} {' '.join(slide.bullets)} {slide.notes}")
            best_ref, best_score = None, 0
            for ref, toks in cands:
                if ref in used:
                    continue
                score = len(slide_toks & toks)
                if score > best_score:
                    best_ref, best_score = ref, score
            if best_ref is not None:
                slide.image = best_ref
                used.add(best_ref)

        # Pass 2: fill any still-imageless content slides with leftover visuals.
        leftovers = [ref for ref, _ in cands if ref not in used]
        for slide in plan.slides[1:]:
            if slide.image is None and leftovers:
                slide.image = leftovers.pop(0)
                used.add(slide.image)

    def _plan_with_llm(self, request, topic, audience, blocks):
        corpus = "\n".join(f"- {b.title}: {b.text[:200]}" for b in blocks[:40])
        data = llm.chat_json(
            system="You are an instructional designer. Build a slide deck plan from "
                   "the provided source blocks ONLY (don't invent facts). Reply JSON: "
                   '{"deck_title": str, "slides": [{"title": str, "bullets": [str], '
                   '"notes": str}]}. 5-9 slides incl. title + summary.',
            user=f"USER REQUEST: {request}\nTOPIC: {topic}\nAUDIENCE: {audience}\n"
                 f"SOURCE BLOCKS:\n{corpus}",
        )
        if not data or not data.get("slides"):
            return None
        slides = [SlidePlan(title=s.get("title", ""),
                            bullets=[str(x) for x in s.get("bullets", [])],
                            notes=s.get("notes", "")) for s in data["slides"]]
        return DeckPlan(deck_title=data.get("deck_title", f"{topic} — Training"),
                        topic=topic, audience=audience, slides=slides)

    def _plan_deterministic(self, topic, audience, blocks) -> DeckPlan:
        slides = [SlidePlan(title=f"{topic}", bullets=[f"Audience: {audience}",
                                                       f"Source blocks: {len(blocks)}"])]
        slides.append(SlidePlan(
            title="Learning Objectives",
            bullets=[f"Understand {topic}", "Follow the key steps", "Apply it in practice"]))
        # chunk blocks into content slides
        chunk: list[str] = []
        for b in blocks:
            line = b.text.strip().replace("\n", " ")
            if line:
                chunk.append(line[:160])
            if len(chunk) == 5:
                slides.append(SlidePlan(title=f"{topic} — details", bullets=chunk))
                chunk = []
        if chunk:
            slides.append(SlidePlan(title=f"{topic} — details", bullets=chunk))
        slides.append(SlidePlan(title="Summary & Next Steps",
                                bullets=["Recap of key points", "Practice on your own",
                                         "Where to get help"]))
        return DeckPlan(deck_title=f"{topic} — Training Deck", topic=topic,
                        audience=audience, slides=slides[:9])
