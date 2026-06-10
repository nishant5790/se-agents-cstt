# Components

Each component is small and single-purpose. This page explains what each one
does, its inputs/outputs, and how to extend it.

---

## Shared runtime — `base.py`, `llm.py`

```mermaid
flowchart LR
    BB[Blackboard\nshared state + save to outputs/]
    AG[Agent base\nname · log · run]
    LLM[llm.chat_json\nAzure OpenAI + offline fallback]
    AG --> BB
    AG -.optional.-> LLM
```

- **Blackboard** ([base.py](../base.py)) — dict-like shared state with a
  `save()` that serialises Pydantic models to `outputs/*.json`.
- **Agent** ([base.py](../base.py)) — base class; every agent implements `run(bb)`.
- **llm** ([llm.py](../llm.py)) — one `chat_json()` helper. Returns `None` when
  Azure OpenAI isn't configured, so callers fall back to deterministic logic.
  Loads the package-local `.env` and auto-trusts corporate TLS certs.

---

## 1. Extraction Agent — `extraction_agent.py` + `extractors.py`

Scans the inputs folder, dispatches each file to a format extractor by suffix,
and assembles the CKM. One bad file is logged and skipped — it never kills the run.

```mermaid
flowchart TD
    DIR[inputs/ files] --> LOOP{for each file}
    LOOP -->|.xlsx/.xls| EXL[extract_xlsx\none block per row]
    LOOP -->|.pdf| EPD[extract_pdf\none block per page]
    LOOP -->|.txt/.md| ETX[extract_text\nsplit on headings]
    LOOP -->|.mp4/.wav/...| EMD[extract_media\nVosk ASR + cache]
    LOOP -->|other| SKIP[skip + log]
    EXL & EPD & ETX & EMD --> CKM[[CKM\nsources + blocks]]
    CKM --> SAVE[(ckm.json)]
```

- **Registry**: `EXTRACTORS` in [extractors.py](../extractors.py) maps suffix → function.
- **Visuals**: PDFs contribute their largest embedded figure per page; videos
  contribute sampled frames grabbed at transcript timestamps (cap via
  `MEDIA_MAX_FRAMES`, default 60). Both are written to `outputs/assets/` and the
  block's `image_ref` points at them.
- **Media cache**: transcripts (and their frame refs) are cached next to the
  source (`<file>.transcript.json`) so re-runs skip re-transcription.
- **Extend**: add a function returning `list[ContentBlock]` and register it in
  `EXTRACTORS`.

---

## 2. Understanding Agent — `understanding_agent.py`

Turns the flat CKM into a graph of topics, concepts, steps and sources. Blocks
are labelled in **LLM batches** (cheap on large corpora) with a keyword fallback.

```mermaid
flowchart TD
    CKM[[CKM blocks]] --> LBL{label blocks}
    LBL -->|LLM available| BATCH[batch of 25\nllm.chat_json]
    LBL -->|no LLM / over budget| KW[keyword extraction]
    BATCH --> LBLS[(topic, concepts) per block]
    KW --> LBLS
    LBLS --> BUILD[build nodes + edges]
    BUILD --> G[[Knowledge Graph]]
    G --> SAVE[(knowledge_graph.json)]
```

**Graph shape**

```mermaid
graph LR
    SRC[source] -- mentions --> BLK[step/concept block]
    BLK -- part_of --> TOP[topic]
    BLK -- relates_to --> CON[concept]
```

- **Budget knobs** (env): `UNDERSTANDING_BATCH` (default 25),
  `UNDERSTANDING_MAX_LLM_BLOCKS` (default 300). Set the latter to `0` to force
  the offline keyword path.
- **Extend**: change node/edge construction or add new relation types in
  [understanding_agent.py](../understanding_agent.py).

---

## 3. Analysis Agent — `analysis_agent.py`

The "data analysis" brain. Takes a natural-language request + the graph and
produces a concrete `DeckPlan`. Interactive: `analyze()` can be called repeatedly.
It also attaches available visuals from the selected blocks to content slides.

```mermaid
flowchart TD
    REQ([user request]) --> TOPIC[pick topic\nmatch request vs graph topics]
    TOPIC --> SEL[collect blocks for topic\nvia part_of edges]
    REQ --> AUD[infer audience\nbeginner/exec/...]
    SEL --> PLAN{plan}
    AUD --> PLAN
    PLAN -->|LLM| LP[LLM slide plan\ngrounded in blocks]
    PLAN -->|fallback| DP[deterministic template]
    LP --> OUT[[DeckPlan]]
    DP --> OUT
    OUT --> SAVE[(deck_plan.json)]
```

- **Grounding**: the LLM is instructed to use source blocks only (no invented facts).
- **Extend**: adjust topic selection, audience detection, or the slide template in
  [analysis_agent.py](../analysis_agent.py).

---

## 4. PPT Agent — `ppt_agent.py`

Renders a `DeckPlan` into a `.pptx`. `python-pptx` is its tool; each slide layout
is a small method so new styles are easy to add. Uses `templates/brand.pptx` as
the master if present.

```mermaid
flowchart TD
    PLAN[[DeckPlan]] --> PRS[new Presentation\nbrand.pptx or default]
    PRS --> LOOP{for each slide}
    LOOP -->|index 0| TITLE[_title_slide]
    LOOP -->|else| CONTENT[_content_slide\nbullets + speaker notes]
    CONTENT -->|slide.image set| IMG[_place_image\nright half of slide]
    TITLE & CONTENT --> SAVE[(DECK--*.pptx)]
```

- **Images**: when a `SlidePlan.image` is set, bullets are narrowed to the left
  half and the visual is placed on the right.
- **Extend**: add a `_xxx_slide()` method and call it from `build()` in
  [ppt_agent.py](../ppt_agent.py).

---

## Orchestrator — `orchestrator.py` + `run.py`

Owns the Blackboard and wires the four agents into the two phases.

```mermaid
stateDiagram-v2
    [*] --> BuildKnowledge
    BuildKnowledge --> Ready: Extraction + Understanding
    Ready --> Authoring: author(request)
    Authoring --> Ready: deck written
    Ready --> [*]: quit
```

- `build_knowledge()` runs Phase A once.
- `author(request)` runs Phase B and returns the deck path; call it repeatedly.
- [run.py](../run.py) is the CLI: `--request` for one-shot, or omit it for an
  interactive prompt loop.
