# Agent Team — multi-format content → learning material

A small,  modular team of agents for ContentSync Training Tetris Orchestrator that, **on the fly**, ingests mixed source
content (xlsx, pdf, video, text), normalises it into a **Canonical Knowledge
Model (CKM)**, builds a **Knowledge Graph**, then generates PowerPoint learning
material from natural-language requests — all coordinated by an **Orchestrator**.
Visuals (video frames, PDF figures) are extracted into the CKM and placed on the
matching slides.

> **Full documentation with diagrams:** see [docs/README.md](docs/README.md) —
> [architecture](docs/architecture.md) · [components](docs/components.md) ·
> [data model](docs/data-model.md) · [usage](docs/usage.md).
>
> **Azure provisioning guide for DevOps:** [azure services](docs/azure-services-devops.md).

```
 inputs/ (xlsx · pdf · mp4 · txt …)
        │
        ▼
 ┌──────────────────┐   Phase A: build knowledge
 │ ExtractionAgent  │ ──► CKM (ckm.json)            format-agnostic blocks
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐
 │ UnderstandingAgent│──► Knowledge Graph (knowledge_graph.json)
 └──────────────────┘     topics · concepts · steps · sources
        │
        ▼  ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
        │   Phase B: author material (repeat per request)
 ┌──────────────────┐
 │  AnalysisAgent   │ ──► DeckPlan (deck_plan.json)  user request → slide plan
 └──────────────────┘
        │
        ▼
 ┌──────────────────┐
 │    PptAgent      │ ──► outputs/DECK--*.pptx       tool: python-pptx
 └──────────────────┘

         all coordinated by  Orchestrator
```

## The team

| Agent | File | Responsibility |
|---|---|---|
| **Extraction** | [extraction_agent.py](extraction_agent.py) | dispatch each file to a format extractor → CKM |
| **Understanding** | [understanding_agent.py](understanding_agent.py) | CKM → knowledge graph (topics/concepts/steps) |
| **Analysis** | [analysis_agent.py](analysis_agent.py) | user request + graph → `DeckPlan` |
| **PPT** | [ppt_agent.py](ppt_agent.py) | `DeckPlan` → branded `.pptx` |
| **Orchestrator** | [orchestrator.py](orchestrator.py) | runs Phase A once, Phase B per request |

Shared pieces: [ckm.py](ckm.py) (CKM + graph schema), [extractors.py](extractors.py)
(per-format extractors), [llm.py](llm.py) (optional Azure OpenAI, offline fallback),
[base.py](base.py) (Blackboard + Agent base).

## Setup

```powershell
cd "agent_team"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # optional — fill Azure OpenAI for higher quality
```

> The team runs **fully offline** without Azure: understanding falls back to
> keyword extraction and deck planning to a deterministic template. Add Azure
> OpenAI credentials in `.env` for LLM-authored topics and slides.
> Video/audio needs the Vosk model (reused from `../tetris_mvp/models`).

## Use

1. Drop any mix of files into `inputs/`.
2. One-shot:

   ```powershell
   python -m agent_team.run --request "beginner deck on opportunity creation"
   ```

3. Interactive (build knowledge once, ask for many decks):

   ```powershell
   python -m agent_team.run
   # > executive summary deck
   # > advanced deck on pricing
   # > quit
   ```

Outputs land in `outputs/`: `ckm.json`, `knowledge_graph.json`,
`deck_plan.json`, and `DECK--*.pptx`.

## Demo UI (chat + sidebar)

A small FastAPI app serves a one-page chat UI with a sidebar to build the
knowledge base, browse topics, and **Create PPT**.

Run locally:

```powershell
pip install -r requirements.txt
uvicorn agent_team.server:app --reload --port 8000
# open http://localhost:8000
```

Or with Docker (from the repo root, `SE Agent/` — starts everything):

```powershell
docker compose -f agent_team/docker-compose.yml up --build
# open http://localhost:8000
```

The container mounts `tetris_mvp/inputs` (sources), `tetris_mvp/models` (Vosk
model), and `agent_team/outputs` (generated decks). Azure OpenAI is optional via
`agent_team/.env`; without it the app runs fully offline.

Flow in the UI: **Build knowledge base** → chat about the content → type a deck
request → **Create PPT** → download the `.pptx`.

## Extending

| Goal | Where |
|---|---|
| New source format | add a function + entry in `extractors.py::EXTRACTORS` |
| New graph relation/node type | edit `understanding_agent.py` |
| Different slide planning logic | edit `analysis_agent.py` |
| New slide layout | add a `_..._slide` method in `ppt_agent.py` |
| Swap transcription engine | replace `extract_media` in `extractors.py` |
