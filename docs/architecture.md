# Architecture

## Design principles

1. **One canonical model** — every source format is normalised into the same
   `ContentBlock` list (the CKM). Downstream agents never touch raw files.
2. **Modular agents over a shared Blackboard** — each agent reads/writes a shared
   in-memory `Blackboard` and persists its artifact to `outputs/`. Stages are
   composable and independently re-runnable.
3. **LLM-optional** — every agent has a deterministic fallback, so the whole team
   runs fully offline. Azure OpenAI, when configured, only improves quality.
4. **Config/registry extension points** — new formats and slide layouts are added
   by registering one function, not by rewriting the pipeline.

## Two phases

The orchestrator separates expensive knowledge-building from cheap authoring, so
you build the knowledge base **once** and generate **many** decks.

```mermaid
flowchart TD
    subgraph A[Phase A — build knowledge once]
        direction LR
        EX[Extraction Agent] -->|ckm.json| UN[Understanding Agent]
        UN -->|knowledge_graph.json| KB[(Knowledge Base)]
    end

    subgraph B[Phase B — author per request, repeatable]
        direction LR
        REQ([user request]) --> AN[Analysis Agent]
        KB --> AN
        AN -->|deck_plan.json| PP[PPT Agent]
        PP -->|DECK--*.pptx| DECK[(Deck)]
    end

    A --> B
```

## End-to-end data flow

```mermaid
flowchart LR
    subgraph Sources
        X[xlsx rows]
        P[pdf pages]
        V[video/audio]
        T[text/markdown]
    end

    X --> XE[extract_xlsx]
    P --> PE[extract_pdf]
    V --> VE[extract_media\nVosk ASR]
    T --> TE[extract_text]

    XE & PE & VE & TE --> CKM[[CKM\nContentBlock list]]

    CKM --> LBL[label blocks\nLLM batched / keywords]
    LBL --> KG[[Knowledge Graph\nnodes + edges]]

    KG --> SEL[select topic + blocks]
    REQ([request]) --> SEL
    SEL --> PLAN[[DeckPlan\nslides]]
    PLAN --> RENDER[python-pptx render]
    RENDER --> FILE[(DECK--*.pptx)]
```

## Runtime sequence

```mermaid
sequenceDiagram
    participant U as User / CLI
    participant O as Orchestrator
    participant E as Extraction
    participant Un as Understanding
    participant A as Analysis
    participant P as PPT

    U->>O: build_knowledge()
    O->>E: run(bb)
    E-->>O: CKM (ckm.json)
    O->>Un: run(bb)
    Un-->>O: Knowledge Graph (knowledge_graph.json)

    loop per deck request
        U->>O: author(request)
        O->>A: run(bb)
        A-->>O: DeckPlan (deck_plan.json)
        O->>P: run(bb)
        P-->>O: DECK--*.pptx
        O-->>U: path to deck
    end
```

## Shared runtime

```mermaid
classDiagram
    class Blackboard {
        +Path workdir
        +dict data
        +get(key)
        +set(key, value)
        +save(name, payload)
    }
    class Agent {
        +str name
        +log(msg)
        +run(bb) Blackboard
    }
    Agent <|-- ExtractionAgent
    Agent <|-- UnderstandingAgent
    Agent <|-- AnalysisAgent
    Agent <|-- PptAgent
    Agent <|-- Orchestrator
    Agent ..> Blackboard : reads/writes
```

The `Blackboard` ([base.py](../base.py)) is the single source of truth passed
between agents. Keys used: `ckm`, `graph`, `topics`, `request`, `deck_plan`,
`pptx_path`.
