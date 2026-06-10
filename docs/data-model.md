# Data Model

All schemas live in [ckm.py](../ckm.py) (Pydantic) and
[analysis_agent.py](../analysis_agent.py) (`DeckPlan`).

## Canonical Knowledge Model (CKM)

The format-agnostic representation every agent works against.

```mermaid
classDiagram
    class CKM {
        +list~str~ sources
        +list~ContentBlock~ blocks
    }
    class ContentBlock {
        +str id
        +str source
        +str modality
        +str title
        +str text
        +float timestamp
        +str image_ref
        +dict metadata
    }
    CKM "1" o-- "many" ContentBlock
```

| Field | Meaning |
|---|---|
| `id` | stable slug, unique per block |
| `source` | originating filename |
| `modality` | `text` · `table_row` · `transcript` · `heading` · `step` |
| `title` | short label |
| `text` | the content |
| `timestamp` | seconds into media (video/audio only) |
| `image_ref` | path to an extracted visual (video frame / pdf figure), or `None` |
| `metadata` | format-specific extras (e.g. xlsx `sheet`, `fields`) |

## Knowledge Graph

```mermaid
classDiagram
    class KnowledgeGraph {
        +list~Node~ nodes
        +list~Edge~ edges
        +neighbors(id)
        +nodes_of_type(type)
    }
    class Node {
        +str id
        +str type
        +str label
        +dict properties
    }
    class Edge {
        +str source
        +str target
        +str relation
    }
    KnowledgeGraph o-- Node
    KnowledgeGraph o-- Edge
```

- **Node types**: `source`, `topic`, `concept`, `step`.
- **Edge relations**: `mentions` (source→block), `part_of` (block→topic),
  `relates_to` (block→concept).
- Node ids are namespaced: `src::`, `topic::`, `concept::`, `blk::`.

## DeckPlan

```mermaid
classDiagram
    class DeckPlan {
        +str deck_title
        +str topic
        +str audience
        +list~SlidePlan~ slides
    }
    class SlidePlan {
        +str title
        +list~str~ bullets
        +str notes
        +str image
    }
    DeckPlan o-- SlidePlan
```

## Artifacts on disk (`outputs/`)

```mermaid
flowchart LR
    E[Extraction] --> A1[ckm.json]
    U[Understanding] --> A2[knowledge_graph.json]
    An[Analysis] --> A3[deck_plan.json]
    P[PPT] --> A4[DECK--*.pptx]
```

| File | Written by | Contents |
|---|---|---|
| `ckm.json` | Extraction | full CKM |
| `knowledge_graph.json` | Understanding | nodes + edges |
| `deck_plan.json` | Analysis | the plan for the last request |
| `DECK--<title>.pptx` | PPT | the generated deck |
| `assets/*.jpg` / `assets/*.png` | Extraction | extracted visuals (video frames, pdf figures) |
| `<media>.transcript.json` | Extraction | cached ASR transcript + frame refs (next to source) |
