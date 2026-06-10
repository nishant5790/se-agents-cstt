# Azure Services Guide for DevOps Provisioning

This document describes the Azure services needed for this application, based on the current codebase and near-term roadmap.

It is written for DevOps and platform teams who need to provision Azure resources, networking, identities, secrets, monitoring, and deployment paths.

## 1) System summary

The application is a Python agent pipeline that:

1. Ingests files (xlsx, pdf, video, text)
2. Builds a Canonical Knowledge Model (CKM)
3. Builds a Knowledge Graph
4. Generates PowerPoint decks from user requests
5. Exposes a FastAPI web API and browser UI

Current runtime architecture is primarily file-based (local inputs/outputs), with optional Azure AI integration.

## 2) Azure service matrix (required vs optional)

| Service | Required now | Purpose in this system | Code status |
|---|---|---|---|
| Azure OpenAI | Optional but strongly recommended | Better topic extraction, chat answers, slide planning quality | Actively wired in llm.py |
| Azure Document Intelligence | Optional (recommended for scanned/complex PDFs) | OCR + layout/table extraction for PDF and office docs | Not implemented in current extraction path |
| Azure Video Indexer | Optional (recommended for production media) | Video transcription + OCR + visual insights for media extraction | Implemented in azure_extractors.py but not default path |
| Azure AI Speech | Optional fallback | Audio transcription fallback path for media | Implemented in azure_extractors.py |
| Azure AI Vision | Optional (future extension) | Advanced frame/image analysis | Env vars present, not actively used in logic |
| Azure AI Search | Not required today | Could index CKM/graph for scalable retrieval | Not implemented in current code |
| Azure AI Foundry | Not required today | Could host/evaluate agent workflows and governance | Not implemented in current code |
| Azure Key Vault | Recommended for production | Secret management for API keys/endpoints | Not yet wired in app code |
| Azure App Service or Azure Container Apps | Recommended for production hosting | Host FastAPI API/UI | Not fixed yet (both viable) |
| Azure Monitor + Application Insights | Recommended for production | Logs, traces, health, alerting | Not yet wired in app code |
| Azure Storage Account | Optional (recommended with cloud deployment) | Persistent inputs/outputs/assets and backups | Not wired in code |
| Azure Container Registry (ACR) | Required if container deployment selected | Store container images | Needed for container-based deployment |

## 3) Recommended target architecture

## 3.1 Baseline production (Phase 1)

1. Host app in Azure App Service (Linux) or Azure Container Apps
2. Use Azure OpenAI for LLM quality improvements
3. Enable Azure Document Intelligence for robust PDF/OCR extraction
4. Keep local/offline extraction fallback enabled
5. Store secrets in Key Vault (or App Service secret settings initially)
6. Enable Application Insights and Log Analytics

## 3.2 Media scale-up (Phase 2)

1. Enable Azure Video Indexer for video-heavy workloads
2. Enable Azure AI Speech fallback for audio-only assets
3. Move large media and outputs to Blob Storage
4. Add queue-based async processing if needed

## 3.3 Enterprise retrieval/governance (Phase 3)

1. Add Azure AI Search for CKM/graph retrieval at scale
2. Add Azure AI Foundry for evaluation and lifecycle governance
3. Add managed identity-based auth across services

## 4) Service-by-service provisioning details

## 4.1 Azure OpenAI (recommended)

Purpose:
- Improve output quality for UnderstandingAgent and chat/planning logic

Used by:
- llm.py via Azure OpenAI Chat Completions

Environment variables expected by app:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION (default in code: 2024-10-21)
- AZURE_OPENAI_DEPLOYMENT (default in code: gpt-4o)

DevOps actions:
1. Provision Azure OpenAI resource in a supported region
2. Create model deployment named to match AZURE_OPENAI_DEPLOYMENT
3. Provide endpoint and key via secret store
4. Restrict network access where possible
5. Set usage quotas and monitoring alerts

Minimum RBAC guidance:
- Runtime identity: Cognitive Services OpenAI User (if moving to Entra auth)
- CI/CD identity: Contributor at resource-group scope only

## 4.2 Azure Video Indexer (optional, production media)

## 4.2 Azure Document Intelligence (optional, recommended for document-heavy workloads)

Purpose:
- Improve extraction quality for scanned PDFs and layout-heavy documents
- Extract tables, paragraphs, headings, and structure for better CKM quality

Current code status:
- Document extraction currently uses pypdf-based local extraction path in extractors.py
- Azure Document Intelligence integration is not wired yet

Environment variables to reserve now:
- AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
- AZURE_DOCUMENT_INTELLIGENCE_KEY
- AZURE_DOCUMENT_INTELLIGENCE_MODEL (optional, default prebuilt-layout)

DevOps actions:
1. Provision Document Intelligence resource in target regions
2. Enable private networking and key rotation policy
3. Set service quotas and cost alerts
4. Provide endpoint/key via Key Vault-backed app settings

When this becomes effectively required:
- If source files include scanned PDFs, image-only pages, or structured forms where local text extraction is insufficient

## 4.3 Azure Video Indexer (optional, production media)

Purpose:
- Replace local Vosk transcription path with cloud video analysis
- Extract transcript and OCR with richer metadata

Used by:
- azure_extractors.py (VideoIndexerClient)

Environment variables expected by code:
- AZURE_VIDEO_INDEXER_ACCOUNT_ID
- AZURE_VIDEO_INDEXER_API_KEY
- AZURE_VIDEO_INDEXER_LOCATION

Important code observation:
- extraction_agent.py currently calls extractors.py default extract_media path
- Azure media extraction exists but is not yet wired as the default runtime path

DevOps actions:
1. Provision Video Indexer-capable cognitive resource/account
2. Create dedicated key rotation policy
3. Set cost guardrails for long video processing
4. Validate region and API endpoint compatibility with code settings

## 4.4 Azure AI Speech (optional fallback)

Purpose:
- Audio transcription fallback and cost-optimized speech path

Used by:
- azure_extractors.py (SpeechServiceClient)

Environment variables:
- AZURE_SPEECH_KEY
- AZURE_SPEECH_REGION

DevOps actions:
1. Provision Speech resource
2. Configure quotas and rate limits
3. Track usage by environment tag (dev/uat/prod)

## 4.5 Azure AI Vision (optional, future)

Purpose:
- Potential advanced frame/image enrichment

Code status:
- Credentials are defined but no active request path currently uses Vision APIs

Environment variables:
- AZURE_VISION_KEY
- AZURE_VISION_ENDPOINT

DevOps recommendation:
- Do not provision until feature scope is confirmed

## 4.6 Azure AI Search (future roadmap)

Purpose in future architecture:
- Index CKM blocks and graph metadata for semantic/hybrid retrieval
- Improve grounded chat and deck planning over larger corpora

Current status:
- Not referenced in code today

When to provision:
- When moving from file-memory retrieval to service-based retrieval

Proposed index shape (future):
- id
- source
- modality
- title
- text
- topic
- concepts
- timestamp
- image_ref
- metadata_json

## 4.7 Azure AI Foundry (future roadmap)

Purpose in future architecture:
- Agent lifecycle management, evaluation, prompt iteration, governance

Current status:
- Not integrated in this repo today

When to provision:
- When you want managed evaluation pipelines, centralized model governance, and agent ops controls

## 4.8 Hosting service choice

Option A: Azure App Service (Web App for Containers or Python runtime)
- Best for straightforward web app hosting
- Good fit for current FastAPI API/UI with moderate scale

Option B: Azure Container Apps
- Best for container-first deployments and burst scaling
- Better fit if adding queue workers for extraction jobs

Decision recommendation:
- Start with App Service if your team prefers simple operations
- Choose Container Apps if async workers/eventing are planned soon

## 4.9 Secrets and identity

Recommended production pattern:

1. Store secrets in Key Vault
2. Assign managed identity to hosting service
3. Grant least-privilege access to Key Vault secrets
4. Remove raw keys from plain .env in deployed environments

Current code note:
- Code reads environment variables directly
- Key Vault references should be injected through host config or a small config layer

## 4.10 Monitoring and operations

Provision and configure:

1. Application Insights (request telemetry, exceptions)
2. Log Analytics workspace
3. Availability checks for health endpoints
4. Alerts for:
   - Error rate spikes
   - High response latency
   - Container restarts/crash loops
   - Token or quota exhaustion (Azure OpenAI)
   - Video indexing failure/timeouts

## 5) Environment variable contract for DevOps

Core app:
- INPUTS_DIR
- OUTPUTS_DIR

Azure OpenAI:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_API_VERSION
- AZURE_OPENAI_DEPLOYMENT

Azure Document Intelligence:
- AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
- AZURE_DOCUMENT_INTELLIGENCE_KEY
- AZURE_DOCUMENT_INTELLIGENCE_MODEL

Azure media services:
- AZURE_VIDEO_INDEXER_ACCOUNT_ID
- AZURE_VIDEO_INDEXER_API_KEY
- AZURE_VIDEO_INDEXER_LOCATION
- AZURE_SPEECH_KEY
- AZURE_SPEECH_REGION
- AZURE_VISION_KEY
- AZURE_VISION_ENDPOINT

Non-Azure media fallback:
- VOSK_MODEL_PATH
- MEDIA_MAX_FRAMES

## 6) Network and security controls

Recommended controls:

1. Separate subscriptions or resource groups per environment (dev/uat/prod)
2. Restrict inbound access to app endpoints (WAF or APIM in front if required)
3. Private endpoint strategy for Key Vault and AI services where supported
4. TLS everywhere, enforce HTTPS-only
5. Secret rotation schedule (30-90 days based on policy)
6. Tagging policy for owner, cost-center, env, data-classification

## 7) Cost planning notes

Primary cost drivers:

1. Azure OpenAI token usage
2. Document Intelligence page processing volume
3. Video Indexer processing minutes
4. Speech transcription hours
5. Hosting compute and storage egress

Cost controls:

1. Separate cost budgets and alerts by environment
2. Rate-limit or batch long-running media jobs
3. Keep offline fallback for non-critical workloads
4. Auto-scale ceilings to avoid runaway spend

## 8) DevOps implementation checklist

## 8.1 Must-have for first production rollout

1. Provision hosting target (App Service or Container Apps)
2. Provision Azure OpenAI + model deployment
3. Provision Azure Document Intelligence (if PDFs are core input)
4. Configure environment variables in host settings
5. Configure secure secret storage (Key Vault recommended)
6. Enable Application Insights and baseline alerts
7. Configure CI/CD pipeline to deploy image/app

## 8.2 Should-have (if media extraction is important)

1. Provision Video Indexer
2. Provision Speech resource
3. Add storage account for media/output persistence
4. Add job timeout and retry policy controls

## 8.3 Future enterprise capabilities

1. Add Azure AI Search retrieval layer
2. Add Azure AI Foundry evaluation/governance workflows
3. Add managed identity-only auth (remove API keys where possible)

## 9) Service request template for DevOps ticket

Use this as a request payload to platform teams.

Project:
- Agent Team (Content extraction -> knowledge graph -> PPT generation)

Environments:
- dev, uat, prod

Required now:
1. Hosting: Azure App Service (Linux) or Azure Container Apps
2. Azure OpenAI resource with deployment name: gpt-4o (or approved equivalent)
3. Azure Document Intelligence resource (if scanned/complex PDFs are in scope)
4. Key Vault for secret storage
5. Application Insights + Log Analytics

Optional now (if video/audio extraction at scale is in-scope):
1. Azure Video Indexer
2. Azure AI Speech

Not required now but requested for roadmap planning:
1. Azure AI Search
2. Azure AI Foundry

Configuration delivery required:
- Securely provide all AZURE_* environment variables listed in Section 5
- Provide RBAC assignments for runtime identity and CI/CD identity
- Provide resource naming convention and region mapping per environment

Operational requirements:
- Alerts for API failure, latency, quota, and extraction timeout
- Budget alerts and cost reports by environment
- Backup/retention policy for generated outputs

## 10) Reality check against current code

What is already active:
1. Azure OpenAI integration path in llm.py
2. Offline fallback path if Azure is missing

What exists but needs runtime wiring decision:
1. Azure Video Indexer and Speech integration in azure_extractors.py
2. extraction_agent.py currently uses extractors.py registry path by default

What is not implemented yet:
1. Azure AI Search integration
2. Azure AI Foundry integration
3. Azure Document Intelligence integration into extraction pipeline
4. Key Vault SDK-based secret retrieval layer
5. Cloud-native storage integration for inputs/outputs

This means DevOps can provision in phases without blocking current app execution.