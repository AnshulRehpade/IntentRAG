# IntentRAG v2

**Intent-Aware Retrieval-Augmented Generation with Self-Healing Pipeline**

A production-ready RAG system that classifies user intent before retrieval, routes to specialized indexes, detects hallucinations, and automatically retries with corrective strategies.

> 18 tests passing · 6 intent categories · 3-method hallucination detection · auto-healing with max 2 retries

---

## Demo

### Sample Query Response

```bash
POST /query
{"query": "Who created Python?"}
```

```json
{
  "success": true,
  "data": {
    "intent": "person",
    "intent_confidence": 0.97,
    "answer": "Python was created by Guido van Rossum. He began working on it in the late 1980s at Centrum Wiskunde & Informatica (CWI) in the Netherlands. [1]",
    "hallucination_check": {
      "is_hallucinated": false,
      "confidence_score": 0.87,
      "llm_verdict": "SUPPORTED"
    },
    "healing": {
      "attempts": 1,
      "was_healed": false,
      "strategies_used": []
    },
    "latency_ms": 1250
  }
}
```

### Self-Healing in Action

```json
{
  "healing": {
    "attempts": 2,
    "was_healed": true,
    "strategies_used": ["strict_grounding"],
    "original_answer": "Python was created in 1989 by Guido...(fabricated date)",
    "improvement_reason": "Strategy 'strict_grounding' resolved the hallucination"
  }
}
```

### Swagger UI

With the server running: **http://localhost:8000/docs**

All endpoints are interactive — register, login, ingest documents, and query directly from the browser.

---

## Architecture

```
Client (React / Postman / Swagger)
│
▼
FastAPI  (/query, /ingest, /health, /eval, /auth, /analytics)
│
▼
RoBERTa Intent Classifier
(factual / person / time / location / explanation / other)
│
▼
LlamaIndex VectorStoreIndex → Qdrant (filtered by tenant_id)
│
┌───────┬───────┬──────┬──────────┬───────┐
factual person  time  location  explan.  other
└───────┴───────┴──────┴──────────┴───────┘
│
▼
Cohere Reranker (rerank-v3.5)
│
▼
OpenAI GPT-4o-mini (intent-specific system prompts)
│
▼
Hallucination Checker (confidence + entropy + LLM verification)
│
├── PASS → Return answer
└── FAIL → Self-Healing: diagnose cause → retry (max 2x)
            ├── Poor retrieval    → expand_retrieval (2x chunks)
            ├── Fabrication       → strict_grounding (temp=0.1)
            ├── High entropy      → reduce_uncertainty (fewer chunks)
            └── 2nd retry         → broaden_search (adjacent intents)
│
▼
LangFuse Trace + PostgreSQL Log + Response
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Intent Classifier | Fine-tuned RoBERTa-base (trained on TREC + SQuAD + SciQ, ~91% accuracy) |
| Vector Store | Qdrant (1 collection per intent, tenant_id metadata filter) |
| Retrieval | LlamaIndex VectorStoreIndex |
| Reranking | Cohere rerank-v3.5 |
| LLM | OpenAI GPT-4o-mini |
| Hallucination Detection | Confidence + entropy (logprobs) + LLM second opinion |
| Self-Healing | Auto-retry with strategy selection based on failure diagnosis |
| Observability | LangFuse (per-step tracing with latency + cost) |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |
| Database | PostgreSQL (tenants, users, query logs, documents) |
| Auth | JWT with RBAC (admin / writer / reader) |

---

## Project Structure

```
IntentRAG/
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── core/                   # Config, DB, auth, security
│   ├── models/                 # SQLAlchemy ORM (Tenant, User, QueryLog, Document)
│   ├── routers/                # API endpoints (auth, query, ingest, eval, analytics, health)
│   ├── schemas/                # Pydantic response models
│   └── services/               # Business logic (classifier, retriever, reranker,
│                               #   generator, hallucination, self_healing, tracing, eval)
├── data/                       # Sample documents (6 intent categories) + eval test set
├── scripts/                    # DB init, seed, ingestion, evaluation, integration tests
├── alembic/                    # Database migrations
├── v1_legacy/                  # Original v1 prototype code (preserved for reference)
├── train_intent_classifier_roberta.py  # Model training script
├── docker-compose.yml          # PostgreSQL + Qdrant
├── requirements.txt            # All dependencies
└── .env.example                # Environment variables (documented below)
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for PostgreSQL and Qdrant)
- OpenAI API key (required for generation + hallucination checks)

### 2. Setup

```bash
cd IntentRAG
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Fill in API keys (see Environment Variables section below)
```

### 3. Start Services

```bash
docker compose up -d          # PostgreSQL + Qdrant
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Seed & Ingest

```bash
python scripts/seed_db.py              # Create demo tenant + admin user
python scripts/ingest_sample_data.py   # Ingest 9 sample documents
```

### 5. Query

```bash
# Login
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@demo.com","password":"password123"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['data']['token'])")

# Query
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "How does the self-attention mechanism work?"}'
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | Public | System health (API + Postgres + Qdrant) |
| POST | `/auth/register` | Public | Register user + tenant |
| POST | `/auth/login` | Public | Get JWT token |
| GET | `/auth/me` | Any | Current user info |
| POST | `/ingest` | Admin, Writer | Upload → chunk → embed → store |
| POST | `/query` | Any | Full self-healing RAG pipeline |
| POST | `/eval` | Admin | RAGAS evaluation on uploaded test set |
| GET | `/analytics/hallucinations` | Admin | Hallucination pattern analysis |

---

## Testing

```bash
# Unit tests — no Docker needed (18/18 passing)
python scripts/test_tenant_isolation_unit.py

# Full integration test — requires Docker + running API
python scripts/test_integration.py

# RAGAS evaluation — requires Docker + running API + ingested data
python scripts/run_evaluation.py
```

**Test coverage:**
- JWT token creation, validation, expiry
- Role-based access control (admin/writer/reader × all endpoints)
- Multi-tenant isolation (tenant A cannot see tenant B's data)
- Token security (invalid, expired, missing tokens rejected)
- Public vs protected endpoint separation
- Request validation (Pydantic)
- Self-healing strategy selection logic

---

## Intent Classifier

Trained on TREC + SQuAD + SciQ datasets using RoBERTa-base, achieving ~91% weighted F1 on held-out test set.

```bash
# Train (outputs to ./intent_classifier_model_roberta/)
python train_intent_classifier_roberta.py

# The API auto-loads the model on first request
# If model directory is missing, falls back to keyword heuristic
```

| Class | Label | Example |
|-------|-------|---------|
| factual | 0 | "What is machine learning?" |
| person | 1 | "Who invented the telephone?" |
| time | 2 | "When was Python released?" |
| location | 3 | "Where is Google headquartered?" |
| explanation | 4 | "How does RAG work?" |
| other | 5 | "Tell me something interesting" |

---

## Environment Variables

```bash
# ─── Required ───────────────────────────────────────────────
OPENAI_API_KEY=sk-...
# Get from: https://platform.openai.com/api-keys
# Cost: ~$0.01 per query (gpt-4o-mini)

DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/intentrag
# Auto-configured by docker-compose.yml — no action needed locally

QDRANT_HOST=localhost
QDRANT_PORT=6333
# Auto-configured by docker-compose.yml — no action needed locally

JWT_SECRET_KEY=your-random-secret-here
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"

# ─── Optional (features degrade gracefully without these) ───
COHERE_API_KEY=...
# Get from: https://dashboard.cohere.com/api-keys (free tier: 1000 calls/month)
# Without it: reranking is skipped, original retrieval order used

LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
# Get from: https://cloud.langfuse.com (free tier available)
# Without it: tracing is a no-op, zero overhead

INTENT_MODEL_PATH=
# Leave empty to use default: ./intent_classifier_model_roberta
# Without model: falls back to keyword-based heuristic classifier
```

---

## Self-Healing Pipeline

When the hallucination checker flags an answer, the system automatically diagnoses and retries:

| Cause | Strategy | What Changes |
|-------|----------|-------------|
| Poor retrieval scores | `expand_retrieval` | top_k doubled to 20, rerank_top_n to 8 |
| Fabricated claims | `strict_grounding` | temp=0.1, "cite sources only" prompt |
| High entropy tokens | `reduce_uncertainty` | Fewer chunks (top 3 only), strict prompt |
| 2nd retry still fails | `broaden_search` | Also queries adjacent intent categories |

- Max 2 retries per query (configurable)
- `"self_heal": false` in request body disables retries
- Healing metadata in response shows what strategy fixed the issue

---

## Hallucination Analytics

`GET /analytics/hallucinations` surfaces patterns:

- Hallucination rate by intent category
- Which intents need more knowledge base coverage
- Latency cost of self-healing retries
- Recent hallucinated queries for debugging
- Auto-generated insights ("explanation intent hallucinates 3x more — add more docs")

---

## Multi-Tenancy

- Every document tagged with `tenant_id` in Qdrant metadata
- Queries always filter by `tenant_id` from JWT — zero cross-tenant leakage
- First user registering a tenant auto-becomes admin
- Roles: **admin** (all access) > **writer** (query + ingest) > **reader** (query only)

---

## Swagger Docs

With the server running: **http://localhost:8000/docs**
