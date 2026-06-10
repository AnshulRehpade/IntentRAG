# IntentRAG v2

**Intent-Aware Retrieval-Augmented Generation with Self-Healing Pipeline**

A production-ready RAG system that classifies user intent before retrieval, routes to specialized vector indexes, detects hallucinations, and automatically retries with corrective strategies — all with zero paid API costs.

> 18 tests passing · 6 intent categories · 3-method hallucination detection · self-healing with max 2 retries · completely free (Groq + local embeddings)

---

## Live Pipeline Output

```bash
POST /query {"query": "Who created Python?"}
```

```json
{
  "intent": "person",
  "intent_confidence": 0.85,
  "answer": "According to the context, Guido van Rossum, a Dutch programmer, created the Python programming language. He began working on Python in the late 1980s at Centrum Wiskunde & Informatica (CWI) in the Netherlands.",
  "model": "llama-3.3-70b-versatile",
  "hallucination_check": {
    "is_hallucinated": false,
    "confidence_score": 0.67,
    "llm_verdict": "SUPPORTED"
  },
  "healing": {
    "attempts": 1,
    "was_healed": false
  },
  "latency_ms": 2679
}
```

### Self-Healing in Action

When hallucination is detected, the system diagnoses the cause and retries:

```json
{
  "healing": {
    "attempts": 2,
    "was_healed": true,
    "strategies_used": ["strict_grounding"],
    "original_answer": "Python was created in 1989...(fabricated date)",
    "improvement_reason": "Strategy 'strict_grounding' resolved the hallucination"
  }
}
```

---

## Architecture

```
Client (React / Swagger)
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
Groq LLM (llama-3.3-70b-versatile, intent-specific prompts)
│
▼
Hallucination Checker (confidence + entropy + LLM verification)
│
├── PASS → Return answer
└── FAIL → Self-Healing: diagnose → retry (max 2x)
            ├── Poor retrieval    → expand_retrieval (2x chunks)
            ├── Fabrication       → strict_grounding (temp=0.1)
            ├── High entropy      → reduce_uncertainty
            └── 2nd retry         → broaden_search (adjacent intents)
│
▼
LangFuse Trace + PostgreSQL Log + Response
```

## Tech Stack

| Layer | Technology | Cost |
|-------|-----------|------|
| API | FastAPI | Free |
| Intent Classifier | Fine-tuned RoBERTa-base (~91% F1) | Free (local) |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free (local) |
| Vector Store | Qdrant | Free (Docker) |
| Retrieval | LlamaIndex VectorStoreIndex | Free |
| Reranking | Cohere rerank-v3.5 | Free tier (1K/month) |
| LLM | Groq llama-3.3-70b-versatile | Free (14.4K req/day) |
| Observability | LangFuse | Free tier (50K obs/month) |
| Evaluation | RAGAS | Free |
| Database | PostgreSQL | Free (Docker) |
| Auth | JWT with RBAC | Free |
| Frontend | React + Vite + Tailwind | Free |

**Total cost: $0/month** — entire system runs on free tiers.

---

## Project Structure

```
IntentRAG/
├── app/
│   ├── main.py                 # FastAPI app + lifespan
│   ├── core/                   # Config, DB, auth, security
│   ├── models/                 # SQLAlchemy ORM (Tenant, User, QueryLog, Document)
│   ├── routers/                # API endpoints
│   │   ├── auth.py             # /auth/register, /auth/login, /auth/me
│   │   ├── health.py           # /health
│   │   ├── ingest.py           # /ingest
│   │   ├── query.py            # /query (self-healing pipeline)
│   │   ├── eval.py             # /eval (RAGAS)
│   │   └── analytics.py       # /analytics/hallucinations
│   ├── schemas/                # Pydantic models
│   └── services/
│       ├── classifier.py       # RoBERTa intent classification
│       ├── retriever.py        # LlamaIndex + Qdrant
│       ├── ingestion.py        # Chunking + embedding
│       ├── reranker.py         # Cohere reranking
│       ├── generator.py        # Groq LLM generation
│       ├── hallucination.py    # 3-method detection
│       ├── self_healing.py     # Diagnose + retry pipeline
│       ├── tracing.py          # LangFuse instrumentation
│       ├── evaluation.py       # RAGAS metrics
│       └── query_logger.py     # PostgreSQL logging
├── frontend/                   # React + Vite + Tailwind
│   └── src/
│       ├── pages/              # Login, Dashboard, Query, Ingest, Analytics
│       └── components/         # Layout with sidebar nav
├── data/                       # Sample documents (6 categories) + eval test set
├── scripts/                    # DB seed, ingestion, evaluation, integration tests
├── v1_legacy/                  # Original prototype (preserved)
├── train_intent_classifier_roberta.py
├── docker-compose.yml          # PostgreSQL + Qdrant
├── requirements.txt
└── .env.example
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- Node.js 18+ (for frontend)
- Groq API key (free: https://console.groq.com)

### Setup

```bash
# Clone
git clone https://github.com/AnshulRehpade/IntentRAG.git
cd IntentRAG

# Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Edit .env → paste your Groq key as OPENAI_API_KEY=gsk_...

# Start services
docker compose up -d

# Start backend
uvicorn app.main:app --port 8000 --reload

# Seed database + ingest sample documents
python scripts/seed_db.py
python scripts/ingest_sample_data.py

# Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Access

- **Frontend**: http://localhost:3000
- **Swagger API docs**: http://localhost:8000/docs
- **Login**: `admin@demo.com` / `password123`

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

# RAGAS evaluation
python scripts/run_evaluation.py
```

**What's tested:**
- JWT token creation, validation, expiry
- Role-based access control (admin/writer/reader × all endpoints)
- Multi-tenant isolation (tenant A cannot see tenant B's data)
- Token security (invalid, expired, missing → rejected)
- Self-healing strategy selection logic
- Request validation

---

## Self-Healing Pipeline

When the hallucination checker flags an answer, the system diagnoses and retries:

| Cause Detected | Strategy | What Changes |
|----------------|----------|-------------|
| Poor retrieval scores | `expand_retrieval` | top_k=20, rerank_top_n=8 |
| Fabricated claims | `strict_grounding` | temp=0.1, "cite sources only" prompt |
| High entropy tokens | `reduce_uncertainty` | Fewer chunks (top 3), strict prompt |
| 2nd retry still fails | `broaden_search` | Also queries adjacent intent categories |

- Max 2 retries per query
- `"self_heal": false` in request body disables retries
- Healing metadata shows what strategy fixed the issue

---

## Hallucination Analytics

`GET /analytics/hallucinations` surfaces patterns:

```json
{
  "summary": {"total_queries": 147, "hallucination_rate": 0.082},
  "by_intent": [
    {"intent": "explanation", "hallucination_rate": 0.184},
    {"intent": "factual", "hallucination_rate": 0.038}
  ],
  "insights": [
    "'explanation' intent has the highest hallucination rate (18%). Add more docs.",
    "Hallucinated responses take 2.1x longer — self-healing adds latency."
  ]
}
```

---

## Multi-Tenancy

- Every document tagged with `tenant_id` in Qdrant metadata
- Queries always filter by `tenant_id` from JWT
- First user registering a tenant auto-becomes admin
- Roles: **admin** (all) > **writer** (query + ingest) > **reader** (query only)

---

## Environment Variables

```bash
# ─── Required ───────────────────────────────────────────────
OPENAI_API_KEY=gsk_...          # Groq key (free: console.groq.com)
OPENAI_BASE_URL=https://api.groq.com/openai/v1
DEFAULT_LLM_MODEL=llama-3.3-70b-versatile
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/intentrag
QDRANT_HOST=localhost
QDRANT_PORT=6333
JWT_SECRET_KEY=<random-string>

# ─── Optional (degrade gracefully without) ──────────────────
COHERE_API_KEY=                 # Free: dashboard.cohere.com
LANGFUSE_PUBLIC_KEY=            # Free: cloud.langfuse.com
LANGFUSE_SECRET_KEY=
```

---

## Intent Classifier

Trained on TREC + SQuAD + SciQ datasets using RoBERTa-base (~91% weighted F1).

```bash
python train_intent_classifier_roberta.py
# Model auto-loads on next API request
# Falls back to keyword heuristic if model directory missing
```

| Intent | Example Query |
|--------|---------------|
| factual | "What is machine learning?" |
| person | "Who invented the telephone?" |
| time | "When was Python released?" |
| location | "Where is Google headquartered?" |
| explanation | "How does RAG work?" |
| other | "Tell me something interesting" |
