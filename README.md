# IntentRAG v2

Intent-Aware Retrieval-Augmented Generation — a production-ready RAG system that classifies user intent before retrieval, routes to specialized indexes, and validates answers for hallucinations.

## Architecture

```
Client (React / Postman)
│
▼
FastAPI  (/query, /ingest, /health, /eval, /auth)
│
▼
RoBERTa Intent Classifier
(factual / person / time / location / explanation / other)
│
▼
LlamaIndex VectorStoreIndex
(routes to correct Qdrant collection based on intent)
│
┌───────┬───────┬──────┬──────────┬───────┐
│       │       │      │          │       │
factual person time  location  explan. other
│       │       │      │          │       │
└───────┴───────┴──────┴──────────┴───────┘
│  ← Qdrant (filtered by tenant_id)
▼
Cohere Reranker (rerank-v3.5)
│
▼
OpenAI GPT (gpt-4o-mini)
│
▼
Hallucination Checker
(confidence + entropy + LLM verification)
│
▼
LangFuse Trace + PostgreSQL Log
│
▼
Response to client
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API | FastAPI |
| Intent Classifier | Fine-tuned RoBERTa (6 classes) |
| Vector Store | Qdrant (1 collection per intent, metadata tenant filter) |
| Retrieval Framework | LlamaIndex (VectorStoreIndex) |
| Reranking | Cohere rerank-v3.5 |
| LLM | OpenAI GPT-4o-mini |
| Hallucination Detection | Confidence + entropy (logprobs) + LLM second opinion |
| Observability | LangFuse (per-step tracing) |
| Evaluation | RAGAS (faithfulness, relevancy, precision, recall) |
| Database | PostgreSQL (users, tenants, query logs, documents) |
| Auth | JWT with role-based access (admin / writer / reader) |
| Deployment | Docker Compose (local), Railway (production) |

## Project Structure

```
IntentRAG/
├── app/
│   ├── main.py                 # FastAPI app entry point
│   ├── core/
│   │   ├── config.py           # Settings from .env
│   │   ├── database.py         # Async SQLAlchemy engine
│   │   ├── dependencies.py     # Auth dependencies (get_current_user, require_role)
│   │   └── security.py         # Password hashing + JWT
│   ├── models/
│   │   └── database.py         # ORM: Tenant, User, QueryLog, Document
│   ├── routers/
│   │   ├── auth.py             # /auth/register, /auth/login, /auth/me
│   │   ├── health.py           # /health (checks API, Postgres, Qdrant)
│   │   ├── ingest.py           # /ingest (upload → chunk → embed → store)
│   │   ├── query.py            # /query (classify → retrieve → rerank → generate)
│   │   └── eval.py             # /eval (RAGAS evaluation)
│   ├── schemas/
│   │   └── response.py         # APIResponse wrapper
│   └── services/
│       ├── classifier.py       # RoBERTa intent classification
│       ├── retriever.py        # LlamaIndex + Qdrant retrieval
│       ├── ingestion.py        # Document chunking + embedding
│       ├── reranker.py         # Cohere reranking
│       ├── generator.py        # OpenAI answer generation
│       ├── hallucination.py    # 3-method hallucination detection
│       ├── tracing.py          # LangFuse instrumentation
│       ├── evaluation.py       # RAGAS metric computation
│       └── query_logger.py     # PostgreSQL query logging
├── alembic/                    # Database migrations
├── data/
│   ├── factual/                # Sample documents per intent
│   ├── person/
│   ├── time/
│   ├── location/
│   ├── explanation/
│   ├── other/
│   └── eval_test_set.json      # RAGAS evaluation test set
├── scripts/
│   ├── init_db.py              # Create database tables
│   ├── seed_db.py              # Seed demo tenant + admin user
│   ├── ingest_sample_data.py   # Ingest all sample documents
│   ├── run_evaluation.py       # Run RAGAS evaluation
│   ├── test_integration.py     # Full E2E multi-tenant test
│   └── test_tenant_isolation_unit.py  # Unit tests (no Docker)
├── docker-compose.yml          # PostgreSQL + Qdrant
├── alembic.ini                 # Migration config
├── requirements.txt            # All dependencies
└── .env.example                # Environment variables template
```

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for PostgreSQL and Qdrant)

### 2. Setup

```bash
# Clone and enter project
cd IntentRAG

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment variables
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Start Services

```bash
# Start PostgreSQL and Qdrant
docker compose up -d

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Seed Data

```bash
# Create demo tenant + admin user
python scripts/seed_db.py

# Ingest sample documents
python scripts/ingest_sample_data.py
```

### 5. Query

```bash
# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "admin@demo.com", "password": "password123"}'

# Query (use the token from login response)
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"query": "Who created Python?"}'
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| GET | `/health` | Public | System health check |
| POST | `/auth/register` | Public | Register user + tenant |
| POST | `/auth/login` | Public | Get JWT token |
| GET | `/auth/me` | Any | Current user info |
| POST | `/ingest` | Admin, Writer | Upload and index documents |
| POST | `/query` | Any | Run full RAG pipeline (self-healing) |
| POST | `/eval` | Admin | Run RAGAS evaluation |
| GET | `/analytics/hallucinations` | Admin | Hallucination pattern analysis |

## Multi-Tenancy

- Every document is tagged with `tenant_id` in Qdrant metadata
- Queries filter by `tenant_id` — one tenant never sees another's data
- `tenant_id` is extracted from JWT (not passed in request body)
- Role-based access: admin > writer > reader

## Query Pipeline Response

```json
{
  "success": true,
  "data": {
    "query_id": "uuid",
    "trace_id": "langfuse-trace-id",
    "query": "Who created Python?",
    "intent": "person",
    "intent_confidence": 0.97,
    "answer": "Python was created by Guido van Rossum...",
    "model": "gpt-4o-mini",
    "usage": {"prompt_tokens": 450, "completion_tokens": 120, "total_tokens": 570},
    "retrieved_chunks": [...],
    "hallucination_check": {
      "is_hallucinated": false,
      "confidence_score": 0.87,
      "llm_verdict": "SUPPORTED",
      "severity": null
    },
    "healing": {
      "attempts": 1,
      "was_healed": false,
      "strategies_used": [],
      "original_answer": null,
      "improvement_reason": null
    },
    "latency_ms": 1250
  }
}
```

## Self-Healing Pipeline

When the hallucination checker flags an answer, the system automatically retries:

```
Generate answer
     ↓
Hallucination detected?
     ↓ YES
Diagnose cause:
├── Poor retrieval      → expand_retrieval (2x chunks, broaden search)
├── Fabrication         → strict_grounding (temp=0.1, "cite sources only")
├── High entropy        → reduce_uncertainty (fewer, higher-quality chunks)
└── 2nd retry fails     → broaden_search (query adjacent intent categories)
     ↓
Return best answer + healing metadata
```

- Max 2 retries per query
- Each retry uses a different corrective strategy
- `self_heal=false` in the request body disables retries
- Healing metadata shows what strategy fixed the issue

## Hallucination Analytics

`GET /analytics/hallucinations` returns:

```json
{
  "summary": {
    "total_queries": 150,
    "total_hallucinations": 12,
    "hallucination_rate": 0.08
  },
  "by_intent": [
    {"intent": "explanation", "hallucination_rate": 0.18, "total_queries": 40},
    {"intent": "factual", "hallucination_rate": 0.03, "total_queries": 60}
  ],
  "insights": [
    "'explanation' intent has the highest hallucination rate (18%). Consider adding more documents.",
    "Hallucinated responses take 2.1x longer on average — self-healing retries add latency."
  ]
}
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/intentrag
QDRANT_HOST=localhost
QDRANT_PORT=6333
JWT_SECRET_KEY=your-secret-key

# Optional (enable for full features)
COHERE_API_KEY=...              # Reranking (fallback: original order)
LANGFUSE_PUBLIC_KEY=pk-lf-...   # Tracing (fallback: no-op)
LANGFUSE_SECRET_KEY=sk-lf-...
INTENT_MODEL_PATH=              # Custom model path (default: ./intent_classifier_model_roberta)
```

## Testing

```bash
# Unit tests (no Docker needed)
python scripts/test_tenant_isolation_unit.py

# Full integration test (requires Docker + running API)
python scripts/test_integration.py

# RAGAS evaluation
python scripts/run_evaluation.py
```

## Training the Intent Classifier

```bash
# Train RoBERTa on TREC + SQuAD + SciQ datasets
python train_intent_classifier_roberta.py

# The model saves to ./intent_classifier_model_roberta/
# The API automatically picks it up on next request
```

## Swagger Docs

With the server running, visit: http://localhost:8000/docs
