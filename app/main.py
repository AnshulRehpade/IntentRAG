"""
IntentRAG v2 - FastAPI Application Entry Point
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.database import engine
from app.models.database import Base
from app.routers import auth, eval, health, ingest, query, analytics
from app.services.tracing import tracing_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: create DB tables (if DB available). Shutdown: dispose engine, flush traces."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    except Exception as e:
        print(f"⚠️  Database not available on startup: {e}")
        print("   Tables will be created when DB becomes available.")
    yield
    tracing_service.flush()
    try:
        await engine.dispose()
    except Exception:
        pass


app = FastAPI(
    title="IntentRAG",
    description="Intent-Aware Retrieval-Augmented Generation System",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(eval.router, prefix="/eval", tags=["Evaluation"])
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
