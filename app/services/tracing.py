"""
LangFuse observability service — traces every step of the pipeline.

Uses LangFuse v4 SDK with the @observe decorator pattern.
Each query gets a parent trace with child spans for:
  - Intent classification
  - Retrieval
  - Reranking
  - Generation
  - Hallucination check

When LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are configured,
traces appear in the LangFuse dashboard automatically.
When keys are missing, tracing is a no-op (no errors).
"""

import time
from typing import Any, Optional
from contextlib import asynccontextmanager

from app.core.config import settings


def _is_enabled() -> bool:
    """Check if LangFuse is configured."""
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def _get_langfuse():
    """Get LangFuse client instance (lazy singleton)."""
    if not _is_enabled():
        return None

    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


class TracingService:
    """
    Manages LangFuse traces for the IntentRAG pipeline.

    Usage:
        trace = tracing_service.start_trace(query, tenant_id, user_id)
        trace.span_classify(intent, confidence, latency_ms)
        trace.span_retrieve(chunks, latency_ms)
        trace.span_rerank(chunks, latency_ms)
        trace.span_generate(prompt, answer, model, usage, latency_ms)
        trace.span_hallucination(result, latency_ms)
        trace.end()
    """

    def __init__(self):
        self._client = None
        self._initialized = False

    def _ensure_client(self):
        """Lazy-initialize the LangFuse client."""
        if not self._initialized:
            self._client = _get_langfuse()
            self._initialized = True
        return self._client

    def start_trace(
        self,
        query: str,
        tenant_id: str,
        user_id: str,
        intent: Optional[str] = None,
    ) -> "PipelineTrace":
        """
        Start a new trace for a query pipeline execution.

        Returns a PipelineTrace object that records spans.
        If LangFuse is not configured, returns a no-op trace.
        """
        client = self._ensure_client()

        if client is None:
            return NoOpTrace()

        trace = client.trace(
            name="intentrag-query",
            input={"query": query},
            metadata={
                "tenant_id": tenant_id,
                "user_id": user_id,
                "intent": intent,
            },
            tags=["intentrag", f"tenant:{tenant_id}"],
        )

        return LangFuseTrace(trace=trace, client=client)

    def flush(self):
        """Flush any pending events to LangFuse."""
        client = self._ensure_client()
        if client:
            client.flush()


class LangFuseTrace:
    """Active LangFuse trace with span helpers for each pipeline step."""

    def __init__(self, trace, client):
        self._trace = trace
        self._client = client
        self._start_time = time.time()

    @property
    def trace_id(self) -> str:
        return self._trace.id

    def span_classify(
        self,
        intent: str,
        confidence: float,
        all_scores: dict,
        latency_ms: int,
    ):
        """Record intent classification span."""
        self._trace.span(
            name="intent-classification",
            input={"query": self._trace.input},
            output={
                "intent": intent,
                "confidence": confidence,
                "all_scores": all_scores,
            },
            metadata={"model": "roberta-base-finetuned", "latency_ms": latency_ms},
        )

    def span_retrieve(
        self,
        intent: str,
        tenant_id: str,
        num_chunks: int,
        top_scores: list[float],
        latency_ms: int,
    ):
        """Record retrieval span."""
        self._trace.span(
            name="retrieval",
            input={"intent": intent, "tenant_id": tenant_id},
            output={"num_chunks": num_chunks, "top_scores": top_scores},
            metadata={
                "vector_store": "qdrant",
                "collection": f"intent_{intent}",
                "latency_ms": latency_ms,
            },
        )

    def span_rerank(
        self,
        num_input: int,
        num_output: int,
        top_scores: list[float],
        latency_ms: int,
    ):
        """Record reranking span."""
        self._trace.span(
            name="reranking",
            input={"num_candidates": num_input},
            output={"num_returned": num_output, "top_scores": top_scores},
            metadata={"model": "cohere-rerank-v3.5", "latency_ms": latency_ms},
        )

    def span_generate(
        self,
        query: str,
        answer: str,
        model: str,
        usage: dict,
        latency_ms: int,
    ):
        """Record LLM generation span."""
        self._trace.generation(
            name="answer-generation",
            input=query,
            output=answer,
            model=model,
            usage=usage or {},
            metadata={"latency_ms": latency_ms},
        )

    def span_hallucination(
        self,
        is_hallucinated: bool,
        confidence_score: float,
        severity: Optional[str],
        details: str,
        latency_ms: int,
    ):
        """Record hallucination check span."""
        self._trace.span(
            name="hallucination-check",
            input={"check_type": "confidence+entropy+llm"},
            output={
                "is_hallucinated": is_hallucinated,
                "confidence_score": confidence_score,
                "severity": severity,
                "details": details,
            },
            metadata={"latency_ms": latency_ms},
        )

    def end(self, output: Optional[dict] = None, latency_ms: Optional[int] = None):
        """Finalize the trace with output and total latency."""
        total_ms = latency_ms or int((time.time() - self._start_time) * 1000)
        self._trace.update(
            output=output or {},
            metadata={"total_latency_ms": total_ms},
        )

    def score(self, name: str, value: float, comment: Optional[str] = None):
        """Attach a score to the trace (e.g., RAGAS score, user feedback)."""
        self._trace.score(name=name, value=value, comment=comment)


class NoOpTrace:
    """No-op trace when LangFuse is not configured. All methods are safe to call."""

    @property
    def trace_id(self) -> Optional[str]:
        return None

    def span_classify(self, *args, **kwargs):
        pass

    def span_retrieve(self, *args, **kwargs):
        pass

    def span_rerank(self, *args, **kwargs):
        pass

    def span_generate(self, *args, **kwargs):
        pass

    def span_hallucination(self, *args, **kwargs):
        pass

    def end(self, *args, **kwargs):
        pass

    def score(self, *args, **kwargs):
        pass


# Singleton instance
tracing_service = TracingService()
