"""
Query endpoint — full IntentRAG pipeline with self-healing (protected).

Pipeline:
1. Classify intent (RoBERTa)
2. Retrieve chunks (Qdrant via LlamaIndex, filtered by tenant_id)
3. Rerank (Cohere rerank-v3.5)
4. Generate answer (OpenAI GPT)
5. Hallucination check (confidence + entropy + LLM verification)
6. Self-healing: if hallucinated → diagnose → retry with corrective strategy
7. Trace in LangFuse (every step end-to-end)
8. Log to PostgreSQL
"""

import time
from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.dependencies import CurrentUser, get_current_user
from app.services.classifier import intent_classifier
from app.services.query_logger import QueryLogger
from app.services.self_healing import self_healing_pipeline
from app.services.tracing import tracing_service

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 10
    rerank_top_n: Optional[int] = 5
    self_heal: Optional[bool] = True  # Enable/disable self-healing


@router.post("")
async def run_query(
    request: QueryRequest,
    user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full IntentRAG pipeline with self-healing.

    When self_heal=true (default):
    - If hallucination is detected, the system diagnoses the cause
    - Retries with a corrective strategy (expand retrieval, strict prompting, etc.)
    - Returns the best answer with healing metadata

    Auth: any authenticated user.
    Tenant isolation enforced via JWT tenant_id.
    """
    start_time = time.time()

    # Start LangFuse trace
    trace = tracing_service.start_trace(
        query=request.query,
        tenant_id=user.tenant_id,
        user_id=user.user_id,
    )

    try:
        # Step 1: Classify intent
        step_start = time.time()
        classification = await intent_classifier.classify(request.query)
        intent = classification["intent"]
        confidence = classification["confidence"]
        classify_ms = int((time.time() - step_start) * 1000)

        trace.span_classify(
            intent=intent,
            confidence=confidence,
            all_scores=classification["all_scores"],
            latency_ms=classify_ms,
        )

        # Steps 2-6: Self-healing pipeline (retrieve → rerank → generate → check → retry)
        step_start = time.time()

        if request.self_heal:
            pipeline_result = await self_healing_pipeline.execute(
                query=request.query,
                intent=intent,
                tenant_id=user.tenant_id,
                top_k=request.top_k or 10,
                rerank_top_n=request.rerank_top_n or 5,
            )
        else:
            # Direct pipeline without self-healing (single pass)
            from app.services.generator import generator_service
            from app.services.hallucination import hallucination_checker
            from app.services.reranker import reranker_service
            from app.services.retriever import retriever_service

            retrieved = await retriever_service.retrieve(
                query=request.query, intent=intent,
                tenant_id=user.tenant_id, top_k=request.top_k or 10,
            )
            reranked = await reranker_service.rerank(
                query=request.query, documents=retrieved, top_n=request.rerank_top_n or 5,
            )
            generation = await generator_service.generate(
                query=request.query, context_chunks=reranked, intent=intent,
            )
            hallucination_result = await hallucination_checker.check(
                query=request.query, answer=generation["answer"],
                context_chunks=reranked, intent=intent,
            )
            pipeline_result = {
                "answer": generation["answer"],
                "model": generation["model"],
                "usage": generation["usage"],
                "retrieved_chunks": reranked,
                "hallucination_check": hallucination_result,
                "healing_metadata": None,
            }

        pipeline_ms = int((time.time() - step_start) * 1000)

        # Trace generation
        trace.span_generate(
            query=request.query,
            answer=pipeline_result["answer"],
            model=pipeline_result["model"],
            usage=pipeline_result["usage"],
            latency_ms=pipeline_ms,
        )

        # Trace hallucination check
        hall_check = pipeline_result["hallucination_check"]
        trace.span_hallucination(
            is_hallucinated=hall_check.get("is_hallucinated", False),
            confidence_score=hall_check.get("confidence_score", 0.0),
            severity=hall_check.get("severity"),
            details=hall_check.get("details", ""),
            latency_ms=0,  # Included in pipeline_ms
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Finalize trace
        healing = pipeline_result.get("healing_metadata")
        trace.end(
            output={
                "intent": intent,
                "answer": pipeline_result["answer"][:500],
                "is_hallucinated": hall_check.get("is_hallucinated", False),
                "was_healed": healing["was_healed"] if healing else False,
                "attempts": healing["attempts"] if healing else 1,
            },
            latency_ms=latency_ms,
        )

        # Log to PostgreSQL
        query_id = None
        try:
            query_id = await QueryLogger.log(
                db=db,
                tenant_id=user.tenant_id,
                user_id=user.user_id,
                query_text=request.query,
                intent=intent,
                retrieved_chunks=[
                    c["content"][:200] for c in pipeline_result["retrieved_chunks"]
                ],
                llm_response=pipeline_result["answer"],
                hallucination_flag=hall_check.get("is_hallucinated", False),
                latency_ms=latency_ms,
            )
        except Exception:
            pass

        return {
            "success": True,
            "data": {
                "query_id": query_id,
                "trace_id": trace.trace_id,
                "query": request.query,
                "intent": intent,
                "intent_confidence": confidence,
                "answer": pipeline_result["answer"],
                "model": pipeline_result["model"],
                "usage": pipeline_result["usage"],
                "retrieved_chunks": [
                    {
                        "content": c["content"],
                        "relevance_score": c["relevance_score"],
                    }
                    for c in pipeline_result["retrieved_chunks"]
                ],
                "hallucination_check": hall_check,
                "healing": healing,
                "latency_ms": latency_ms,
            },
            "error": None,
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        trace.end(output={"error": str(e)}, latency_ms=latency_ms)
        return {
            "success": False,
            "data": {"latency_ms": latency_ms},
            "error": f"Query failed: {str(e)}",
        }
