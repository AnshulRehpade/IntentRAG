"""
Evaluation endpoint — run RAGAS evaluation on an uploaded test set (protected).

Accepts a JSON body with test items (question + ground_truth),
runs each through the full pipeline, then computes RAGAS metrics.
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.core.dependencies import CurrentUser, require_role
from app.services.evaluation import evaluation_service

router = APIRouter()


class EvalItem(BaseModel):
    question: str
    ground_truth: str
    intent_category: Optional[str] = None


class EvalRequest(BaseModel):
    test_set: List[EvalItem]


@router.post("")
async def run_evaluation(
    request: EvalRequest,
    user: CurrentUser = Depends(require_role("admin")),
):
    """
    Run RAGAS evaluation on the provided test set.

    For each item:
    1. Classify intent (or use provided intent_category)
    2. Retrieve chunks from Qdrant (filtered by tenant_id)
    3. Rerank with Cohere
    4. Generate answer with OpenAI
    5. Compute RAGAS metrics against ground_truth

    Metrics returned:
    - faithfulness: Is the answer faithful to retrieved context?
    - answer_relevancy: Is the answer relevant to the question?
    - context_precision: Is the retrieved context precise?
    - context_recall: Does the context contain the ground truth?

    Auth: admin only.
    Tenant isolation enforced via JWT tenant_id.
    """
    start_time = time.time()

    try:
        # Convert request to dict format
        test_items = [
            {
                "question": item.question,
                "ground_truth": item.ground_truth,
                "intent_category": item.intent_category,
            }
            for item in request.test_set
        ]

        # Run evaluation
        result = await evaluation_service.run_evaluation(
            test_set=test_items,
            tenant_id=user.tenant_id,
        )

        latency_ms = int((time.time() - start_time) * 1000)

        return {
            "success": True,
            "data": {
                "tenant_id": user.tenant_id,
                "aggregate_scores": result["aggregate_scores"],
                "per_item_scores": result["per_item_scores"],
                "num_items_evaluated": result["num_items"],
                "num_items_submitted": len(request.test_set),
                "errors": result["errors"],
                "latency_ms": latency_ms,
            },
            "error": None,
        }

    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "data": {"latency_ms": latency_ms},
            "error": f"Evaluation failed: {str(e)}",
        }
