"""
RAGAS evaluation service — computes RAG quality metrics.

Metrics computed:
- Faithfulness: Is the answer faithful to the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: How precise is the retrieved context?
- Context Recall: Does the context contain the ground truth?

Uses the RAGAS library with OpenAI as the evaluation LLM.
NOTE: ragas + datasets are optional dependencies. If not installed,
the /eval endpoint returns a helpful error instead of crashing.
"""

from typing import Optional

from app.core.config import settings
from app.services.classifier import intent_classifier
from app.services.generator import generator_service
from app.services.reranker import reranker_service
from app.services.retriever import retriever_service

# Lazy import — ragas is heavy and optional in production
_ragas_available = False
try:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    _ragas_available = True
except ImportError:
    pass


class EvaluationService:
    """
    Runs RAGAS evaluation on a test set by:
    1. Running each question through the full pipeline
    2. Collecting retrieved contexts and generated answers
    3. Computing RAGAS metrics against ground truth
    """

    async def run_evaluation(
        self,
        test_set: list[dict],
        tenant_id: str,
    ) -> dict:
        """
        Run evaluation on the provided test set.

        Args:
            test_set: List of {"question": str, "ground_truth": str, "intent_category": str | None}
            tenant_id: Tenant to evaluate against

        Returns:
            {
                "aggregate_scores": {metric: float},
                "per_item_scores": [{question, scores, intent, ...}],
                "num_items": int,
                "errors": [str],
            }
        """
        if not _ragas_available:
            return {
                "aggregate_scores": {},
                "per_item_scores": [],
                "num_items": 0,
                "errors": ["RAGAS not installed. This feature requires: pip install ragas datasets"],
            }
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        per_item = []
        errors = []

        for i, item in enumerate(test_set):
            question = item["question"]
            ground_truth = item["ground_truth"]
            intent_override = item.get("intent_category")

            try:
                # Run the pipeline for this question
                result = await self._run_single(question, tenant_id, intent_override)

                questions.append(question)
                answers.append(result["answer"])
                contexts.append(result["contexts"])
                ground_truths.append(ground_truth)

                per_item.append({
                    "index": i,
                    "question": question,
                    "intent": result["intent"],
                    "answer": result["answer"],
                    "num_contexts": len(result["contexts"]),
                    "ground_truth": ground_truth,
                })

            except Exception as e:
                errors.append(f"Item {i} ({question[:50]}): {str(e)}")

        if not questions:
            return {
                "aggregate_scores": {},
                "per_item_scores": [],
                "num_items": 0,
                "errors": errors,
            }

        # Build RAGAS dataset
        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Run RAGAS evaluation
        try:
            ragas_result = evaluate(
                dataset=eval_dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                ],
            )

            aggregate_scores = {
                "faithfulness": ragas_result.get("faithfulness", None),
                "answer_relevancy": ragas_result.get("answer_relevancy", None),
                "context_precision": ragas_result.get("context_precision", None),
                "context_recall": ragas_result.get("context_recall", None),
            }

            # Extract per-item scores from the result dataframe if available
            try:
                df = ragas_result.to_pandas()
                for i, row in df.iterrows():
                    if i < len(per_item):
                        per_item[i]["scores"] = {
                            "faithfulness": _safe_float(row.get("faithfulness")),
                            "answer_relevancy": _safe_float(row.get("answer_relevancy")),
                            "context_precision": _safe_float(row.get("context_precision")),
                            "context_recall": _safe_float(row.get("context_recall")),
                        }
            except Exception:
                # per-item extraction failed, aggregate is still valid
                pass

        except Exception as e:
            errors.append(f"RAGAS evaluation failed: {str(e)}")
            aggregate_scores = {}

        return {
            "aggregate_scores": aggregate_scores,
            "per_item_scores": per_item,
            "num_items": len(questions),
            "errors": errors,
        }

    async def _run_single(
        self, question: str, tenant_id: str, intent_override: Optional[str] = None
    ) -> dict:
        """
        Run the pipeline for a single evaluation question.

        Returns:
            {"answer": str, "contexts": list[str], "intent": str}
        """
        # Classify intent (or use override)
        if intent_override:
            intent = intent_override
        else:
            classification = await intent_classifier.classify(question)
            intent = classification["intent"]

        # Retrieve
        retrieved_chunks = await retriever_service.retrieve(
            query=question,
            intent=intent,
            tenant_id=tenant_id,
            top_k=10,
        )

        # Rerank
        reranked_chunks = await reranker_service.rerank(
            query=question,
            documents=retrieved_chunks,
            top_n=5,
        )

        # Generate answer
        generation = await generator_service.generate(
            query=question,
            context_chunks=reranked_chunks,
            intent=intent,
        )

        # Extract context strings for RAGAS
        contexts = [c["content"] for c in reranked_chunks]

        return {
            "answer": generation["answer"],
            "contexts": contexts,
            "intent": intent,
        }


def _safe_float(value) -> Optional[float]:
    """Safely convert to float, return None if not possible."""
    if value is None:
        return None
    try:
        import math
        f = float(value)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


# Singleton instance
evaluation_service = EvaluationService()
