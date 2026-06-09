"""
Self-Healing RAG Pipeline — critique, diagnose, and retry.

When a hallucination is detected, this service:
1. Diagnoses the root cause from the hallucination check result
2. Selects a corrective strategy based on the cause
3. Retries the pipeline with adjusted parameters
4. Returns the best answer (original or retried)

Strategies:
- LOW_RETRIEVAL: Expand query + increase top_k + try adjacent intents
- FABRICATION: Re-prompt with stricter grounding instructions
- INSUFFICIENT_CONTEXT: Broaden search across multiple intent categories
- HIGH_ENTROPY: Lower temperature + add explicit "only use context" constraint

Max retries: 2 (to avoid infinite loops and high latency)
"""

from typing import Optional

from app.services.classifier import intent_classifier
from app.services.generator import generator_service
from app.services.hallucination import hallucination_checker
from app.services.reranker import reranker_service
from app.services.retriever import retriever_service


# Strict system prompt used during retries to reduce hallucination
STRICT_GROUNDING_PROMPT = (
    "You are a precise assistant that ONLY answers based on the provided context. "
    "Rules:\n"
    "1. If the context does not contain the answer, say 'I don't have enough information to answer this.'\n"
    "2. Do NOT add any information beyond what is explicitly stated in the context.\n"
    "3. Do NOT make inferences or extrapolations.\n"
    "4. Cite which part of the context supports each claim.\n"
    "5. If you are uncertain about any detail, omit it rather than guess."
)

# Adjacent intent categories for broadening search
ADJACENT_INTENTS = {
    "factual": ["explanation", "other"],
    "person": ["factual", "time"],
    "time": ["factual", "person"],
    "location": ["factual", "other"],
    "explanation": ["factual", "other"],
    "other": ["factual", "explanation"],
}

MAX_RETRIES = 2


class SelfHealingPipeline:
    """
    Wraps the query pipeline with self-correction capabilities.

    Flow:
    1. Run normal pipeline → get answer + hallucination check
    2. If hallucinated → diagnose cause → select strategy → retry
    3. Return best result with healing metadata
    """

    async def execute(
        self,
        query: str,
        intent: str,
        tenant_id: str,
        top_k: int = 10,
        rerank_top_n: int = 5,
    ) -> dict:
        """
        Execute the self-healing pipeline.

        Returns:
            {
                "answer": str,
                "model": str,
                "usage": dict,
                "retrieved_chunks": list,
                "hallucination_check": dict,
                "healing_metadata": {
                    "attempts": int,
                    "was_healed": bool,
                    "strategies_used": list[str],
                    "original_answer": str | None,
                    "improvement_reason": str | None,
                }
            }
        """
        # --- Attempt 1: Normal pipeline ---
        result = await self._run_pipeline(
            query=query,
            intent=intent,
            tenant_id=tenant_id,
            top_k=top_k,
            rerank_top_n=rerank_top_n,
            strict_mode=False,
        )

        hallucination = result["hallucination_check"]

        # If not hallucinated or check was skipped, return immediately
        if not hallucination.get("is_hallucinated", False):
            result["healing_metadata"] = {
                "attempts": 1,
                "was_healed": False,
                "strategies_used": [],
                "original_answer": None,
                "improvement_reason": None,
            }
            return result

        # --- Hallucination detected — diagnose and retry ---
        original_answer = result["answer"]
        strategies_used = []

        for retry_num in range(MAX_RETRIES):
            # Diagnose cause and pick strategy
            strategy = self._diagnose_and_select_strategy(hallucination, retry_num)
            strategies_used.append(strategy["name"])

            # Execute retry with corrective parameters
            retry_result = await self._run_pipeline(
                query=strategy.get("expanded_query", query),
                intent=strategy.get("intent", intent),
                tenant_id=tenant_id,
                top_k=strategy.get("top_k", top_k),
                rerank_top_n=strategy.get("rerank_top_n", rerank_top_n),
                strict_mode=strategy.get("strict_mode", True),
                additional_intents=strategy.get("additional_intents", []),
            )

            retry_hallucination = retry_result["hallucination_check"]

            # Check if retry is better
            if not retry_hallucination.get("is_hallucinated", False):
                # Healed — return the retry result
                retry_result["healing_metadata"] = {
                    "attempts": retry_num + 2,  # +1 for original, +1 for 0-index
                    "was_healed": True,
                    "strategies_used": strategies_used,
                    "original_answer": original_answer,
                    "improvement_reason": f"Strategy '{strategy['name']}' resolved the hallucination",
                }
                return retry_result

            # Retry didn't fix it — check if confidence improved
            original_confidence = hallucination.get("confidence_score", 0)
            retry_confidence = retry_hallucination.get("confidence_score", 0)

            if retry_confidence > original_confidence:
                # Partial improvement — use the better result for next iteration
                result = retry_result
                hallucination = retry_hallucination

        # --- All retries exhausted — return best result with warning ---
        result["healing_metadata"] = {
            "attempts": MAX_RETRIES + 1,
            "was_healed": False,
            "strategies_used": strategies_used,
            "original_answer": original_answer,
            "improvement_reason": "Retries did not fully resolve hallucination — returning best attempt",
        }
        return result

    def _diagnose_and_select_strategy(
        self, hallucination: dict, retry_num: int
    ) -> dict:
        """
        Analyze the hallucination result and select a corrective strategy.

        Different strategies are tried on successive retries.
        """
        causes = hallucination.get("likely_causes", [])
        retrieval_quality = hallucination.get("retrieval_quality", "unknown")
        hallucination_type = hallucination.get("hallucination_type")
        confidence = hallucination.get("confidence_score", 0)

        # First retry: address the primary cause
        if retry_num == 0:
            if retrieval_quality in ("poor", "marginal") or "LOW_RETRIEVAL_QUALITY" in causes:
                return {
                    "name": "expand_retrieval",
                    "top_k": 20,  # Double the retrieval
                    "rerank_top_n": 8,
                    "strict_mode": True,
                    "additional_intents": [],  # Will be filled by adjacent intents
                }

            if hallucination_type in ("FABRICATION", "EXTRAPOLATION") or "FABRICATED_DETAILS" in causes:
                return {
                    "name": "strict_grounding",
                    "strict_mode": True,
                    "top_k": 10,
                    "rerank_top_n": 5,
                }

            if "HIGH_ENTROPY_GENERATION" in causes or "MULTIPLE_LOW_CONFIDENCE_TOKENS" in causes:
                return {
                    "name": "reduce_uncertainty",
                    "strict_mode": True,
                    "top_k": 15,
                    "rerank_top_n": 3,  # Fewer but higher quality chunks
                }

            # Default first strategy: strict grounding
            return {
                "name": "strict_grounding",
                "strict_mode": True,
                "top_k": 10,
                "rerank_top_n": 5,
            }

        # Second retry: broaden search across intents
        return {
            "name": "broaden_search",
            "top_k": 20,
            "rerank_top_n": 8,
            "strict_mode": True,
            "additional_intents": ["factual", "explanation"],  # Always include these as fallback
        }

    async def _run_pipeline(
        self,
        query: str,
        intent: str,
        tenant_id: str,
        top_k: int,
        rerank_top_n: int,
        strict_mode: bool,
        additional_intents: Optional[list[str]] = None,
        expanded_query: Optional[str] = None,
    ) -> dict:
        """
        Run a single pass of the retrieve → rerank → generate → check pipeline.
        """
        use_query = expanded_query or query

        # Retrieve from primary intent
        retrieved_chunks = await retriever_service.retrieve(
            query=use_query,
            intent=intent,
            tenant_id=tenant_id,
            top_k=top_k,
        )

        # Optionally retrieve from adjacent intents
        if additional_intents:
            for adj_intent in additional_intents:
                if adj_intent != intent:
                    adj_chunks = await retriever_service.retrieve(
                        query=use_query,
                        intent=adj_intent,
                        tenant_id=tenant_id,
                        top_k=top_k // 2,  # Half allocation for adjacent
                    )
                    retrieved_chunks.extend(adj_chunks)

        # Rerank (combined set if broadened)
        reranked_chunks = await reranker_service.rerank(
            query=use_query,
            documents=retrieved_chunks,
            top_n=rerank_top_n,
        )

        # Generate with optional strict mode
        if strict_mode:
            generation = await self._generate_strict(query, reranked_chunks, intent)
        else:
            generation = await generator_service.generate(
                query=query,
                context_chunks=reranked_chunks,
                intent=intent,
            )

        # Hallucination check
        hallucination_result = await hallucination_checker.check(
            query=query,
            answer=generation["answer"],
            context_chunks=reranked_chunks,
            intent=intent,
        )

        return {
            "answer": generation["answer"],
            "model": generation["model"],
            "usage": generation["usage"],
            "retrieved_chunks": reranked_chunks,
            "hallucination_check": hallucination_result,
        }

    async def _generate_strict(
        self, query: str, context_chunks: list[dict], intent: str
    ) -> dict:
        """Generate with strict grounding — overrides the system prompt."""
        from openai import AsyncOpenAI
        from app.core.config import settings

        if not settings.openai_api_key:
            return {"answer": "[No API key]", "model": "gpt-4o-mini", "usage": {}, "context_used": 0}

        # Build context
        context_parts = []
        for i, chunk in enumerate(context_chunks[:5], 1):
            context_parts.append(f"[{i}] {chunk['content'].strip()}")
        context_str = "\n\n".join(context_parts)

        client = AsyncOpenAI(api_key=settings.openai_api_key)

        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": STRICT_GROUNDING_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context_str}\n\n"
                            f"Question: {query}\n\n"
                            "Answer STRICTLY from the context above. Cite [source numbers]."
                        ),
                    },
                ],
                temperature=0.1,  # Very low temperature for strict grounding
                max_tokens=1024,
            )

            choice = response.choices[0]
            usage = response.usage

            return {
                "answer": choice.message.content,
                "model": "gpt-4o-mini",
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "context_used": len(context_chunks),
            }

        except Exception as e:
            return {
                "answer": f"[Strict generation failed: {str(e)}]",
                "model": "gpt-4o-mini",
                "usage": {},
                "context_used": 0,
            }


# Singleton instance
self_healing_pipeline = SelfHealingPipeline()
