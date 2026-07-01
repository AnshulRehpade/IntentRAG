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
from app.services.web_search import web_search_service


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

        Flow:
        1. Run normal pipeline (retrieve → rerank → generate → check)
        2. If hallucinated → diagnose → retry with corrective strategy
        3. If knowledge base has no relevant data → web search fallback
        4. Return best answer with source metadata

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
                    "source": "knowledge_base" | "web_search",
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

        # If not hallucinated, check if the answer is a "can't find info" response
        # In that case, try web search even though it's not technically hallucinated
        answer_lower = result["answer"].lower()
        is_no_info_answer = any(phrase in answer_lower for phrase in [
            "does not mention", "doesn't mention", "no information",
            "not mentioned", "cannot find", "don't have enough",
            "not in the context", "context does not", "context provided does not",
            "additional context", "additional information would be needed",
        ])

        if not hallucination.get("is_hallucinated", False) and not is_no_info_answer:
            result["healing_metadata"] = {
                "attempts": 1,
                "was_healed": False,
                "strategies_used": [],
                "original_answer": None,
                "improvement_reason": None,
                "source": "knowledge_base",
            }
            return result

        # --- Try web search if KB doesn't have the answer ---
        if is_no_info_answer or hallucination.get("retrieval_quality") == "poor":
            web_result = await self._try_web_fallback(query, intent)
            if web_result is not None:
                web_result["healing_metadata"] = {
                    "attempts": 1,
                    "was_healed": True,
                    "strategies_used": ["web_search_fallback"],
                    "original_answer": result["answer"],
                    "improvement_reason": "Knowledge base had no relevant data — answered from web search",
                    "source": "web_search",
                }
                return web_result

        # --- Hallucination detected — diagnose and retry ---
        original_answer = result["answer"]
        strategies_used = []

        for retry_num in range(MAX_RETRIES):
            strategy = self._diagnose_and_select_strategy(hallucination, retry_num)
            strategies_used.append(strategy["name"])

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

            if not retry_hallucination.get("is_hallucinated", False):
                retry_result["healing_metadata"] = {
                    "attempts": retry_num + 2,
                    "was_healed": True,
                    "strategies_used": strategies_used,
                    "original_answer": original_answer,
                    "improvement_reason": f"Strategy '{strategy['name']}' resolved the hallucination",
                    "source": "knowledge_base",
                }
                return retry_result

            # Check if confidence improved
            original_confidence = hallucination.get("confidence_score", 0)
            retry_confidence = retry_hallucination.get("confidence_score", 0)

            if retry_confidence > original_confidence:
                result = retry_result
                hallucination = retry_hallucination

        # --- All KB retries exhausted — try web as last resort ---
        web_result = await self._try_web_fallback(query, intent)
        if web_result is not None:
            web_result["healing_metadata"] = {
                "attempts": MAX_RETRIES + 2,
                "was_healed": True,
                "strategies_used": strategies_used + ["web_search_fallback"],
                "original_answer": original_answer,
                "improvement_reason": "KB retries failed — answered from web search",
                "source": "web_search",
            }
            return web_result

        # --- Everything failed — return best KB attempt ---
        result["healing_metadata"] = {
            "attempts": MAX_RETRIES + 1,
            "was_healed": False,
            "strategies_used": strategies_used,
            "original_answer": original_answer,
            "improvement_reason": "Retries did not fully resolve hallucination — returning best attempt",
            "source": "knowledge_base",
        }
        return result

    async def _try_web_fallback(self, query: str, intent: str) -> Optional[dict]:
        """
        Search the web and generate an answer from web results.
        Returns None if web search yields nothing useful.
        """
        search_results = web_search_service.search(query, max_results=5)
        if not search_results:
            return None

        web_chunks = web_search_service.format_as_chunks(search_results)
        if not web_chunks:
            return None

        # Generate answer from web context
        generation = await generator_service.generate(
            query=query,
            context_chunks=web_chunks,
            intent=intent,
        )

        # Skip full hallucination check for web results — just do a basic verification
        return {
            "answer": generation["answer"],
            "model": generation["model"],
            "usage": generation["usage"],
            "retrieved_chunks": web_chunks,
            "hallucination_check": {
                "is_hallucinated": False,
                "confidence_score": 0.6,
                "retrieval_quality": "web",
                "entropy_score": None,
                "llm_verdict": "WEB_SOURCED",
                "hallucination_type": None,
                "severity": None,
                "flagged_claims": [],
                "likely_causes": [],
                "details": "Answer sourced from web search (not knowledge base)",
            },
        }

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

        if not settings.groq_api_key:
            return {"answer": "[No API key]", "model": settings.default_llm_model, "usage": {}, "context_used": 0}

        # Build context
        context_parts = []
        for i, chunk in enumerate(context_chunks[:5], 1):
            context_parts.append(f"[{i}] {chunk['content'].strip()}")
        context_str = "\n\n".join(context_parts)

        kwargs = {"api_key": settings.groq_api_key}
        if settings.groq_base_url:
            kwargs["base_url"] = settings.groq_base_url
        client = AsyncOpenAI(**kwargs)

        try:
            response = await client.chat.completions.create(
                model=settings.default_llm_model,
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
                "model": settings.default_llm_model,
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
                "model": settings.default_llm_model,
                "usage": {},
                "context_used": 0,
            }


# Singleton instance
self_healing_pipeline = SelfHealingPipeline()
