"""
Hallucination detection service — wraps v1 logic.

Combines three detection methods:
1. Model confidence check (retrieval score threshold)
2. Entropy-based detection (token probability analysis via logprobs)
3. Second LLM review step (verify answer against context)

Runs on every response before returning to the client.
"""

import json
import re
from typing import Optional

import numpy as np
from openai import AsyncOpenAI

from app.core.config import settings


class HallucinationChecker:
    """
    Multi-method hallucination detection pipeline.

    Methods (in order):
    1. Confidence check — if retrieval scores are too low, flag immediately
    2. Entropy analysis — re-prompt with logprobs, check token confidence
    3. LLM verification — ask a second LLM call to verify answer vs context
    """

    # Thresholds
    LOW_RETRIEVAL_SCORE = 0.15  # Below this, retrieval quality is suspect
    HIGH_ENTROPY_THRESHOLD = 3.0  # Bits — above this indicates uncertainty
    LOW_TOKEN_PROB_THRESHOLD = 0.1  # Token probability below this is suspicious

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize the async OpenAI-compatible client."""
        if self._client is None:
            kwargs = {"api_key": settings.groq_api_key}
            if settings.groq_base_url:
                kwargs["base_url"] = settings.groq_base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def check(
        self,
        query: str,
        answer: str,
        context_chunks: list[dict],
        intent: str = "unknown",
    ) -> dict:
        """
        Run the full hallucination detection pipeline.

        Args:
            query: Original user query
            answer: Generated answer from the LLM
            context_chunks: Reranked chunks used for generation
            intent: Classified intent category

        Returns:
            {
                "is_hallucinated": bool,
                "confidence_score": float,  # 0.0 (no confidence) to 1.0 (fully confident)
                "retrieval_quality": str,   # "good" | "marginal" | "poor"
                "entropy_score": float | None,
                "llm_verdict": str,         # "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED"
                "hallucination_type": str | None,
                "severity": str | None,     # "MINOR" | "MODERATE" | "SEVERE"
                "flagged_claims": list[str],
                "likely_causes": list[str],
                "details": str,
            }
        """
        if not settings.groq_api_key or not answer:
            return self._default_result("No API key or empty answer — skipping check")

        # --- Method 1: Retrieval Confidence Check ---
        retrieval_quality, avg_score = self._check_retrieval_confidence(context_chunks)

        # --- Method 2: Entropy Analysis (logprobs) ---
        entropy_result = await self._check_entropy(answer, context_chunks)

        # --- Method 3: LLM Verification ---
        llm_result = await self._llm_verify(query, answer, context_chunks)

        # --- Combine signals ---
        confidence_score = self._calculate_confidence(
            retrieval_quality=retrieval_quality,
            avg_retrieval_score=avg_score,
            entropy_result=entropy_result,
            llm_result=llm_result,
        )

        # Hallucination requires multiple bad signals (not just one)
        bad_signals = 0
        if llm_result["support_level"] == "UNSUPPORTED":
            bad_signals += 1
        if confidence_score < 0.4:
            bad_signals += 1
        if entropy_result["entropy"] is not None and entropy_result["entropy"] > self.HIGH_ENTROPY_THRESHOLD:
            bad_signals += 1
        if retrieval_quality == "poor":
            bad_signals += 1

        is_hallucinated = bad_signals >= 2

        severity = self._determine_severity(confidence_score, llm_result, entropy_result)
        likely_causes = self._identify_causes(
            retrieval_quality, entropy_result, llm_result, intent
        )

        return {
            "is_hallucinated": is_hallucinated,
            "confidence_score": round(confidence_score, 4),
            "retrieval_quality": retrieval_quality,
            "entropy_score": entropy_result["entropy"],
            "llm_verdict": llm_result["support_level"],
            "hallucination_type": llm_result.get("hallucination_type"),
            "severity": severity if is_hallucinated else None,
            "flagged_claims": llm_result.get("unsupported_claims", []),
            "likely_causes": likely_causes if is_hallucinated else [],
            "details": self._build_details(
                retrieval_quality, entropy_result, llm_result, confidence_score
            ),
        }

    # ------------------------------------------------------------------
    # Method 1: Retrieval confidence
    # ------------------------------------------------------------------

    def _check_retrieval_confidence(self, context_chunks: list[dict]) -> tuple[str, float]:
        """Evaluate retrieval quality from chunk scores."""
        if not context_chunks:
            return "poor", 0.0

        scores = [c.get("relevance_score", c.get("score", 0.0)) for c in context_chunks]
        avg_score = float(np.mean(scores)) if scores else 0.0

        # Thresholds calibrated for all-MiniLM-L6-v2 (produces lower scores than OpenAI)
        if avg_score >= 0.4:
            return "good", avg_score
        elif avg_score >= self.LOW_RETRIEVAL_SCORE:
            return "marginal", avg_score
        else:
            return "poor", avg_score

    # ------------------------------------------------------------------
    # Method 2: Entropy analysis via logprobs
    # ------------------------------------------------------------------

    async def _check_entropy(self, answer: str, context_chunks: list[dict]) -> dict:
        """
        Re-prompt with the answer for verification and analyze logprobs.
        High entropy on verification tokens = model is uncertain about answer validity.
        """
        default = {"entropy": None, "avg_probability": None, "low_prob_token_count": 0}

        context_str = "\n".join(c.get("content", "")[:300] for c in context_chunks[:3])
        if not context_str:
            return default

        prompt = (
            f"Given this context, verify if the following answer is accurate.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Answer to verify:\n{answer}\n\n"
            f"Respond with ONLY one word: VERIFIED or UNVERIFIED"
        )

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=settings.default_llm_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )

            # Try to get logprobs (not all providers support this)
            choice = response.choices[0]
            if not hasattr(choice, 'logprobs') or not choice.logprobs or not choice.logprobs.content:
                return default

            # Extract token probabilities
            token_probs = []
            low_prob_count = 0
            for token_info in choice.logprobs.content:
                prob = np.exp(token_info.logprob)
                token_probs.append(prob)
                if prob < self.LOW_TOKEN_PROB_THRESHOLD:
                    low_prob_count += 1

            if not token_probs:
                return default

            # Calculate Shannon entropy
            probs_array = np.array(token_probs)
            entropy = float(-np.sum(probs_array * np.log2(probs_array + 1e-10)))

            return {
                "entropy": round(entropy, 4),
                "avg_probability": round(float(np.mean(probs_array)), 4),
                "low_prob_token_count": low_prob_count,
            }

        except Exception:
            return default

    # ------------------------------------------------------------------
    # Method 3: LLM verification (second opinion)
    # ------------------------------------------------------------------

    async def _llm_verify(self, query: str, answer: str, context_chunks: list[dict]) -> dict:
        """Ask a separate LLM call to verify answer against context."""
        default = {
            "support_level": "UNKNOWN",
            "hallucination_type": None,
            "unsupported_claims": [],
        }

        context_str = "\n\n".join(c.get("content", "")[:1000] for c in context_chunks[:5])
        if not context_str:
            return {**default, "support_level": "UNSUPPORTED"}

        prompt = (
            "You are a hallucination detection expert for RAG systems.\n\n"
            f"CONTEXT (Retrieved Information):\n{context_str}\n\n"
            f"USER QUERY:\n{query}\n\n"
            f"GENERATED ANSWER:\n{answer}\n\n"
            "Analyze if the ANSWER's claims are supported by the CONTEXT.\n"
            "IMPORTANT: If the answer explicitly states something that appears in the context, "
            "even if it's a small part of a larger passage, classify it as SUPPORTED.\n"
            "Only classify as UNSUPPORTED if the answer makes claims that clearly cannot be "
            "found anywhere in the context.\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "support_level": "SUPPORTED" | "PARTIALLY_SUPPORTED" | "UNSUPPORTED",\n'
            '  "hallucination_type": "FACTUAL" | "ATTRIBUTION" | "CONTRADICTION" | "FABRICATION" | "EXTRAPOLATION" | null,\n'
            '  "unsupported_claims": ["list of claims not in context"]\n'
            "}"
        )

        try:
            client = self._get_client()
            response = await client.chat.completions.create(
                model=settings.default_llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=300,
            )

            raw = response.choices[0].message.content.strip()

            # Parse JSON (handle markdown wrapping)
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            result = json.loads(raw)
            return {
                "support_level": result.get("support_level", "UNKNOWN"),
                "hallucination_type": result.get("hallucination_type"),
                "unsupported_claims": result.get("unsupported_claims", []),
            }

        except Exception:
            return default

    # ------------------------------------------------------------------
    # Scoring and classification
    # ------------------------------------------------------------------

    def _calculate_confidence(
        self,
        retrieval_quality: str,
        avg_retrieval_score: float,
        entropy_result: dict,
        llm_result: dict,
    ) -> float:
        """
        Calculate combined confidence score (0.0 - 1.0).

        Weights:
        - LLM verification: 50% (primary signal)
        - Retrieval quality: 20%
        - Entropy analysis: 30%
        """
        score = 0.0

        # LLM verification (50%)
        support = llm_result.get("support_level", "UNKNOWN")
        if support == "SUPPORTED":
            score += 0.5
        elif support == "PARTIALLY_SUPPORTED":
            score += 0.3
        elif support == "UNKNOWN":
            score += 0.2

        # Retrieval quality (20%)
        score += 0.2 * min(avg_retrieval_score / 0.4, 1.0)  # Normalize to 0.4 as "good"

        # Entropy (30%)
        entropy = entropy_result.get("entropy")
        if entropy is not None:
            if entropy < 1.0:
                score += 0.3
            elif entropy < 2.0:
                score += 0.2
            elif entropy < self.HIGH_ENTROPY_THRESHOLD:
                score += 0.1
        else:
            score += 0.2  # No entropy data, give benefit of doubt

        return min(score, 1.0)

    def _determine_severity(
        self, confidence: float, llm_result: dict, entropy_result: dict
    ) -> str:
        """Determine hallucination severity."""
        if confidence < 0.2:
            return "SEVERE"
        elif confidence < 0.4:
            return "MODERATE"
        else:
            return "MINOR"

    def _identify_causes(
        self,
        retrieval_quality: str,
        entropy_result: dict,
        llm_result: dict,
        intent: str,
    ) -> list[str]:
        """Identify likely root causes."""
        causes = []

        if retrieval_quality == "poor":
            causes.append("LOW_RETRIEVAL_QUALITY")
        elif retrieval_quality == "marginal":
            causes.append("MARGINAL_RETRIEVAL_QUALITY")

        entropy = entropy_result.get("entropy")
        if entropy is not None and entropy > self.HIGH_ENTROPY_THRESHOLD:
            causes.append("HIGH_ENTROPY_GENERATION")

        if entropy_result.get("low_prob_token_count", 0) > 3:
            causes.append("MULTIPLE_LOW_CONFIDENCE_TOKENS")

        if llm_result.get("hallucination_type") == "CONTRADICTION":
            causes.append("CONTRADICTS_CONTEXT")
        elif llm_result.get("hallucination_type") == "FABRICATION":
            causes.append("FABRICATED_DETAILS")

        if intent == "explanation":
            causes.append("COMPLEX_EXPLANATION_QUERY")

        return causes if causes else ["UNKNOWN"]

    def _build_details(
        self, retrieval_quality: str, entropy_result: dict, llm_result: dict, confidence: float
    ) -> str:
        """Build a human-readable details string."""
        parts = [f"Confidence: {confidence:.2f}"]
        parts.append(f"Retrieval: {retrieval_quality}")
        parts.append(f"LLM verdict: {llm_result.get('support_level', 'N/A')}")
        if entropy_result.get("entropy") is not None:
            parts.append(f"Entropy: {entropy_result['entropy']:.2f} bits")
        return " | ".join(parts)

    def _default_result(self, reason: str) -> dict:
        """Return a default no-op result."""
        return {
            "is_hallucinated": False,
            "confidence_score": 0.0,
            "retrieval_quality": "unknown",
            "entropy_score": None,
            "llm_verdict": "SKIPPED",
            "hallucination_type": None,
            "severity": None,
            "flagged_claims": [],
            "likely_causes": [],
            "details": reason,
        }


# Singleton instance
hallucination_checker = HallucinationChecker()
