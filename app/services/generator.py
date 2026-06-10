"""
Answer generation service — OpenAI GPT.

Takes the user query + reranked context chunks and generates a final answer.
Uses structured prompts tailored by intent category.
"""

from typing import Optional

from openai import AsyncOpenAI

from app.core.config import settings

# System prompts per intent category for better generation quality
INTENT_PROMPTS = {
    "factual": (
        "You are a precise factual assistant. Answer the question based strictly on the "
        "provided context. If the context doesn't contain enough information, say so clearly. "
        "Cite specific facts from the context."
    ),
    "person": (
        "You are a knowledgeable assistant specializing in information about people. "
        "Answer questions about individuals based on the provided context. Include relevant "
        "biographical details, roles, and contributions mentioned in the context."
    ),
    "time": (
        "You are a chronologically precise assistant. Answer time-related questions using "
        "the provided context. Include specific dates, durations, and timelines when available."
    ),
    "location": (
        "You are a geographically aware assistant. Answer location-related questions using "
        "the provided context. Include specific places, addresses, and spatial relationships."
    ),
    "explanation": (
        "You are a clear and thorough explainer. Provide detailed explanations based on "
        "the provided context. Break down complex topics step by step. Use examples from "
        "the context when available."
    ),
    "other": (
        "You are a helpful assistant. Answer the question based on the provided context. "
        "If the context doesn't contain enough information, say so clearly."
    ),
}


class GeneratorService:
    """
    Generates the final answer using an LLM with retrieved + reranked context.
    Supports OpenAI, Groq, or any OpenAI-compatible API.
    """

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize the async OpenAI-compatible client."""
        if self._client is None:
            kwargs = {"api_key": settings.openai_api_key}
            if settings.openai_base_url:
                kwargs["base_url"] = settings.openai_base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def generate(
        self,
        query: str,
        context_chunks: list[dict],
        intent: str,
        model: Optional[str] = None,
    ) -> dict:
        """
        Generate an answer given the query and reranked context.

        Args:
            query: The user's original query
            context_chunks: Reranked chunks with "content" and "relevance_score"
            intent: Classified intent category (determines system prompt)
            model: Optional model override

        Returns:
            {
                "answer": str,
                "model": str,
                "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
                "context_used": int  # number of chunks fed to the model
            }
        """
        if not settings.openai_api_key:
            return {
                "answer": "[LLM API key not configured]",
                "model": settings.default_llm_model,
                "usage": {},
                "context_used": 0,
            }

        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
                "model": settings.default_llm_model,
                "usage": {},
                "context_used": 0,
            }

        # Build the context string from reranked chunks
        context_str = self._build_context(context_chunks)

        # Select system prompt based on intent
        system_prompt = INTENT_PROMPTS.get(intent, INTENT_PROMPTS["other"])

        # Build messages
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_str}\n\n"
                    f"Question: {query}\n\n"
                    "Answer based on the context above. If the context doesn't provide "
                    "enough information, clearly state what's missing."
                ),
            },
        ]

        try:
            client = self._get_client()
            use_model = model or settings.default_llm_model

            response = await client.chat.completions.create(
                model=use_model,
                messages=messages,
                temperature=0.3,
                max_tokens=1024,
            )

            choice = response.choices[0]
            usage = response.usage

            return {
                "answer": choice.message.content,
                "model": use_model,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                },
                "context_used": len(context_chunks),
            }

        except Exception as e:
            return {
                "answer": f"[Generation failed: {str(e)}]",
                "model": settings.default_llm_model,
                "usage": {},
                "context_used": len(context_chunks),
            }

    def _build_context(self, chunks: list[dict], max_chunks: int = 5) -> str:
        """
        Format reranked chunks into a numbered context string.
        Limits to max_chunks to stay within token budget.
        """
        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks], 1):
            score = chunk.get("relevance_score", 0.0)
            content = chunk["content"].strip()
            context_parts.append(f"[{i}] (relevance: {score:.3f})\n{content}")

        return "\n\n".join(context_parts)


# Singleton instance
generator_service = GeneratorService()
