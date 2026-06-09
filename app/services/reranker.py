"""
Reranking service — Cohere Rerank API.

Takes retrieved chunks from Qdrant and re-scores them using Cohere's
cross-encoder reranker for better relevance ordering.
"""

from typing import Optional

import cohere

from app.core.config import settings


class RerankerService:
    """
    Uses Cohere's rerank-v3.5 to re-score and reorder retrieved chunks.
    """

    def __init__(self, model: str = "rerank-v3.5"):
        self._client: Optional[cohere.ClientV2] = None
        self._model = model

    def _get_client(self) -> cohere.ClientV2:
        """Lazy-initialize the Cohere client."""
        if self._client is None:
            self._client = cohere.ClientV2(api_key=settings.cohere_api_key)
        return self._client

    async def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Rerank retrieved documents using Cohere.

        Args:
            query: The user query
            documents: Retrieved chunks, each with "content", "score", "metadata"
            top_n: Number of top results to return after reranking

        Returns:
            Reranked list of {"content": str, "relevance_score": float, "metadata": dict, "original_index": int}
        """
        if not documents:
            return []

        if not settings.cohere_api_key:
            # No API key configured — return documents as-is (sorted by original score)
            return [
                {
                    "content": doc["content"],
                    "relevance_score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {}),
                    "original_index": i,
                }
                for i, doc in enumerate(documents[:top_n])
            ]

        # Extract text content for reranking
        doc_texts = [doc["content"] for doc in documents]

        try:
            client = self._get_client()
            response = client.rerank(
                model=self._model,
                query=query,
                documents=doc_texts,
                top_n=min(top_n, len(documents)),
            )

            # Map results back to original documents with scores
            reranked = []
            for result in response.results:
                original_doc = documents[result.index]
                reranked.append(
                    {
                        "content": original_doc["content"],
                        "relevance_score": result.relevance_score,
                        "metadata": original_doc.get("metadata", {}),
                        "original_index": result.index,
                    }
                )

            return reranked

        except Exception as e:
            # Graceful fallback — return original order if reranking fails
            return [
                {
                    "content": doc["content"],
                    "relevance_score": doc.get("score", 0.0),
                    "metadata": doc.get("metadata", {}),
                    "original_index": i,
                    "_rerank_error": str(e),
                }
                for i, doc in enumerate(documents[:top_n])
            ]


# Singleton instance
reranker_service = RerankerService()
