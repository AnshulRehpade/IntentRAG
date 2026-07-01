"""
Web search fallback — DuckDuckGo (free, no API key needed).

Used when the knowledge base doesn't have relevant content for a query.
Results are clearly marked as "web-sourced" in the response.
"""

from typing import Optional

from ddgs import DDGS


class WebSearchService:
    """
    Searches the web via DuckDuckGo when the knowledge base
    doesn't have relevant content (retrieval quality is poor).
    """

    # Minimum retrieval score to skip web search
    RETRIEVAL_THRESHOLD = 0.2

    def __init__(self):
        self._ddgs = None

    def _get_client(self) -> DDGS:
        if self._ddgs is None:
            self._ddgs = DDGS()
        return self._ddgs

    def should_fallback(self, retrieved_chunks: list[dict]) -> bool:
        """
        Determine if web search fallback is needed.

        Returns True if:
        - No chunks retrieved, OR
        - All chunks have very low relevance scores
        """
        if not retrieved_chunks:
            return True

        scores = [c.get("relevance_score", c.get("score", 0.0)) for c in retrieved_chunks]
        max_score = max(scores) if scores else 0.0
        return max_score < self.RETRIEVAL_THRESHOLD

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        """
        Search the web using DuckDuckGo.

        Returns:
            List of {"title": str, "content": str, "url": str}
        """
        try:
            client = self._get_client()
            results = list(client.text(query, max_results=max_results))

            return [
                {
                    "title": r.get("title", ""),
                    "content": r.get("body", ""),
                    "url": r.get("href", r.get("link", "")),
                }
                for r in results
            ]
        except Exception as e:
            print(f"⚠️ Web search failed: {e}")
            return []

    def format_as_chunks(self, search_results: list[dict]) -> list[dict]:
        """
        Format web search results to look like retrieved chunks
        so the generator can use them in the same pipeline.
        """
        return [
            {
                "content": f"{r['title']}\n{r['content']}",
                "relevance_score": 0.5,  # Default score for web results
                "metadata": {"source": "web", "url": r["url"]},
            }
            for r in search_results
            if r["content"]
        ]


# Singleton instance
web_search_service = WebSearchService()
