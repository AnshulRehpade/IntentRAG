"""
Document ingestion service — chunking, embedding, and storage.

Handles:
1. Reading uploaded file content
2. Splitting into chunks (sentence-based with overlap)
3. Delegating to RetrieverService for embedding + Qdrant storage
4. Logging document metadata to PostgreSQL
"""

from typing import Optional

from llama_index.core.node_parser import SentenceSplitter

from app.services.retriever import retriever_service


class IngestionService:
    """
    Processes uploaded documents: chunk → embed → store in Qdrant.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    async def ingest(
        self,
        content: str,
        intent_category: str,
        tenant_id: str,
        source: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> dict:
        """
        Ingest a document's content into the vector store.

        Args:
            content: Raw text content of the document
            intent_category: Which intent collection to store in
            tenant_id: Tenant isolation
            source: Optional source label (filename, URL, etc.)
            doc_id: Optional document ID for cross-reference

        Returns:
            {
                "chunks_created": int,
                "node_ids": list[str],
                "intent_category": str,
            }
        """
        # Split content into chunks
        chunks = self._split_text(content)

        if not chunks:
            return {
                "chunks_created": 0,
                "node_ids": [],
                "intent_category": intent_category,
            }

        # Build metadata for each chunk
        metadata = {}
        if source:
            metadata["source"] = source
        if doc_id:
            metadata["doc_id"] = doc_id

        # Ingest into Qdrant via retriever service
        node_ids = await retriever_service.ingest_nodes(
            texts=chunks,
            intent=intent_category,
            tenant_id=tenant_id,
            metadata=metadata,
        )

        return {
            "chunks_created": len(chunks),
            "node_ids": node_ids,
            "intent_category": intent_category,
        }

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks using sentence splitter."""
        if not text.strip():
            return []

        # SentenceSplitter expects a list of text or can split a single string
        chunks = self._splitter.split_text(text)
        return chunks


# Singleton instance
ingestion_service = IngestionService()
