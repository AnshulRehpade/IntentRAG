"""
Retrieval service — LlamaIndex with Qdrant backend.

Architecture:
- One Qdrant collection per intent category (6 total):
  factual, person, time, location, explanation, other
- Each collection stores chunks from ALL tenants
- Tenant isolation enforced via metadata filter on `tenant_id`
- LlamaIndex VectorStoreIndex wraps each collection
- The router dispatches to the correct index based on classified intent
"""

from typing import Optional

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import TextNode, NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    FilterOperator,
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client import AsyncQdrantClient

from app.core.config import settings

# Intent categories — one collection per category
INTENT_CATEGORIES = settings.intent_categories


class RetrieverService:
    """
    Manages Qdrant vector stores and LlamaIndex query engines per intent category.
    Provides retrieval with tenant_id filtering.
    """

    def __init__(self):
        self._qdrant_client: Optional[QdrantClient] = None
        self._vector_stores: dict[str, QdrantVectorStore] = {}
        self._indexes: dict[str, VectorStoreIndex] = {}
        self._embed_model: Optional[OpenAIEmbedding] = None
        self._initialized = False

    async def initialize(self):
        """
        Set up Qdrant client, ensure collections exist, and create
        LlamaIndex indexes for each intent category.
        """
        if self._initialized:
            return

        # Initialize Qdrant client
        if settings.qdrant_api_key:
            # Qdrant Cloud
            self._qdrant_client = QdrantClient(
                url=f"https://{settings.qdrant_host}",
                api_key=settings.qdrant_api_key,
            )
            self._async_client = AsyncQdrantClient(
                url=f"https://{settings.qdrant_host}",
                api_key=settings.qdrant_api_key,
            )
        else:
            # Local Docker
            self._qdrant_client = QdrantClient(
                host=settings.qdrant_host, port=settings.qdrant_port
            )
            self._async_client = AsyncQdrantClient(
                host=settings.qdrant_host, port=settings.qdrant_port
            )

        # Initialize embedding model (FastEmbed — lightweight ONNX, no PyTorch needed)
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        self._embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Create collections and indexes for each intent category
        for category in INTENT_CATEGORIES:
            collection_name = f"intent_{category}"

            # Ensure collection exists in Qdrant
            self._ensure_collection(collection_name)

            # Create QdrantVectorStore
            vector_store = QdrantVectorStore(
                client=self._qdrant_client,
                aclient=self._async_client,
                collection_name=collection_name,
            )
            self._vector_stores[category] = vector_store

            # Create LlamaIndex VectorStoreIndex
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=self._embed_model,
                storage_context=storage_context,
            )
            self._indexes[category] = index

        self._initialized = True

    def _ensure_collection(self, collection_name: str):
        """Create a Qdrant collection if it doesn't exist."""
        collections = self._qdrant_client.get_collections().collections
        existing_names = [c.name for c in collections]

        if collection_name not in existing_names:
            self._qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # all-MiniLM-L6-v2 dimension
                    distance=models.Distance.COSINE,
                ),
            )

    async def retrieve(
        self, query: str, intent: str, tenant_id: str, top_k: int = 10
    ) -> list[dict]:
        """
        Retrieve relevant chunks from the intent-specific collection,
        filtered by tenant_id.

        Args:
            query: The user's query
            intent: Classified intent category
            tenant_id: For tenant isolation
            top_k: Number of chunks to retrieve

        Returns:
            List of {"content": str, "score": float, "metadata": dict}
        """
        if not self._initialized:
            await self.initialize()

        # Validate intent category
        if intent not in INTENT_CATEGORIES:
            intent = "other"

        index = self._indexes[intent]

        # Build retriever with tenant filter
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="tenant_id",
                    value=tenant_id,
                    operator=FilterOperator.EQ,
                )
            ]
        )

        retriever = index.as_retriever(
            similarity_top_k=top_k,
            filters=filters,
        )

        # Execute retrieval
        nodes: list[NodeWithScore] = await retriever.aretrieve(query)

        # Format results
        results = []
        for node_with_score in nodes:
            node = node_with_score.node
            results.append(
                {
                    "content": node.get_content(),
                    "score": node_with_score.score,
                    "metadata": node.metadata,
                    "node_id": node.node_id,
                }
            )

        return results

    async def ingest_nodes(
        self, texts: list[str], intent: str, tenant_id: str, metadata: Optional[dict] = None
    ) -> list[str]:
        """
        Ingest text chunks into the intent-specific collection.

        Args:
            texts: List of text chunks to embed and store
            intent: Intent category → determines which collection
            tenant_id: For tenant isolation metadata
            metadata: Additional metadata to attach to each node

        Returns:
            List of node IDs that were stored
        """
        if not self._initialized:
            await self.initialize()

        if intent not in INTENT_CATEGORIES:
            intent = "other"

        # Build nodes with tenant_id in metadata
        nodes = []
        for text in texts:
            node_metadata = {"tenant_id": tenant_id}
            if metadata:
                node_metadata.update(metadata)
            node = TextNode(text=text, metadata=node_metadata)
            nodes.append(node)

        # Insert into the appropriate index
        index = self._indexes[intent]
        index.insert_nodes(nodes)

        return [node.node_id for node in nodes]

    def is_healthy(self) -> bool:
        """Check if Qdrant is reachable."""
        try:
            if self._qdrant_client is None:
                return False
            self._qdrant_client.get_collections()
            return True
        except Exception:
            return False


# Singleton instance
retriever_service = RetrieverService()
