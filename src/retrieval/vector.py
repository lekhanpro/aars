"""Vector retriever backed by ChromaDB and sentence-transformers embeddings."""

from __future__ import annotations

from typing import Any

import structlog

try:
    import chromadb
except ImportError:
    chromadb = Any  # type: ignore[assignment]

from config.settings import ChromaSettings, EmbeddingSettings, RetrieverSettings
from src.api.schemas.common import Document
from src.retrieval.base import BaseRetriever
from src.utils.embeddings import EmbeddingModel

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class VectorRetriever(BaseRetriever):
    """Retrieve documents via dense-vector similarity search.

    At startup the retriever connects to a ChromaDB instance and obtains
    (or creates) the configured collection.  Queries are embedded through
    :class:`EmbeddingModel` and the nearest neighbours are returned as
    :class:`Document` objects.
    """

    def __init__(
        self,
        chroma_settings: ChromaSettings | None = None,
        embedding_settings: EmbeddingSettings | None = None,
        retriever_settings: RetrieverSettings | None = None,
        chroma_client: Any | None = None,
    ) -> None:
        self._chroma_settings = chroma_settings or ChromaSettings()
        self._embedding_settings = embedding_settings or EmbeddingSettings()
        self._retriever_settings = retriever_settings or RetrieverSettings()
        self._provided_client = chroma_client

        self._embedding_model: EmbeddingModel | None = None
        self._client: Any | None = None
        self._collections: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Connect to ChromaDB and load the embedding model."""
        logger.info(
            "vector_retriever.initialize",
            chroma_host=self._chroma_settings.host,
            chroma_port=self._chroma_settings.port,
            embedding_model=self._embedding_settings.model,
        )
        # Embedding model (singleton)
        self._embedding_model = EmbeddingModel.get(self._embedding_settings.model)

        # ChromaDB connection
        if self._provided_client is not None:
            self._client = self._provided_client
        else:
            if chromadb is Any:
                raise RuntimeError(
                    "chromadb is required for vector retrieval when no custom client is provided."
                )
            self._client = chromadb.HttpClient(
                host=self._chroma_settings.host,
                port=self._chroma_settings.port,
            )
        default_collection = self._client.get_or_create_collection(
            name=self._chroma_settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collections[self._chroma_settings.collection_name] = default_collection
        logger.info(
            "vector_retriever.ready",
            collection=self._chroma_settings.collection_name,
            count=default_collection.count(),
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        collection: str = "default",
    ) -> list[Document]:
        """Embed *query* and return the closest documents from ChromaDB.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Number of results.  Falls back to ``RetrieverSettings.top_k``.
        collection:
            Chroma collection to query. Falls back to the configured default
            collection when omitted or empty.
        """
        if self._client is None or self._embedding_model is None:
            raise RuntimeError(
                "VectorRetriever has not been initialised. Call initialize() first."
            )

        effective_top_k = top_k if top_k is not None else self._retriever_settings.top_k
        collection_name = collection or self._chroma_settings.collection_name
        chroma_collection = self._get_collection(collection_name)

        logger.debug(
            "vector_retriever.query",
            query=query[:120],
            top_k=effective_top_k,
            collection=collection_name,
        )

        query_embedding = self._embedding_model.embed([query])[0]

        results: dict[str, Any] = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=effective_top_k,
            include=["documents", "metadatas", "distances"],
        )

        return self._parse_results(results)

    def _get_collection(self, name: str) -> Any:
        """Return a cached Chroma collection handle."""
        if self._client is None:
            raise RuntimeError("VectorRetriever has not been initialised.")
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_results(results: dict[str, Any]) -> list[Document]:
        """Convert raw ChromaDB query output to a list of :class:`Document`."""
        documents: list[Document] = []

        ids_batch: list[list[str]] = results.get("ids", [[]])
        docs_batch: list[list[str | None]] = results.get("documents", [[]])
        metas_batch: list[list[dict[str, Any] | None]] = results.get("metadatas", [[]])
        distances_batch: list[list[float]] = results.get("distances", [[]])

        if not ids_batch:
            return documents

        ids = ids_batch[0]
        docs = docs_batch[0] if docs_batch else []
        metas = metas_batch[0] if metas_batch else []
        distances = distances_batch[0] if distances_batch else []

        for idx, doc_id in enumerate(ids):
            # ChromaDB returns cosine *distance*; convert to a similarity score
            # in [0, 1] so that higher == better.
            distance = distances[idx] if idx < len(distances) else 0.0
            score = max(0.0, 1.0 - distance)

            content = (docs[idx] if idx < len(docs) else None) or ""
            metadata = (metas[idx] if idx < len(metas) else None) or {}

            documents.append(
                Document(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                )
            )

        logger.debug("vector_retriever.results", count=len(documents))
        return documents
