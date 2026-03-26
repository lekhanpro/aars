"""Ingestion pipeline: load -> chunk -> embed -> index."""

from __future__ import annotations

import os
from uuid import uuid4

import structlog
from fastapi import UploadFile

from config.settings import Settings
from src.api.schemas.common import Document
from src.api.schemas.responses import IngestResponse
from src.ingestion.graph_builder import GraphBuilder
from src.ingestion.chunkers.recursive import RecursiveChunker
from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.loaders.text_loader import TextLoader
from src.ingestion.loaders.image_loader import ImageLoader
from src.ingestion.loaders.video_loader import VideoLoader
from src.ingestion.multimodal import MultimodalSegregator, detect_modality, ContentModality
from src.retrieval.graph import GraphRetriever
from src.retrieval.keyword import KeywordRetriever
from src.utils.embeddings import EmbeddingModel

logger = structlog.get_logger(__name__)

# Map lowercase file extensions to loader instances.
_LOADER_REGISTRY: dict[str, PDFLoader | TextLoader | ImageLoader | VideoLoader] = {}


def _get_loader(extension: str) -> PDFLoader | TextLoader | ImageLoader | VideoLoader:
    """Return a cached loader instance for the given file extension."""
    ext = extension.lower()
    if ext not in _LOADER_REGISTRY:
        if ext in PDFLoader.SUPPORTED_EXTENSIONS:
            _LOADER_REGISTRY[ext] = PDFLoader()
        elif ext in TextLoader.SUPPORTED_EXTENSIONS:
            _LOADER_REGISTRY[ext] = TextLoader()
        elif ext in ImageLoader.SUPPORTED_EXTENSIONS:
            _LOADER_REGISTRY[ext] = ImageLoader()
        elif ext in VideoLoader.SUPPORTED_EXTENSIONS:
            _LOADER_REGISTRY[ext] = VideoLoader()
        else:
            raise ValueError(
                f"Unsupported file type '{ext}'. Supported types: "
                f"{sorted(PDFLoader.SUPPORTED_EXTENSIONS | TextLoader.SUPPORTED_EXTENSIONS | ImageLoader.SUPPORTED_EXTENSIONS | VideoLoader.SUPPORTED_EXTENSIONS)}"
            )
    return _LOADER_REGISTRY[ext]


class IngestionPipeline:
    """Orchestrates document ingestion: load, chunk, embed, and store.

    Args:
        chroma_client: An initialised ChromaDB ``HttpClient`` (or ``None`` if
            ChromaDB is unavailable).
        settings: Application-wide :class:`Settings` instance.
    """

    def __init__(self, chroma_client: object | None, settings: Settings) -> None:
        self._chroma_client = chroma_client
        self._settings = settings
        self._keyword_retriever: KeywordRetriever | None = None
        self._graph_builder: GraphBuilder | None = None
        self._graph_retriever: GraphRetriever | None = None

    def attach_runtime(
        self,
        *,
        keyword_retriever: KeywordRetriever | None = None,
        graph_builder: GraphBuilder | None = None,
        graph_retriever: GraphRetriever | None = None,
    ) -> IngestionPipeline:
        """Attach shared runtime services used during ingestion."""
        self._keyword_retriever = keyword_retriever
        self._graph_builder = graph_builder
        self._graph_retriever = graph_retriever
        return self

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def ingest(
        self,
        file: UploadFile,
        collection: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> IngestResponse:
        """Run the full ingestion pipeline for an uploaded file.

        Args:
            file: FastAPI ``UploadFile`` from the request.
            collection: Target ChromaDB collection name.
            chunk_size: Override for the default chunk size from settings.
            chunk_overlap: Override for the default chunk overlap from settings.

        Returns:
            An :class:`IngestResponse` summarising the operation.

        Raises:
            RuntimeError: If ChromaDB is not available or any pipeline step
                fails unrecoverably.
            ValueError: If the file type is unsupported or the file is empty.
        """
        if self._chroma_client is None:
            raise RuntimeError(
                "ChromaDB client is not available. Cannot ingest documents."
            )

        filename = file.filename or "unknown"
        logger.info("ingestion_started", filename=filename, collection=collection)

        # ---- 1. Read raw bytes -----------------------------------------
        try:
            file_bytes = await file.read()
        except Exception as exc:
            logger.error("file_read_failed", filename=filename, error=str(exc))
            raise RuntimeError(f"Failed to read uploaded file '{filename}': {exc}") from exc

        if not file_bytes:
            raise ValueError(f"Uploaded file '{filename}' is empty.")

        # ---- 2. Load ---------------------------------------------------
        ext = os.path.splitext(filename)[1]
        loader = _get_loader(ext)
        documents = await loader.load(file_bytes, filename)

        logger.info(
            "documents_loaded",
            filename=filename,
            num_documents=len(documents),
        )

        # ---- 3. Chunk --------------------------------------------------
        effective_chunk_size = chunk_size or self._settings.chunker.size
        effective_chunk_overlap = chunk_overlap or self._settings.chunker.overlap

        chunker = RecursiveChunker(
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
        )

        all_chunks: list[dict] = []
        for doc in documents:
            chunks = chunker.chunk(doc["content"], doc["metadata"])
            all_chunks.extend(chunks)

        if not all_chunks:
            raise ValueError(
                f"Chunking produced no chunks for '{filename}'. "
                "The file may contain only whitespace."
            )

        logger.info(
            "chunks_created",
            filename=filename,
            num_chunks=len(all_chunks),
            chunk_size=effective_chunk_size,
            chunk_overlap=effective_chunk_overlap,
        )

        # ---- 4. Embed --------------------------------------------------
        texts = [c["content"] for c in all_chunks]
        embed_model = EmbeddingModel.get(self._settings.embedding.model)
        embeddings = embed_model.embed(texts, batch_size=self._settings.embedding.batch_size)

        logger.info("embeddings_generated", filename=filename, num_embeddings=len(embeddings))

        chunk_ids = [str(uuid4()) for _ in all_chunks]
        metadatas = [c["metadata"] for c in all_chunks]
        chunk_documents = [
            Document(
                id=chunk_id,
                content=chunk["content"],
                metadata=chunk["metadata"],
            )
            for chunk_id, chunk in zip(chunk_ids, all_chunks, strict=True)
        ]

        # ---- 5. Store in ChromaDB --------------------------------------

        try:
            chroma_collection = self._chroma_client.get_or_create_collection(  # type: ignore[union-attr]
                name=collection,
            )
        except Exception as exc:
            logger.error(
                "chroma_collection_failed",
                collection=collection,
                error=str(exc),
            )
            raise RuntimeError(
                f"Failed to get or create ChromaDB collection '{collection}': {exc}"
            ) from exc

        # ChromaDB has a per-call size limit; batch upserts to stay safe.
        chroma_batch = 512
        total = len(chunk_ids)
        for start in range(0, total, chroma_batch):
            end = min(start + chroma_batch, total)
            try:
                chroma_collection.upsert(
                    ids=chunk_ids[start:end],
                    documents=texts[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=metadatas[start:end],
                )
            except Exception as exc:
                logger.error(
                    "chroma_upsert_failed",
                    collection=collection,
                    batch_start=start,
                    batch_end=end,
                    error=str(exc),
                )
                raise RuntimeError(
                    f"ChromaDB upsert failed for batch [{start}:{end}]: {exc}"
                ) from exc

        logger.info(
            "ingestion_complete",
            filename=filename,
            collection=collection,
            chunks_stored=total,
        )

        # ---- 6. Update in-memory retrievers -----------------------------
        if self._keyword_retriever is not None:
            self._keyword_retriever.add_documents(chunk_documents, collection=collection)

        if self._graph_builder is not None and self._graph_retriever is not None:
            self._graph_builder.add_documents(chunk_documents, collection=collection)
            self._graph_retriever.set_graph(
                self._graph_builder.get_graph(collection),
                collection=collection,
            )

        return IngestResponse(
            collection=collection,
            documents_ingested=len(documents),
            chunks_created=total,
            message=(
                f"Successfully ingested '{filename}': {len(documents)} document(s), "
                f"{total} chunk(s) stored in collection '{collection}'."
            ),
        )
