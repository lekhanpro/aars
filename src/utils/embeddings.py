"""Embedding model singleton backed by sentence-transformers."""

from __future__ import annotations

import threading
from typing import ClassVar

import structlog

logger = structlog.get_logger(__name__)


class EmbeddingModel:
    """Thread-safe singleton wrapper around :class:`SentenceTransformer`.

    Usage::

        model = EmbeddingModel.get("all-MiniLM-L6-v2")
        vectors = model.embed(["hello world", "another sentence"], batch_size=64)
    """

    _instance: ClassVar[EmbeddingModel | None] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _model_name: str
    _model: object  # SentenceTransformer, typed loosely to defer import

    def __init__(self, model_name: str) -> None:
        # Import lazily so that the module can be imported without
        # sentence-transformers installed (e.g. during linting).
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            logger.error("sentence_transformers_not_installed")
            raise RuntimeError(
                "sentence-transformers is required for embeddings. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model_name = model_name
        logger.info("embedding_model_loading", model=model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("embedding_model_loaded", model=model_name)

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> EmbeddingModel:
        """Return (or create) the singleton instance for *model_name*.

        If the requested model differs from the currently loaded one the
        singleton is replaced.
        """
        if cls._instance is not None and cls._instance._model_name == model_name:
            return cls._instance

        with cls._lock:
            # Double-checked locking.
            if cls._instance is not None and cls._instance._model_name == model_name:
                return cls._instance
            cls._instance = cls(model_name)
            return cls._instance

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def embed(
        self,
        texts: list[str],
        batch_size: int = 64,
    ) -> list[list[float]]:
        """Encode *texts* into dense vectors, processing in batches.

        Args:
            texts: Strings to embed.
            batch_size: Number of texts per encoding batch.

        Returns:
            List of float vectors, one per input text.

        Raises:
            RuntimeError: If the underlying model fails to encode.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch = texts[start:end]
            try:
                vectors = self._model.encode(  # type: ignore[union-attr]
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_embeddings.extend(vectors.tolist())
            except Exception as exc:
                logger.error(
                    "embedding_batch_failed",
                    batch_start=start,
                    batch_end=end,
                    error=str(exc),
                )
                raise RuntimeError(
                    f"Embedding failed for batch [{start}:{end}]: {exc}"
                ) from exc

            logger.debug(
                "embedding_batch_done",
                batch_start=start,
                batch_end=end,
                total=total,
            )

        num_batches = (total + batch_size - 1) // batch_size
        logger.info("embedding_complete", num_texts=total, num_batches=num_batches)
        return all_embeddings
