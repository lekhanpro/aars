"""Embedding model singleton backed by sentence-transformers."""

from __future__ import annotations

import hashlib
import math
import re
import threading
from typing import ClassVar

import numpy as np
import structlog

logger = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class _HashingSentenceTransformer:
    """Deterministic lightweight fallback when sentence-transformers is unavailable."""

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    def encode(
        self,
        texts: list[str],
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> np.ndarray:
        del show_progress_bar
        del convert_to_numpy
        rows: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self._dimensions
            for token in _TOKEN_RE.findall(text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self._dimensions
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vector[index] += sign
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
            rows.append(vector)
        return np.array(rows, dtype=np.float64)


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
            logger.warning(
                "sentence_transformers_not_installed",
                model=model_name,
                fallback="hashing",
            )
            self._model_name = model_name
            self._model = _HashingSentenceTransformer()
            return

        self._model_name = model_name
        logger.info("embedding_model_loading", model=model_name)
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            logger.warning(
                "embedding_model_fallback",
                model=model_name,
                error=str(exc),
                fallback="hashing",
            )
            self._model_name = model_name
            self._model = _HashingSentenceTransformer()
            return
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
