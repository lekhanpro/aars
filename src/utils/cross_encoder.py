"""Cross-encoder model singleton for reranking and NLI tasks."""

from __future__ import annotations

import random
import threading
from typing import ClassVar

import structlog

logger = structlog.get_logger(__name__)


class _FallbackCrossEncoder:
    """Deterministic fallback when sentence-transformers is unavailable."""

    def predict(self, sentence_pairs: list[list[str]]) -> list[float]:
        rng = random.Random(42)
        return [rng.uniform(0.0, 1.0) for _ in sentence_pairs]


class CrossEncoderModel:
    """Thread-safe singleton wrapper for sentence-transformers CrossEncoder.

    Usage::

        model = CrossEncoderModel.get("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = model.score("query", ["doc1", "doc2"])
    """

    _instances: ClassVar[dict[str, CrossEncoderModel]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        try:
            from sentence_transformers import CrossEncoder

            logger.info("cross_encoder_loading", model=model_name)
            self._model = CrossEncoder(model_name)
            logger.info("cross_encoder_loaded", model=model_name)
        except ImportError:
            logger.warning(
                "sentence_transformers_not_installed",
                model=model_name,
                fallback="random_scores",
            )
            self._model = _FallbackCrossEncoder()
        except Exception as exc:
            logger.warning(
                "cross_encoder_fallback",
                model=model_name,
                error=str(exc),
                fallback="random_scores",
            )
            self._model = _FallbackCrossEncoder()

    @classmethod
    def get(cls, model_name: str) -> CrossEncoderModel:
        """Return (or create) the singleton for *model_name*."""
        if model_name in cls._instances:
            return cls._instances[model_name]

        with cls._lock:
            if model_name in cls._instances:
                return cls._instances[model_name]
            instance = cls(model_name)
            cls._instances[model_name] = instance
            return instance

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Score each document against *query*.

        Returns a list of relevance scores aligned with *documents*.
        Higher is more relevant.
        """
        if not documents:
            return []
        pairs = [[query, doc] for doc in documents]
        raw_scores = self._model.predict(pairs)
        return [float(s) for s in raw_scores]

    def predict(self, premise: str, hypotheses: list[str]) -> list[float]:
        """Score NLI entailment of each hypothesis given *premise*.

        Used for hallucination detection with NLI models.
        """
        if not hypotheses:
            return []
        pairs = [[premise, h] for h in hypotheses]
        raw_scores = self._model.predict(pairs)
        return [float(s) for s in raw_scores]
