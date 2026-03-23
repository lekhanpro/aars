"""Evaluation metrics for answer quality and retrieval effectiveness.

All text-comparison metrics apply consistent normalisation (lowercasing,
punctuation and article removal) to reduce surface-form mismatches.
"""

from __future__ import annotations

import math
import re
import string

import numpy as np


# ------------------------------------------------------------------
# Text normalisation helpers
# ------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def _normalise(text: str) -> str:
    """Lowercase, strip articles, remove punctuation, and collapse whitespace."""
    text = text.lower()
    text = _ARTICLES_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    return " ".join(text.split())


def _tokenise(text: str) -> list[str]:
    """Normalise then split into whitespace-delimited tokens."""
    return _normalise(text).split()


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------


class Metrics:
    """Static methods implementing answer-quality and retrieval metrics."""

    # ------------------------------------------------------------------
    # Answer quality
    # ------------------------------------------------------------------

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> float:
        """Normalised exact-match score.

        Returns ``1.0`` if the normalised prediction string equals the
        normalised ground truth, ``0.0`` otherwise.

        Parameters
        ----------
        prediction:
            Model-generated answer string.
        ground_truth:
            Gold-standard answer string.
        """
        return 1.0 if _normalise(prediction) == _normalise(ground_truth) else 0.0

    @staticmethod
    def exact_match_multi(prediction: str, ground_truths: list[str]) -> float:
        """Exact-match against multiple acceptable answers.

        Returns ``1.0`` if the prediction matches *any* of the ground
        truth strings after normalisation.
        """
        norm_pred = _normalise(prediction)
        for gt in ground_truths:
            if norm_pred == _normalise(gt):
                return 1.0
        return 0.0

    @staticmethod
    def token_f1(prediction: str, ground_truth: str) -> float:
        """Token-level F1 score between prediction and ground truth.

        The text is normalised before tokenisation.  If either string is
        empty after normalisation the score is ``0.0`` (unless both are
        empty, in which case it is ``1.0``).

        Parameters
        ----------
        prediction:
            Model-generated answer string.
        ground_truth:
            Gold-standard answer string.
        """
        pred_tokens = _tokenise(prediction)
        truth_tokens = _tokenise(ground_truth)

        if not pred_tokens and not truth_tokens:
            return 1.0
        if not pred_tokens or not truth_tokens:
            return 0.0

        common = set(pred_tokens) & set(truth_tokens)
        if not common:
            return 0.0

        # Count token overlaps (multiset intersection).
        pred_counts: dict[str, int] = {}
        for t in pred_tokens:
            pred_counts[t] = pred_counts.get(t, 0) + 1
        truth_counts: dict[str, int] = {}
        for t in truth_tokens:
            truth_counts[t] = truth_counts.get(t, 0) + 1

        overlap = sum(min(pred_counts.get(t, 0), truth_counts.get(t, 0)) for t in common)

        precision = overlap / len(pred_tokens)
        recall = overlap / len(truth_tokens)

        if precision + recall == 0.0:
            return 0.0

        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def token_f1_multi(prediction: str, ground_truths: list[str]) -> float:
        """Maximum token-F1 across multiple acceptable answers."""
        if not ground_truths:
            return 0.0
        return max(Metrics.token_f1(prediction, gt) for gt in ground_truths)

    # ------------------------------------------------------------------
    # Retrieval quality
    # ------------------------------------------------------------------

    @staticmethod
    def recall_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Recall@K — fraction of relevant items found in the top-K retrieved.

        Parameters
        ----------
        retrieved_ids:
            Ordered list of retrieved document identifiers.
        relevant_ids:
            Set of ground-truth relevant document identifiers.
        k:
            Cut-off rank.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.
        """
        if not relevant_ids:
            return 0.0

        top_k = set(retrieved_ids[:k])
        relevant = set(relevant_ids)
        return len(top_k & relevant) / len(relevant)

    @staticmethod
    def precision_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Precision@K — fraction of top-K retrieved items that are relevant.

        Parameters
        ----------
        retrieved_ids:
            Ordered list of retrieved document identifiers.
        relevant_ids:
            Set of ground-truth relevant document identifiers.
        k:
            Cut-off rank.
        """
        if k <= 0:
            return 0.0

        top_k = retrieved_ids[:k]
        if not top_k:
            return 0.0

        relevant = set(relevant_ids)
        return sum(1 for doc_id in top_k if doc_id in relevant) / len(top_k)

    @staticmethod
    def mrr_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Mean Reciprocal Rank@K.

        Returns ``1 / rank`` of the first relevant document in the top-K,
        or ``0.0`` if none is found.

        Parameters
        ----------
        retrieved_ids:
            Ordered list of retrieved document identifiers.
        relevant_ids:
            Set of ground-truth relevant document identifiers.
        k:
            Cut-off rank.
        """
        relevant = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(
        retrieved_ids: list[str],
        relevant_ids: list[str],
        k: int,
    ) -> float:
        """Normalised Discounted Cumulative Gain@K.

        Uses binary relevance: ``rel(d) = 1`` if ``d in relevant_ids`` else ``0``.

        Parameters
        ----------
        retrieved_ids:
            Ordered list of retrieved document identifiers.
        relevant_ids:
            Set of ground-truth relevant document identifiers.
        k:
            Cut-off rank.

        Returns
        -------
        float
            Value in ``[0.0, 1.0]``.
        """
        if not relevant_ids:
            return 0.0

        relevant = set(relevant_ids)

        # DCG for the retrieved ranking.
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
            if doc_id in relevant:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG — all relevant items ranked at the top.
        ideal_count = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    # ------------------------------------------------------------------
    # Aggregate helpers
    # ------------------------------------------------------------------

    @staticmethod
    def aggregate(scores: list[float]) -> dict[str, float]:
        """Compute mean, standard deviation, median, min, and max.

        Parameters
        ----------
        scores:
            A list of per-sample metric values.

        Returns
        -------
        dict
            Keys: ``mean``, ``std``, ``median``, ``min``, ``max``, ``count``.
        """
        if not scores:
            return {
                "mean": 0.0,
                "std": 0.0,
                "median": 0.0,
                "min": 0.0,
                "max": 0.0,
                "count": 0,
            }

        arr = np.array(scores, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(scores),
        }
