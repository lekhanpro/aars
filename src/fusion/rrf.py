"""Reciprocal Rank Fusion — merges multiple ranked result lists."""

from __future__ import annotations

from collections import defaultdict

import structlog

from src.api.schemas.common import Document

logger = structlog.get_logger()


class ReciprocalRankFusion:
    """Merge multiple ranked document lists using Reciprocal Rank Fusion.

    RRF is a simple, parameter-light fusion technique.  For each document
    *d* appearing in one or more result lists, its fused score is:

        score(d) = sum_{r in result_lists} 1 / (k + rank_r(d))

    where *rank_r(d)* is the 1-based rank of *d* in result list *r*,
    and *k* is a smoothing constant (default 60).
    """

    def __init__(self, k: int = 60) -> None:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")
        self._k = k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fuse(self, result_lists: list[list[Document]]) -> list[Document]:
        """Fuse *result_lists* into a single ranked list.

        Parameters
        ----------
        result_lists:
            Two or more ranked document lists to merge.  Each inner list
            is ordered by descending relevance.

        Returns
        -------
        list[Document]
            De-duplicated documents sorted by descending fused score.
            Each :pyattr:`Document.score` is overwritten with the RRF
            score.
        """
        if not result_lists:
            return []

        # Accumulate RRF scores keyed by document id.
        scores: dict[str, float] = defaultdict(float)
        # Keep the first occurrence of each document to preserve metadata.
        doc_map: dict[str, Document] = {}

        for list_idx, ranked_list in enumerate(result_lists):
            for rank, doc in enumerate(ranked_list, start=1):
                scores[doc.id] += 1.0 / (self._k + rank)
                if doc.id not in doc_map:
                    doc_map[doc.id] = doc

        # Build output list sorted by descending fused score.
        fused: list[Document] = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            original = doc_map[doc_id]
            fused.append(
                Document(
                    id=original.id,
                    content=original.content,
                    metadata=original.metadata,
                    score=score,
                )
            )

        logger.info(
            "rrf.fused",
            input_lists=len(result_lists),
            unique_docs=len(fused),
            k=self._k,
        )
        return fused
