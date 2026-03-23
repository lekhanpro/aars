"""Maximal Marginal Relevance — diversity-aware reranking."""

from __future__ import annotations

import numpy as np
import structlog

from src.api.schemas.common import Document

logger = structlog.get_logger()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D vectors.

    Returns 0.0 when either vector has zero magnitude, avoiding division
    by zero.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class MaximalMarginalRelevance:
    """Rerank documents using Maximal Marginal Relevance (MMR).

    MMR balances *relevance* to the query with *diversity* among the
    selected documents:

        MMR(d) = lambda * sim(d, q) - (1 - lambda) * max_{s in S} sim(d, s)

    where *S* is the set of already-selected documents.
    """

    def __init__(self, lambda_param: float = 0.5) -> None:
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda_param must be in [0, 1], got {lambda_param}")
        self._lambda = lambda_param

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rerank(
        self,
        documents: list[Document],
        query_embedding: list[float],
        doc_embeddings: list[list[float]],
        top_k: int = 5,
    ) -> list[Document]:
        """Rerank *documents* for relevance and diversity.

        Parameters
        ----------
        documents:
            Candidate documents to rerank.
        query_embedding:
            Embedding vector for the query.
        doc_embeddings:
            Embedding vectors for each document, aligned by index with
            *documents*.
        top_k:
            Number of documents to return.

        Returns
        -------
        list[Document]
            Up to *top_k* documents ordered by MMR score.  Each
            :pyattr:`Document.score` is overwritten with the MMR score.

        Raises
        ------
        ValueError
            If *documents* and *doc_embeddings* have different lengths.
        """
        if len(documents) != len(doc_embeddings):
            raise ValueError(
                f"documents ({len(documents)}) and doc_embeddings "
                f"({len(doc_embeddings)}) must have the same length"
            )

        if not documents:
            return []

        top_k = min(top_k, len(documents))
        query_vec = np.asarray(query_embedding, dtype=np.float64)
        doc_vecs = np.asarray(doc_embeddings, dtype=np.float64)

        # Pre-compute relevance scores: sim(d_i, q) for every candidate.
        relevance = np.array(
            [_cosine_similarity(doc_vecs[i], query_vec) for i in range(len(documents))],
            dtype=np.float64,
        )

        selected_indices: list[int] = []
        remaining = set(range(len(documents)))

        for _ in range(top_k):
            best_idx = -1
            best_mmr = -float("inf")

            for idx in remaining:
                rel_score = relevance[idx]

                # Maximum similarity to any already-selected document.
                if selected_indices:
                    max_sim = max(
                        _cosine_similarity(doc_vecs[idx], doc_vecs[s])
                        for s in selected_indices
                    )
                else:
                    max_sim = 0.0

                mmr_score = self._lambda * rel_score - (1.0 - self._lambda) * max_sim

                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            if best_idx == -1:
                break  # pragma: no cover — safety net

            selected_indices.append(best_idx)
            remaining.discard(best_idx)

        # Build output with MMR scores assigned.
        result: list[Document] = []
        for rank, idx in enumerate(selected_indices):
            doc = documents[idx]
            # Recompute final MMR score for logging/downstream use.
            if rank == 0:
                mmr_final = self._lambda * relevance[idx]
            else:
                max_sim = max(
                    _cosine_similarity(doc_vecs[idx], doc_vecs[s])
                    for s in selected_indices[:rank]
                )
                mmr_final = self._lambda * relevance[idx] - (1.0 - self._lambda) * max_sim

            result.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=float(mmr_final),
                )
            )

        logger.info(
            "mmr.reranked",
            candidates=len(documents),
            selected=len(result),
            lambda_param=self._lambda,
        )
        return result
