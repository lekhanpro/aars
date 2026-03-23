"""Fusion pipeline — chains RRF and MMR into a single reranking step."""

from __future__ import annotations

import structlog

from src.api.schemas.common import Document
from src.fusion.mmr import MaximalMarginalRelevance
from src.fusion.rrf import ReciprocalRankFusion

logger = structlog.get_logger()


class FusionPipeline:
    """Two-stage fusion: Reciprocal Rank Fusion followed by MMR reranking.

    Stage 1 (RRF) merges multiple ranked result lists into a single
    de-duplicated list ordered by fused relevance score.

    Stage 2 (MMR) reranks the merged list to balance relevance against
    diversity, returning the final *top_k* documents.
    """

    def __init__(
        self,
        rrf: ReciprocalRankFusion,
        mmr: MaximalMarginalRelevance,
        final_top_k: int = 5,
    ) -> None:
        self._rrf = rrf
        self._mmr = mmr
        self._final_top_k = final_top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def fuse(
        self,
        result_lists: list[list[Document]],
        query_embedding: list[float],
        doc_embeddings: list[list[float]],
    ) -> list[Document]:
        """Merge and rerank *result_lists* into a final document set.

        Parameters
        ----------
        result_lists:
            Multiple ranked document lists produced by different
            retrieval strategies.
        query_embedding:
            Embedding vector for the user query.
        doc_embeddings:
            Embedding vectors for every *unique* document across all
            result lists, **keyed by document id** — i.e. one embedding
            per unique document.  The caller must align these with the
            de-duplicated order produced by RRF.  In practice, it is
            simpler to supply a mapping and let the pipeline look up
            embeddings after RRF.  This parameter expects a flat list
            aligned with the RRF-merged output (same order, same
            length).

        Returns
        -------
        list[Document]
            Up to ``final_top_k`` documents sorted by MMR score.
        """
        log = logger.bind(
            input_lists=len(result_lists),
            final_top_k=self._final_top_k,
        )
        log.info("fusion_pipeline.start")

        # Stage 1: Reciprocal Rank Fusion
        merged = self._rrf.fuse(result_lists)
        if not merged:
            log.info("fusion_pipeline.empty_after_rrf")
            return []

        log.info("fusion_pipeline.rrf_complete", merged_count=len(merged))

        # Align doc_embeddings with the merged order.  If the caller
        # provided fewer embeddings than merged docs (e.g. embeddings
        # only for a subset), we truncate to the common length.
        usable = min(len(merged), len(doc_embeddings))
        if usable == 0:
            log.warning("fusion_pipeline.no_embeddings_for_mmr")
            return merged[: self._final_top_k]

        merged_subset = merged[:usable]
        embeddings_subset = doc_embeddings[:usable]

        # Stage 2: Maximal Marginal Relevance
        result = self._mmr.rerank(
            documents=merged_subset,
            query_embedding=query_embedding,
            doc_embeddings=embeddings_subset,
            top_k=self._final_top_k,
        )

        log.info("fusion_pipeline.complete", result_count=len(result))
        return result
