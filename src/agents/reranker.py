"""Cross-encoder reranker agent — reranks retrieved documents using a cross-encoder model."""

from __future__ import annotations

import asyncio

import structlog

from src.api.schemas.common import Document
from src.utils.cross_encoder import CrossEncoderModel

logger = structlog.get_logger()

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks documents using a cross-encoder relevance model.

    This agent sits between fusion and generation, providing a more
    accurate relevance ranking than vector similarity or BM25 alone.
    Cross-encoders jointly encode the query and document, producing
    higher-quality scores at the cost of being slower.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: CrossEncoderModel | None = None

    def _ensure_model(self) -> CrossEncoderModel:
        if self._model is None:
            self._model = CrossEncoderModel.get(self._model_name)
        return self._model

    async def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[Document]:
        """Rerank *documents* by cross-encoder relevance to *query*.

        Parameters
        ----------
        query:
            The user query.
        documents:
            Documents to rerank (typically from fusion output).
        top_k:
            Maximum number of documents to return.

        Returns
        -------
        list[Document]
            Documents sorted by cross-encoder score, limited to *top_k*.
        """
        if not documents:
            return []

        log = logger.bind(query=query[:80], doc_count=len(documents))
        log.info("reranker.reranking")

        model = self._ensure_model()
        contents = [doc.content for doc in documents]

        scores = await asyncio.to_thread(model.score, query, contents)

        scored_docs = sorted(
            zip(documents, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )

        results: list[Document] = []
        for doc, score in scored_docs[:top_k]:
            results.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score,
                    modality=doc.modality,
                )
            )

        log.info("reranker.reranked", result_count=len(results))
        return results
