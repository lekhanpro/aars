"""Keyword retriever using BM25 (Okapi) ranking."""

from __future__ import annotations

import re
import threading

import structlog
from rank_bm25 import BM25Okapi

from config.settings import RetrieverSettings
from src.api.schemas.common import Document
from src.retrieval.base import BaseRetriever

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Pre-compiled regex for lightweight tokenisation.
_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _tokenize(text: str) -> list[str]:
    """Lower-case, strip punctuation, and split on whitespace."""
    cleaned = _NON_ALPHA.sub("", text.lower())
    return cleaned.split()


class KeywordRetriever(BaseRetriever):
    """BM25-based sparse retriever.

    Documents must be added explicitly via :pymethod:`add_documents` before
    querying.  The BM25 index is rebuilt each time the corpus changes.

    Thread-safety is guaranteed by a :class:`threading.Lock` around index
    mutations.
    """

    def __init__(self, retriever_settings: RetrieverSettings | None = None) -> None:
        self._settings = retriever_settings or RetrieverSettings()
        self._documents: list[Document] = []
        self._tokenized_corpus: list[list[str]] = []
        self._index: BM25Okapi | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[Document]) -> None:
        """Append *documents* to the corpus and rebuild the BM25 index.

        Parameters
        ----------
        documents:
            One or more :class:`Document` instances.  Duplicates (by id) are
            silently skipped.
        """
        if not documents:
            return

        with self._lock:
            existing_ids = {doc.id for doc in self._documents}
            new_docs = [d for d in documents if d.id not in existing_ids]
            if not new_docs:
                logger.debug("keyword_retriever.add_documents.no_new")
                return

            self._documents.extend(new_docs)
            self._tokenized_corpus.extend(_tokenize(d.content) for d in new_docs)
            self._rebuild_index()

        logger.info(
            "keyword_retriever.add_documents",
            added=len(new_docs),
            total=len(self._documents),
        )

    def clear(self) -> None:
        """Remove all documents and reset the index."""
        with self._lock:
            self._documents.clear()
            self._tokenized_corpus.clear()
            self._index = None
        logger.info("keyword_retriever.cleared")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Score every document against *query* and return the top-k.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Number of results.  Falls back to ``RetrieverSettings.bm25_top_k``.
        """
        effective_top_k = top_k if top_k is not None else self._settings.bm25_top_k

        if self._index is None or not self._documents:
            logger.debug("keyword_retriever.empty_index", query=query[:120])
            return []

        tokenized_query = _tokenize(query)
        if not tokenized_query:
            logger.debug("keyword_retriever.empty_query")
            return []

        logger.debug(
            "keyword_retriever.query",
            query=query[:120],
            top_k=effective_top_k,
            corpus_size=len(self._documents),
        )

        scores: list[float] = self._index.get_scores(tokenized_query).tolist()

        # Pair each document with its BM25 score, sort descending.
        scored = sorted(
            zip(self._documents, scores, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )

        results: list[Document] = []
        for doc, score in scored[:effective_top_k]:
            if score <= 0.0:
                break
            results.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score,
                )
            )

        logger.debug("keyword_retriever.results", count=len(results))
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild_index(self) -> None:
        """Rebuild the BM25 index from the current tokenised corpus.

        Must be called while ``self._lock`` is held.
        """
        if not self._tokenized_corpus:
            self._index = None
            return
        self._index = BM25Okapi(self._tokenized_corpus)
        logger.debug("keyword_retriever.index_rebuilt", size=len(self._tokenized_corpus))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of documents currently in the index."""
        return len(self._documents)
