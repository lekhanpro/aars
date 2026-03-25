"""Keyword retriever using BM25 (Okapi) ranking."""

from __future__ import annotations

import re
import threading

import structlog

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class _ScoreList(list[float]):
        def tolist(self) -> list[float]:
            return list(self)

    class BM25Okapi:  # type: ignore[override]
        """Lightweight fallback scorer when rank-bm25 is unavailable."""

        def __init__(self, corpus: list[list[str]]) -> None:
            self._corpus = corpus

        def get_scores(self, query_tokens: list[str]) -> _ScoreList:
            query_set = set(query_tokens)
            scores = _ScoreList()
            for document in self._corpus:
                if not document:
                    scores.append(0.0)
                    continue
                hits = sum(1 for token in document if token in query_set)
                scores.append(hits / len(document))
            return scores

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
        self._documents: dict[str, list[Document]] = {}
        self._tokenized_corpus: dict[str, list[list[str]]] = {}
        self._indices: dict[str, BM25Okapi] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Corpus management
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: list[Document],
        collection: str = "default",
    ) -> None:
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
            docs = self._documents.setdefault(collection, [])
            tokenized = self._tokenized_corpus.setdefault(collection, [])
            existing_ids = {doc.id for doc in docs}
            new_docs = [d for d in documents if d.id not in existing_ids]
            if not new_docs:
                logger.debug("keyword_retriever.add_documents.no_new")
                return

            docs.extend(new_docs)
            tokenized.extend(_tokenize(d.content) for d in new_docs)
            self._rebuild_index(collection)

        logger.info(
            "keyword_retriever.add_documents",
            added=len(new_docs),
            total=len(self._documents[collection]),
            collection=collection,
        )

    def clear(self, collection: str | None = None) -> None:
        """Remove all documents and reset the index."""
        with self._lock:
            if collection is None:
                self._documents.clear()
                self._tokenized_corpus.clear()
                self._indices.clear()
            else:
                self._documents.pop(collection, None)
                self._tokenized_corpus.pop(collection, None)
                self._indices.pop(collection, None)
        logger.info("keyword_retriever.cleared", collection=collection or "*")

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        collection: str = "default",
    ) -> list[Document]:
        """Score every document against *query* and return the top-k.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Number of results.  Falls back to ``RetrieverSettings.bm25_top_k``.
        """
        effective_top_k = top_k if top_k is not None else self._settings.bm25_top_k
        documents = self._documents.get(collection, [])
        index = self._indices.get(collection)
        if index is None or not documents:
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
            corpus_size=len(documents),
            collection=collection,
        )

        scores: list[float] = index.get_scores(tokenized_query).tolist()

        # Pair each document with its BM25 score, sort descending.
        scored = sorted(
            zip(documents, scores, strict=True),
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

    def _rebuild_index(self, collection: str) -> None:
        """Rebuild the BM25 index from the current tokenised corpus.

        Must be called while ``self._lock`` is held.
        """
        tokenized = self._tokenized_corpus.get(collection, [])
        if not tokenized:
            self._indices.pop(collection, None)
            return
        self._indices[collection] = BM25Okapi(tokenized)
        logger.debug("keyword_retriever.index_rebuilt", size=len(tokenized), collection=collection)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def corpus_size(self) -> int:
        """Number of documents currently in the index."""
        return sum(len(documents) for documents in self._documents.values())
