"""Abstract base class for all retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.api.schemas.common import Document


class BaseRetriever(ABC):
    """Interface that every retriever must implement.

    Subclasses provide strategy-specific document lookup (vector, keyword,
    graph, etc.) while the base class defines the common contract consumed
    by the pipeline and the :class:`RetrieverRegistry`.
    """

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Return the *top_k* most relevant documents for *query*.

        Parameters
        ----------
        query:
            Natural-language query string.
        top_k:
            Maximum number of documents to return.

        Returns
        -------
        list[Document]
            Ranked list of documents with populated ``score`` fields.
        """
        ...

    async def initialize(self) -> None:
        """Optional async setup hook (e.g. loading models, connecting to DBs).

        Called once by :pymethod:`RetrieverRegistry.initialize_all` at
        application startup.  The default implementation is a no-op.
        """
