"""Central registry that maps strategy names to retriever instances."""

from __future__ import annotations

import structlog

from src.retrieval.base import BaseRetriever

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class RetrieverRegistry:
    """Stores named :class:`BaseRetriever` instances and exposes them by key.

    Usage::

        registry = RetrieverRegistry()
        registry.register("vector", VectorRetriever(...))
        retriever = registry.get("vector")
    """

    def __init__(self) -> None:
        self._retrievers: dict[str, BaseRetriever] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, retriever: BaseRetriever) -> None:
        """Register a retriever under *name* (case-insensitive).

        Raises
        ------
        ValueError
            If a retriever is already registered under the same name.
        """
        key = name.lower()
        if key in self._retrievers:
            raise ValueError(
                f"Retriever already registered under name '{key}'. "
                "Unregister it first or choose a different name."
            )
        self._retrievers[key] = retriever
        logger.info("retriever_registry.registered", name=key)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, strategy: str) -> BaseRetriever:
        """Return the retriever registered under *strategy* (case-insensitive).

        Raises
        ------
        KeyError
            If no retriever is registered for the given strategy.
        """
        key = strategy.lower()
        try:
            return self._retrievers[key]
        except KeyError:
            available = ", ".join(sorted(self._retrievers)) or "(none)"
            raise KeyError(
                f"No retriever registered for strategy '{key}'. "
                f"Available: {available}"
            ) from None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize_all(self) -> None:
        """Call :pymethod:`initialize` on every registered retriever."""
        for name, retriever in self._retrievers.items():
            logger.info("retriever_registry.initializing", name=name)
            await retriever.initialize()
            logger.info("retriever_registry.initialized", name=name)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        """Return sorted list of registered retriever names."""
        return sorted(self._retrievers)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._retrievers

    def __len__(self) -> int:
        return len(self._retrievers)

    def __repr__(self) -> str:
        return f"RetrieverRegistry(retrievers={self.names})"
