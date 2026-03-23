"""Pass-through retriever that returns no documents.

Used when the planner determines the query can be answered directly by the
LLM without any external context (e.g. conversational / opinion queries).
"""

from __future__ import annotations

import structlog

from src.api.schemas.common import Document
from src.retrieval.base import BaseRetriever

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class NoneRetriever(BaseRetriever):
    """Retriever that always returns an empty document list."""

    async def retrieve(self, query: str, top_k: int = 10) -> list[Document]:
        """Return an empty list — no retrieval needed."""
        logger.debug("none_retriever.skip", query=query[:120])
        return []
