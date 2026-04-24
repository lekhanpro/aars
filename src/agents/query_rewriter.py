"""Query rewriter agent — rewrites failed queries to improve retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts" / "query_rewriter.txt"


class RewrittenQuery(BaseModel):
    rewritten: str
    technique: str
    reasoning: str


class QueryRewriter:
    """Rewrites queries when retrieval fails to find relevant documents.

    Applies three techniques: synonym expansion, abstraction to broader
    concepts, or contextual enrichment. Tracks rewrite attempts to
    avoid infinite loops.
    """

    MAX_REWRITES = 3

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._template = self._load_template()

    async def rewrite(self, query: str, context: str = "") -> RewrittenQuery:
        """Produce a rewritten query for better retrieval.

        Parameters
        ----------
        query:
            The original or previously rewritten query.
        context:
            Description of why previous retrieval failed (e.g.,
            "No relevant documents found" or "Low confidence score").
        """
        prompt = self._template.format(query=query, context=context or "No relevant documents found")
        log = logger.bind(query=query)
        log.info("query_rewriter.rewriting")

        try:
            result = await self._llm.structured_output(
                prompt=prompt,
                output_model=RewrittenQuery,
            )
        except Exception:
            log.exception("query_rewriter.rewrite_failed")
            return RewrittenQuery(
                rewritten=query,
                technique="none",
                reasoning="Rewrite failed; returning original query.",
            )

        log.info(
            "query_rewriter.rewritten",
            technique=result.technique,
            rewritten=result.rewritten[:120],
        )
        return result

    @staticmethod
    def _load_template() -> str:
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("query_rewriter.template_not_found", path=str(_PROMPT_PATH))
            raise
