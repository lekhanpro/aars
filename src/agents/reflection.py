"""Reflection agent — evaluates whether retrieved documents suffice."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.common import Document, ReflectionResult

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts" / "reflection.txt"


class ReflectionAgent:
    """Evaluates the quality of retrieved documents against a user query.

    Determines whether the current document set is sufficient to generate
    a high-quality answer, and—if not—suggests a follow-up query and
    strategy.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        max_iterations: int = 3,
    ) -> None:
        self._llm = llm_client
        self.max_iterations = max_iterations
        self._template = self._load_template()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        query: str,
        documents: list[Document],
    ) -> ReflectionResult:
        """Evaluate whether *documents* are sufficient to answer *query*.

        Parameters
        ----------
        query:
            The original user query.
        documents:
            Retrieved documents to evaluate.

        Returns
        -------
        ReflectionResult
            Assessment including sufficiency flag, confidence score,
            missing-information description, and optional next-query
            suggestion.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed into a valid
            ``ReflectionResult``.
        """
        documents_text = self._format_documents(documents)
        prompt = self._template.format(query=query, documents=documents_text)

        log = logger.bind(query=query, doc_count=len(documents))
        log.info("reflection.evaluating")

        try:
            result = await self._llm.structured_output(
                prompt=prompt,
                output_model=ReflectionResult,
            )
        except Exception:
            log.exception("reflection.evaluation_failed")
            raise

        log.info(
            "reflection.evaluation_complete",
            sufficient=result.sufficient,
            confidence=result.confidence,
            has_next_query=bool(result.next_query),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        """Render documents into a numbered text block for the prompt."""
        if not documents:
            return "(No documents retrieved.)"

        parts: list[str] = []
        for idx, doc in enumerate(documents, start=1):
            header = f"[{doc.id}] (score: {doc.score:.4f})"
            parts.append(f"--- Document {idx}: {header} ---\n{doc.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _load_template() -> str:
        """Read the reflection prompt template from disk."""
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("reflection.template_not_found", path=str(_PROMPT_PATH))
            raise
