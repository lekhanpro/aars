"""Answer generator — produces grounded, cited answers from retrieved documents."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from src.api.schemas.common import Citation, Document

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts" / "answer.txt"


# ------------------------------------------------------------------
# Local response model
# ------------------------------------------------------------------


class AnswerResult(BaseModel):
    """Structured output returned by the answer-generation step."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


# ------------------------------------------------------------------
# Generator
# ------------------------------------------------------------------


class AnswerGenerator:
    """Generate grounded answers with citations using an LLM.

    Loads the answer prompt template from ``config/prompts/answer.txt``,
    formats retrieved documents with their IDs, and uses
    :meth:`LLMClient.structured_output` to produce an
    :class:`AnswerResult`.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._template = self._load_template()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        query: str,
        documents: list[Document],
    ) -> AnswerResult:
        """Generate a cited answer for *query* using *documents*.

        Parameters
        ----------
        query:
            The user's question.
        documents:
            Retrieved (and reranked) documents to ground the answer.

        Returns
        -------
        AnswerResult
            The generated answer text, supporting citations, a
            confidence score, and reasoning.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed into a valid
            ``AnswerResult``.
        """
        documents_text = self._format_documents(documents)
        prompt = self._template.format(query=query, documents=documents_text)

        log = logger.bind(query=query, doc_count=len(documents))
        log.info("answer_generator.generating")

        try:
            result = await self._llm.structured_output(
                prompt=prompt,
                output_model=AnswerResult,
            )
        except Exception:
            log.exception("answer_generator.generation_failed")
            raise

        log.info(
            "answer_generator.complete",
            confidence=result.confidence,
            citation_count=len(result.citations),
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        """Render documents into a numbered text block for the prompt."""
        if not documents:
            return "(No documents provided.)"

        parts: list[str] = []
        for idx, doc in enumerate(documents, start=1):
            header = f"[{doc.id}]"
            parts.append(f"--- Document {idx}: {header} ---\n{doc.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _load_template() -> str:
        """Read the answer prompt template from disk."""
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("answer_generator.template_not_found", path=str(_PROMPT_PATH))
            raise
