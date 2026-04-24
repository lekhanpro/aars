"""Relevance grader agent — grades individual documents for query relevance."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from src.api.schemas.common import Document, GradedDocument

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts" / "relevance_grader.txt"


class _GradingResult(BaseModel):
    grades: list[GradedDocument]


class RelevanceGrader:
    """Grades each retrieved document as relevant or not relevant.

    Unlike the bulk sufficiency check in :class:`ReflectionAgent`, this
    agent evaluates every document independently, enabling fine-grained
    filtering before fusion and generation.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._template = self._load_template()

    async def grade(
        self, query: str, documents: list[Document]
    ) -> list[GradedDocument]:
        """Grade each document for relevance to *query*.

        Returns a list of :class:`GradedDocument` aligned with the
        input documents.
        """
        if not documents:
            return []

        documents_text = self._format_documents(documents)
        prompt = self._template.format(query=query, documents=documents_text)

        log = logger.bind(query=query, doc_count=len(documents))
        log.info("relevance_grader.grading")

        try:
            result = await self._llm.structured_output(
                prompt=prompt,
                output_model=_GradingResult,
            )
        except Exception:
            log.exception("relevance_grader.grading_failed")
            return [
                GradedDocument(doc_id=doc.id, relevant=True, reasoning="Grading failed; assuming relevant.")
                for doc in documents
            ]

        log.info(
            "relevance_grader.graded",
            relevant=sum(1 for g in result.grades if g.relevant),
            total=len(result.grades),
        )
        return result.grades

    @staticmethod
    def _format_documents(documents: list[Document]) -> str:
        if not documents:
            return "(No documents.)"
        parts: list[str] = []
        for idx, doc in enumerate(documents, start=1):
            parts.append(f"--- Document {idx}: [{doc.id}] ---\n{doc.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _load_template() -> str:
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("relevance_grader.template_not_found", path=str(_PROMPT_PATH))
            raise
