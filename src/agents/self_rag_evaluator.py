"""Self-RAG evaluator agent — computes inline RAGAS-style quality scores."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.common import Document, SelfRAGEvaluation

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "prompts" / "self_rag_evaluator.txt"
)


class SelfRAGEvaluator:
    """Computes RAGAS-style evaluation scores inline during query processing.

    Scores four dimensions: faithfulness, answer_relevancy,
    context_precision, and context_recall. The overall score is a
    weighted average (faithfulness 0.3, relevancy 0.3, precision 0.2,
    recall 0.2).
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._template = self._load_template()

    async def evaluate(
        self,
        question: str,
        answer: str,
        documents: list[Document],
    ) -> SelfRAGEvaluation:
        """Evaluate the quality of *answer* for *question* given *documents*.

        Returns
        -------
        SelfRAGEvaluation
            Scores for faithfulness, answer_relevancy, context_precision,
            context_recall, and an overall weighted score.
        """
        documents_text = self._format_documents(documents)
        prompt = self._template.format(
            question=question,
            answer=answer,
            documents=documents_text,
        )

        log = logger.bind(question=question[:80])
        log.info("self_rag_evaluator.evaluating")

        try:
            result = await self._llm.structured_output(
                prompt=prompt,
                output_model=SelfRAGEvaluation,
            )
        except Exception:
            log.exception("self_rag_evaluator.evaluation_failed")
            return SelfRAGEvaluation(
                faithfulness=0.5,
                answer_relevancy=0.5,
                context_precision=0.5,
                context_recall=0.5,
                overall=0.5,
            )

        log.info(
            "self_rag_evaluator.evaluated",
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            overall=result.overall,
        )
        return result

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
            logger.error("self_rag_evaluator.template_not_found", path=str(_PROMPT_PATH))
            raise
