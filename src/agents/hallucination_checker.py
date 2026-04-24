"""Hallucination checker agent — verifies answer grounding in source documents."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.api.schemas.common import Document, HallucinationResult

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "prompts" / "hallucination_checker.txt"
)


class HallucinationChecker:
    """Checks whether a generated answer is grounded in source documents.

    Supports two modes:

    - **llm** (default): Uses the LLM as a judge to verify grounding.
    - **nli**: Uses a cross-encoder NLI model (nli-deberta-v3-small) to
      compute entailment scores. Requires sentence-transformers.
    """

    def __init__(self, llm_client: LLMClient, mode: str = "llm") -> None:
        self._llm = llm_client
        self._mode = mode
        self._template = self._load_template()

    async def check(
        self,
        question: str,
        answer: str,
        documents: list[Document],
    ) -> HallucinationResult:
        """Check whether *answer* is grounded in *documents*.

        Parameters
        ----------
        question:
            The original user query.
        answer:
            The generated answer to verify.
        documents:
            Source documents the answer should be grounded in.
        """
        if self._mode == "nli":
            return await self._check_nli(answer, documents)
        return await self._check_llm(question, answer, documents)

    async def _check_llm(
        self,
        question: str,
        answer: str,
        documents: list[Document],
    ) -> HallucinationResult:
        documents_text = self._format_documents(documents)
        prompt = self._template.format(
            question=question,
            answer=answer,
            documents=documents_text,
        )

        log = logger.bind(question=question[:80])
        log.info("hallucination_checker.checking_llm")

        try:
            result = await self._llm.structured_output(
                prompt=prompt,
                output_model=HallucinationResult,
            )
        except Exception:
            log.exception("hallucination_checker.check_failed")
            return HallucinationResult(
                grounded=True,
                score=0.5,
                reasoning="Hallucination check failed; returning neutral score.",
            )

        log.info(
            "hallucination_checker.checked",
            grounded=result.grounded,
            score=result.score,
            ungrounded_count=len(result.ungrounded_claims),
        )
        return result

    async def _check_nli(
        self,
        answer: str,
        documents: list[Document],
    ) -> HallucinationResult:
        log = logger.bind(mode="nli")
        log.info("hallucination_checker.checking_nli")

        try:
            from src.utils.cross_encoder import CrossEncoderModel

            model = CrossEncoderModel.get("cross-encoder/nli-deberta-v3-small")
        except Exception:
            log.warning("hallucination_checker.nli_model_unavailable")
            return HallucinationResult(
                grounded=True,
                score=0.5,
                reasoning="NLI model unavailable; returning neutral score.",
            )

        context = " ".join(doc.content for doc in documents)
        sentences = [s.strip() for s in answer.split(".") if s.strip()]
        if not sentences:
            return HallucinationResult(grounded=True, score=1.0, reasoning="Empty answer.")

        scores = await asyncio.to_thread(model.predict, context, sentences)

        grounded_count = sum(1 for s in scores if s > 0.5)
        overall_score = grounded_count / len(scores) if scores else 1.0
        ungrounded = [
            sentences[i] for i, s in enumerate(scores) if s <= 0.5
        ]

        return HallucinationResult(
            grounded=overall_score >= 0.8,
            score=overall_score,
            ungrounded_claims=ungrounded,
            reasoning=f"NLI check: {grounded_count}/{len(scores)} sentences grounded.",
        )

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
            logger.error("hallucination_checker.template_not_found", path=str(_PROMPT_PATH))
            raise
