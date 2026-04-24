"""Intent router agent — classifies query intent before retrieval planning."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel

from src.api.schemas.common import IntentType

if TYPE_CHECKING:
    from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts" / "intent_router.txt"


class _IntentClassification(BaseModel):
    intent: IntentType
    reasoning: str


class IntentRouter:
    """Classifies a user query into an intent type before retrieval.

    Routes queries as simple, complex, multi_hop, or direct. Direct
    queries skip retrieval entirely, saving cost on questions the LLM
    can answer from training data.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._template = self._load_template()

    async def classify(self, query: str) -> IntentType:
        """Classify *query* into an intent type.

        Returns
        -------
        IntentType
            One of simple, complex, multi_hop, or direct.
        """
        prompt = self._template.format(query=query)
        log = logger.bind(query=query)
        log.info("intent_router.classifying")

        try:
            result = await self._llm.structured_output(
                prompt=prompt,
                output_model=_IntentClassification,
            )
        except Exception:
            log.exception("intent_router.classification_failed")
            return IntentType.SIMPLE

        log.info(
            "intent_router.classified",
            intent=result.intent.value,
            reasoning=result.reasoning[:100],
        )
        return result.intent

    @staticmethod
    def _load_template() -> str:
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("intent_router.template_not_found", path=str(_PROMPT_PATH))
            raise
