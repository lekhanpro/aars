"""Planner agent — classifies queries and selects retrieval strategies."""

from __future__ import annotations

from pathlib import Path

import structlog

from src.api.schemas.common import RetrievalPlan
from src.llm.client import LLMClient

logger = structlog.get_logger()

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "config" / "prompts" / "planner.txt"


class PlannerAgent:
    """Analyzes a user query and produces a structured retrieval plan.

    Uses the LLM to classify query type, complexity, and optimal retrieval
    strategy, returning a fully-populated :class:`RetrievalPlan`.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._template = self._load_template()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def plan(self, query: str) -> RetrievalPlan:
        """Generate a retrieval plan for *query*.

        Parameters
        ----------
        query:
            The raw user query to analyze.

        Returns
        -------
        RetrievalPlan
            Structured plan including query type, complexity, strategy,
            rewritten query, optional decomposed sub-queries, and reasoning.

        Raises
        ------
        ValueError
            If the LLM response cannot be parsed into a valid
            ``RetrievalPlan``.
        """
        prompt = self._template.format(query=query)
        log = logger.bind(query=query)
        log.info("planner.generating_plan")

        try:
            plan = await self._llm.structured_output(
                prompt=prompt,
                output_model=RetrievalPlan,
            )
        except Exception:
            log.exception("planner.plan_failed")
            raise

        log.info(
            "planner.plan_complete",
            query_type=plan.query_type.value,
            complexity=plan.complexity.value,
            strategy=plan.strategy.value,
            decomposed_count=len(plan.decomposed_queries),
        )
        return plan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_template() -> str:
        """Read the planner prompt template from disk."""
        try:
            return _PROMPT_PATH.read_text(encoding="utf-8")
        except FileNotFoundError:
            logger.error("planner.template_not_found", path=str(_PROMPT_PATH))
            raise
