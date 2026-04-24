"""Tests for the SelfRAGEvaluator agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.self_rag_evaluator import SelfRAGEvaluator
from src.api.schemas.common import Document, SelfRAGEvaluation


class TestSelfRAGEvaluator:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.structured_output = AsyncMock()
        self.evaluator = SelfRAGEvaluator(self.mock_llm)

    @pytest.mark.asyncio
    async def test_evaluate_high_quality(self):
        self.mock_llm.structured_output.return_value = SelfRAGEvaluation(
            faithfulness=0.95,
            answer_relevancy=0.90,
            context_precision=0.88,
            context_recall=0.85,
            overall=0.90,
        )
        docs = [Document(id="1", content="Relevant content.", score=0.9)]
        result = await self.evaluator.evaluate(
            question="test query",
            answer="Good answer grounded in documents.",
            documents=docs,
        )
        assert result.overall == 0.90
        assert result.faithfulness == 0.95

    @pytest.mark.asyncio
    async def test_evaluate_low_quality(self):
        self.mock_llm.structured_output.return_value = SelfRAGEvaluation(
            faithfulness=0.2,
            answer_relevancy=0.3,
            context_precision=0.4,
            context_recall=0.1,
            overall=0.25,
        )
        docs = [Document(id="1", content="Irrelevant.", score=0.2)]
        result = await self.evaluator.evaluate(
            question="test query",
            answer="Fabricated answer.",
            documents=docs,
        )
        assert result.overall == 0.25
        assert result.faithfulness == 0.2

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        self.mock_llm.structured_output.side_effect = RuntimeError("LLM failed")
        docs = [Document(id="1", content="Content", score=0.9)]
        result = await self.evaluator.evaluate(
            question="test", answer="answer", documents=docs,
        )
        assert result.overall == 0.5
        assert result.faithfulness == 0.5
