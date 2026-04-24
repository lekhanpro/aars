"""Tests for the HallucinationChecker agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.hallucination_checker import HallucinationChecker
from src.api.schemas.common import Document, HallucinationResult


class TestHallucinationChecker:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.structured_output = AsyncMock()
        self.checker = HallucinationChecker(self.mock_llm, mode="llm")

    @pytest.mark.asyncio
    async def test_grounded_answer(self):
        self.mock_llm.structured_output.return_value = HallucinationResult(
            grounded=True,
            score=0.95,
            ungrounded_claims=[],
            reasoning="All claims supported by documents.",
        )
        docs = [Document(id="1", content="Transformers were introduced in 2017.", score=0.9)]
        result = await self.checker.check(
            question="When were transformers introduced?",
            answer="Transformers were introduced in 2017.",
            documents=docs,
        )
        assert result.grounded is True
        assert result.score == 0.95
        assert result.ungrounded_claims == []

    @pytest.mark.asyncio
    async def test_ungrounded_answer(self):
        self.mock_llm.structured_output.return_value = HallucinationResult(
            grounded=False,
            score=0.3,
            ungrounded_claims=["GPT-5 was released in 2024."],
            reasoning="The claim about GPT-5 is not in the documents.",
        )
        docs = [Document(id="1", content="GPT-4 was released in 2023.", score=0.9)]
        result = await self.checker.check(
            question="When was GPT-5 released?",
            answer="GPT-5 was released in 2024.",
            documents=docs,
        )
        assert result.grounded is False
        assert len(result.ungrounded_claims) == 1

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        self.mock_llm.structured_output.side_effect = RuntimeError("LLM failed")
        docs = [Document(id="1", content="Content", score=0.9)]
        result = await self.checker.check(
            question="test", answer="test answer", documents=docs,
        )
        assert result.grounded is True
        assert result.score == 0.5
