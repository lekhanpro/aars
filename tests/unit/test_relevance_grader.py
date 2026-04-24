"""Tests for the RelevanceGrader agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.relevance_grader import RelevanceGrader, _GradingResult
from src.api.schemas.common import Document, GradedDocument


class TestRelevanceGrader:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.structured_output = AsyncMock()
        self.grader = RelevanceGrader(self.mock_llm)

    @pytest.mark.asyncio
    async def test_grade_mixed_documents(self):
        self.mock_llm.structured_output.return_value = _GradingResult(
            grades=[
                GradedDocument(doc_id="doc_1", relevant=True, reasoning="On topic."),
                GradedDocument(doc_id="doc_2", relevant=False, reasoning="Off topic."),
            ]
        )
        docs = [
            Document(id="doc_1", content="Relevant content", score=0.9),
            Document(id="doc_2", content="Irrelevant content", score=0.3),
        ]
        result = await self.grader.grade("test query", docs)
        assert len(result) == 2
        assert result[0].relevant is True
        assert result[1].relevant is False

    @pytest.mark.asyncio
    async def test_grade_empty_documents(self):
        result = await self.grader.grade("test query", [])
        assert result == []
        self.mock_llm.structured_output.assert_not_called()

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        self.mock_llm.structured_output.side_effect = RuntimeError("LLM failed")
        docs = [Document(id="doc_1", content="Content", score=0.9)]
        result = await self.grader.grade("test query", docs)
        assert len(result) == 1
        assert result[0].relevant is True
