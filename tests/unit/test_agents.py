"""Tests for Planner and Reflection agents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.planner import PlannerAgent
from src.agents.reflection import ReflectionAgent
from src.api.schemas import Document, ReflectionResult, RetrievalPlan


class TestPlannerAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.structured_output = AsyncMock()
        self.planner = PlannerAgent(self.mock_llm)

    @pytest.mark.asyncio
    async def test_plan_factual_query(self):
        expected = RetrievalPlan(
            query_type="factual",
            complexity="simple",
            strategy="keyword",
            rewritten_query="transformer architecture year introduced",
            decomposed_queries=[],
            reasoning="Simple factual query about a date.",
        )
        self.mock_llm.structured_output.return_value = expected

        result = await self.planner.plan("When was the transformer architecture introduced?")
        assert result.strategy.value == "keyword"
        assert result.query_type.value == "factual"
        self.mock_llm.structured_output.assert_called_once()

    @pytest.mark.asyncio
    async def test_plan_multi_hop_query(self):
        expected = RetrievalPlan(
            query_type="multi_hop",
            complexity="complex",
            strategy="graph",
            rewritten_query="BERT relation to transformers",
            decomposed_queries=["What is BERT?", "How does BERT use transformers?"],
            reasoning="Multi-hop query requiring entity relationship traversal.",
        )
        self.mock_llm.structured_output.return_value = expected

        result = await self.planner.plan("How does BERT relate to the transformer architecture?")
        assert result.strategy.value == "graph"
        assert len(result.decomposed_queries) == 2

    @pytest.mark.asyncio
    async def test_plan_conversational_query(self):
        expected = RetrievalPlan(
            query_type="conversational",
            complexity="simple",
            strategy="none",
            rewritten_query="",
            decomposed_queries=[],
            reasoning="Conversational query, no retrieval needed.",
        )
        self.mock_llm.structured_output.return_value = expected

        result = await self.planner.plan("Hello, how are you?")
        assert result.strategy.value == "none"


class TestReflectionAgent:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.structured_output = AsyncMock()
        self.agent = ReflectionAgent(self.mock_llm, max_iterations=3)

    @pytest.mark.asyncio
    async def test_sufficient_documents(self):
        expected = ReflectionResult(
            sufficient=True,
            confidence=0.95,
            missing_information="",
            next_query="",
            next_strategy="",
        )
        self.mock_llm.structured_output.return_value = expected

        docs = [Document(id="1", content="Relevant content here.", score=0.9)]
        result = await self.agent.evaluate("test query", docs)
        assert result.sufficient is True
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_insufficient_documents(self):
        expected = ReflectionResult(
            sufficient=False,
            confidence=0.3,
            missing_information="No information about attention weights.",
            next_query="attention weights computation",
            next_strategy="vector",
        )
        self.mock_llm.structured_output.return_value = expected

        docs = [Document(id="1", content="Basic transformer info.", score=0.5)]
        result = await self.agent.evaluate("How are attention weights computed?", docs)
        assert result.sufficient is False
        assert result.next_strategy == "vector"

    @pytest.mark.asyncio
    async def test_empty_documents(self):
        expected = ReflectionResult(
            sufficient=False,
            confidence=0.0,
            missing_information="No documents retrieved.",
            next_query="attention mechanism",
            next_strategy="vector",
        )
        self.mock_llm.structured_output.return_value = expected

        result = await self.agent.evaluate("test query", [])
        assert result.sufficient is False
