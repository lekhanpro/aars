"""Tests for the CrossEncoderReranker agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agents.reranker import CrossEncoderReranker
from src.api.schemas.common import Document


class TestCrossEncoderReranker:
    def setup_method(self):
        self.reranker = CrossEncoderReranker()

    @pytest.mark.asyncio
    async def test_rerank_orders_by_score(self):
        mock_model = MagicMock()
        mock_model.score = MagicMock(return_value=[0.3, 0.9, 0.6])
        self.reranker._model = mock_model

        docs = [
            Document(id="a", content="Low relevance", score=0.9),
            Document(id="b", content="High relevance", score=0.5),
            Document(id="c", content="Medium relevance", score=0.7),
        ]
        result = await self.reranker.rerank("test query", docs, top_k=2)

        assert len(result) == 2
        assert result[0].id == "b"
        assert result[1].id == "c"

    @pytest.mark.asyncio
    async def test_rerank_empty_documents(self):
        result = await self.reranker.rerank("test query", [], top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_respects_top_k(self):
        mock_model = MagicMock()
        mock_model.score = MagicMock(return_value=[0.9, 0.8, 0.7, 0.6])
        self.reranker._model = mock_model

        docs = [
            Document(id=f"doc_{i}", content=f"Content {i}", score=0.5)
            for i in range(4)
        ]
        result = await self.reranker.rerank("test query", docs, top_k=2)
        assert len(result) == 2
