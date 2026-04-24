"""Tests for the IntentRouter agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.intent_router import IntentRouter, _IntentClassification
from src.api.schemas.common import IntentType


class TestIntentRouter:
    def setup_method(self):
        self.mock_llm = MagicMock()
        self.mock_llm.structured_output = AsyncMock()
        self.router = IntentRouter(self.mock_llm)

    @pytest.mark.asyncio
    async def test_classify_simple(self):
        self.mock_llm.structured_output.return_value = _IntentClassification(
            intent=IntentType.SIMPLE,
            reasoning="Single-fact question.",
        )
        result = await self.router.classify("What year was Python created?")
        assert result == IntentType.SIMPLE

    @pytest.mark.asyncio
    async def test_classify_complex(self):
        self.mock_llm.structured_output.return_value = _IntentClassification(
            intent=IntentType.COMPLEX,
            reasoning="Requires reasoning over multiple chunks.",
        )
        result = await self.router.classify("Compare BERT and GPT architectures")
        assert result == IntentType.COMPLEX

    @pytest.mark.asyncio
    async def test_classify_multi_hop(self):
        self.mock_llm.structured_output.return_value = _IntentClassification(
            intent=IntentType.MULTI_HOP,
            reasoning="Sequential retrieval steps needed.",
        )
        result = await self.router.classify("What did the CEO of OpenAI say about AGI?")
        assert result == IntentType.MULTI_HOP

    @pytest.mark.asyncio
    async def test_classify_direct(self):
        self.mock_llm.structured_output.return_value = _IntentClassification(
            intent=IntentType.DIRECT,
            reasoning="Can answer from training data.",
        )
        result = await self.router.classify("What is 2 + 2?")
        assert result == IntentType.DIRECT

    @pytest.mark.asyncio
    async def test_fallback_on_error(self):
        self.mock_llm.structured_output.side_effect = RuntimeError("LLM failed")
        result = await self.router.classify("test query")
        assert result == IntentType.SIMPLE
