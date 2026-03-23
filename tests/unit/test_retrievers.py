"""Tests for retriever implementations."""

from __future__ import annotations

import pytest

from src.api.schemas import Document
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.none import NoneRetriever
from src.retrieval.registry import RetrieverRegistry


class TestKeywordRetriever:
    def setup_method(self):
        self.retriever = KeywordRetriever()

    @pytest.mark.asyncio
    async def test_empty_index(self):
        result = await self.retriever.retrieve("test query", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_add_and_retrieve(self):
        docs = [
            Document(id="1", content="The quick brown fox jumps over the lazy dog"),
            Document(id="2", content="A fast red fox leaps across the sleeping hound"),
            Document(id="3", content="Python programming language features"),
        ]
        self.retriever.add_documents(docs)
        result = await self.retriever.retrieve("quick brown fox", top_k=2)
        assert len(result) <= 2
        assert result[0].id == "1"  # Best match

    @pytest.mark.asyncio
    async def test_top_k_limit(self):
        docs = [Document(id=str(i), content=f"document about topic {i}") for i in range(20)]
        self.retriever.add_documents(docs)
        result = await self.retriever.retrieve("document topic", top_k=5)
        assert len(result) <= 5

    @pytest.mark.asyncio
    async def test_no_match(self):
        docs = [Document(id="1", content="completely unrelated content xyz")]
        self.retriever.add_documents(docs)
        result = await self.retriever.retrieve("quantum physics equations", top_k=5)
        # BM25 always returns results based on corpus, but scores may be low
        assert isinstance(result, list)


class TestNoneRetriever:
    @pytest.mark.asyncio
    async def test_returns_empty(self):
        retriever = NoneRetriever()
        result = await retriever.retrieve("any query", top_k=10)
        assert result == []


class TestRetrieverRegistry:
    def test_register_and_get(self):
        registry = RetrieverRegistry()
        retriever = NoneRetriever()
        registry.register("none", retriever)
        assert registry.get("none") is retriever

    def test_get_missing(self):
        registry = RetrieverRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_register_multiple(self):
        registry = RetrieverRegistry()
        r1 = NoneRetriever()
        r2 = KeywordRetriever()
        registry.register("none", r1)
        registry.register("keyword", r2)
        assert registry.get("none") is r1
        assert registry.get("keyword") is r2
