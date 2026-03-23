"""Shared test fixtures."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.settings import Settings
from src.api.schemas import Document, ReflectionResult, RetrievalPlan


@pytest.fixture
def settings():
    return Settings(anthropic_api_key="test-key")


@pytest.fixture
def sample_documents():
    return [
        Document(
            id="doc_1",
            content="The transformer architecture was introduced in 2017 by Vaswani et al.",
            metadata={"source": "test.pdf", "page": 1},
            score=0.95,
        ),
        Document(
            id="doc_2",
            content="Attention mechanisms allow models to focus on relevant parts of the input.",
            metadata={"source": "test.pdf", "page": 2},
            score=0.88,
        ),
        Document(
            id="doc_3",
            content="BERT uses bidirectional transformers for language understanding tasks.",
            metadata={"source": "test2.pdf", "page": 1},
            score=0.82,
        ),
        Document(
            id="doc_4",
            content="GPT models use autoregressive transformers for text generation.",
            metadata={"source": "test2.pdf", "page": 3},
            score=0.78,
        ),
        Document(
            id="doc_5",
            content="Retrieval-augmented generation combines retrieval with language models.",
            metadata={"source": "test3.pdf", "page": 1},
            score=0.75,
        ),
    ]


@pytest.fixture
def sample_plan():
    return RetrievalPlan(
        query_type="factual",
        complexity="simple",
        strategy="keyword",
        rewritten_query="transformer architecture introduction year",
        decomposed_queries=[],
        reasoning="Factual query about a specific fact, keyword retrieval is optimal.",
    )


@pytest.fixture
def sample_reflection_sufficient():
    return ReflectionResult(
        sufficient=True,
        confidence=0.92,
        missing_information="",
        next_query="",
        next_strategy="",
    )


@pytest.fixture
def sample_reflection_insufficient():
    return ReflectionResult(
        sufficient=False,
        confidence=0.4,
        missing_information="Missing details about the specific attention mechanism used.",
        next_query="multi-head self-attention mechanism details",
        next_strategy="vector",
    )


@pytest.fixture
def mock_llm_client():
    client = MagicMock()
    client.generate = AsyncMock(return_value="Test response")
    client.structured_output = AsyncMock()
    client.total_tokens = 0
    client.total_calls = 0
    client.reset_counters = MagicMock()
    return client


@pytest.fixture
def mock_chroma_client():
    client = MagicMock()
    collection = MagicMock()
    collection.query = MagicMock(
        return_value={
            "ids": [["doc_1", "doc_2"]],
            "documents": [
                [
                    "The transformer architecture was introduced in 2017.",
                    "Attention mechanisms allow models to focus on input.",
                ]
            ],
            "metadatas": [[{"source": "test.pdf"}, {"source": "test.pdf"}]],
            "distances": [[0.1, 0.2]],
        }
    )
    collection.count = MagicMock(return_value=10)
    client.get_or_create_collection = MagicMock(return_value=collection)
    client.heartbeat = MagicMock(return_value=True)
    client.list_collections = MagicMock(return_value=[])
    return client
