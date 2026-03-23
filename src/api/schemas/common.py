"""Common Pydantic models used across the API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class QueryType(str, Enum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    MULTI_HOP = "multi_hop"
    OPINION = "opinion"
    CONVERSATIONAL = "conversational"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class RetrievalStrategy(str, Enum):
    KEYWORD = "keyword"
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"
    NONE = "none"


class Document(BaseModel):
    id: str
    content: str
    metadata: dict = Field(default_factory=dict)
    score: float = 0.0


class Citation(BaseModel):
    doc_id: str
    text: str


class RetrievalPlan(BaseModel):
    query_type: QueryType
    complexity: Complexity
    strategy: RetrievalStrategy
    rewritten_query: str
    decomposed_queries: list[str] = Field(default_factory=list)
    reasoning: str


class ReflectionResult(BaseModel):
    sufficient: bool
    confidence: float = Field(ge=0.0, le=1.0)
    missing_information: str = ""
    next_query: str = ""
    next_strategy: str = ""
