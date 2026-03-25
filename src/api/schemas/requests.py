"""Request models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import RetrievalStrategy


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user query")
    collection: str = Field(default="default", description="Document collection to query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    enable_planner: bool = Field(default=True, description="Enable planner-based strategy selection")
    enable_reflection: bool = Field(default=True, description="Enable reflection loop")
    enable_fusion: bool = Field(default=True, description="Enable reciprocal-rank fusion")
    enable_mmr: bool = Field(default=True, description="Enable diversity reranking after fusion")
    enable_keyword: bool = Field(default=True, description="Allow keyword retrieval")
    enable_graph: bool = Field(default=True, description="Allow graph retrieval")
    default_strategy: RetrievalStrategy = Field(
        default=RetrievalStrategy.VECTOR,
        description="Fallback retrieval strategy when planner is disabled or unavailable",
    )
    enable_trace: bool = Field(default=True, description="Return pipeline execution trace")


class IngestRequest(BaseModel):
    collection: str = Field(default="default", description="Target collection name")
    chunk_size: int | None = Field(default=None, description="Override default chunk size")
    chunk_overlap: int | None = Field(default=None, description="Override default chunk overlap")
