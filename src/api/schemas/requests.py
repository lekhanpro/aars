"""Request models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user query")
    collection: str = Field(default="default", description="Document collection to query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    enable_reflection: bool = Field(default=True, description="Enable reflection loop")
    enable_trace: bool = Field(default=True, description="Return pipeline execution trace")


class IngestRequest(BaseModel):
    collection: str = Field(default="default", description="Target collection name")
    chunk_size: int | None = Field(default=None, description="Override default chunk size")
    chunk_overlap: int | None = Field(default=None, description="Override default chunk overlap")
