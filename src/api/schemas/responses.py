"""Response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .common import (
    Citation,
    Document,
    GradedDocument,
    HallucinationResult,
    ReflectionResult,
    RetrievalPlan,
    SelfRAGEvaluation,
)


class TraceStep(BaseModel):
    step: str
    duration_ms: float
    details: dict = Field(default_factory=dict)


class PipelineTrace(BaseModel):
    trace_id: str
    steps: list[TraceStep] = Field(default_factory=list)
    total_duration_ms: float = 0.0
    total_tokens: int = 0
    total_api_calls: int = 0


class QueryResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)
    retrieval_plan: RetrievalPlan | None = None
    reflection_results: list[ReflectionResult] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    graded_documents: list[GradedDocument] = Field(default_factory=list)
    hallucination_result: HallucinationResult | None = None
    self_rag_evaluation: SelfRAGEvaluation | None = None
    reranker_applied: bool = False
    trace: PipelineTrace | None = None


class IngestResponse(BaseModel):
    collection: str
    documents_ingested: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0"
    chromadb_connected: bool = False


class CollectionInfo(BaseModel):
    name: str
    document_count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
