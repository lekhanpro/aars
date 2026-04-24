"""LangGraph state schema for the AARS pipeline."""

from __future__ import annotations

from typing import Any, TypedDict

from src.api.schemas.common import (
    Citation,
    Document,
    GradedDocument,
    HallucinationResult,
    IntentType,
    ReflectionResult,
    RetrievalPlan,
    SelfRAGEvaluation,
)


class AARSState(TypedDict, total=False):
    """State passed through the LangGraph pipeline.

    All fields are optional (total=False) so nodes can return partial
    updates. LangGraph merges each node's return dict into the
    running state.
    """

    # Input
    query: str
    collection: str
    top_k: int

    # Intent + planning
    intent: IntentType
    plan: RetrievalPlan

    # Retrieval
    current_query: str
    current_strategy: str
    queries_to_run: list[str]
    result_lists: list[list[Document]]
    documents: list[Document]

    # Grading
    graded_documents: list[GradedDocument]

    # Fusion + reranking
    fused_documents: list[Document]
    reranker_applied: bool

    # Generation
    answer: str
    confidence: float
    citations: list[Citation]

    # Verification
    hallucination_result: HallucinationResult
    self_rag_evaluation: SelfRAGEvaluation

    # Loop control
    reflection_results: list[ReflectionResult]
    iteration: int
    max_iterations: int

    # Config flags
    enable_planner: bool
    enable_reflection: bool
    enable_fusion: bool
    enable_mmr: bool
    enable_keyword: bool
    enable_graph: bool
    enable_reranker: bool
    enable_hallucination_check: bool
    enable_grading: bool

    # Trace
    trace_steps: list[dict[str, Any]]
    error: str
