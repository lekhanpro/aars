"""Graph pipeline runner — adapts LangGraph execution to QueryRequest/QueryResponse."""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

import structlog

from src.api.schemas import (
    PipelineTrace,
    QueryRequest,
    QueryResponse,
    TraceStep,
)
from src.pipeline.graph import AARSGraph
from src.pipeline.state import AARSState

if TYPE_CHECKING:
    from src.pipeline.orchestrator import PipelineOrchestrator

logger = structlog.get_logger()


class GraphPipelineRunner:
    """Runs the AARS LangGraph pipeline and converts results to API responses.

    This is the drop-in replacement for calling
    :meth:`PipelineOrchestrator.run` directly. It converts a
    :class:`QueryRequest` into initial graph state, invokes the
    compiled LangGraph, and converts the final state into a
    :class:`QueryResponse`.
    """

    def __init__(self, orchestrator: PipelineOrchestrator) -> None:
        self._orch = orchestrator
        self._graph = AARSGraph(orchestrator)

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the LangGraph pipeline for *request*."""
        await self._orch._ensure_initialized()
        self._orch.llm_client.reset_counters()

        initial_state: AARSState = {
            "query": request.query,
            "collection": request.collection,
            "top_k": request.top_k,
            "max_iterations": self._orch.settings.pipeline.max_reflection_iterations,
            "iteration": 0,
            "enable_planner": request.enable_planner,
            "enable_reflection": request.enable_reflection,
            "enable_fusion": request.enable_fusion,
            "enable_mmr": request.enable_mmr,
            "enable_keyword": request.enable_keyword,
            "enable_graph": request.enable_graph,
            "enable_reranker": request.enable_reranker,
            "enable_hallucination_check": request.enable_hallucination_check,
            "enable_grading": request.enable_grading,
            "documents": [],
            "result_lists": [],
            "reflection_results": [],
            "graded_documents": [],
            "trace_steps": [],
        }

        try:
            final_state = await self._graph.ainvoke(initial_state)
        except Exception as e:
            logger.error("graph_pipeline_error", error=str(e))
            return QueryResponse(
                answer=f"An error occurred while processing your query: {e}",
                confidence=0.0,
            )

        return self._build_response(final_state, request)

    def _build_response(self, state: AARSState, request: QueryRequest) -> QueryResponse:
        trace = None
        if request.enable_trace:
            steps = [
                TraceStep(
                    step=s.get("step", ""),
                    duration_ms=round(s.get("duration_ms", 0.0), 2),
                    details={k: v for k, v in s.items() if k not in ("step", "duration_ms")},
                )
                for s in state.get("trace_steps", [])
            ]
            total_ms = sum(s.duration_ms for s in steps)
            trace = PipelineTrace(
                trace_id=str(uuid.uuid4()),
                steps=steps,
                total_duration_ms=round(total_ms, 2),
                total_tokens=self._orch.llm_client.total_tokens,
                total_api_calls=self._orch.llm_client.total_calls,
            )

        return QueryResponse(
            answer=state.get("answer", ""),
            confidence=state.get("confidence", 0.0),
            citations=state.get("citations", []),
            retrieval_plan=state.get("plan"),
            reflection_results=state.get("reflection_results", []),
            documents=state.get("fused_documents", []),
            graded_documents=state.get("graded_documents", []),
            hallucination_result=state.get("hallucination_result"),
            self_rag_evaluation=state.get("self_rag_evaluation"),
            reranker_applied=state.get("reranker_applied", False),
            trace=trace,
        )
