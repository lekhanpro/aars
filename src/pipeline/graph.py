"""LangGraph StateGraph implementation wrapping the AARS pipeline components."""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

import structlog
from langgraph.graph import END, StateGraph

from src.api.schemas.common import (
    Complexity,
    IntentType,
    QueryType,
    RetrievalPlan,
    RetrievalStrategy,
)
from src.pipeline.state import AARSState
from src.utils.embeddings import EmbeddingModel

if TYPE_CHECKING:
    from src.pipeline.orchestrator import PipelineOrchestrator

logger = structlog.get_logger()


class AARSGraph:
    """LangGraph-based AARS pipeline.

    Each node delegates to existing components on the
    :class:`PipelineOrchestrator`, keeping all business logic in the
    agents and retrievers rather than duplicating it here.
    """

    def __init__(self, orchestrator: PipelineOrchestrator) -> None:
        self._orch = orchestrator
        self._compiled = self._build()

    def _build(self) -> Any:
        graph = StateGraph(AARSState)

        graph.add_node("route_intent", self._route_intent)
        graph.add_node("plan_query", self._plan_query)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_relevance", self._grade_relevance)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("fuse", self._fuse)
        graph.add_node("rerank", self._rerank)
        graph.add_node("generate", self._generate)
        graph.add_node("check_hallucination", self._check_hallucination)
        graph.add_node("evaluate", self._evaluate)

        graph.set_entry_point("route_intent")
        graph.add_edge("route_intent", "plan_query")
        graph.add_edge("plan_query", "retrieve")
        graph.add_edge("retrieve", "grade_relevance")

        graph.add_conditional_edges(
            "grade_relevance",
            self._should_rewrite,
            {"fuse": "fuse", "rewrite": "rewrite_query"},
        )
        graph.add_edge("rewrite_query", "retrieve")

        graph.add_edge("fuse", "rerank")
        graph.add_edge("rerank", "generate")
        graph.add_edge("generate", "check_hallucination")
        graph.add_edge("check_hallucination", "evaluate")
        graph.add_edge("evaluate", END)

        return graph.compile()

    async def ainvoke(self, state: AARSState) -> AARSState:
        return await self._compiled.ainvoke(state)

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    async def _route_intent(self, state: AARSState) -> dict[str, Any]:
        t0 = time.monotonic()
        intent = await self._orch.intent_router.classify(state["query"])
        return {
            "intent": intent,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": "route_intent", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"intent": intent.value}}],
        }

    async def _plan_query(self, state: AARSState) -> dict[str, Any]:
        from src.api.schemas import QueryRequest

        t0 = time.monotonic()
        intent = state.get("intent", IntentType.SIMPLE)

        if intent == IntentType.DIRECT:
            plan = RetrievalPlan(
                query_type=QueryType.CONVERSATIONAL,
                complexity=Complexity.SIMPLE,
                strategy=RetrievalStrategy.NONE,
                rewritten_query=state["query"],
                decomposed_queries=[],
                reasoning="Direct intent — LLM can answer without retrieval.",
            )
        elif state.get("enable_planner", True):
            plan = await self._orch.planner.plan(state["query"])
        else:
            plan = RetrievalPlan(
                query_type=QueryType.FACTUAL,
                complexity=Complexity.SIMPLE,
                strategy=RetrievalStrategy.VECTOR,
                rewritten_query=state["query"],
                decomposed_queries=[],
                reasoning="Planner disabled; using default strategy.",
            )

        queries = plan.decomposed_queries or [plan.rewritten_query or state["query"]]

        return {
            "plan": plan,
            "current_query": plan.rewritten_query or state["query"],
            "current_strategy": plan.strategy.value,
            "queries_to_run": queries,
            "iteration": 0,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": "plan_query", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"strategy": plan.strategy.value}}],
        }

    async def _retrieve(self, state: AARSState) -> dict[str, Any]:
        from src.api.schemas import QueryRequest

        t0 = time.monotonic()
        request = self._build_request(state)
        strategy = state.get("current_strategy", "vector")
        queries = state.get("queries_to_run", [state["query"]])
        top_k = state.get("top_k", 5)

        iter_result_lists = await self._orch._retrieve(
            queries=queries,
            strategy=strategy,
            collection=state.get("collection", "default"),
            top_k=top_k,
            request=request,
        )
        iter_docs = [doc for rl in iter_result_lists for doc in rl]
        prev_lists = state.get("result_lists", [])
        prev_docs = state.get("documents", [])

        return {
            "result_lists": prev_lists + iter_result_lists,
            "documents": prev_docs + iter_docs,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": f"retrieve_iter_{state.get('iteration', 0)}", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"strategy": strategy, "num_docs": len(iter_docs)}}],
        }

    async def _grade_relevance(self, state: AARSState) -> dict[str, Any]:
        if not state.get("enable_grading", True):
            return {}

        docs = state.get("documents", [])
        if not docs:
            return {}

        t0 = time.monotonic()
        grades = await self._orch.relevance_grader.grade(state["query"], docs)
        prev_grades = state.get("graded_documents", [])

        relevant_ids = {g.doc_id for g in grades if g.relevant}
        filtered = [d for d in docs if d.id in relevant_ids]

        return {
            "graded_documents": prev_grades + grades,
            "documents": filtered if filtered else docs,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": f"grading_iter_{state.get('iteration', 0)}", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"relevant": len(filtered), "total": len(docs)}}],
        }

    def _should_rewrite(self, state: AARSState) -> str:
        if not state.get("enable_reflection", True):
            return "fuse"
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 3)
        if iteration >= max_iterations:
            return "fuse"
        plan = state.get("plan")
        if plan and plan.strategy == RetrievalStrategy.NONE:
            return "fuse"
        docs = state.get("documents", [])
        if not docs:
            return "rewrite" if iteration < max_iterations else "fuse"
        return "fuse"

    async def _rewrite_query(self, state: AARSState) -> dict[str, Any]:
        t0 = time.monotonic()
        iteration = state.get("iteration", 0)

        reflection = await self._orch.reflection.evaluate(
            state["query"], state.get("documents", [])
        )
        prev_reflections = state.get("reflection_results", [])

        rewrite_result = await self._orch.query_rewriter.rewrite(
            query=state.get("current_query", state["query"]),
            context=reflection.missing_information or "No relevant documents found",
        )

        return {
            "reflection_results": prev_reflections + [reflection],
            "current_query": rewrite_result.rewritten,
            "queries_to_run": [rewrite_result.rewritten],
            "current_strategy": reflection.next_strategy or state.get("current_strategy", "vector"),
            "iteration": iteration + 1,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": f"rewrite_iter_{iteration}", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"technique": rewrite_result.technique}}],
        }

    async def _fuse(self, state: AARSState) -> dict[str, Any]:
        t0 = time.monotonic()
        docs = state.get("documents", [])
        result_lists = state.get("result_lists", [])
        top_k = state.get("top_k", 5)
        plan = state.get("plan")

        if not docs or (plan and plan.strategy == RetrievalStrategy.NONE):
            return {"fused_documents": docs[:top_k]}

        if state.get("enable_fusion", True) and result_lists:
            merged = self._orch.fusion.merge(result_lists)
        else:
            merged = self._orch._deduplicate_documents(docs)

        if state.get("enable_mmr", True) and merged:
            embedding_model = EmbeddingModel.get(self._orch.settings.embedding.model)
            query_embedding = embedding_model.embed([state["query"]])[0]
            doc_embeddings = embedding_model.embed([d.content for d in merged])
            fused = self._orch.fusion.rerank_merged(
                merged_documents=merged,
                query_embedding=query_embedding,
                doc_embeddings=doc_embeddings,
            )
        else:
            fused = merged[:top_k]

        return {
            "fused_documents": fused,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": "fusion", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"in": len(docs), "out": len(fused)}}],
        }

    async def _rerank(self, state: AARSState) -> dict[str, Any]:
        fused = state.get("fused_documents", [])
        if not state.get("enable_reranker", True) or not fused:
            return {"reranker_applied": False}

        t0 = time.monotonic()
        reranked = await self._orch.reranker.rerank(
            query=state["query"],
            documents=fused,
            top_k=state.get("top_k", 5),
        )
        return {
            "fused_documents": reranked,
            "reranker_applied": True,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": "reranking", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"num_docs": len(reranked)}}],
        }

    async def _generate(self, state: AARSState) -> dict[str, Any]:
        t0 = time.monotonic()
        docs = state.get("fused_documents", [])
        result = await self._orch.generator.generate(state["query"], docs)
        return {
            "answer": result.answer,
            "confidence": result.confidence,
            "citations": result.citations,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": "generation", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"confidence": result.confidence}}],
        }

    async def _check_hallucination(self, state: AARSState) -> dict[str, Any]:
        if not state.get("enable_hallucination_check", True):
            return {}

        docs = state.get("fused_documents", [])
        answer = state.get("answer", "")
        if not docs or not answer:
            return {}

        t0 = time.monotonic()
        result = await self._orch.hallucination_checker.check(
            question=state["query"],
            answer=answer,
            documents=docs,
        )
        return {
            "hallucination_result": result,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": "hallucination_check", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"grounded": result.grounded, "score": result.score}}],
        }

    async def _evaluate(self, state: AARSState) -> dict[str, Any]:
        t0 = time.monotonic()
        docs = state.get("fused_documents", [])
        answer = state.get("answer", "")

        result = await self._orch.self_rag_evaluator.evaluate(
            question=state["query"],
            answer=answer,
            documents=docs,
        )
        return {
            "self_rag_evaluation": result,
            "trace_steps": state.get("trace_steps", [])
            + [{"step": "self_rag_evaluation", "duration_ms": (time.monotonic() - t0) * 1000, "details": {"overall": result.overall}}],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_request(self, state: AARSState) -> Any:
        from src.api.schemas import QueryRequest

        return QueryRequest(
            query=state["query"],
            collection=state.get("collection", "default"),
            top_k=state.get("top_k", 5),
            enable_planner=state.get("enable_planner", True),
            enable_reflection=state.get("enable_reflection", True),
            enable_fusion=state.get("enable_fusion", True),
            enable_mmr=state.get("enable_mmr", True),
            enable_keyword=state.get("enable_keyword", True),
            enable_graph=state.get("enable_graph", True),
            enable_reranker=state.get("enable_reranker", True),
            enable_hallucination_check=state.get("enable_hallucination_check", True),
            enable_grading=state.get("enable_grading", True),
        )
