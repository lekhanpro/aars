"""Core pipeline orchestrator — coordinates all AARS components."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from src.agents.planner import PlannerAgent
from src.agents.reflection import ReflectionAgent
from src.api.schemas import (
    Complexity,
    Document,
    QueryRequest,
    QueryResponse,
    QueryType,
    RetrievalPlan,
    ReflectionResult,
    RetrievalStrategy,
)
from src.fusion.fusion_pipeline import FusionPipeline
from src.fusion.mmr import MaximalMarginalRelevance
from src.fusion.rrf import ReciprocalRankFusion
from src.generation.answer_generator import AnswerGenerator
from src.pipeline.trace import TraceRecorder
from src.retrieval.graph import GraphRetriever
from src.retrieval.keyword import KeywordRetriever
from src.retrieval.none import NoneRetriever
from src.retrieval.registry import RetrieverRegistry
from src.retrieval.vector import VectorRetriever
from src.utils.embeddings import EmbeddingModel

if TYPE_CHECKING:
    from config.settings import Settings
    from src.llm.client import LLMClient

logger = structlog.get_logger()


class PipelineOrchestrator:
    """Orchestrates: query → plan → retrieve → reflect → fuse → generate → response."""

    def __init__(
        self,
        llm_client: LLMClient,
        chroma_client: object | None,
        settings: Settings,
        keyword_retriever: KeywordRetriever | None = None,
        graph_retriever: GraphRetriever | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.chroma_client = chroma_client
        self.settings = settings
        self.keyword_retriever = keyword_retriever or KeywordRetriever(
            retriever_settings=settings.retriever
        )
        self.graph_retriever = graph_retriever or GraphRetriever(
            retriever_settings=settings.retriever
        )

        # Initialize agents
        self.planner = PlannerAgent(llm_client)
        self.reflection = ReflectionAgent(
            llm_client, max_iterations=settings.pipeline.max_reflection_iterations
        )
        self.generator = AnswerGenerator(llm_client)

        # Initialize fusion
        rrf = ReciprocalRankFusion(k=settings.fusion.rrf_k)
        mmr = MaximalMarginalRelevance(lambda_param=settings.fusion.mmr_lambda)
        self.fusion = FusionPipeline(rrf=rrf, mmr=mmr, final_top_k=settings.fusion.final_top_k)

        # Initialize retriever registry
        self.registry = RetrieverRegistry()
        self._setup_retrievers()

    def _setup_retrievers(self) -> None:
        """Register all available retrievers."""
        if self.chroma_client:
            self.registry.register(
                "vector",
                VectorRetriever(
                    chroma_settings=self.settings.chroma,
                    embedding_settings=self.settings.embedding,
                    retriever_settings=self.settings.retriever,
                    chroma_client=self.chroma_client,
                ),
            )
        self.registry.register("keyword", self.keyword_retriever)
        self.registry.register("graph", self.graph_retriever)
        self.registry.register("none", NoneRetriever())
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Initialize all retrievers on first use."""
        if not self._initialized:
            await self.registry.initialize_all()
            self._initialized = True

    async def run(self, request: QueryRequest) -> QueryResponse:
        """Execute the full AARS pipeline."""
        await self._ensure_initialized()
        trace = TraceRecorder()
        self.llm_client.reset_counters()
        reflection_results: list[ReflectionResult] = []

        try:
            # Step 1: Planning
            t0 = time.monotonic()
            plan = await self._build_plan(request)
            trace.record(
                "planning",
                (time.monotonic() - t0) * 1000,
                strategy=plan.strategy.value,
                query_type=plan.query_type.value,
                complexity=plan.complexity.value,
            )
            logger.info(
                "plan_complete",
                strategy=plan.strategy.value,
                query_type=plan.query_type.value,
            )

            # Step 2: Retrieval (with reflection loop)
            all_documents: list[Document] = []
            all_result_lists: list[list[Document]] = []
            current_query = plan.rewritten_query or request.query
            current_strategy = self._resolve_strategy(plan.strategy.value, request)
            queries_to_run = plan.decomposed_queries or [current_query]

            for iteration in range(self.settings.pipeline.max_reflection_iterations + 1):
                t0 = time.monotonic()
                iter_result_lists = await self._retrieve(
                    queries=queries_to_run,
                    strategy=current_strategy,
                    collection=request.collection,
                    top_k=request.top_k,
                    request=request,
                )
                iter_docs = [doc for result_list in iter_result_lists for doc in result_list]
                all_result_lists.extend(iter_result_lists)
                all_documents.extend(iter_docs)
                trace.record(
                    f"retrieval_iter_{iteration}",
                    (time.monotonic() - t0) * 1000,
                    strategy=current_strategy,
                    num_docs=len(iter_docs),
                )

                # Step 3: Reflection (skip on last iteration or if disabled)
                max_iters = self.settings.pipeline.max_reflection_iterations
                if not request.enable_reflection or iteration >= max_iters:
                    break

                if plan.strategy == RetrievalStrategy.NONE:
                    break

                t0 = time.monotonic()
                reflection = await self.reflection.evaluate(request.query, all_documents)
                reflection_results.append(reflection)
                trace.record(
                    f"reflection_iter_{iteration}",
                    (time.monotonic() - t0) * 1000,
                    sufficient=reflection.sufficient,
                    confidence=reflection.confidence,
                )

                if reflection.sufficient:
                    break

                # Update for next iteration
                current_query = reflection.next_query or current_query
                current_strategy = self._resolve_strategy(
                    reflection.next_strategy or current_strategy,
                    request,
                )
                queries_to_run = [current_query]
                logger.info(
                    "reflection_retry",
                    iteration=iteration + 1,
                    next_strategy=current_strategy,
                )

            # Step 4: Fusion (deduplicate and rerank)
            t0 = time.monotonic()
            if all_documents and plan.strategy != RetrievalStrategy.NONE:
                if request.enable_fusion and all_result_lists:
                    merged_docs = self.fusion.merge(all_result_lists)
                else:
                    merged_docs = self._deduplicate_documents(all_documents)

                if request.enable_mmr and merged_docs:
                    embedding_model = EmbeddingModel.get(self.settings.embedding.model)
                    query_embedding = embedding_model.embed([request.query])[0]
                    doc_embeddings = embedding_model.embed([d.content for d in merged_docs])
                    fused_docs = self.fusion.rerank_merged(
                        merged_documents=merged_docs,
                        query_embedding=query_embedding,
                        doc_embeddings=doc_embeddings,
                    )
                else:
                    fused_docs = merged_docs[:request.top_k]
            else:
                fused_docs = all_documents[:request.top_k]
            trace.record(
                "fusion",
                (time.monotonic() - t0) * 1000,
                num_docs_in=len(all_documents),
                num_docs_out=len(fused_docs),
            )

            # Step 5: Answer generation
            t0 = time.monotonic()
            answer_result = await self.generator.generate(request.query, fused_docs)
            trace.record(
                "generation",
                (time.monotonic() - t0) * 1000,
                confidence=answer_result.confidence,
            )

            # Finalize trace
            trace.add_tokens(self.llm_client.total_tokens)
            trace.add_api_call()
            pipeline_trace = trace.finalize()

            return QueryResponse(
                answer=answer_result.answer,
                confidence=answer_result.confidence,
                citations=answer_result.citations,
                retrieval_plan=plan,
                reflection_results=reflection_results,
                documents=fused_docs,
                trace=pipeline_trace if request.enable_trace else None,
            )

        except Exception as e:
            logger.error("pipeline_error", error=str(e))
            trace.record("error", 0, error=str(e))
            pipeline_trace = trace.finalize()
            return QueryResponse(
                answer=f"An error occurred while processing your query: {e}",
                confidence=0.0,
                trace=pipeline_trace if request.enable_trace else None,
            )

    async def _retrieve(
        self,
        queries: list[str],
        strategy: str,
        collection: str,
        top_k: int,
        request: QueryRequest,
    ) -> list[list[Document]]:
        """Execute retrieval for one or more queries using the specified strategy."""
        result_lists: list[list[Document]] = []

        strategies = [strategy]
        if strategy == "hybrid":
            strategies = []
            strategies.append("vector")
            if request.enable_keyword:
                strategies.append("keyword")
            if request.enable_graph:
                strategies.append("graph")

        for strat in strategies:
            strat = self._resolve_strategy(strat, request)
            if strat == "none":
                continue
            try:
                retriever = self.registry.get(strat)
            except KeyError:
                logger.warning("retriever_not_found", strategy=strat)
                continue
            for query in queries:
                try:
                    docs = await retriever.retrieve(
                        query,
                        top_k=top_k,
                        collection=collection,
                    )
                    if docs:
                        result_lists.append(docs)
                except Exception as e:
                    logger.error("retrieval_error", strategy=strat, error=str(e))

        return result_lists

    async def _build_plan(self, request: QueryRequest) -> RetrievalPlan:
        """Return either an LLM-generated plan or a deterministic fallback plan."""
        if request.enable_planner:
            plan = await self.planner.plan(request.query)
            strategy = self._resolve_strategy(plan.strategy.value, request)
            return RetrievalPlan(
                query_type=plan.query_type,
                complexity=plan.complexity,
                strategy=RetrievalStrategy(strategy),
                rewritten_query=plan.rewritten_query,
                decomposed_queries=plan.decomposed_queries,
                reasoning=plan.reasoning,
            )
        return self._fallback_plan(request)

    def _fallback_plan(self, request: QueryRequest) -> RetrievalPlan:
        """Create a synthetic plan when the planner is disabled."""
        strategy = self._resolve_strategy(request.default_strategy.value, request)
        return RetrievalPlan(
            query_type=QueryType.FACTUAL,
            complexity=Complexity.SIMPLE,
            strategy=RetrievalStrategy(strategy),
            rewritten_query=request.query,
            decomposed_queries=[],
            reasoning="Planner disabled; using the request's default retrieval strategy.",
        )

    def _resolve_strategy(self, strategy: str, request: QueryRequest) -> str:
        """Map a desired strategy to one that is currently enabled and available."""
        resolved = strategy
        if resolved == "graph" and not request.enable_graph:
            resolved = request.default_strategy.value
        if resolved == "keyword" and not request.enable_keyword:
            resolved = "vector"
        if resolved == "hybrid":
            available = ["vector"]
            if request.enable_keyword:
                available.append("keyword")
            if request.enable_graph:
                available.append("graph")
            if len(available) == 1:
                resolved = available[0]
        if resolved not in self.registry and resolved not in {"none", "hybrid"}:
            resolved = "vector" if "vector" in self.registry else "keyword"
        if resolved == "keyword" and not request.enable_keyword:
            resolved = "vector"
        if resolved == "graph" and not request.enable_graph:
            resolved = "vector"
        return resolved

    @staticmethod
    def _deduplicate_documents(documents: list[Document]) -> list[Document]:
        """Preserve the highest-scoring instance of each document id."""
        by_id: dict[str, Document] = {}
        for doc in documents:
            existing = by_id.get(doc.id)
            if existing is None or doc.score > existing.score:
                by_id[doc.id] = doc
        return sorted(by_id.values(), key=lambda doc: doc.score, reverse=True)
