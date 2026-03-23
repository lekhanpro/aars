"""Core pipeline orchestrator — coordinates all AARS components."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

from src.agents.planner import PlannerAgent
from src.agents.reflection import ReflectionAgent
from src.api.schemas import (
    Document,
    QueryRequest,
    QueryResponse,
    ReflectionResult,
    RetrievalStrategy,
)
from src.fusion.fusion_pipeline import FusionPipeline
from src.fusion.mmr import MaximalMarginalRelevance
from src.fusion.rrf import ReciprocalRankFusion
from src.generation.answer_generator import AnswerGenerator
from src.pipeline.trace import TraceRecorder
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
    ) -> None:
        self.llm_client = llm_client
        self.chroma_client = chroma_client
        self.settings = settings

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
                ),
            )
        self.registry.register(
            "keyword",
            KeywordRetriever(retriever_settings=self.settings.retriever),
        )
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
            plan = await self.planner.plan(request.query)
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
            current_query = plan.rewritten_query or request.query
            current_strategy = plan.strategy.value
            queries_to_run = plan.decomposed_queries or [current_query]

            for iteration in range(self.settings.pipeline.max_reflection_iterations + 1):
                t0 = time.monotonic()
                iter_docs = await self._retrieve(
                    queries=queries_to_run,
                    strategy=current_strategy,
                    collection=request.collection,
                    top_k=request.top_k,
                )
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
                current_strategy = reflection.next_strategy or current_strategy
                queries_to_run = [current_query]
                logger.info(
                    "reflection_retry",
                    iteration=iteration + 1,
                    next_strategy=current_strategy,
                )

            # Step 4: Fusion (deduplicate and rerank)
            t0 = time.monotonic()
            if all_documents and plan.strategy != RetrievalStrategy.NONE:
                embedding_model = EmbeddingModel.get(self.settings.embedding.model)
                query_embedding = embedding_model.embed([request.query])[0]
                doc_embeddings = embedding_model.embed(
                    [d.content for d in all_documents]
                )
                fused_docs = await self.fusion.fuse(
                    result_lists=[all_documents],
                    query_embedding=query_embedding,
                    doc_embeddings=doc_embeddings,
                )
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
    ) -> list[Document]:
        """Execute retrieval for one or more queries using the specified strategy."""
        all_docs: list[Document] = []

        strategies = [strategy]
        if strategy == "hybrid":
            strategies = ["vector", "keyword"]

        for strat in strategies:
            try:
                retriever = self.registry.get(strat)
            except KeyError:
                logger.warning("retriever_not_found", strategy=strat)
                continue
            for query in queries:
                try:
                    docs = await retriever.retrieve(query, top_k=top_k)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.error("retrieval_error", strategy=strat, error=str(e))

        return all_docs
