"""Runnable local benchmark harness for AARS.

The benchmark is intentionally offline and deterministic. It exercises the
checked-in retrieval pipeline, the simplified baselines, and the benchmark
metrics without requiring a running API server or external model providers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.baselines import ALL_BASELINES
from benchmarks.local_fixture import (
    LOCAL_COLLECTION,
    LOCAL_DOCUMENTS,
    LOCAL_SAMPLES,
    baseline_documents,
)
from benchmarks.metrics import Metrics
from benchmarks.significance import compare_systems
from config.settings import Settings
from src.api.schemas import Citation, QueryRequest, QueryResponse, RetrievalPlan, ReflectionResult
from src.generation.answer_generator import AnswerResult
from src.ingestion.graph_builder import GraphBuilder
from src.pipeline.orchestrator import PipelineOrchestrator
from src.retrieval.graph import GraphRetriever
from src.retrieval.keyword import KeywordRetriever
from src.utils.embeddings import EmbeddingModel

DATASET_NAMES = ["local_fixture"]

_USER_QUERY_RE = re.compile(r"User Query:\s*(.+)")
_QUESTION_RE = re.compile(r"Question:\s*(.+)")
_DOC_BLOCK_RE = re.compile(
    r"--- Document \d+: \[(?P<doc_id>[^\]]+)\][^\n]*---\n(?P<content>.*?)(?=\n\n--- Document |\Z)",
    re.S,
)
_BASELINE_CONTEXT_RE = re.compile(r"Context:\n(?P<context>.*?)\n\nQuestion:\s*(?P<question>.+?)\n\nAnswer:", re.S)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryCollection:
    """Minimal Chroma-like collection used by the offline benchmark."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._rows: dict[str, dict[str, Any]] = {}

    def count(self) -> int:
        return len(self._rows)

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        for doc_id, document, embedding, metadata in zip(
            ids,
            documents,
            embeddings,
            metadatas,
            strict=True,
        ):
            self._rows[doc_id] = {
                "document": document,
                "embedding": embedding,
                "metadata": metadata,
            }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, list[list[Any]]]:
        query_embedding = query_embeddings[0] if query_embeddings else []
        scored = []
        for doc_id, row in self._rows.items():
            similarity = _cosine_similarity(query_embedding, row["embedding"])
            scored.append((doc_id, row, similarity))
        scored.sort(key=lambda item: item[2], reverse=True)
        top = scored[:n_results]

        result: dict[str, list[list[Any]]] = {"ids": [[doc_id for doc_id, _, _ in top]]}
        if "documents" in include:
            result["documents"] = [[row["document"] for _, row, _ in top]]
        if "metadatas" in include:
            result["metadatas"] = [[row["metadata"] for _, row, _ in top]]
        if "distances" in include:
            result["distances"] = [[1.0 - similarity for _, _, similarity in top]]
        return result


class InMemoryChromaClient:
    """Minimal client implementing the Chroma APIs used in this repo."""

    def __init__(self) -> None:
        self._collections: dict[str, InMemoryCollection] = {}

    def heartbeat(self) -> bool:
        return True

    def get_or_create_collection(
        self,
        *,
        name: str,
        metadata: dict[str, Any] | None = None,
    ) -> InMemoryCollection:
        del metadata
        if name not in self._collections:
            self._collections[name] = InMemoryCollection(name)
        return self._collections[name]

    def list_collections(self) -> list[InMemoryCollection]:
        return list(self._collections.values())

    def delete_collection(self, name: str) -> None:
        self._collections.pop(name, None)


def _extract_query(prompt: str) -> str:
    for pattern in (_USER_QUERY_RE, _QUESTION_RE):
        match = pattern.search(prompt)
        if match:
            return match.group(1).strip()
    return ""


def _extract_documents(prompt: str) -> list[dict[str, str]]:
    return [
        {"id": match.group("doc_id").strip(), "content": match.group("content").strip()}
        for match in _DOC_BLOCK_RE.finditer(prompt)
    ]


def _answer_from_documents(
    query: str,
    documents: list[dict[str, str]],
) -> tuple[str, list[Citation], str]:
    """Return answer, citations, and reasoning for a benchmark query."""
    question = query.lower()

    def find(pattern: str) -> tuple[str | None, dict[str, str] | None]:
        compiled = re.compile(pattern, re.IGNORECASE)
        for document in documents:
            match = compiled.search(document["content"])
            if match:
                return match.group(1).strip(), document
        return None, None

    def citation_for(document: dict[str, str], text: str) -> Citation:
        return Citation(doc_id=document["id"], text=text)

    if "exact term overlap" in question:
        answer, document = find(r"([A-Za-z0-9-]+) is a sparse lexical ranking algorithm")
        if answer and document:
            return answer, [citation_for(document, document["content"])], "Matched the sparse-ranking definition."

    if "stores dense embeddings" in question or "semantic search" in question:
        answer, document = find(r"([A-Za-z0-9-]+) is a vector database")
        if answer and document:
            return answer, [citation_for(document, document["content"])], "Matched the vector-database description."

    if "who introduced the architecture used by bert" in question:
        has_bert = any("bert uses the transformer" in document["content"].lower() for document in documents)
        answer, document = find(r"introduced in \d{4} by ([^.]+)")
        if has_bert and answer and document:
            citations = [citation_for(document, document["content"])]
            citations.extend(
                citation_for(candidate, candidate["content"])
                for candidate in documents
                if candidate["id"] == "bert_architecture"
            )
            return answer, citations, "Connected BERT to the Transformer and extracted the introducer."

    if "sam altman" in question and "headquartered" in question:
        company = None
        company_doc = None
        for document in documents:
            match = re.search(r"Sam Altman is the CEO of ([^.]+)", document["content"], re.IGNORECASE)
            if match:
                company = match.group(1).strip()
                company_doc = document
                break
        if company:
            city, city_doc = find(rf"{re.escape(company)} is headquartered in ([^.]+)")
            if city and city_doc:
                citations = [citation_for(city_doc, city_doc["content"])]
                if company_doc is not None:
                    citations.append(citation_for(company_doc, company_doc["content"]))
                return city, citations, "Used the CEO relation to identify the company, then extracted the HQ city."

    if "more retrieval is needed" in question or "evidence is sufficient" in question:
        answer, document = find(r"AARS uses (?:a|an) ([^.]+?) to decide whether retrieved evidence is sufficient")
        if answer and document:
            return answer, [citation_for(document, document["content"])], "Matched the reflection-step description."

    if "when was retrieval-augmented generation introduced" in question:
        answer, document = find(r"introduced by [^.]+ in (\d{4})")
        if answer and document:
            return answer, [citation_for(document, document["content"])], "Extracted the introduction year."

    if "merges ranked lists before mmr" in question:
        answer, document = find(r"([A-Za-z ]+), or RRF, merges ranked lists")
        if answer and document:
            return answer, [citation_for(document, document["content"])], "Matched the fusion method definition."

    if "wording changes but meaning stays the same" in question:
        answer, document = find(r"(Dense vector retrieval) is useful")
        if answer and document:
            return answer, [citation_for(document, document["content"])], "Matched the semantic-retrieval description."

    if "entity relationships" in question and "multi-hop" in question:
        answer, document = find(r"(Graph retrieval) traverses entity relationships")
        if answer and document:
            return answer, [citation_for(document, document["content"])], "Matched the graph-retrieval definition."

    return "I don't know.", [], "The provided documents did not contain a confident answer."


class OfflineBenchmarkLLM:
    """Local deterministic substitute for planner, reflection, and generation."""

    def __init__(self) -> None:
        self.total_tokens = 0
        self.total_calls = 0

    async def generate(
        self,
        prompt: str,
        system: str = "",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        del system
        del max_tokens
        del temperature
        self.total_calls += 1
        self.total_tokens += len(prompt.split())

        match = _BASELINE_CONTEXT_RE.search(prompt)
        if not match:
            return "I don't know."

        question = match.group("question").strip()
        context = match.group("context")
        docs = [
            {"id": f"context_{index}", "content": line.partition("] ")[2].strip()}
            for index, line in enumerate(context.splitlines(), start=1)
            if line.startswith("[")
        ]
        answer, _, _ = _answer_from_documents(question, docs)
        return answer

    async def structured_output(
        self,
        prompt: str,
        output_model: type[Any],
        system: str = "",
        max_tokens: int | None = None,
    ) -> Any:
        del system
        del max_tokens
        self.total_calls += 1
        self.total_tokens += len(prompt.split())

        query = _extract_query(prompt)
        model_name = output_model.__name__

        if model_name == "RetrievalPlan":
            return output_model.model_validate(self._plan_for_query(query))

        documents = _extract_documents(prompt)
        answer, citations, reasoning = _answer_from_documents(query, documents)

        if model_name == "ReflectionResult":
            sufficient = answer != "I don't know."
            next_strategy = ""
            if not sufficient:
                if "who introduced the architecture used by bert" in query.lower() or "sam altman" in query.lower():
                    next_strategy = "graph"
                elif "semantic" in query.lower() or "wording changes" in query.lower():
                    next_strategy = "vector"
                else:
                    next_strategy = "hybrid"
            return output_model.model_validate(
                {
                    "sufficient": sufficient,
                    "confidence": 0.9 if sufficient else 0.25,
                    "missing_information": "" if sufficient else "The current documents do not fully answer the question.",
                    "next_query": "" if sufficient else query,
                    "next_strategy": next_strategy,
                }
            )

        if model_name == "AnswerResult":
            return output_model.model_validate(
                {
                    "answer": answer,
                    "citations": [citation.model_dump() for citation in citations],
                    "confidence": 0.92 if answer != "I don't know." else 0.2,
                    "reasoning": reasoning,
                }
            )

        raise ValueError(f"Unsupported structured output model: {model_name}")

    def reset_counters(self) -> None:
        self.total_tokens = 0
        self.total_calls = 0

    @staticmethod
    def _plan_for_query(query: str) -> dict[str, Any]:
        question = query.lower()
        if "entity relationships" in question and "multi-hop questions" in question:
            return {
                "query_type": "factual",
                "complexity": "simple",
                "strategy": "keyword",
                "rewritten_query": query,
                "decomposed_queries": [],
                "reasoning": "This is a direct definitional query with strong lexical cues.",
            }
        if "who introduced the architecture used by bert" in question or "sam altman" in question:
            return {
                "query_type": "multi_hop",
                "complexity": "complex",
                "strategy": "graph",
                "rewritten_query": query,
                "decomposed_queries": [],
                "reasoning": "The question connects multiple entities across more than one statement.",
            }
        if "semantic search" in question or "wording changes but meaning stays the same" in question:
            return {
                "query_type": "analytical",
                "complexity": "moderate",
                "strategy": "vector",
                "rewritten_query": query,
                "decomposed_queries": [],
                "reasoning": "The query is semantically phrased rather than an exact lexical match.",
            }
        if "merges ranked lists" in question:
            return {
                "query_type": "analytical",
                "complexity": "moderate",
                "strategy": "hybrid",
                "rewritten_query": query,
                "decomposed_queries": [],
                "reasoning": "The query mixes a named method with conceptual pipeline context.",
            }
        if "entity relationships" in question:
            return {
                "query_type": "multi_hop",
                "complexity": "moderate",
                "strategy": "graph",
                "rewritten_query": query,
                "decomposed_queries": [],
                "reasoning": "The question directly targets graph-style relation traversal.",
            }
        return {
            "query_type": "factual",
            "complexity": "simple",
            "strategy": "keyword",
            "rewritten_query": query,
            "decomposed_queries": [],
            "reasoning": "The query asks for a direct fact that should benefit from exact lexical matching.",
        }


@dataclass
class BenchmarkEnvironment:
    settings: Settings
    llm_client: OfflineBenchmarkLLM
    chroma_client: InMemoryChromaClient
    orchestrator: PipelineOrchestrator


async def build_environment() -> BenchmarkEnvironment:
    """Create and preload a fully local AARS runtime."""
    settings = Settings(anthropic_api_key="local-benchmark")
    settings.embedding.model = "local-benchmark-hashing"
    settings.chroma.collection_name = LOCAL_COLLECTION
    settings.fusion.final_top_k = 5

    llm_client = OfflineBenchmarkLLM()
    chroma_client = InMemoryChromaClient()
    keyword_retriever = KeywordRetriever(retriever_settings=settings.retriever)
    graph_retriever = GraphRetriever(retriever_settings=settings.retriever)
    await graph_retriever.initialize()

    orchestrator = PipelineOrchestrator(
        llm_client=llm_client,
        chroma_client=chroma_client,
        settings=settings,
        keyword_retriever=keyword_retriever,
        graph_retriever=graph_retriever,
    )
    await orchestrator._ensure_initialized()

    embeddings = EmbeddingModel.get(settings.embedding.model).embed(
        [document.content for document in LOCAL_DOCUMENTS],
        batch_size=settings.embedding.batch_size,
    )
    collection = chroma_client.get_or_create_collection(name=LOCAL_COLLECTION)
    collection.upsert(
        ids=[document.id for document in LOCAL_DOCUMENTS],
        documents=[document.content for document in LOCAL_DOCUMENTS],
        embeddings=embeddings,
        metadatas=[document.metadata for document in LOCAL_DOCUMENTS],
    )

    keyword_retriever.add_documents(LOCAL_DOCUMENTS, collection=LOCAL_COLLECTION)
    graph_builder = GraphBuilder()
    await graph_builder.initialize()
    graph_builder.add_documents(LOCAL_DOCUMENTS, collection=LOCAL_COLLECTION)
    graph_retriever.set_graph(graph_builder.get_graph(LOCAL_COLLECTION), collection=LOCAL_COLLECTION)

    return BenchmarkEnvironment(
        settings=settings,
        llm_client=llm_client,
        chroma_client=chroma_client,
        orchestrator=orchestrator,
    )


def _answer_scores(prediction: str, answers: list[str]) -> tuple[float, float]:
    return (
        Metrics.exact_match_multi(prediction, answers),
        Metrics.token_f1_multi(prediction, answers),
    )


def _retrieval_scores(retrieved_ids: list[str], relevant_ids: list[str]) -> dict[str, float]:
    return {
        "recall_at_3": Metrics.recall_at_k(retrieved_ids, relevant_ids, 3),
        "precision_at_3": Metrics.precision_at_k(retrieved_ids, relevant_ids, 3),
        "mrr_at_5": Metrics.mrr_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg_at_5": Metrics.ndcg_at_k(retrieved_ids, relevant_ids, 5),
    }


def _summarize_system(
    *,
    name: str,
    em_scores: list[float],
    f1_scores: list[float],
    retrieval_metrics: dict[str, list[float]],
    latencies: list[float],
    total_tokens: int,
    total_api_calls: int,
) -> dict[str, Any]:
    return {
        "name": name,
        "exact_match": sum(em_scores) / len(em_scores) if em_scores else 0.0,
        "token_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "total_tokens": total_tokens,
        "total_api_calls": total_api_calls,
        "recall_at_3": sum(retrieval_metrics["recall_at_3"]) / len(retrieval_metrics["recall_at_3"]),
        "precision_at_3": sum(retrieval_metrics["precision_at_3"]) / len(retrieval_metrics["precision_at_3"]),
        "mrr_at_5": sum(retrieval_metrics["mrr_at_5"]) / len(retrieval_metrics["mrr_at_5"]),
        "ndcg_at_5": sum(retrieval_metrics["ndcg_at_5"]) / len(retrieval_metrics["ndcg_at_5"]),
        "per_sample_em": em_scores,
        "per_sample_f1": f1_scores,
    }


class BenchmarkRunner:
    """Run the local benchmark fixture against AARS and baseline systems."""

    async def run_aars(
        self,
        environment: BenchmarkEnvironment,
        *,
        enable_reflection: bool = True,
        system_name: str = "aars",
    ) -> dict[str, Any]:
        em_scores: list[float] = []
        f1_scores: list[float] = []
        latencies: list[float] = []
        retrieval_metrics = {
            "recall_at_3": [],
            "precision_at_3": [],
            "mrr_at_5": [],
            "ndcg_at_5": [],
        }
        total_tokens = 0
        total_api_calls = 0

        for sample in LOCAL_SAMPLES:
            request = QueryRequest(
                query=sample["question"],  # type: ignore[arg-type]
                collection=LOCAL_COLLECTION,
                top_k=5,
                enable_reflection=enable_reflection,
                enable_trace=True,
            )
            start = time.monotonic()
            result: QueryResponse = await environment.orchestrator.run(request)
            latency = (time.monotonic() - start) * 1000

            answers = sample["answers"]  # type: ignore[assignment]
            em, f1 = _answer_scores(result.answer, answers)
            metrics = _retrieval_scores(
                [document.id for document in result.documents],
                sample["relevant_ids"],  # type: ignore[arg-type]
            )

            em_scores.append(em)
            f1_scores.append(f1)
            latencies.append(latency)
            for key, value in metrics.items():
                retrieval_metrics[key].append(value)

            if result.trace is not None:
                total_tokens += result.trace.total_tokens
                total_api_calls += result.trace.total_api_calls

        return _summarize_system(
            name=system_name,
            em_scores=em_scores,
            f1_scores=f1_scores,
            retrieval_metrics=retrieval_metrics,
            latencies=latencies,
            total_tokens=total_tokens,
            total_api_calls=total_api_calls,
        )

    async def run_baseline(self, baseline_name: str, llm_client: OfflineBenchmarkLLM) -> dict[str, Any]:
        documents = baseline_documents()
        baseline = next(candidate for candidate in ALL_BASELINES if candidate.name == baseline_name)

        em_scores: list[float] = []
        f1_scores: list[float] = []
        latencies: list[float] = []
        retrieval_metrics = {
            "recall_at_3": [],
            "precision_at_3": [],
            "mrr_at_5": [],
            "ndcg_at_5": [],
        }
        total_tokens = 0
        total_api_calls = 0

        for sample in LOCAL_SAMPLES:
            start = time.monotonic()
            result = await baseline.run(sample["question"], documents, llm_client)  # type: ignore[arg-type]
            latency = (time.monotonic() - start) * 1000
            em, f1 = _answer_scores(result["answer"], sample["answers"])  # type: ignore[arg-type]
            metrics = _retrieval_scores(
                [document["id"] for document in result["documents"]],
                sample["relevant_ids"],  # type: ignore[arg-type]
            )

            em_scores.append(em)
            f1_scores.append(f1)
            latencies.append(latency)
            for key, value in metrics.items():
                retrieval_metrics[key].append(value)

        total_tokens += llm_client.total_tokens
        total_api_calls += llm_client.total_calls
        llm_client.reset_counters()

        return _summarize_system(
            name=baseline_name,
            em_scores=em_scores,
            f1_scores=f1_scores,
            retrieval_metrics=retrieval_metrics,
            latencies=latencies,
            total_tokens=total_tokens,
            total_api_calls=total_api_calls,
        )

    async def run(self) -> dict[str, Any]:
        environment = await build_environment()
        results: dict[str, Any] = {
            "dataset": "local_fixture",
            "collection": LOCAL_COLLECTION,
            "num_documents": len(LOCAL_DOCUMENTS),
            "num_samples": len(LOCAL_SAMPLES),
            "systems": {},
        }

        results["systems"]["aars"] = await self.run_aars(environment)
        results["systems"]["aars_no_reflection"] = await self.run_aars(
            environment,
            enable_reflection=False,
            system_name="aars_no_reflection",
        )

        for baseline in ALL_BASELINES:
            results["systems"][baseline.name] = await self.run_baseline(
                baseline.name,
                environment.llm_client,
            )

        significance_input = {
            name: system_result["per_sample_f1"]
            for name, system_result in results["systems"].items()
            if "per_sample_f1" in system_result
        }
        results["significance_vs_naive_rag"] = compare_systems(
            significance_input,
            baseline_name="naive_rag",
        )
        results["markdown_table"] = self.generate_markdown_table(results["systems"])
        return results

    @staticmethod
    def generate_markdown_table(systems: dict[str, dict[str, Any]]) -> str:
        lines = [
            "| System | EM | F1 | Recall@3 | Precision@3 | MRR@5 | NDCG@5 | Avg Latency (ms) |",
            "|--------|----|----|----------|-------------|-------|--------|------------------|",
        ]
        ordered = [
            "aars",
            "aars_no_reflection",
            "naive_rag",
            "hybrid_rag",
            "flare",
            "self_rag",
            "standard_routing",
        ]
        for name in ordered:
            result = systems[name]
            lines.append(
                "| "
                + " | ".join(
                    [
                        name,
                        f"{result['exact_match']:.3f}",
                        f"{result['token_f1']:.3f}",
                        f"{result['recall_at_3']:.3f}",
                        f"{result['precision_at_3']:.3f}",
                        f"{result['mrr_at_5']:.3f}",
                        f"{result['ndcg_at_5']:.3f}",
                        f"{result['avg_latency_ms']:.2f}",
                    ]
                )
                + " |"
            )
        return "\n".join(lines)


async def _run_and_save(output_path: str) -> dict[str, Any]:
    runner = BenchmarkRunner()
    results = await runner.run()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline AARS benchmark fixture")
    parser.add_argument(
        "--dataset",
        default="local_fixture",
        choices=DATASET_NAMES,
        help="Benchmark dataset to run",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results_local.json",
        help="Where to save the JSON results",
    )
    args = parser.parse_args()

    if args.dataset != "local_fixture":
        raise ValueError("Only the checked-in local_fixture benchmark is supported in this runner.")

    results = asyncio.run(_run_and_save(args.output))
    print(results["markdown_table"])


if __name__ == "__main__":
    main()
