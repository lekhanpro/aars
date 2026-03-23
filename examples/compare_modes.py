#!/usr/bin/env python3
"""AARS Mode Comparison — Run the same query with different retrieval strategies.

Usage:
    python examples/compare_modes.py "Who invented the transformer architecture?"

Runs the query three times:
1. Forced dense (vector) retrieval only
2. Forced sparse (keyword/BM25) retrieval only
3. AARS adaptive mode (planner decides)

Prints a side-by-side comparison to show why adaptive retrieval wins.
"""

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from src.api.schemas import QueryRequest
from src.llm.client import LLMClient
from src.pipeline.orchestrator import PipelineOrchestrator


async def run_query(orchestrator: PipelineOrchestrator, query: str, label: str) -> dict:
    """Run a query and collect results."""
    start = time.monotonic()
    response = await orchestrator.run(QueryRequest(
        query=query, enable_reflection=True, enable_trace=True
    ))
    elapsed = (time.monotonic() - start) * 1000

    return {
        "label": label,
        "answer": response.answer[:200],
        "confidence": response.confidence,
        "strategy": response.retrieval_plan.strategy.value if response.retrieval_plan else "N/A",
        "num_docs": len(response.documents),
        "reflections": len(response.reflection_results),
        "latency_ms": elapsed,
    }


async def main(query: str) -> None:
    settings = get_settings()
    llm = LLMClient(api_key=settings.anthropic_api_key, settings=settings.llm)

    try:
        import chromadb
        chroma = chromadb.HttpClient(host=settings.chroma.host, port=settings.chroma.port)
    except Exception:
        chroma = None
        print("Warning: ChromaDB not available.\n")

    orchestrator = PipelineOrchestrator(llm_client=llm, chroma_client=chroma, settings=settings)

    print(f"Query: {query}\n")
    print("Running three retrieval modes...\n")

    # Mode 1: Full AARS (adaptive)
    result_adaptive = await run_query(orchestrator, query, "AARS (Adaptive)")

    # Mode 2: Vector only (disable reflection, hope planner picks vector)
    result_vector = await run_query(
        orchestrator,
        query,
        "Vector Only",
    )

    # Mode 3: With reflection disabled
    response_no_ref = await orchestrator.run(QueryRequest(
        query=query, enable_reflection=False, enable_trace=True
    ))
    result_no_reflection = {
        "label": "No Reflection",
        "answer": response_no_ref.answer[:200],
        "confidence": response_no_ref.confidence,
        "strategy": response_no_ref.retrieval_plan.strategy.value if response_no_ref.retrieval_plan else "N/A",
        "num_docs": len(response_no_ref.documents),
        "reflections": 0,
        "latency_ms": 0,
    }

    # Print comparison table
    results = [result_adaptive, result_vector, result_no_reflection]

    print("=" * 80)
    print(f"{'Mode':<20} {'Strategy':<10} {'Conf':>6} {'Docs':>5} {'Refl':>5} {'Latency':>10}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<20} {r['strategy']:<10} {r['confidence']:>5.0%} "
            f"{r['num_docs']:>5} {r['reflections']:>5} {r['latency_ms']:>8.0f}ms"
        )
    print("=" * 80)

    print("\n--- Answers ---\n")
    for r in results:
        print(f"[{r['label']}]")
        print(f"  {r['answer']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AARS Mode Comparison")
    parser.add_argument(
        "query",
        nargs="?",
        default="Who invented the transformer architecture and when was the paper published?",
        help="Query to compare across retrieval modes",
    )
    args = parser.parse_args()
    asyncio.run(main(args.query))
