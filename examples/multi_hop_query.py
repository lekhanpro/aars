#!/usr/bin/env python3
"""AARS Multi-hop Query Demo — Shows query decomposition and graph retrieval.

Usage:
    python examples/multi_hop_query.py "How does BERT relate to transformers and what impact did it have on NLP?"

Demonstrates:
- Planner detecting multi_hop query type
- Query decomposition into sub-queries
- Graph retrieval for entity relationships
- Reflection loop iterations
- Final answer with citations
"""

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import get_settings
from src.api.schemas import QueryRequest
from src.llm.client import LLMClient
from src.pipeline.orchestrator import PipelineOrchestrator


async def main(query: str) -> None:
    settings = get_settings()
    llm = LLMClient(api_key=settings.anthropic_api_key, settings=settings.llm)

    try:
        import chromadb
        chroma = chromadb.HttpClient(host=settings.chroma.host, port=settings.chroma.port)
    except Exception:
        chroma = None
        print("Warning: ChromaDB not available. Running with limited retrieval.\n")

    orchestrator = PipelineOrchestrator(llm_client=llm, chroma_client=chroma, settings=settings)
    response = await orchestrator.run(QueryRequest(query=query, enable_reflection=True))

    # Display planner decision
    if response.retrieval_plan:
        plan = response.retrieval_plan
        print("=" * 60)
        print("PLANNER DECISION")
        print("=" * 60)
        print(f"  Query Type:   {plan.query_type.value}")
        print(f"  Complexity:   {plan.complexity.value}")
        print(f"  Strategy:     {plan.strategy.value}")
        print(f"  Rewritten:    {plan.rewritten_query}")
        if plan.decomposed_queries:
            print(f"  Sub-queries:")
            for i, sq in enumerate(plan.decomposed_queries, 1):
                print(f"    {i}. {sq}")
        print(f"  Reasoning:    {plan.reasoning}")

    # Display reflection iterations
    if response.reflection_results:
        print(f"\n{'=' * 60}")
        print(f"REFLECTION LOOP ({len(response.reflection_results)} iterations)")
        print("=" * 60)
        for i, ref in enumerate(response.reflection_results, 1):
            status = "SUFFICIENT" if ref.sufficient else "INSUFFICIENT"
            print(f"\n  Iteration {i}: [{status}]")
            print(f"    Confidence:  {ref.confidence:.0%}")
            if ref.missing_information:
                print(f"    Missing:     {ref.missing_information}")
            if ref.next_query:
                print(f"    Next query:  {ref.next_query}")
            if ref.next_strategy:
                print(f"    Next strat:  {ref.next_strategy}")

    # Display answer
    print(f"\n{'=' * 60}")
    print("ANSWER")
    print("=" * 60)
    print(f"\n{response.answer}\n")
    print(f"Confidence: {response.confidence:.0%}")

    # Display citations
    if response.citations:
        print(f"\nCitations:")
        for cite in response.citations:
            print(f"  [{cite.doc_id}] {cite.text[:100]}...")

    # Display trace summary
    if response.trace:
        print(f"\nTrace: {response.trace.total_duration_ms:.0f}ms total, "
              f"{response.trace.total_tokens} tokens, "
              f"{response.trace.total_api_calls} API calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AARS Multi-hop Query Demo")
    parser.add_argument(
        "query",
        nargs="?",
        default="How does the attention mechanism in BERT relate to the original transformer architecture, and what improvements did it introduce for NLP tasks?",
        help="A complex multi-hop question",
    )
    args = parser.parse_args()
    asyncio.run(main(args.query))
