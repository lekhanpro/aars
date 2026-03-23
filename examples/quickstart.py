#!/usr/bin/env python3
"""AARS Quickstart — Ingest one file and run one query.

Usage:
    python examples/quickstart.py path/to/document.pdf "What is the main topic?"

The simplest possible AARS usage in 10 lines of meaningful code.
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


async def main(file_path: str, query: str) -> None:
    # Initialize
    settings = get_settings()
    llm = LLMClient(api_key=settings.anthropic_api_key, settings=settings.llm)

    # Ingest the document
    from src.ingestion.pipeline import IngestionPipeline
    import chromadb

    chroma = chromadb.HttpClient(host=settings.chroma.host, port=settings.chroma.port)
    pipeline = IngestionPipeline(chroma_client=chroma, settings=settings)

    from fastapi import UploadFile
    from io import BytesIO

    with open(file_path, "rb") as f:
        content = f.read()
    upload = UploadFile(filename=os.path.basename(file_path), file=BytesIO(content))
    result = await pipeline.ingest(file=upload, collection="default")
    print(f"Ingested: {result.chunks_created} chunks from {file_path}")

    # Query
    orchestrator = PipelineOrchestrator(llm_client=llm, chroma_client=chroma, settings=settings)
    response = await orchestrator.run(QueryRequest(query=query))
    print(f"\nAnswer: {response.answer}")
    print(f"Confidence: {response.confidence:.0%}")
    print(f"Strategy: {response.retrieval_plan.strategy.value if response.retrieval_plan else 'N/A'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AARS Quickstart")
    parser.add_argument("file", help="Path to a PDF or text file to ingest")
    parser.add_argument("query", help="Question to ask about the document")
    args = parser.parse_args()
    asyncio.run(main(args.file, args.query))
