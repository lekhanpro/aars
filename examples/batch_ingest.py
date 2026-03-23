#!/usr/bin/env python3
"""AARS Batch Ingestion — Ingest an entire folder of documents in parallel.

Usage:
    python examples/batch_ingest.py /path/to/documents/ --collection my_collection

Supports PDF (.pdf) and text (.txt, .md) files.
Shows a progress bar and prints a summary of chunks indexed per file.
"""

import argparse
import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".rst"}


def find_documents(folder: str) -> list[Path]:
    """Recursively find all supported documents in a folder."""
    root = Path(folder)
    if not root.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


async def ingest_file(
    pipeline, file_path: Path, collection: str
) -> dict:
    """Ingest a single file and return results."""
    from fastapi import UploadFile

    try:
        content = file_path.read_bytes()
        upload = UploadFile(filename=file_path.name, file=BytesIO(content))
        result = await pipeline.ingest(file=upload, collection=collection)
        return {
            "file": file_path.name,
            "status": "success",
            "chunks": result.chunks_created,
            "documents": result.documents_ingested,
        }
    except Exception as e:
        return {
            "file": file_path.name,
            "status": "error",
            "error": str(e),
            "chunks": 0,
            "documents": 0,
        }


async def main(folder: str, collection: str, workers: int) -> None:
    from config.settings import get_settings
    from src.ingestion.pipeline import IngestionPipeline

    settings = get_settings()

    try:
        import chromadb
        chroma = chromadb.HttpClient(host=settings.chroma.host, port=settings.chroma.port)
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        sys.exit(1)

    pipeline = IngestionPipeline(chroma_client=chroma, settings=settings)
    files = find_documents(folder)

    if not files:
        print(f"No supported files found in {folder}")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"Found {len(files)} documents in {folder}")
    print(f"Collection: {collection}")
    print(f"Workers: {workers}")
    print("-" * 60)

    # Progress tracking
    results = []
    total = len(files)

    try:
        from tqdm import tqdm
        progress = tqdm(total=total, desc="Ingesting", unit="file")
    except ImportError:
        progress = None
        print("(Install tqdm for a progress bar: pip install tqdm)")

    # Process files with bounded concurrency
    semaphore = asyncio.Semaphore(workers)

    async def bounded_ingest(file_path: Path) -> dict:
        async with semaphore:
            result = await ingest_file(pipeline, file_path, collection)
            if progress:
                progress.update(1)
            else:
                status_icon = "+" if result["status"] == "success" else "!"
                print(f"  [{status_icon}] {result['file']}: {result['chunks']} chunks")
            return result

    tasks = [bounded_ingest(f) for f in files]
    results = await asyncio.gather(*tasks)

    if progress:
        progress.close()

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"{'File':<40} {'Status':<10} {'Chunks':>8}")
    print("-" * 60)

    total_chunks = 0
    success_count = 0
    for r in results:
        status_str = "OK" if r["status"] == "success" else "FAILED"
        print(f"{r['file']:<40} {status_str:<10} {r['chunks']:>8}")
        total_chunks += r["chunks"]
        if r["status"] == "success":
            success_count += 1

    print("-" * 60)
    print(f"Total: {success_count}/{total} files, {total_chunks} chunks indexed")

    # Show errors
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for r in errors:
            print(f"  {r['file']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AARS Batch Document Ingestion")
    parser.add_argument("folder", help="Path to folder containing documents")
    parser.add_argument("--collection", default="default", help="Target collection name")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()
    asyncio.run(main(args.folder, args.collection, args.workers))
