# AARS

Adaptive multi-strategy RAG backend with planner, reflection, fusion, and a reproducible offline benchmark.

AARS treats retrieval as a query-time decision instead of a fixed pipeline. It can plan before retrieving, switch between keyword, vector, graph, or hybrid retrieval, reflect on whether the evidence is sufficient, then fuse and rerank before answer generation.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-14222b?style=flat-square)](https://lekhanpro.github.io/aars/)
[![Benchmark](https://img.shields.io/badge/benchmark-local_fixture-115e59?style=flat-square)](https://github.com/lekhanpro/aars/tree/main/benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

Documentation site: [lekhanpro.github.io/aars](https://lekhanpro.github.io/aars/)

![AARS Pipeline](assets/aars-pipeline.svg)

## How It Works

1. `Plan` classifies the query and selects `keyword`, `vector`, `graph`, `hybrid`, or `none`.
2. `Retrieve` runs the selected retrievers against the requested collection.
3. `Reflect` checks whether the current evidence is sufficient to answer.
4. `Retry` can revise query and strategy when retrieval is not good enough.
5. `Fuse` merges ranked lists with reciprocal-rank fusion and optionally reranks with MMR.
6. `Generate` returns grounded answers with citations, documents, confidence, and optional trace data.

## Why AARS Instead Of Fixed-Pipeline RAG?

Traditional RAG stacks usually force one retrieval mode onto every question. AARS is built around the idea that factual, semantic, and multi-hop questions should not all be handled the same way.

![AARS vs Existing RAG](assets/aars-vs-rag.svg)

- Query planning is request-aware instead of hard-coded.
- Retrieval is collection-aware across vector, keyword, and graph state.
- Reflection can trigger another retrieval round instead of answering from weak context.
- Fusion and MMR are first-class parts of the runtime path, not presentation-layer claims.

## System Architecture

The checked-in runtime is actually wired end to end: startup builds the shared orchestrator, ingestion updates all retrievers for the target collection, and query execution reuses shared state instead of rebuilding components per request.

![AARS System Architecture](assets/architecture.svg)

### Implemented Runtime Capabilities

- Shared startup services for orchestrator, ingestion pipeline, keyword retriever, and graph retriever.
- Collection-aware vector, keyword, and graph retrieval.
- Planner, reflection, fusion, MMR, keyword, graph, and default-strategy toggles in the query schema.
- Deterministic hashing embeddings fallback when `sentence-transformers` is unavailable.
- Deterministic entity-extraction fallback when spaCy or `en_core_web_sm` is unavailable.
- A real offline benchmark runner in [`benchmarks/runner.py`](benchmarks/runner.py) with checked-in fixture data and results.

## Reproducible Local Benchmark

This repository does not claim an external public benchmark result. The checked-in benchmark is the local offline fixture only, designed for reproducibility and regression testing.

![AARS Local Benchmark](assets/benchmarks.svg)

- Documents: `12`
- Questions: `9`
- Systems: `AARS`, `AARS without reflection`, `NaiveRAG`, `HybridRAG`, `FLARE-style`, `Self-RAG-style`, `StandardRouting`
- Result file: [`benchmarks/results_local.json`](benchmarks/results_local.json)

### Current Benchmark Summary

| System | EM | F1 | Recall@3 | Precision@3 | MRR@5 | NDCG@5 | Avg Latency (ms) |
|--------|----|----|----------|-------------|-------|--------|------------------|
| AARS | 1.000 | 1.000 | 1.000 | 0.537 | 0.944 | 0.959 | 1.67 |
| AARS no reflection | 1.000 | 1.000 | 1.000 | 0.537 | 0.944 | 0.959 | 1.78 |
| NaiveRAG | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| HybridRAG | 1.000 | 1.000 | 1.000 | 0.444 | 1.000 | 0.991 | 0.00 |
| FLARE-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| Self-RAG-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| StandardRouting | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |

Latency values are local in-process measurements and will vary by machine and run. The ranking metrics and answer metrics are the stable part of this fixture.

### Run The Benchmark

```bash
python benchmarks/runner.py --output benchmarks/results_local.json
```

## Quick Start

### Install

```bash
git clone https://github.com/lekhanpro/aars.git
cd aars
pip install -e ".[dev,ui]"
```

### Optional Runtime Extras

```bash
python -m spacy download en_core_web_sm
```

If `sentence-transformers` is missing, AARS falls back to deterministic hashing embeddings. If spaCy is missing, graph extraction falls back to a simple heuristic instead of failing outright.

### Start Chroma And The API

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env

docker run -p 8001:8000 chromadb/chroma:latest
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Ingest A Document

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -F "file=@my_document.txt" \
  -F "collection=demo"
```

### Query AARS

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What sparse ranking algorithm rewards exact term overlap?",
    "collection": "demo",
    "top_k": 5,
    "enable_planner": true,
    "enable_reflection": true,
    "enable_fusion": true,
    "enable_mmr": true,
    "enable_keyword": true,
    "enable_graph": true,
    "default_strategy": "vector",
    "enable_trace": true
  }'
```

## API Surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/ingest` | Load a text or PDF file into a collection and update all retrievers |
| `POST` | `/api/v1/query` | Run planning, retrieval, reflection, fusion, and answer generation |
| `GET` | `/api/v1/health` | Report API and Chroma connectivity |
| `GET` | `/api/v1/collections` | List available collections |
| `DELETE` | `/api/v1/collections/{name}` | Delete a collection |
| `GET` | `/api/v1/debug/trace/{id}` | Fetch a stored execution trace |

## Project Layout

```text
aars/
├── assets/
├── benchmarks/
├── config/
├── docs/
├── examples/
├── src/
├── tests/
├── ui/
└── README.md
```

## Development

```bash
python -m compileall src benchmarks tests examples ui config scripts
pytest -q
```

## Current Limitations

- Live answer generation still depends on an Anthropic-compatible key.
- The reproducible benchmark is local-fixture only, not HotpotQA or another external public dataset.
- `paper/main.pdf` may still exist locally if Windows has it locked open, but it is ignored and no longer part of the tracked project surface.

## License

MIT. See [LICENSE](LICENSE).
