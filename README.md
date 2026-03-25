# AARS

Agentic Adaptive Retrieval System: a query-aware RAG backend that plans before retrieving, can retry retrieval through reflection, and now has a reproducible checked-in benchmark.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

![AARS Pipeline](assets/aars-pipeline.svg)

## What is implemented

The checked-in project now supports a real end-to-end runtime path instead of disconnected components:

- shared startup services for the orchestrator, ingestion pipeline, keyword retriever, and graph retriever
- collection-aware vector, keyword, and graph retrieval
- ingestion that updates Chroma-compatible storage, BM25 state, and the entity graph together
- planner, reflection, fusion, and MMR toggles in the query request schema
- deterministic fallback embeddings when `sentence-transformers` is not installed
- deterministic fallback entity extraction when spaCy or `en_core_web_sm` is not installed
- a runnable offline benchmark in `benchmarks/runner.py` with checked-in results in `benchmarks/results_local.json`

## Architecture

At query time AARS runs this loop:

1. planner chooses `keyword`, `vector`, `graph`, `hybrid`, or `none`
2. one or more retrievers run against the requested collection
3. reflection decides whether the evidence is sufficient
4. retrieval can retry with a revised strategy
5. reciprocal-rank fusion merges result lists
6. MMR reranks for diversity before grounded answer generation

The API returns the answer, citations, retrieved documents, retrieval plan, reflection results, and an optional execution trace.

## Reproducible benchmark

The only checked-in benchmark result in this repository is the local offline fixture benchmark. It is small by design and exists to be rerunnable without network access or external APIs.

- documents: 12
- questions: 9
- systems compared: AARS, AARS without reflection, NaiveRAG, HybridRAG, FLARE-style, Self-RAG-style, StandardRouting
- results file: `benchmarks/results_local.json`

### Current benchmark summary

| System | EM | F1 | Recall@3 | Precision@3 | MRR@5 | NDCG@5 | Avg Latency (ms) |
|--------|----|----|----------|-------------|-------|--------|------------------|
| AARS | 1.000 | 1.000 | 1.000 | 0.537 | 0.944 | 0.959 | 1.67 |
| AARS no reflection | 1.000 | 1.000 | 1.000 | 0.537 | 0.944 | 0.959 | 1.78 |
| NaiveRAG | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| HybridRAG | 1.000 | 1.000 | 1.000 | 0.444 | 1.000 | 0.991 | 0.00 |
| FLARE-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| Self-RAG-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| StandardRouting | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |

These numbers are from the local fixture only. They are useful for regression checking and verifying the retrieval pipeline, but they are not a substitute for large external QA benchmarks.
Latency values are in-process measurements on the local machine and should be treated as relative rather than portable.

### Run the benchmark

```bash
python benchmarks/runner.py --output benchmarks/results_local.json
```

The runner prints a markdown table and writes a JSON report with per-system metrics and significance-test scaffolding.

## Quick start

### 1. Install

```bash
git clone https://github.com/lekhanpro/aars.git
cd aars
pip install -e ".[dev,ui]"
```

### 2. Optional runtime extras

These improve retrieval quality but are no longer hard requirements for basic local benchmarking:

```bash
python -m spacy download en_core_web_sm
```

If `sentence-transformers` is missing, AARS falls back to a deterministic hashing embedder. If spaCy or the English model is missing, graph extraction falls back to a simple capitalized-entity heuristic.

### 3. Start Chroma and the API

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env

docker run -p 8001:8000 chromadb/chroma:latest
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Ingest a document

```bash
curl -X POST http://localhost:8000/api/v1/ingest ^
  -F "file=@my_document.txt" ^
  -F "collection=demo"
```

### 5. Query AARS

```bash
curl -X POST http://localhost:8000/api/v1/query ^
  -H "Content-Type: application/json" ^
  -d "{
    \"query\": \"What sparse ranking algorithm rewards exact term overlap?\",
    \"collection\": \"demo\",
    \"top_k\": 5,
    \"enable_planner\": true,
    \"enable_reflection\": true,
    \"enable_fusion\": true,
    \"enable_mmr\": true,
    \"enable_keyword\": true,
    \"enable_graph\": true,
    \"default_strategy\": \"vector\",
    \"enable_trace\": true
  }"
```

## API surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/ingest` | Load a text or PDF file into a collection and update all retrievers |
| `POST` | `/api/v1/query` | Run planning, retrieval, reflection, fusion, and answer generation |
| `GET` | `/api/v1/health` | Report API and Chroma connectivity |
| `GET` | `/api/v1/collections` | List available collections |
| `DELETE` | `/api/v1/collections/{name}` | Delete a collection |
| `GET` | `/api/v1/debug/trace/{id}` | Fetch a stored execution trace |

## Project layout

```text
aars/
├── benchmarks/
│   ├── baselines.py
│   ├── datasets.py
│   ├── local_fixture.py
│   ├── metrics.py
│   ├── results_local.json
│   ├── runner.py
│   └── significance.py
├── config/
│   ├── prompts/
│   └── settings.py
├── src/
│   ├── agents/
│   ├── api/
│   ├── fusion/
│   ├── generation/
│   ├── ingestion/
│   ├── pipeline/
│   ├── retrieval/
│   └── utils/
├── tests/
├── ui/
└── README.md
```

## Development

Basic verification:

```bash
python -m compileall src benchmarks tests examples ui config scripts
```

If `pytest` is installed in the environment:

```bash
pytest
```

## Current limitations

- answer generation in the API still depends on an Anthropic-compatible key for live use
- the checked-in reproducible benchmark is a local fixture, not HotpotQA or another external dataset
- external dataset loader code still exists under `benchmarks/datasets.py`, but there are no checked-in public results for those datasets

## License

MIT. See [LICENSE](LICENSE).
