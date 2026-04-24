```
     _    _    ____  ____
    / \  / \  |  _ \/ ___|
   / _ \/ _ \ | |_) \___ \
  / ___ / ___ \|  _ < ___) |
 /_/  \_\_/  \_\_| \_\____/
```

# AARS — Adaptive Agentic Retrieval System

**Self-correcting, intent-aware RAG with built-in benchmarking.**

AARS treats retrieval as a query-time decision instead of a fixed pipeline. It classifies intent, plans retrieval strategy, retrieves with hybrid vector + keyword + graph search, grades relevance per-document, rewrites failed queries, reranks with a cross-encoder, generates grounded answers with citations, checks for hallucinations, and self-evaluates quality — all in a single adaptive pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4B32C3?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-FF6F61?style=flat-square)](https://www.trychroma.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-evaluation-115e59?style=flat-square)](https://github.com/explodinggradients/ragas)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-14222b?style=flat-square)](https://lekhanpro.github.io/aars/)
[![Benchmark](https://img.shields.io/badge/benchmark-local_fixture-115e59?style=flat-square)](https://github.com/lekhanpro/aars/tree/main/benchmarks)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CI](https://github.com/lekhanpro/aars/actions/workflows/ci.yml/badge.svg)](https://github.com/lekhanpro/aars/actions/workflows/ci.yml)

Documentation site: [lekhanpro.github.io/aars](https://lekhanpro.github.io/aars/)

---

## Pipeline Overview

![AARS Pipeline](assets/aars-pipeline.svg)

### How It Works

1. **Classify** — Intent Router classifies the query as simple, complex, multi-hop, or direct.
2. **Plan** — Query Planner selects `keyword`, `vector`, `graph`, `hybrid`, or `none` strategy and decomposes multi-hop queries.
3. **Retrieve** — runs the selected retrievers against the requested collection.
4. **Grade** — Relevance Grader evaluates each document individually (YES/NO).
5. **Reflect** — checks whether the current evidence is sufficient to answer.
6. **Rewrite** — Query Rewriter rewrites failed queries with synonym expansion, abstraction, or context enrichment.
7. **Fuse** — merges ranked lists with Reciprocal Rank Fusion and optionally reranks with MMR.
8. **Rerank** — Cross-Encoder Reranker (ms-marco-MiniLM) provides fine-grained relevance scoring.
9. **Generate** — returns grounded answers with citations, documents, confidence, and trace data.
10. **Verify** — Hallucination Checker verifies grounding; Self-RAG Evaluator computes RAGAS-style scores.

---

## Architecture

```
USER QUERY
    │
    ▼
┌─────────────────┐
│  INTENT ROUTER  │ ← Classifies: simple / complex / multi_hop / direct
│  (Agent 0)      │   Direct queries skip retrieval entirely
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  QUERY PLANNER  │ ← Selects strategy + decomposes multi-hop queries
│  (Agent 1)      │   keyword / vector / graph / hybrid / none
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────┐
│         RETRIEVAL LAYER          │
│  ┌──────────┐  ┌──────────────┐  │
│  │  Vector  │  │  Keyword     │  │
│  │ (Chroma) │  │  (BM25)      │  │
│  └──────────┘  └──────────────┘  │
│  ┌──────────┐  ┌──────────────┐  │
│  │  Graph   │  │   Hybrid     │  │
│  │(NetworkX)│  │  (RRF Fusion)│  │
│  └──────────┘  └──────────────┘  │
└──────────────────────────────────┘
         │
         ▼
┌────────────────────┐
│  RELEVANCE GRADER  │ ← Per-document YES/NO grading
│  (Agent 2)         │   Filters irrelevant chunks
└────────┬───────────┘
         │
    ┌────┴────┐
    ▼         ▼
RELEVANT   NOT RELEVANT
    │         │
    │    ┌────▼──────────┐
    │    │ QUERY REWRITER │ ← Synonym expansion / abstraction / context
    │    │ (Agent 3)      │   Max 3 rewrite attempts, then fallback
    │    └───────┬────────┘
    │            │ (loops back to retrieval)
    ▼
┌────────────────────┐
│  CROSS-ENCODER     │ ← ms-marco-MiniLM-L-6-v2
│  RERANKER (Agent 4)│   Reranks by query-document relevance
└────────┬───────────┘
         │
         ▼
┌─────────────────────┐
│    GENERATOR (LLM)  │ ← Grounded answer with citations
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  HALLUCINATION      │ ← NLI or LLM-as-judge grounding check
│  CHECKER (Agent 5)  │   Flags ungrounded claims
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  SELF-RAG EVALUATOR │ ← RAGAS-style inline scoring
│  (Agent 6)          │   Faithfulness, Relevancy, Precision, Recall
└──────────┬──────────┘
           │
           ▼
    FINAL ANSWER + CONFIDENCE + RAGAS SCORES
```

---

## Why AARS Instead Of Fixed-Pipeline RAG?

Traditional RAG stacks usually force one retrieval mode onto every question. AARS is built around the idea that factual, semantic, and multi-hop questions should not all be handled the same way.

![AARS vs Existing RAG](assets/aars-vs-rag.svg)

- Query planning is request-aware instead of hard-coded.
- Retrieval is collection-aware across vector, keyword, and graph state.
- Reflection can trigger another retrieval round instead of answering from weak context.
- Fusion and MMR are first-class parts of the runtime path, not presentation-layer claims.

---

## System Architecture

The checked-in runtime is actually wired end to end: startup builds the shared orchestrator, ingestion updates all retrievers for the target collection, and query execution reuses shared state instead of rebuilding components per request.

![AARS System Architecture](assets/architecture.svg)

### Implemented Runtime Capabilities

- Shared startup services for orchestrator, ingestion pipeline, keyword retriever, and graph retriever.
- Collection-aware vector, keyword, and graph retrieval.
- 7 agents: Intent Router, Query Planner, Relevance Grader, Query Rewriter, Cross-Encoder Reranker, Hallucination Checker, Self-RAG Evaluator.
- LangGraph StateGraph pipeline with conditional edges for retry loops.
- Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) and NLI hallucination detection.
- Deterministic hashing embeddings fallback when `sentence-transformers` is unavailable.
- Deterministic entity-extraction fallback when spaCy or `en_core_web_sm` is unavailable.
- A real offline benchmark runner in [`benchmarks/runner.py`](benchmarks/runner.py) with checked-in fixture data and results.
- RAGAS + DeepEval evaluation framework integration.

---

## Benchmark Results

### AARS Local Benchmark

Reproducible local benchmark with checked-in fixture data (12 documents, 9 questions, 7 systems):

![AARS Local Benchmark](assets/benchmarks.svg)

| System | EM | F1 | Recall@3 | Precision@3 | MRR@5 | NDCG@5 | Avg Latency (ms) |
|--------|----|----|----------|-------------|-------|--------|------------------|
| **AARS** | **1.000** | **1.000** | **1.000** | **0.537** | **0.944** | **0.959** | 3.44 |
| AARS no reflection | 1.000 | 1.000 | 1.000 | 0.537 | 0.944 | 0.959 | 3.56 |
| NaiveRAG | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| HybridRAG | 1.000 | 1.000 | 1.000 | 0.444 | 1.000 | 0.991 | 0.00 |
| FLARE-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| Self-RAG-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |
| StandardRouting | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 | 0.00 |

Latency values are local in-process measurements and will vary by machine and run. The ranking metrics and answer metrics are the stable part of this fixture. Result file: [`benchmarks/results_local.json`](benchmarks/results_local.json)

### AARS vs Other RAG Approaches

| Feature | Naive RAG | Advanced RAG | [TreeDex](https://github.com/mithun50/treedex) | AARS |
|---------|-----------|-------------|---------|------|
| **Retrieval** | Vector only | Vector + keyword | Tree-based (vectorless) | Adaptive (vector + keyword + graph + hybrid) |
| **Query classification** | None | Sometimes | None | Intent Router + Query Planner |
| **Strategy selection** | Fixed | Fixed | Fixed (tree traversal) | Dynamic per-query |
| **Multi-hop support** | None | None | Limited (tree depth) | Graph retrieval + query decomposition |
| **Self-correction** | None | None | None | Reflection loop + query rewriting |
| **Reranking** | None | Sometimes | None | Cross-encoder (ms-marco) |
| **Hallucination check** | None | None | None | NLI + LLM-as-judge |
| **Inline evaluation** | None | None | None | RAGAS-style scores per query |
| **Page attribution** | Chunk-level | Chunk-level | Exact page citations | Chunk-level with document IDs |
| **Infrastructure** | Vector DB | Vector DB | None (JSON index) | Vector DB + optional graph |
| **Index format** | Opaque embeddings | Opaque embeddings | Human-readable JSON tree | Embeddings + BM25 + entity graph |
| **Agent observability** | None | None | None | Full step-by-step trace |
| **Embedding required** | Yes | Yes | No | Yes (with deterministic fallback) |

### Comparison Philosophy

- **Naive RAG**: Embed → retrieve → generate. No intelligence in the pipeline.
- **Advanced RAG**: Adds hybrid search or reranking but remains a fixed pipeline.
- **[TreeDex](https://github.com/mithun50/treedex)** by [@mithun50](https://github.com/mithun50): Takes a completely different approach — no vectors, no embeddings. Indexes documents into navigable tree structures (using PDF ToC or LLM-detected headings) and retrieves by tree traversal. Excellent for structured documents with clear hierarchies. Supports 14+ LLM providers and produces human-readable JSON indexes that work across Python and Node.js.
- **AARS**: Adaptive multi-strategy pipeline with 7 agents. Classifies intent *before* retrieval, selects strategy per-query, self-corrects when retrieval fails, verifies grounding after generation, and evaluates its own answer quality inline.

### Run The Benchmark

```bash
python benchmarks/runner.py --output benchmarks/results_local.json

# With RAGAS evaluation (requires pip install -e ".[eval]")
python benchmarks/runner.py --ragas --output benchmarks/results_local.json

# With DeepEval regression testing
python benchmarks/runner.py --deepeval --output benchmarks/results_local.json
```

---

## Agent Overview

| # | Agent | Role | When It Runs |
|---|-------|------|-------------|
| 0 | **Intent Router** | Classifies query as simple/complex/multi_hop/direct | Every query |
| 1 | **Query Planner** | Selects retrieval strategy, decomposes multi-hop queries | Every query (unless disabled) |
| 2 | **Relevance Grader** | Grades each retrieved document as relevant/irrelevant | After retrieval |
| 3 | **Query Rewriter** | Rewrites failed queries with synonym expansion or abstraction | When grading/reflection fails |
| 4 | **Cross-Encoder Reranker** | Reranks documents using ms-marco cross-encoder | After fusion |
| 5 | **Hallucination Checker** | Verifies answer grounding in source documents | After generation |
| 6 | **Self-RAG Evaluator** | Computes faithfulness, relevancy, precision, recall | Every query |

All agents are behind feature flags (`enable_*`) and can be toggled per request.

---

## LangGraph Pipeline

AARS ships with two pipeline modes:

- **`/query`** — Manual orchestrator (default, battle-tested)
- **`/query/graph`** — LangGraph StateGraph with conditional edges

The LangGraph pipeline uses the same agents and components but orchestrates them as a state machine with automatic routing decisions at each node.

```
route_intent → plan_query → retrieve → grade_relevance
    → [fuse | rewrite_query → retrieve (loop)]
    → rerank → generate → check_hallucination → evaluate → END
```

---

## Quick Start

### Install

```bash
git clone https://github.com/lekhanpro/aars.git
cd aars
pip install -e ".[dev,ui]"
```

### Optional Runtime Extras

```bash
# spaCy model for graph retrieval
python -m spacy download en_core_web_sm

# RAGAS + DeepEval for evaluation
pip install -e ".[eval]"
```

If `sentence-transformers` is missing, AARS falls back to deterministic hashing embeddings. If spaCy is missing, graph extraction falls back to a simple heuristic instead of failing outright.

### Start ChromaDB And The API

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
    "enable_reranker": true,
    "enable_hallucination_check": true,
    "enable_grading": true,
    "enable_trace": true
  }'
```

### Query With LangGraph Pipeline

```bash
curl -X POST http://localhost:8000/api/v1/query/graph \
  -H "Content-Type: application/json" \
  -d '{"query": "How do transformers work?", "collection": "demo"}'
```

### Launch The Dashboard

```bash
streamlit run ui/app.py
```

---

## API Surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/query` | Main RAG pipeline (orchestrator) |
| `POST` | `/api/v1/query/graph` | LangGraph-based RAG pipeline |
| `POST` | `/api/v1/ingest` | Load a text or PDF file into a collection and update all retrievers |
| `GET` | `/api/v1/health` | Report API and ChromaDB connectivity |
| `GET` | `/api/v1/collections` | List available collections |
| `DELETE` | `/api/v1/collections/{name}` | Delete a collection |
| `GET` | `/api/v1/documents/{collection}` | List documents in a collection |
| `GET` | `/api/v1/debug/trace/{id}` | Fetch a stored execution trace |

---

## Evaluation Frameworks

### Built-in Self-RAG Evaluation (per query)
Every query returns inline scores: faithfulness, answer_relevancy, context_precision, context_recall.

### RAGAS (batch evaluation)
```bash
python benchmarks/runner.py --ragas --output benchmarks/results_local.json
```

### DeepEval (regression testing)
```bash
python benchmarks/runner.py --deepeval --output benchmarks/results_local.json
```

---

## Project Layout

```text
aars/
├── assets/               # SVG diagrams (pipeline, architecture, benchmarks, comparison)
├── benchmarks/           # Local fixture, metrics, baselines, RAGAS/DeepEval wrappers
├── config/               # Settings, logging, prompt templates for all 7 agents
├── docs/                 # GitHub Pages documentation site
├── examples/             # Usage examples (quickstart, batch ingest, multi-hop, compare modes)
├── src/
│   ├── agents/           # 7 agent modules
│   ├── api/              # FastAPI endpoints and Pydantic schemas
│   ├── fusion/           # RRF + MMR fusion pipeline
│   ├── generation/       # Answer generator with citations
│   ├── ingestion/        # Document loaders, chunkers, graph builder
│   ├── llm/              # Anthropic SDK wrapper
│   ├── pipeline/         # Orchestrator, LangGraph state/graph, trace recorder
│   ├── retrieval/        # Vector (ChromaDB), keyword (BM25), graph (NetworkX)
│   └── utils/            # Embedding model, cross-encoder model
├── tests/                # Unit + integration tests
├── ui/                   # 3-tab Streamlit dashboard (Query / Benchmarks / Documents)
└── README.md
```

---

## Multimodal Support

AARS supports ingestion of images and video files alongside text documents. During ingestion, multimodal files are automatically segregated: images (`.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.webp`, `.tiff`) and videos (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) are detected by extension, stored with appropriate metadata, and indexed into the target collection. This allows retrieval queries to surface multimodal evidence when relevant, while keeping text-based chunking and embedding pipelines separate from binary media handling.

---

## Development

```bash
pip install -e ".[dev]"
python -m compileall src benchmarks tests examples ui config scripts
pytest -q
```

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lekhanpro/aars&type=Date)](https://star-history.com/#lekhanpro/aars&Date)

---

## Current Limitations

- Live answer generation still depends on an Anthropic-compatible key.
- The reproducible benchmark is local-fixture only, not HotpotQA or another external public dataset.
- Cross-encoder reranking and NLI hallucination detection require `sentence-transformers` (falls back to random scores when unavailable).
- LangGraph pipeline (`/query/graph`) requires the `langgraph` package.

---

## Citation

If you use AARS in your research, please cite:

```bibtex
@software{aars2025,
  title     = {AARS: Adaptive Agentic Retrieval System},
  author    = {Lekhan},
  year      = {2025},
  url       = {https://github.com/lekhanpro/aars},
  note      = {Self-correcting, intent-aware RAG with 7-agent pipeline and built-in benchmarking}
}
```

---

## Contributing

Contributions welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Run `pytest -q` and `python -m compileall src` before submitting
5. Open a pull request with a clear description

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

## License

MIT. See [LICENSE](LICENSE).
