```
     _    _    ____  ____
    / \  / \  |  _ \/ ___|
   / _ \/ _ \ | |_) \___ \
  / ___ / ___ \|  _ < ___) |
 /_/  \_\_/  \_\_| \_\____/
```

# AARS — Adaptive Agentic Retrieval System

**Self-correcting, intent-aware RAG with built-in benchmarking.**

AARS treats retrieval as a query-time decision. It classifies intent, plans retrieval strategy, retrieves with hybrid vector + keyword + graph search, grades relevance per-document, rewrites failed queries, reranks with a cross-encoder, generates grounded answers with citations, checks for hallucinations, and self-evaluates quality — all in a single adaptive pipeline.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-pipeline-4B32C3?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-FF6F61?style=flat-square)](https://www.trychroma.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-evaluation-115e59?style=flat-square)](https://github.com/explodinggradients/ragas)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CI](https://github.com/lekhanpro/aars/actions/workflows/ci.yml/badge.svg)](https://github.com/lekhanpro/aars/actions/workflows/ci.yml)

Documentation: [lekhanpro.github.io/aars](https://lekhanpro.github.io/aars/)

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

## What Makes AARS Different

| Feature | Naive RAG | Advanced RAG | AARS |
|---------|-----------|-------------|------|
| Query classification | No | Sometimes | Intent router + query planner |
| Retrieval strategy | Fixed (vector only) | Hybrid | Adaptive (keyword/vector/graph/hybrid/none) |
| Self-correction | No | No | Reflection loop + query rewriting |
| Reranking | No | Sometimes | Cross-encoder (ms-marco) |
| Hallucination check | No | No | NLI + LLM-as-judge |
| Inline evaluation | No | No | RAGAS-style scores per query |
| Agent trace | No | No | Full step-by-step observability |

---

## Benchmark Results

Local reproducible benchmark (12 documents, 9 questions, 7 systems):

| System | EM | F1 | Recall@3 | Precision@3 | MRR@5 | NDCG@5 |
|--------|----|----|----------|-------------|-------|--------|
| **AARS** | **1.000** | **1.000** | **1.000** | **0.537** | **0.944** | **0.959** |
| AARS no reflection | 1.000 | 1.000 | 1.000 | 0.537 | 0.944 | 0.959 |
| NaiveRAG | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 |
| HybridRAG | 1.000 | 1.000 | 1.000 | 0.444 | 1.000 | 0.991 |
| FLARE-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 |
| Self-RAG-style | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 |
| StandardRouting | 1.000 | 1.000 | 1.000 | 0.444 | 0.944 | 0.959 |

Additional evaluation available with RAGAS (faithfulness, answer relevancy, context precision, context recall) and DeepEval (regression testing).

---

## Quick Start

```bash
git clone https://github.com/lekhanpro/aars.git
cd aars
pip install -e ".[dev,ui]"

# Optional: spaCy model for graph retrieval
python -m spacy download en_core_web_sm

# Optional: RAGAS + DeepEval for evaluation
pip install -e ".[eval]"
```

### Start ChromaDB and the API

```bash
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

docker run -p 8001:8000 chromadb/chroma:latest
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### Ingest a Document

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

### Query with LangGraph Pipeline

```bash
curl -X POST http://localhost:8000/api/v1/query/graph \
  -H "Content-Type: application/json" \
  -d '{"query": "How do transformers work?", "collection": "demo"}'
```

### Launch the Dashboard

```bash
streamlit run ui/app.py
```

---

## Agent Overview

| Agent | Role | When It Runs |
|-------|------|-------------|
| **Intent Router** | Classifies query as simple/complex/multi_hop/direct | Every query |
| **Query Planner** | Selects retrieval strategy, decomposes multi-hop queries | Every query (unless disabled) |
| **Relevance Grader** | Grades each retrieved document as relevant/irrelevant | After retrieval |
| **Query Rewriter** | Rewrites failed queries with synonym expansion or abstraction | When grading/reflection fails |
| **Cross-Encoder Reranker** | Reranks documents using ms-marco cross-encoder | After fusion |
| **Hallucination Checker** | Verifies answer grounding in source documents | After generation |
| **Self-RAG Evaluator** | Computes faithfulness, relevancy, precision, recall | Every query |

All agents are behind feature flags (`enable_*`) and can be toggled per request.

---

## LangGraph Pipeline

AARS ships with two pipeline modes:

- **`/query`** — Manual orchestrator (default, battle-tested)
- **`/query/graph`** — LangGraph StateGraph with conditional edges

The LangGraph pipeline uses the same agents and components but orchestrates them as a state machine with automatic routing decisions at each node.

---

## API Surface

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `POST` | `/api/v1/query` | Main RAG pipeline (orchestrator) |
| `POST` | `/api/v1/query/graph` | LangGraph-based RAG pipeline |
| `POST` | `/api/v1/ingest` | Load a file into a collection |
| `GET` | `/api/v1/health` | API and ChromaDB connectivity |
| `GET` | `/api/v1/collections` | List available collections |
| `DELETE` | `/api/v1/collections/{name}` | Delete a collection |
| `GET` | `/api/v1/documents/{collection}` | List documents in a collection |
| `GET` | `/api/v1/debug/trace/{id}` | Fetch an execution trace |

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

```
aars/
├── src/
│   ├── agents/           # 7 agent modules (intent, planner, grader, rewriter, reranker, hallucination, evaluator)
│   ├── api/              # FastAPI endpoints and Pydantic schemas
│   ├── fusion/           # RRF + MMR fusion pipeline
│   ├── generation/       # Answer generator with citations
│   ├── ingestion/        # Document loaders, chunkers, graph builder
│   ├── llm/              # Anthropic SDK wrapper
│   ├── pipeline/         # Orchestrator, LangGraph state/graph, trace recorder
│   ├── retrieval/        # Vector (ChromaDB), keyword (BM25), graph (NetworkX)
│   └── utils/            # Embedding model, cross-encoder model
├── benchmarks/           # Local fixture, metrics, baselines, RAGAS/DeepEval wrappers
├── config/               # Settings, logging, prompt templates
├── tests/                # Unit + integration tests
├── ui/                   # 3-tab Streamlit dashboard
├── examples/             # Usage examples
└── docs/                 # Documentation site source
```

---

## Development

```bash
pip install -e ".[dev]"
python -m compileall src benchmarks tests examples ui config scripts
pytest -q
```

---

## Run the Benchmark

```bash
python benchmarks/runner.py --output benchmarks/results_local.json
```

---

## Citation

If you use AARS in your research, please cite:

```bibtex
@software{aars2025,
  title     = {AARS: Adaptive Agentic Retrieval System},
  author    = {Lekhan},
  year      = {2025},
  url       = {https://github.com/lekhanpro/aars},
  note      = {Self-correcting, intent-aware RAG with built-in benchmarking}
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
