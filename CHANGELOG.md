# Changelog

All notable changes to AARS are documented in this file.

## [0.1.0] - 2026-03-23

### Added

- **Planner Agent**: LLM-based query classification and dynamic retrieval strategy selection (keyword, vector, graph, hybrid, none)
- **Reflection Agent**: Iterative retrieval sufficiency evaluation with re-retrieval support (up to 3 iterations)
- **Multi-Strategy Retrieval**:
  - Vector retriever (ChromaDB + sentence-transformers)
  - Keyword retriever (BM25 via rank-bm25)
  - Graph retriever (NetworkX + spaCy NER entity traversal)
  - Hybrid mode (parallel vector + keyword)
  - None mode (direct LLM answer without retrieval)
- **Fusion Pipeline**: Reciprocal Rank Fusion (RRF) for cross-retriever score merging + Maximal Marginal Relevance (MMR) for diversity-aware reranking
- **Answer Generator**: Claude-based grounded answer generation with inline citations and confidence scoring
- **Document Ingestion**: PDF (PyMuPDF) and text file loading, recursive character chunking, batch embedding, ChromaDB indexing
- **Entity Graph Builder**: Automatic entity-relationship graph construction from documents using spaCy NER
- **Pipeline Trace**: Full execution trace recording with timing, token usage, and per-step details
- **REST API**: FastAPI endpoints for query, ingest, health check, collection management, and debug trace retrieval
- **Streamlit UI**: Interactive demo with query input, answer display, citation viewer, pipeline trace visualization, and document upload
- **Benchmark Suite**: Dataset loaders (HotpotQA, NQ, TriviaQA, MS MARCO), evaluation metrics (EM, Token F1, Recall@K, MRR, NDCG), 5 baselines, 6 ablation configurations, paired bootstrap significance testing
- **Research Paper**: Complete Springer LNCS paper with methodology, experiments, and results
- **Docker Deployment**: Multi-service Docker Compose (FastAPI + ChromaDB + Streamlit)
- **GitHub Pages**: Project documentation site
- **CI/CD**: GitHub Actions for testing, linting, Docker build verification, and Pages deployment
- **Examples**: Quickstart, multi-hop query, mode comparison, and batch ingestion scripts
