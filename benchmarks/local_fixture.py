"""Deterministic local benchmark fixture for AARS.

The fixture is intentionally small and fully checked into the repository so
the benchmark can be rerun without network access, external APIs, or dataset
downloads. It exercises factual, semantic, and multi-hop retrieval behaviors.
"""

from __future__ import annotations

from src.api.schemas import Document

LOCAL_COLLECTION = "local_fixture"

LOCAL_DOCUMENTS: list[Document] = [
    Document(
        id="transformer_history",
        content="The Transformer architecture was introduced in 2017 by Ashish Vaswani and colleagues.",
        metadata={"topic": "transformers"},
    ),
    Document(
        id="bert_architecture",
        content="BERT uses the Transformer encoder to build bidirectional language representations.",
        metadata={"topic": "transformers"},
    ),
    Document(
        id="rag_origin",
        content="Retrieval-Augmented Generation, or RAG, was introduced by Patrick Lewis in 2020.",
        metadata={"topic": "rag"},
    ),
    Document(
        id="bm25",
        content="BM25 is a sparse lexical ranking algorithm that rewards exact term overlap.",
        metadata={"topic": "retrieval"},
    ),
    Document(
        id="chroma",
        content="ChromaDB is a vector database used by AARS to store dense document embeddings for semantic search.",
        metadata={"topic": "aars"},
    ),
    Document(
        id="rrf",
        content="Reciprocal Rank Fusion, or RRF, merges ranked lists from multiple retrievers.",
        metadata={"topic": "fusion"},
    ),
    Document(
        id="mmr",
        content="Maximal Marginal Relevance, or MMR, reranks retrieved passages to reduce redundancy and improve diversity.",
        metadata={"topic": "fusion"},
    ),
    Document(
        id="reflection",
        content="AARS uses a reflection step to decide whether retrieved evidence is sufficient before another retrieval round.",
        metadata={"topic": "aars"},
    ),
    Document(
        id="openai_ceo",
        content="Sam Altman is the CEO of OpenAI.",
        metadata={"topic": "company"},
    ),
    Document(
        id="openai_hq",
        content="OpenAI is headquartered in San Francisco.",
        metadata={"topic": "company"},
    ),
    Document(
        id="graph_retrieval",
        content="Graph retrieval traverses entity relationships to answer multi-hop questions.",
        metadata={"topic": "retrieval"},
    ),
    Document(
        id="dense_retrieval",
        content="Dense vector retrieval is useful when a question is phrased semantically rather than with exact keywords.",
        metadata={"topic": "retrieval"},
    ),
]

LOCAL_SAMPLES: list[dict[str, object]] = [
    {
        "id": "q_bm25",
        "question": "What sparse ranking algorithm rewards exact term overlap?",
        "answers": ["BM25"],
        "relevant_ids": ["bm25"],
    },
    {
        "id": "q_chroma",
        "question": "Which component in AARS stores dense embeddings for semantic search?",
        "answers": ["ChromaDB"],
        "relevant_ids": ["chroma"],
    },
    {
        "id": "q_bert_transformer",
        "question": "Who introduced the architecture used by BERT?",
        "answers": ["Ashish Vaswani and colleagues", "Ashish Vaswani"],
        "relevant_ids": ["transformer_history", "bert_architecture"],
    },
    {
        "id": "q_openai_city",
        "question": "In which city is the company led by Sam Altman headquartered?",
        "answers": ["San Francisco"],
        "relevant_ids": ["openai_ceo", "openai_hq"],
    },
    {
        "id": "q_reflection",
        "question": "What AARS step decides whether more retrieval is needed?",
        "answers": ["reflection step", "reflection"],
        "relevant_ids": ["reflection"],
    },
    {
        "id": "q_rag_year",
        "question": "When was Retrieval-Augmented Generation introduced?",
        "answers": ["2020"],
        "relevant_ids": ["rag_origin"],
    },
    {
        "id": "q_rrf",
        "question": "Which method merges ranked lists before MMR reranking?",
        "answers": ["Reciprocal Rank Fusion", "RRF"],
        "relevant_ids": ["rrf", "mmr"],
    },
    {
        "id": "q_dense",
        "question": "What search method is helpful when wording changes but meaning stays the same?",
        "answers": ["Dense vector retrieval", "vector retrieval"],
        "relevant_ids": ["dense_retrieval"],
    },
    {
        "id": "q_graph",
        "question": "What retrieval style traverses entity relationships for multi-hop questions?",
        "answers": ["Graph retrieval"],
        "relevant_ids": ["graph_retrieval"],
    },
]


def baseline_documents() -> list[dict[str, object]]:
    """Return fixture documents in the dict format expected by baselines."""
    return [
        {
            "id": document.id,
            "content": document.content,
            "metadata": document.metadata,
        }
        for document in LOCAL_DOCUMENTS
    ]
