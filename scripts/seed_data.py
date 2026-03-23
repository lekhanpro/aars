"""Seed ChromaDB with sample documents for testing."""

from __future__ import annotations

import argparse
import sys

import chromadb

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_transformer_1",
        "content": (
            "The Transformer architecture was introduced in 2017 by Vaswani et al. "
            "in the paper 'Attention Is All You Need'. It replaced recurrent layers "
            "with self-attention mechanisms, enabling parallel processing of sequences."
        ),
        "metadata": {"source": "transformers_overview.txt", "topic": "transformers"},
    },
    {
        "id": "doc_attention_1",
        "content": (
            "Self-attention, or intra-attention, computes representations of a sequence "
            "by relating different positions of the same sequence. Multi-head attention "
            "runs multiple attention functions in parallel, allowing the model to attend "
            "to information from different representation subspaces."
        ),
        "metadata": {"source": "attention_mechanisms.txt", "topic": "attention"},
    },
    {
        "id": "doc_bert_1",
        "content": (
            "BERT (Bidirectional Encoder Representations from Transformers) was introduced "
            "by Devlin et al. in 2019. It uses masked language modeling and next sentence "
            "prediction for pre-training, enabling bidirectional context understanding."
        ),
        "metadata": {"source": "bert_overview.txt", "topic": "bert"},
    },
    {
        "id": "doc_rag_1",
        "content": (
            "Retrieval-Augmented Generation (RAG) was proposed by Lewis et al. in 2020. "
            "It combines a pre-trained retriever (DPR) with a pre-trained generator (BART) "
            "to incorporate external knowledge into language model outputs."
        ),
        "metadata": {"source": "rag_paper.txt", "topic": "rag"},
    },
    {
        "id": "doc_rag_2",
        "content": (
            "RAG addresses the limitation of parametric-only models by providing access "
            "to external knowledge bases at inference time. This enables factual grounding "
            "and reduces hallucination in generated responses."
        ),
        "metadata": {"source": "rag_paper.txt", "topic": "rag"},
    },
    {
        "id": "doc_selfrag_1",
        "content": (
            "Self-RAG by Asai et al. (2023) introduces reflection tokens that allow the "
            "model to decide when to retrieve and to assess the quality of retrieved passages. "
            "This self-reflective approach improves both retrieval precision and generation quality."
        ),
        "metadata": {"source": "selfrag_paper.txt", "topic": "self-rag"},
    },
    {
        "id": "doc_flare_1",
        "content": (
            "FLARE (Forward-Looking Active REtrieval) by Jiang et al. (2023) uses the model's "
            "generation confidence to trigger retrieval. When the model is uncertain about its "
            "next tokens, it actively retrieves relevant information to guide generation."
        ),
        "metadata": {"source": "flare_paper.txt", "topic": "flare"},
    },
    {
        "id": "doc_bm25_1",
        "content": (
            "BM25 (Best Match 25) is a probabilistic retrieval function based on term frequency "
            "and inverse document frequency. It extends TF-IDF with document length normalization "
            "and saturation, making it one of the most effective keyword retrieval algorithms."
        ),
        "metadata": {"source": "retrieval_methods.txt", "topic": "bm25"},
    },
    {
        "id": "doc_embeddings_1",
        "content": (
            "Dense passage retrieval uses learned vector representations (embeddings) to capture "
            "semantic similarity between queries and documents. Models like sentence-transformers "
            "map text to dense vectors where cosine similarity reflects semantic relatedness."
        ),
        "metadata": {"source": "retrieval_methods.txt", "topic": "embeddings"},
    },
    {
        "id": "doc_fusion_1",
        "content": (
            "Reciprocal Rank Fusion (RRF) combines rankings from multiple retrieval systems "
            "using the formula score(d) = sum(1/(k + rank_r(d))). It is simple yet effective "
            "for merging results from heterogeneous retrieval methods."
        ),
        "metadata": {"source": "fusion_methods.txt", "topic": "fusion"},
    },
]


def seed(host: str = "localhost", port: int = 8001, collection: str = "default") -> None:
    client = chromadb.HttpClient(host=host, port=port)
    col = client.get_or_create_collection(name=collection)

    ids = [d["id"] for d in SAMPLE_DOCUMENTS]
    contents = [d["content"] for d in SAMPLE_DOCUMENTS]
    metadatas = [d["metadata"] for d in SAMPLE_DOCUMENTS]

    col.add(ids=ids, documents=contents, metadatas=metadatas)
    print(f"Seeded {len(SAMPLE_DOCUMENTS)} documents into collection '{collection}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed ChromaDB with sample data")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--collection", default="default")
    args = parser.parse_args()
    seed(args.host, args.port, args.collection)
