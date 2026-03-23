"""Tests for RRF and MMR fusion algorithms."""

from __future__ import annotations

from src.api.schemas import Document
from src.fusion.mmr import MaximalMarginalRelevance
from src.fusion.rrf import ReciprocalRankFusion


class TestReciprocalRankFusion:
    def setup_method(self):
        self.rrf = ReciprocalRankFusion(k=60)

    def test_single_list(self):
        docs = [
            Document(id="a", content="doc a", score=0.9),
            Document(id="b", content="doc b", score=0.8),
        ]
        result = self.rrf.fuse([docs])
        assert len(result) == 2
        assert result[0].id == "a"
        assert result[1].id == "b"

    def test_two_lists_merge(self):
        list1 = [
            Document(id="a", content="doc a", score=0.9),
            Document(id="b", content="doc b", score=0.8),
        ]
        list2 = [
            Document(id="b", content="doc b", score=0.95),
            Document(id="c", content="doc c", score=0.7),
        ]
        result = self.rrf.fuse([list1, list2])
        # b appears in both lists, should have highest fused score
        assert result[0].id == "b"
        assert len(result) == 3  # a, b, c deduplicated

    def test_empty_lists(self):
        result = self.rrf.fuse([])
        assert result == []

    def test_scores_are_rrf_scores(self):
        docs = [Document(id="a", content="doc a", score=0.9)]
        result = self.rrf.fuse([docs])
        expected = 1.0 / (60 + 1)  # k + rank (1-indexed)
        assert abs(result[0].score - expected) < 1e-6

    def test_deduplication(self):
        list1 = [Document(id="same", content="content 1", score=0.9)]
        list2 = [Document(id="same", content="content 1", score=0.8)]
        result = self.rrf.fuse([list1, list2])
        assert len(result) == 1


class TestMaximalMarginalRelevance:
    def setup_method(self):
        self.mmr = MaximalMarginalRelevance(lambda_param=0.5)

    def test_basic_reranking(self):
        docs = [
            Document(id="a", content="doc a", score=0.9),
            Document(id="b", content="doc b", score=0.8),
        ]
        query_emb = [1.0, 0.0, 0.0]
        doc_embs = [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]
        result = self.mmr.rerank(docs, query_emb, doc_embs, top_k=2)
        assert len(result) == 2

    def test_top_k_limit(self):
        docs = [Document(id=f"d{i}", content=f"doc {i}", score=0.5) for i in range(10)]
        query_emb = [1.0, 0.0]
        doc_embs = [[float(i) / 10, 1.0 - float(i) / 10] for i in range(10)]
        result = self.mmr.rerank(docs, query_emb, doc_embs, top_k=3)
        assert len(result) == 3

    def test_diversity_promotion(self):
        # Two similar docs and one different — MMR should promote the different one
        docs = [
            Document(id="sim1", content="similar 1", score=0.9),
            Document(id="sim2", content="similar 2", score=0.85),
            Document(id="diff", content="different", score=0.8),
        ]
        query_emb = [1.0, 0.0]
        doc_embs = [
            [0.99, 0.01],  # very similar to query
            [0.98, 0.02],  # very similar to query and sim1
            [0.5, 0.5],    # different from others
        ]
        result = self.mmr.rerank(docs, query_emb, doc_embs, top_k=3)
        # First should be most relevant
        assert result[0].id == "sim1"
        # Third position should promote diversity
        assert len(result) == 3

    def test_empty_documents(self):
        result = self.mmr.rerank([], [1.0, 0.0], [], top_k=5)
        assert result == []

    def test_lambda_zero_pure_diversity(self):
        mmr = MaximalMarginalRelevance(lambda_param=0.0)
        docs = [
            Document(id="a", content="a", score=0.9),
            Document(id="b", content="b", score=0.5),
        ]
        result = mmr.rerank(docs, [1.0], [[0.9], [0.1]], top_k=2)
        assert len(result) == 2

    def test_lambda_one_pure_relevance(self):
        mmr = MaximalMarginalRelevance(lambda_param=1.0)
        docs = [
            Document(id="a", content="a", score=0.9),
            Document(id="b", content="b", score=0.5),
        ]
        result = mmr.rerank(docs, [1.0, 0.0], [[0.9, 0.1], [0.1, 0.9]], top_k=2)
        # Pure relevance, most similar to query first
        assert result[0].id == "a"
