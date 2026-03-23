"""Tests for benchmark metrics."""

from __future__ import annotations

from benchmarks.metrics import Metrics


class TestExactMatch:
    def test_exact_match(self):
        assert Metrics.exact_match("the answer", "the answer") == 1.0

    def test_case_insensitive(self):
        assert Metrics.exact_match("The Answer", "the answer") == 1.0

    def test_article_removal(self):
        assert Metrics.exact_match("a cat", "cat") == 1.0
        assert Metrics.exact_match("the dog", "dog") == 1.0

    def test_no_match(self):
        assert Metrics.exact_match("cat", "dog") == 0.0

    def test_whitespace_normalization(self):
        assert Metrics.exact_match("  the  answer  ", "the answer") == 1.0


class TestTokenF1:
    def test_perfect_match(self):
        assert Metrics.token_f1("the cat sat", "the cat sat") == 1.0

    def test_partial_overlap(self):
        score = Metrics.token_f1("the cat sat on mat", "the cat sat")
        assert 0 < score < 1

    def test_no_overlap(self):
        assert Metrics.token_f1("abc", "xyz") == 0.0

    def test_empty_prediction(self):
        assert Metrics.token_f1("", "answer") == 0.0

    def test_empty_truth(self):
        assert Metrics.token_f1("answer", "") == 0.0


class TestRecallAtK:
    def test_perfect_recall(self):
        assert Metrics.recall_at_k(["a", "b", "c"], ["a", "b"], k=3) == 1.0

    def test_partial_recall(self):
        assert Metrics.recall_at_k(["a", "b", "c"], ["a", "d"], k=3) == 0.5

    def test_no_recall(self):
        assert Metrics.recall_at_k(["a", "b"], ["c", "d"], k=2) == 0.0

    def test_k_limit(self):
        assert Metrics.recall_at_k(["a", "b", "c", "d"], ["c", "d"], k=2) == 0.0

    def test_empty_relevant(self):
        assert Metrics.recall_at_k(["a"], [], k=5) == 0.0


class TestMRRAtK:
    def test_first_position(self):
        assert Metrics.mrr_at_k(["a", "b", "c"], ["a"], k=3) == 1.0

    def test_second_position(self):
        assert Metrics.mrr_at_k(["b", "a", "c"], ["a"], k=3) == 0.5

    def test_not_found(self):
        assert Metrics.mrr_at_k(["a", "b"], ["c"], k=2) == 0.0


class TestNDCGAtK:
    def test_perfect_ranking(self):
        score = Metrics.ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=3)
        assert abs(score - 1.0) < 1e-6

    def test_no_relevant(self):
        assert Metrics.ndcg_at_k(["a", "b"], ["c", "d"], k=2) == 0.0

    def test_partial_relevant(self):
        score = Metrics.ndcg_at_k(["a", "b", "c"], ["b", "c"], k=3)
        assert 0 < score < 1
