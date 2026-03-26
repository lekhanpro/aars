"""Baseline RAG systems for comparison against AARS.

Each baseline exposes an async ``run`` method with a uniform interface so
the benchmark runner can iterate over them in a single loop.  The
baselines intentionally use simplified logic — they exist to demonstrate
the value added by individual AARS components (planner, reflection,
fusion, etc.).
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Common result type
# ------------------------------------------------------------------

class BaselineResult:
    """Container for a baseline system's output on a single query."""

    __slots__ = ("answer", "documents", "strategy_used", "metadata")

    def __init__(
        self,
        answer: str,
        documents: list[dict[str, Any]],
        strategy_used: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.answer = answer
        self.documents = documents
        self.strategy_used = strategy_used
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "documents": self.documents,
            "strategy_used": self.strategy_used,
            "metadata": self.metadata,
        }


# ------------------------------------------------------------------
# Abstract base
# ------------------------------------------------------------------

class BaseBaseline(ABC):
    """Interface that every baseline must implement."""

    name: str = "base"

    @abstractmethod
    async def run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        llm_client: Any,
    ) -> dict[str, Any]:
        """Execute the baseline on a single query.

        Parameters
        ----------
        query:
            Natural-language question.
        documents:
            Corpus of candidate documents (each a dict with ``id``,
            ``content``, and optional ``metadata``).
        llm_client:
            An LLM client with an async ``generate(prompt)`` method.

        Returns
        -------
        dict
            Keys: ``answer``, ``documents`` (list of selected docs),
            ``strategy_used``.
        """
        ...


# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------

_NON_ALPHA = re.compile(r"[^a-z0-9\s]")


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    return _NON_ALPHA.sub("", text.lower()).split()


def _bm25_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Minimal BM25-ish overlap score (TF only, no IDF)."""
    if not doc_tokens:
        return 0.0
    query_set = set(query_tokens)
    hits = sum(1 for t in doc_tokens if t in query_set)
    return hits / len(doc_tokens)


def _format_context(docs: list[dict[str, Any]], max_docs: int = 5) -> str:
    """Render a list of document dicts into a numbered context block."""
    parts: list[str] = []
    for idx, doc in enumerate(docs[:max_docs], start=1):
        parts.append(f"[{idx}] {doc.get('content', '')}")
    return "\n\n".join(parts)


async def _generate_answer(llm_client: Any, query: str, context: str) -> str:
    """Ask the LLM to answer *query* given *context*."""
    prompt = (
        f"Answer the following question using ONLY the provided context. "
        f"If the context does not contain the answer, say 'I don't know.'\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return await llm_client.generate(prompt)


# ------------------------------------------------------------------
# 1. Naive RAG
# ------------------------------------------------------------------

class NaiveRAG(BaseBaseline):
    """Always uses vector-similarity retrieval with no reflection.

    This is the simplest possible RAG pipeline: embed the query, retrieve
    the top-K documents by cosine similarity, and feed them directly to
    the LLM.
    """

    name = "naive_rag"

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    async def run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        llm_client: Any,
    ) -> dict[str, Any]:
        """Retrieve by simple token overlap (proxy for vector search) and generate."""
        query_tokens = _tokenise(query)

        scored = [
            (doc, _bm25_score(query_tokens, _tokenise(doc.get("content", ""))))
            for doc in documents
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [doc for doc, _ in scored[: self._top_k]]

        context = _format_context(selected)
        answer = await _generate_answer(llm_client, query, context)

        return BaselineResult(
            answer=answer,
            documents=selected,
            strategy_used="vector_only",
        ).to_dict()


# ------------------------------------------------------------------
# 2. Hybrid RAG
# ------------------------------------------------------------------

class HybridRAG(BaseBaseline):
    """Vector + keyword retrieval with fixed Reciprocal Rank Fusion.

    Combines two ranked lists using RRF but does not perform reflection
    or adaptive strategy selection.
    """

    name = "hybrid_rag"

    def __init__(self, top_k: int = 5, rrf_k: int = 60) -> None:
        self._top_k = top_k
        self._rrf_k = rrf_k

    async def run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        llm_client: Any,
    ) -> dict[str, Any]:
        """Fuse keyword and vector rankings with RRF, then generate."""
        query_tokens = _tokenise(query)

        # Simulate two separate rankings.
        # "Vector" ranking — full token overlap.
        vector_scored = sorted(
            documents,
            key=lambda d: _bm25_score(query_tokens, _tokenise(d.get("content", ""))),
            reverse=True,
        )

        # "Keyword" ranking — exact query-token match ratio.
        keyword_scored = sorted(
            documents,
            key=lambda d: sum(
                1 for t in query_tokens if t in d.get("content", "").lower()
            ),
            reverse=True,
        )

        # RRF fusion
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}
        for rank, doc in enumerate(vector_scored, start=1):
            doc_id = doc.get("id", str(rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank)
            doc_map[doc_id] = doc
        for rank, doc in enumerate(keyword_scored, start=1):
            doc_id = doc.get("id", str(rank))
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank)
            doc_map[doc_id] = doc

        fused_ids = sorted(rrf_scores, key=lambda did: rrf_scores[did], reverse=True)
        selected = [doc_map[did] for did in fused_ids[: self._top_k]]

        context = _format_context(selected)
        answer = await _generate_answer(llm_client, query, context)

        return BaselineResult(
            answer=answer,
            documents=selected,
            strategy_used="hybrid_rrf",
        ).to_dict()


# ------------------------------------------------------------------
# 3. FLARE-style baseline
# ------------------------------------------------------------------

class FLAREBaseline(BaseBaseline):
    """Confidence-based active retrieval inspired by FLARE.

    Generates a preliminary answer, estimates confidence via a simple
    heuristic (answer length and hedging-word presence), and retrieves
    additional context if confidence is below a threshold.
    """

    name = "flare"

    _HEDGING_WORDS = frozenset({
        "maybe", "perhaps", "possibly", "might", "could", "uncertain",
        "unclear", "not sure", "i don't know", "unknown",
    })

    def __init__(
        self,
        top_k: int = 5,
        confidence_threshold: float = 0.5,
        max_iterations: int = 2,
    ) -> None:
        self._top_k = top_k
        self._confidence_threshold = confidence_threshold
        self._max_iterations = max_iterations

    async def run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        llm_client: Any,
    ) -> dict[str, Any]:
        """Generate, assess confidence, re-retrieve if needed."""
        query_tokens = _tokenise(query)

        # Initial retrieval
        scored = sorted(
            documents,
            key=lambda d: _bm25_score(query_tokens, _tokenise(d.get("content", ""))),
            reverse=True,
        )
        selected = [doc for doc in scored[: self._top_k]]
        iterations_used = 1

        for _ in range(self._max_iterations):
            context = _format_context(selected)
            answer = await _generate_answer(llm_client, query, context)

            # Heuristic confidence estimation.
            confidence = self._estimate_confidence(answer)
            if confidence >= self._confidence_threshold:
                break

            # Low confidence: expand retrieval window.
            iterations_used += 1
            expanded_k = min(self._top_k * iterations_used, len(scored))
            selected = [doc for doc in scored[:expanded_k]]
        else:
            # Final generation with expanded context.
            context = _format_context(selected)
            answer = await _generate_answer(llm_client, query, context)

        return BaselineResult(
            answer=answer,
            documents=selected,
            strategy_used="flare_active",
            metadata={"iterations": iterations_used},
        ).to_dict()

    def _estimate_confidence(self, answer: str) -> float:
        """Heuristic confidence based on hedging words and answer length."""
        lower = answer.lower()
        hedge_count = sum(1 for w in self._HEDGING_WORDS if w in lower)

        # Short answers or answers with many hedges are low confidence.
        length_factor = min(len(answer.split()) / 20.0, 1.0)
        hedge_penalty = 0.2 * hedge_count

        return max(0.0, min(1.0, length_factor - hedge_penalty))


# ------------------------------------------------------------------
# 4. Self-RAG baseline
# ------------------------------------------------------------------

class SelfRAGBaseline(BaseBaseline):
    """Self-reflective retrieval with explicit relevance assessment.

    After each retrieval round the LLM is asked to judge whether the
    documents are relevant and sufficient.  If not, a refined query is
    generated and retrieval is repeated.
    """

    name = "self_rag"

    def __init__(self, top_k: int = 5, max_iterations: int = 2) -> None:
        self._top_k = top_k
        self._max_iterations = max_iterations

    async def run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        llm_client: Any,
    ) -> dict[str, Any]:
        """Retrieve, assess, optionally refine, then generate."""
        current_query = query
        selected: list[dict[str, Any]] = []
        iterations_used = 0

        for iteration in range(self._max_iterations):
            iterations_used = iteration + 1

            # Retrieval
            query_tokens = _tokenise(current_query)
            scored = sorted(
                documents,
                key=lambda d: _bm25_score(query_tokens, _tokenise(d.get("content", ""))),
                reverse=True,
            )
            selected = [doc for doc in scored[: self._top_k]]

            # Self-assessment via LLM
            assessment = await self._assess_relevance(
                llm_client, current_query, selected
            )

            if assessment.get("sufficient", False):
                break

            # Refine the query based on assessment feedback.
            refined = assessment.get("refined_query", "")
            if refined and refined != current_query:
                current_query = refined
            else:
                break

        context = _format_context(selected)
        answer = await _generate_answer(llm_client, query, context)

        return BaselineResult(
            answer=answer,
            documents=selected,
            strategy_used="self_rag",
            metadata={"iterations": iterations_used},
        ).to_dict()

    @staticmethod
    async def _assess_relevance(
        llm_client: Any,
        query: str,
        docs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Ask the LLM whether retrieved documents are sufficient."""
        context = _format_context(docs)
        prompt = (
            f"You are evaluating whether the retrieved documents can answer "
            f"the question below.\n\n"
            f"Question: {query}\n\n"
            f"Documents:\n{context}\n\n"
            f"Are these documents sufficient to answer the question?\n"
            f"If YES, respond with: SUFFICIENT\n"
            f"If NO, respond with: INSUFFICIENT | <refined search query>\n"
        )
        response = await llm_client.generate(prompt)
        response_upper = response.strip().upper()

        if "SUFFICIENT" in response_upper and "INSUFFICIENT" not in response_upper:
            return {"sufficient": True}

        # Extract refined query after the pipe.
        parts = response.split("|", maxsplit=1)
        refined = parts[1].strip() if len(parts) > 1 else ""
        return {"sufficient": False, "refined_query": refined}


# ------------------------------------------------------------------
# 5. Standard Routing
# ------------------------------------------------------------------

class StandardRouting(BaseBaseline):
    """Rule-based query router using keyword-matching heuristics.

    Routes queries to different "strategies" based on surface-level
    keyword cues, without an LLM-based planner.
    """

    name = "standard_routing"

    # Mapping from keyword patterns to strategies.
    _ROUTING_RULES: list[tuple[list[str], str]] = [
        (["who", "person", "founder", "author", "ceo"], "entity_lookup"),
        (["compare", "difference", "versus", "vs"], "multi_doc"),
        (["when", "date", "year", "time"], "keyword"),
        (["how many", "count", "number", "total"], "keyword"),
        (["why", "explain", "reason", "cause"], "vector"),
        (["what is", "define", "meaning"], "vector"),
    ]

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    async def run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        llm_client: Any,
    ) -> dict[str, Any]:
        """Route the query, retrieve, and generate."""
        strategy = self._route(query)
        query_tokens = _tokenise(query)

        # All strategies ultimately use the same retrieval (token overlap)
        # but the naming reflects which heuristic was triggered.
        scored = sorted(
            documents,
            key=lambda d: _bm25_score(query_tokens, _tokenise(d.get("content", ""))),
            reverse=True,
        )

        # For "multi_doc" strategy, retrieve more documents.
        effective_k = self._top_k * 2 if strategy == "multi_doc" else self._top_k
        selected = [doc for doc in scored[:effective_k]]

        context = _format_context(selected, max_docs=effective_k)
        answer = await _generate_answer(llm_client, query, context)

        return BaselineResult(
            answer=answer,
            documents=selected,
            strategy_used=f"routing_{strategy}",
        ).to_dict()

    def _route(self, query: str) -> str:
        """Determine the routing strategy from query keywords."""
        lower = query.lower()
        for keywords, strategy in self._ROUTING_RULES:
            if any(kw in lower for kw in keywords):
                return strategy
        return "vector"  # default fallback


# ------------------------------------------------------------------
# 6. TreeDex baseline
# ------------------------------------------------------------------

class TreeDexBaseline(BaseBaseline):
    """Tree-based vectorless retrieval inspired by TreeDex.

    Builds a hierarchical document index using heading detection and
    retrieves by matching query terms against the tree structure,
    prioritizing deeper (more specific) matches.
    """

    name = "treedex"

    def __init__(self, top_k: int = 5) -> None:
        self._top_k = top_k

    async def run(
        self,
        query: str,
        documents: list[dict[str, Any]],
        llm_client: Any,
    ) -> dict[str, Any]:
        """Build tree index, traverse for matches, generate answer."""
        query_tokens = _tokenise(query)

        # Build a simple hierarchical index: detect "sections" by splitting
        # on sentence boundaries and scoring by structural depth.
        scored_segments: list[tuple[dict[str, Any], float]] = []

        for doc in documents:
            content = doc.get("content", "")
            sentences = [s.strip() for s in content.split(". ") if s.strip()]

            # Score each document by hierarchical term matching:
            # - Terms in first sentence (title/heading) get 2x weight
            # - Terms in body get 1x weight
            # - Longer documents with matching terms get a depth bonus
            heading_tokens = set(_tokenise(sentences[0])) if sentences else set()
            body_tokens = set(_tokenise(content))

            heading_overlap = len(set(query_tokens) & heading_tokens)
            body_overlap = len(set(query_tokens) & body_tokens)

            # Hierarchical scoring: heading matches are worth more
            tree_score = (heading_overlap * 2.0 + body_overlap * 1.0)

            # Depth bonus: longer, more specific documents rank higher
            # when they have matches (simulates tree depth traversal)
            if tree_score > 0:
                depth_factor = min(len(sentences) / 5.0, 2.0)
                tree_score *= (1.0 + 0.1 * depth_factor)

            if tree_score > 0:
                scored_segments.append((doc, tree_score))

        # Sort by tree score descending
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        selected = [doc for doc, _ in scored_segments[:self._top_k]]

        # If no matches, fall back to first documents
        if not selected:
            selected = documents[:self._top_k]

        context = _format_context(selected)
        answer = await _generate_answer(llm_client, query, context)

        return BaselineResult(
            answer=answer,
            documents=selected,
            strategy_used="tree_index",
        ).to_dict()


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

ALL_BASELINES: list[BaseBaseline] = [
    NaiveRAG(),
    HybridRAG(),
    FLAREBaseline(),
    SelfRAGBaseline(),
    StandardRouting(),
    TreeDexBaseline(),
]


def get_baseline_by_name(name: str) -> BaseBaseline | None:
    """Look up a baseline by its ``name`` attribute."""
    for baseline in ALL_BASELINES:
        if baseline.name == name:
            return baseline
    return None
