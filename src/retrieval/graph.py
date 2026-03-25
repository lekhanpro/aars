"""Graph retriever backed by a shared NetworkX entity-relationship graph."""

from __future__ import annotations

import re
from typing import Any

import networkx as nx
import structlog

from config.settings import RetrieverSettings
from src.api.schemas.common import Document
from src.retrieval.base import BaseRetriever

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)
_IGNORED_ENTITY_WORDS = {
    "a",
    "an",
    "how",
    "in",
    "the",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
}


class GraphRetriever(BaseRetriever):
    """Retrieve documents by traversing an entity-relationship graph.

    Workflow:

    1. Extract named entities from the query using spaCy NER.
    2. For each entity found in the graph, perform a breadth-first traversal
       up to ``graph_max_hops`` hops.
    3. Collect all document IDs associated with the discovered entity nodes.
    4. Return the corresponding :class:`Document` objects, ranked by the
       number of unique paths that led to them (i.e. more connections =
       higher score).
    """

    # spaCy entity types we care about for knowledge-graph lookups.
    _ENTITY_LABELS: frozenset[str] = frozenset({
        "PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT",
        "WORK_OF_ART", "LAW", "NORP", "FAC",
    })

    def __init__(self, retriever_settings: RetrieverSettings | None = None) -> None:
        self._settings = retriever_settings or RetrieverSettings()
        self._graphs: dict[str, nx.DiGraph] = {}
        self._nlp: Any = None  # spacy.Language — loaded lazily

    # ------------------------------------------------------------------
    # Graph management
    # ------------------------------------------------------------------

    def set_graph(self, graph: nx.DiGraph, collection: str = "default") -> None:
        """Assign the shared entity-relationship graph.

        This is expected to be called at application startup (or whenever
        the graph is rebuilt by the ingestion layer).
        """
        self._graphs[collection] = graph
        logger.info(
            "graph_retriever.set_graph",
            collection=collection,
            nodes=graph.number_of_nodes(),
            edges=graph.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load the spaCy NLP model (small English pipeline)."""
        model_name = "en_core_web_sm"
        try:
            import spacy  # deferred import to keep startup fast when not needed

            self._nlp = spacy.load(model_name)
        except OSError:
            logger.warning(
                "graph_retriever.spacy_model_missing",
                model=model_name,
                hint="Falling back to simple entity extraction",
            )
            self._nlp = None
            return
        except Exception as exc:
            logger.warning(
                "graph_retriever.spacy_unavailable",
                model=model_name,
                error=str(exc),
                hint="Falling back to simple entity extraction",
            )
            self._nlp = None
            return

        logger.info("graph_retriever.initialized", spacy_model=model_name)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        collection: str = "default",
    ) -> list[Document]:
        """Extract entities from *query*, traverse the graph, return documents.

        Parameters
        ----------
        query:
            Natural-language query.
        top_k:
            Maximum number of documents.  Falls back to
            ``RetrieverSettings.graph_top_k``.
        """
        effective_top_k = top_k if top_k is not None else self._settings.graph_top_k
        max_hops: int = self._settings.graph_max_hops

        # 1. Extract entities ------------------------------------------------
        entities = self._extract_entities(query)
        if not entities:
            logger.debug("graph_retriever.no_entities", query=query[:120])
            return []

        logger.debug(
            "graph_retriever.entities",
            query=query[:120],
            entities=entities,
            max_hops=max_hops,
            collection=collection,
        )

        # 2. Traverse the graph -----------------------------------------------
        graph = self._graphs.get(collection)
        if graph is None or graph.number_of_nodes() == 0:
            logger.debug("graph_retriever.empty_graph")
            return []

        # Collect doc_ids reachable from each entity, tracking hit counts.
        doc_hits: dict[str, int] = {}
        doc_map: dict[str, Document] = {}

        for entity in entities:
            normalised = entity.lower()
            if normalised not in graph:
                continue

            visited = self._bfs(graph, normalised, max_hops)

            for node in visited:
                node_data: dict[str, Any] = graph.nodes[node]
                node_doc_ids: list[str] = node_data.get("doc_ids", [])
                node_documents: dict[str, Document] = node_data.get("documents", {})

                for doc_id in node_doc_ids:
                    doc_hits[doc_id] = doc_hits.get(doc_id, 0) + 1
                    # Keep the first Document object we encounter for each id.
                    if doc_id not in doc_map and doc_id in node_documents:
                        doc_map[doc_id] = node_documents[doc_id]

        if not doc_hits:
            logger.debug("graph_retriever.no_docs_found", entities=entities)
            return []

        # 3. Rank by hit count and return top-k --------------------------------
        ranked_ids = sorted(doc_hits, key=lambda did: doc_hits[did], reverse=True)

        results: list[Document] = []
        for doc_id in ranked_ids[:effective_top_k]:
            if doc_id in doc_map:
                src_doc = doc_map[doc_id]
                results.append(
                    Document(
                        id=src_doc.id,
                        content=src_doc.content,
                        metadata=src_doc.metadata,
                        score=float(doc_hits[doc_id]),
                    )
                )
            else:
                # We know the doc_id but do not have its content cached on the
                # node.  Return a stub so the caller can hydrate it later.
                results.append(
                    Document(
                        id=doc_id,
                        content="",
                        metadata={"source": "graph"},
                        score=float(doc_hits[doc_id]),
                    )
                )

        logger.debug("graph_retriever.results", count=len(results))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Return deduplicated entity strings from *text* via spaCy NER."""
        if self._nlp is None:
            return self._simple_entities(text)

        doc = self._nlp(text)
        seen: set[str] = set()
        entities: list[str] = []
        for ent in doc.ents:
            if ent.label_ in self._ENTITY_LABELS:
                normalised = ent.text.strip()
                if normalised and normalised.lower() not in seen:
                    seen.add(normalised.lower())
                    entities.append(normalised)
        return entities

    def _simple_entities(self, text: str) -> list[str]:
        """Fallback entity extraction based on title-cased and all-caps phrases."""
        pattern = re.compile(
            r"\b(?:[A-Z][a-zA-Z-]*|[A-Z]{2,})(?:\s+(?:[A-Z][a-zA-Z-]*|[A-Z]{2,}))*\b"
        )
        seen: set[str] = set()
        entities: list[str] = []
        for match in pattern.finditer(text):
            entity = match.group(0).strip()
            normalized = re.sub(r"^(?:The|A|An)\s+", "", entity, flags=re.IGNORECASE).strip()
            if not normalized or normalized.lower() in _IGNORED_ENTITY_WORDS:
                continue
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                entities.append(normalized)
        return entities

    def _bfs(self, graph: nx.DiGraph, start_node: str, max_hops: int) -> set[str]:
        """Breadth-first traversal up to *max_hops* from *start_node*.

        Returns the set of all visited nodes (including *start_node*).
        """
        visited: set[str] = {start_node}
        frontier: set[str] = {start_node}

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                # Follow both outgoing and incoming edges.
                for neighbour in graph.successors(node):
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
                for neighbour in graph.predecessors(node):
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
            if not next_frontier:
                break
            visited |= next_frontier
            frontier = next_frontier

        return visited
