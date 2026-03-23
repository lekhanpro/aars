"""Build and maintain an entity-relationship graph from ingested documents.

The graph is a :class:`networkx.DiGraph` where:

* **Nodes** represent named entities (lowercased).  Each node carries a
  ``doc_ids`` list and a ``documents`` dict mapping doc-id to
  :class:`Document`.
* **Edges** represent co-occurrence relationships between entities within
  the same sentence.  Each edge stores a ``weight`` (co-occurrence count)
  and the ``doc_ids`` in which the pair co-occurred.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import networkx as nx
import structlog

from src.api.schemas.common import Document

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Entity types we consider relevant for the knowledge graph.
_ENTITY_LABELS: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "EVENT", "PRODUCT",
    "WORK_OF_ART", "LAW", "NORP", "FAC",
})


class GraphBuilder:
    """Incrementally construct an entity-relationship graph with spaCy NER.

    Usage::

        builder = GraphBuilder()
        await builder.initialize()
        builder.add_document(doc)
        graph = builder.graph  # the live NetworkX DiGraph
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nlp: Any = None  # spacy.Language

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load the spaCy NLP model."""
        import spacy

        model_name = "en_core_web_sm"
        try:
            self._nlp = spacy.load(model_name)
        except OSError:
            logger.warning(
                "graph_builder.spacy_model_missing",
                model=model_name,
                hint="Run: python -m spacy download en_core_web_sm",
            )
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Install it with: python -m spacy download {model_name}"
            ) from None

        logger.info("graph_builder.initialized", spacy_model=model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Return the underlying :class:`nx.DiGraph`."""
        return self._graph

    def add_document(self, document: Document) -> None:
        """Extract entities from *document* and update the graph.

        For every sentence in the document, each pair of recognised entities
        is connected with a directed edge (both directions) whose weight
        reflects co-occurrence frequency.
        """
        if self._nlp is None:
            raise RuntimeError(
                "GraphBuilder has not been initialised. Call initialize() first."
            )

        doc = self._nlp(document.content)
        total_entities = 0

        for sent in doc.sents:
            entities = self._unique_entities(sent)
            if not entities:
                continue

            total_entities += len(entities)

            # Register each entity node.
            for entity_text in entities:
                self._ensure_entity_node(entity_text, document)

            # Create edges for every co-occurring pair within the sentence.
            for ent_a, ent_b in combinations(entities, 2):
                self._add_cooccurrence_edge(ent_a, ent_b, document.id)

        logger.debug(
            "graph_builder.add_document",
            doc_id=document.id,
            entities_found=total_entities,
            nodes=self._graph.number_of_nodes(),
            edges=self._graph.number_of_edges(),
        )

    def add_documents(self, documents: list[Document]) -> None:
        """Convenience wrapper — add multiple documents at once."""
        for document in documents:
            self.add_document(document)
        logger.info(
            "graph_builder.add_documents",
            count=len(documents),
            nodes=self._graph.number_of_nodes(),
            edges=self._graph.number_of_edges(),
        )

    def clear(self) -> None:
        """Remove all nodes and edges from the graph."""
        self._graph.clear()
        logger.info("graph_builder.cleared")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _unique_entities(span: Any) -> list[str]:
        """Return deduplicated, normalised entity strings from a spaCy span."""
        seen: set[str] = set()
        entities: list[str] = []
        for ent in span.ents:
            if ent.label_ not in _ENTITY_LABELS:
                continue
            normalised = ent.text.strip().lower()
            if normalised and normalised not in seen:
                seen.add(normalised)
                entities.append(normalised)
        return entities

    def _ensure_entity_node(self, entity: str, document: Document) -> None:
        """Create the entity node if needed and associate *document* with it."""
        if entity not in self._graph:
            self._graph.add_node(
                entity,
                doc_ids=[],
                documents={},
                label=entity,
            )

        node_data: dict[str, Any] = self._graph.nodes[entity]

        if document.id not in node_data["doc_ids"]:
            node_data["doc_ids"].append(document.id)

        # Cache the Document object so downstream consumers (e.g.
        # GraphRetriever) can return it directly without a second lookup.
        node_data["documents"][document.id] = document

    def _add_cooccurrence_edge(
        self, entity_a: str, entity_b: str, doc_id: str
    ) -> None:
        """Add or update a bidirectional co-occurrence edge between two entities."""
        for src, dst in ((entity_a, entity_b), (entity_b, entity_a)):
            if self._graph.has_edge(src, dst):
                edge_data: dict[str, Any] = self._graph.edges[src, dst]
                edge_data["weight"] = edge_data.get("weight", 1) + 1
                if doc_id not in edge_data.get("doc_ids", []):
                    edge_data.setdefault("doc_ids", []).append(doc_id)
            else:
                self._graph.add_edge(
                    src,
                    dst,
                    relationship="co_occurs",
                    weight=1,
                    doc_ids=[doc_id],
                )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, int]:
        """Return basic graph statistics."""
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
            "documents": len({
                doc_id
                for _, data in self._graph.nodes(data=True)
                for doc_id in data.get("doc_ids", [])
            }),
        }
