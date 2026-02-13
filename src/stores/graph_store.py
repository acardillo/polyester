"""Graph store using NetworkX for relationship-based retrieval."""

import re
from typing import Optional

import networkx as nx

from src.classifiers import classify_structural_intent
from src.core import Document, Relationship

from .base import MemoryStore

# Fallback when sklearn classifier not available (keywords, edge_type, use_successors)
STRUCTURAL_PATTERNS = [
    (["what functions call", "callers of", "who calls"], "calls", False),
    (["what does", "call", "calls", "invoke", "internally"], "calls", True),
    (["inherit", "inherits", "subclass", "base class", "parent class"], "base_class", True),
    (["depend", "depends", "use", "import"], None, True),
]


class GraphStore(MemoryStore):
    """
    Graph store using NetworkX for relationship traversal.

    Documents become nodes, relationships become edges.
    Excels at queries involving connections between documents.
    """

    def __init__(self):
        """
        Initialize graph store with directed graph.

        Args:
            **kwargs: Additional configuration (unused currently)
        """
        self.graph = nx.DiGraph()
        self.inverted_index = {}

    def index(self, documents: list[Document]) -> None:
        """
        Index documents as graph nodes with relationship edges.

        Args:
            documents: List of Document objects to index
        """
        self.graph.clear()
        self.inverted_index = {}

        for doc in documents:
            self._add_node(doc)
            self._index_document(doc)

        # Add edges for all relationships (second pass ensures all nodes are accounted for)
        for doc in documents:
            self._add_edges(doc.relationships)

    def query(self, query_text: str, n_results: int = 5) -> list[Document]:
        """
        Query graph using multiple strategies.

        For structural queries (e.g. "what does X call?", "what does X inherit from?"),
        prioritizes graph neighbors of the resolved subject. Otherwise: exact ID,
        keyword match, then neighbor expansion.
        """
        results: list[Document] = []
        seen_ids: set[str] = set()

        intent = classify_structural_intent(query_text)
        if intent is None:
            intent = self._detect_structural_intent(query_text)
        is_structural, edge_type, use_successors = intent

        if is_structural:
            seeds = self._resolve_structural_seeds(query_text, limit=5)
            if seeds:
                structural_docs = self._get_structural_neighbors(
                    seeds, edge_type=edge_type, limit=n_results, use_successors=use_successors
                )
                for doc in structural_docs:
                    if doc.id not in seen_ids:
                        results.append(doc)
                        seen_ids.add(doc.id)

        # Exact ID match (if not already in results)
        if len(results) < n_results:
            exact = self._find_by_id(query_text)
            if exact and exact.id not in seen_ids:
                results.append(exact)
                seen_ids.add(exact.id)

        # Keyword search to fill remaining slots
        if len(results) < n_results:
            keyword_matches = self._find_by_keyword(query_text, limit=n_results * 2)
            for doc in keyword_matches:
                if doc.id not in seen_ids:
                    results.append(doc)
                    seen_ids.add(doc.id)
                    if len(results) >= n_results:
                        break

        # Expand to neighbors of top matches
        if len(results) < n_results:
            for match in list(results):
                neighbors = self.get_neighbors(match.id)
                for neighbor in neighbors:
                    if neighbor.id not in seen_ids:
                        results.append(neighbor)
                        seen_ids.add(neighbor.id)
                        if len(results) >= n_results:
                            break
                if len(results) >= n_results:
                    break

        return results[:n_results]

    def clear(self) -> None:
        """Clear all nodes and edges from graph."""
        self.graph.clear()
        self.inverted_index = {}

    def size(self) -> int:
        """Return number of nodes in graph."""
        return self.graph.number_of_nodes()

    def _add_node(self, doc: Document) -> None:
        """Add a single document as graph node."""
        self.graph.add_node(doc.id, content=doc.content, metadata=doc.metadata, document=doc)

    def _add_edges(self, relationships: list[Relationship]) -> None:
        """Add edges for all relationships of a document."""
        for relationship in relationships:
            if relationship.source_id in self.graph and relationship.target_id in self.graph:
                self.graph.add_edge(
                    relationship.source_id,
                    relationship.target_id,
                    type=relationship.relationship_type,
                )

    def _index_document(self, doc: Document) -> None:
        """Add document to inverted index."""
        words = doc.content.lower().split()
        for word in words:
            if word not in self.inverted_index:
                self.inverted_index[word] = set()
            self.inverted_index[word].add(doc.id)

    def _find_by_id(self, node_id: str) -> Optional[Document]:
        """Find document by exact ID."""
        if node_id in self.graph:
            return self.graph.nodes[node_id]["document"]
        return None

    def _find_by_keyword(self, keyword: str, limit: int = 10) -> list[Document]:
        """Find documents using inverted index with scoring."""
        keyword_lower = keyword.lower()
        query_words = keyword_lower.split()

        doc_scores = {}

        for word in query_words:
            if word in self.inverted_index:
                for doc_id in self.inverted_index[word]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1

        sorted_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

        results = []
        for doc_id in sorted_ids[:limit]:
            if doc_id in self.graph:
                results.append(self.graph.nodes[doc_id]["document"])

        return results

    def get_neighbors(self, node_id: str, edge_type: Optional[str] = None) -> list[Document]:
        """
        Get neighboring documents connected by edges.

        Args:
            node_id: ID of source node
            edge_type: Optional filter by relationship type (e.g., "calls", "inherits")

        Returns:
            List of connected Documents
        """
        if node_id not in self.graph:
            return []

        neighbors = []
        for neighbor_id in self.graph.successors(node_id):
            # Filter by edge type if specified
            if edge_type:
                edge_data = self.graph[node_id][neighbor_id]
                if edge_data.get("type") != edge_type:
                    continue

            neighbors.append(self.graph.nodes[neighbor_id]["document"])

        return neighbors

    def get_predecessors(self, node_id: str, edge_type: Optional[str] = None) -> list[Document]:
        """Get nodes that have an edge to this node (e.g. callers of this function)."""
        if node_id not in self.graph:
            return []
        neighbors = []
        for pred_id in self.graph.predecessors(node_id):
            if edge_type:
                edge_data = self.graph[pred_id][node_id]
                if edge_data.get("type") != edge_type:
                    continue
            neighbors.append(self.graph.nodes[pred_id]["document"])
        return neighbors

    def _resolve_structural_seeds(self, query_text: str, limit: int = 5) -> list[str]:
        """Resolve seed node IDs for structural queries: dotted names + top keyword hits."""
        seeds: list[str] = []
        # Try dotted identifiers (e.g. "json.load", "pathlib.Path") as exact IDs
        for prefix in ("stdlib.", ""):
            for match in re.finditer(r"\b([a-zA-Z_]+\.[a-zA-Z_.]+)\b", query_text):
                candidate = (prefix + match.group(1)) if prefix else ("stdlib." + match.group(1))
                if candidate not in seeds and self._find_by_id(candidate):
                    seeds.append(candidate)
                    if len(seeds) >= limit:
                        return seeds
        # Fall back to top keyword matches
        keyword_matches = self._find_by_keyword(query_text, limit=limit)
        for doc in keyword_matches:
            if doc.id not in seeds and doc.id in self.graph:
                seeds.append(doc.id)
                if len(seeds) >= limit:
                    break
        return seeds

    def _detect_structural_intent(self, query_text: str) -> tuple[bool, Optional[str], bool]:
        """Return (is_structural, edge_type_or_none, use_successors)."""
        q = query_text.lower()
        for keywords, edge_type, use_successors in STRUCTURAL_PATTERNS:
            if any(kw in q for kw in keywords):
                return True, edge_type, use_successors
        return False, None, True

    def _get_structural_neighbors(
        self,
        seed_ids: list[str],
        edge_type: Optional[str] = None,
        limit: int = 5,
        use_successors: bool = True,
    ) -> list[Document]:
        """Return neighbor docs from all seeds (successors or predecessors), deduped."""
        seen: set[str] = set()
        out: list[Document] = []
        for nid in seed_ids:
            docs = (
                self.get_neighbors(nid, edge_type=edge_type)
                if use_successors
                else self.get_predecessors(nid, edge_type=edge_type)
            )
            for doc in docs:
                if doc.id not in seen:
                    seen.add(doc.id)
                    out.append(doc)
        return out[:limit]
