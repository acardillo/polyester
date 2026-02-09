"""Graph store using NetworkX for relationship-based retrieval."""

import networkx as nx
from typing import Any, Optional
from .base import MemoryStore
from src.core import Document, Relationship


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
            self._add_edges(doc.relationships)
            self._index_document(doc)

    
    def query(self, query_text: str, n_results: int = 5) -> list[Document]:
        """
        Query graph using multiple strategies.
        
        Strategies:
        1. Exact ID match (highest priority)
        2. Content/keyword match
        3. Related nodes (via edges)
        
        Args:
            query_text: Search query
            n_results: Max results to return
            
        Returns:
            List of Document objects, ranked by relevance
        """
        results = []
        seen_ids = set()
        
        # Strategy 1: Exact ID match
        exact = self._find_by_id(query_text)
        if exact:
            results.append(exact)
            seen_ids.add(exact.id)

        # Strategy 2: Keyword search (scored by word matches)
        if len(results) < n_results:
            keyword_matches = self._find_by_keyword(query_text, limit=n_results * 2)
            for doc in keyword_matches:
                if doc.id not in seen_ids:
                    results.append(doc)
                    seen_ids.add(doc.id)
                    if len(results) >= n_results:
                        break

        # Strategy 3: Expand to neighbors of top matches
        if len(results) < n_results:
            for match in results.copy():
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
        self.graph.add_node(
            doc.id,
            content=doc.content,
            metadata=doc.metadata,
            document=doc
        )


    def _add_edges(self, relationships: list[Relationship]) -> None:
        """Add edges for all relationships of a document."""
        for relationship in relationships:
            if relationship.source_id in self.graph and relationship.target_id in self.graph:
                self.graph.add_edge(
                    relationship.source_id,
                    relationship.target_id,
                    type=relationship.relationship_type
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
            return self.graph.nodes[node_id]['document']
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
                results.append(self.graph.nodes[doc_id]['document'])
        
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
                if edge_data.get('type') != edge_type:
                    continue
            
            neighbors.append(self.graph.nodes[neighbor_id]['document'])
        
        return neighbors