"""Hybrid store combining vector, graph, and BM25 retrieval."""

from collections import defaultdict

from src.core import Document

from .base import MemoryStore
from .bm25_store import BM25Store
from .graph_store import GraphStore
from .vector_store import VectorStore


class HybridStore(MemoryStore):
    """
    Hybrid retrieval combining semantic, structural, and keyword search.

    Combines three retrieval strategies:
    1. VectorStore: Semantic similarity via embeddings
    2. GraphStore: Relationship traversal via graph structure
    3. BM25Store: Keyword matching via BM25 algorithm
    """

    def __init__(
        self,
        collection_name: str = "hybrid_collection",
        vector_weight: float = 0.4,
        graph_weight: float = 0.3,
        bm25_weight: float = 0.3,
    ):
        """
        Initialize hybrid store with three sub-stores.

        Args:
            collection_name: Name for vector store collection
            vector_weight: Weight for vector scores (0-1)
            graph_weight: Weight for graph scores (0-1)
            bm25_weight: Weight for BM25 scores (0-1)

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        # Validate weights
        total_weight = vector_weight + graph_weight + bm25_weight
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight:.3f}")

        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.bm25_weight = bm25_weight

        self.vector_store = VectorStore(collection_name=collection_name)
        self.graph_store = GraphStore()
        self.bm25_store = BM25Store()

    def index(self, documents: list[Document]) -> None:
        """
        Index documents in all three stores.

        Args:
            documents: List of Document objects to index
        """
        self.vector_store.index(documents)
        self.graph_store.index(documents)
        self.bm25_store.index(documents)

    def query(self, query_text: str, n_results: int = 5) -> list[Document]:
        """
        Query all three stores in parallel and merge with weighted scoring.

        Process:
        1. Query each store (vector, graph, BM25)
        2. Merge rankings with weighted Reciprocal Rank Fusion.
        3. Return top N.
        """
        vector_results = self.vector_store.query(query_text, n_results=n_results * 2)
        graph_results = self.graph_store.query(query_text, n_results=n_results * 2)
        bm25_results = self.bm25_store.query(query_text, n_results=n_results * 2)

        ranked_docs = self._rank_with_weighted_rrf(
            vector_results=vector_results,
            graph_results=graph_results,
            bm25_results=bm25_results,
        )

        return ranked_docs[:n_results]

    def _rank_with_weighted_rrf(
        self,
        vector_results: list[Document],
        graph_results: list[Document],
        bm25_results: list[Document],
        k: int = 60,
    ) -> list[Document]:
        """
        Rank results from multiple stores using Weighted Reciprocal Rank Fusion.

        Args:
            vector_results: List of vector store results
            graph_results: List of graph store results
            bm25_results: List of BM25 store results
            k: RRF smoothing constant (default 60)

        Returns:
            Ranked list of fused Documents
        """

        doc_lookup = {}
        scores = defaultdict(float)

        # 1. For each store's ranked list, add weight * 1/(k + rank) per document
        for docs, weight in [
            (vector_results, self.vector_weight),
            (graph_results, self.graph_weight),
            (bm25_results, self.bm25_weight),
        ]:
            for rank, doc in enumerate(docs, start=1):
                scores[doc.id] += weight * (1.0 / (k + rank))
                doc_lookup[doc.id] = doc

        # 2. Sort by score
        sorted_ids = sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)
        return [doc_lookup[doc_id] for doc_id in sorted_ids]

    def clear(self) -> None:
        """Clear all three stores."""
        self.vector_store.clear()
        self.graph_store.clear()
        self.bm25_store.clear()

    def size(self) -> int:
        """Return number of documents (should be same across all stores)."""
        return self.vector_store.size()
