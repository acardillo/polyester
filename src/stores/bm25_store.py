"""BM25 keyword-based retrieval store."""

from rank_bm25 import BM25Okapi
from typing import Optional
from .base import MemoryStore
from src.core import Document


class BM25Store(MemoryStore):
    """
    BM25 keyword-based retrieval using rank_bm25.
    
    Uses BM25Okapi algorithm for ranking documents based on term frequency,
    inverse document frequency, and document length normalization.
    
    Good for:
    - Exact keyword matching
    - Traditional information retrieval
    - Complementing semantic search
    """
    
    def __init__(self):
        """
        Initialize BM25 store.
        
        Note: BM25 index is built during index() call, not here.
        """
        self.bm25_index = None
        self.documents = []
    
    def index(self, documents: list[Document]) -> None:
        """
        Index documents using BM25.
        
        Args:
            documents: List of Document objects to index
            
        Process:
        1. Store documents for later retrieval
        2. Tokenize document content (lowercase, split on whitespace)
        3. Build BM25Okapi index from tokenized corpus
        """
        self.documents = documents

        tokenized_documents = [
            doc.content.lower().split()
            for doc in documents
        ]

        self.bm25_index = BM25Okapi(tokenized_documents)
    
    def query(self, query_text: str, n_results: int = 5) -> list[Document]:
        """
        Query documents using BM25 ranking.
        
        Args:
            query_text: Search query string
            n_results: Maximum number of results to return
            
        Returns:
            List of Documents, ranked by BM25 score (highest first)
            
        Process:
        1. Tokenize query (same as documents: lowercase, split)
        2. Get BM25 scores for all documents
        3. Find top N documents by score
        4. Return documents in ranked order
        """
        if self.bm25_index is None or not self.documents:
            return []

        tokenized_query = query_text.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)

        top_n_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], # Sort by score and use index as value
            reverse=True
        )[:n_results]

        return [self.documents[i] for i in top_n_indices]
    
    def clear(self) -> None:
        """Clear the BM25 index and all documents."""
        self.bm25_index = None
        self.documents = []
    
    def size(self) -> int:
        """Return number of indexed documents."""
        return len(self.documents)