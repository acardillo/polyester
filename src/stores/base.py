from abc import ABC, abstractmethod

from src.core import Document


class MemoryStore(ABC):
    """
    Abstract base class for memory/retrieval stores.

    All store implementations (vector, graph, hybrid) must implement
    the index, query, and clear methods to enable document retrieval.
    """

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """
        Index documents for retrieval.

        Args:
            documents: List of Document objects to index
        """
        pass

    @abstractmethod
    def query(self, query_text: str, n_results: int = 5) -> list[Document]:
        """
        Query store and return the most relevant documents.

        Args:
            query_text: Natural language query string
            n_results: Number of results to return (default: 5)

        Returns:
            List of Document objects, ordered by relevance
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed documents from the store."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return number of indexed documents."""
        pass
