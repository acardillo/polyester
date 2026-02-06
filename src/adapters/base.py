"""Abstract base class for data adapters."""

from abc import ABC, abstractmethod
from typing import Any
from src.core import Document


class DataAdapter(ABC):
    """
    Abstract base class for converting domain-specific data into Documents.
    
    All adapters must implement load_documents() which transforms raw data
    from their source into a list of Document objects that stores can index.
    
    Example implementations:
    - PythonDocsAdapter: stdlib docs → Documents
    - MusicAdapter: song metadata → Documents
    - WikipediaAdapter: articles → Documents
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize the adapter with configuration.
        
        Args:
            **kwargs: Adapter-specific configuration (paths, URLs, options, etc.)
        """
        self.config = kwargs

    @abstractmethod
    def load_documents(self) -> list[Document]:
        """
        Load raw data from source and convert to Document objects.
        
        Subclasses must implement this method to transform their specific
        data source into Documents. Include relationships if the source
        provides them. Leave embeddings as None (stores generate them).
        
        Returns:
            List of Document objects ready for indexing.
        """
        pass

    def validate_source(self) -> bool:
        """Check if data source is valid/accessible."""
        return True

    def get_metadata(self) -> dict[str, Any]:
        """Return info about the dataset (size, version, etc.)."""
        return {}