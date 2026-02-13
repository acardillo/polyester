"""Adapter for Python standard library documentation."""

import json
from pathlib import Path
from typing import Any

from .base import DataAdapter
from src.core import Document, Relationship


class PythonDocsAdapter(DataAdapter):
    """
    Adapter for Python stdlib documentation extracted from inspect/AST.
    
    Expects JSON file with structure:
    {
        "metadata": {...},
        "data": [
            {
                "id": "stdlib.module.name",
                "name": "function_name",
                "module": "module_name",
                "type": "function" or "class",
                "signature": "(...)",
                "description": "docstring",
                "relationships": [...]
            }
        ]
    }
    
    Usage:
        adapter = PythonDocsAdapter(data_path="data/python_docs.json")
        documents = adapter.load_documents()
    """
    
    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize adapter.
        
        Args:
            **kwargs: Must include 'data_path' - path to JSON file
        
        Raises:
            ValueError: If data_path not provided
        """
        super().__init__(**kwargs)

        self.data_path = kwargs.get('data_path')
        if self.data_path is None:
            raise ValueError("data_path is required")

        self.data_path = Path(self.data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"data_path {self.data_path} does not exist")


    def load_documents(self) -> list[Document]:
        """
        Load Python docs from JSON and convert to Documents.
        
        Returns:
            List of Document objects with relationships
        
        Raises:
            FileNotFoundError: If data file doesn't exist
            json.JSONDecodeError: If JSON is malformed
        """
        with self.data_path.open() as f:
            data = json.load(f)

        return [self._convert_to_document(item) for item in data.get('data', [])]
    

    def _convert_to_document(self, doc_data: dict[str, Any]) -> Document:
        """
        Convert a single JSON entry to a Document.
        
        Args:
            doc_data: Dict from JSON with id, name, description, etc.
        
        Returns:
            Document object
        """

        return Document(
            id=doc_data['id'],
            content=self._build_content(doc_data),
            metadata={
                "name": doc_data['name'],
                "module": doc_data['module'],
                "type": doc_data['type'],
                "signature": doc_data['signature']
            },
            embedding=None,
            relationships=self._convert_relationships(doc_data['id'], doc_data['relationships'])
        )
    
    
    def _convert_relationships(
        self, 
        source_id: str, 
        relationships: list[dict[str, str]]
    ) -> list[Relationship]:
        """
        Convert relationship dicts to Relationship objects.
        
        Args:
            source_id: ID of the source document
            relationships: List of {"target": "...", "type": "..."}
        
        Returns:
            List of Relationship objects
        """

        result = []
        for relationship in relationships:
            target_id = relationship['target']
            
            if source_id == target_id:
                continue
            
            result.append(Relationship(
                source_id=source_id,
                target_id=target_id,
                relationship_type=relationship['type'],
                metadata=relationship.get('metadata', {})
            ))

        return result

    def _build_content(self, doc_data: dict[str, Any]) -> str:
        """Build the content string for a document.
        
        Args:
            doc_data: Dict from JSON with name, signature, and description
        
        Returns:
            Content string
        """
        parts = [doc_data['name'], doc_data['signature']]
        if doc_data.get('description'):
            parts.append(f": {doc_data['description']}")

        return " ".join(parts)
