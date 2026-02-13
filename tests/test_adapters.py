"""Unit tests for data adapters."""

import json
from pathlib import Path

import pytest

from src.adapters import DataAdapter, PythonDocsAdapter
from src.core import Document, Relationship


class TestDataAdapter:
    """Tests for DataAdapter ABC."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that DataAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataAdapter()


class TestPythonDocsAdapter:
    """Tests for PythonDocsAdapter."""

    @pytest.fixture
    def sample_data_file(self, tmp_path):
        """Create a temporary JSON file with sample data."""
        data = {
            "metadata": {"python_version": "3.9", "function_count": 2},
            "data": [
                {
                    "id": "stdlib.json.load",
                    "name": "load",
                    "module": "json",
                    "type": "function",
                    "signature": "(fp, *, cls=None)",
                    "description": "Deserialize fp to a Python object",
                    "relationships": [{"target": "loads", "type": "calls"}],
                },
                {
                    "id": "stdlib.json.loads",
                    "name": "loads",
                    "module": "json",
                    "type": "function",
                    "signature": "(s, *, cls=None)",
                    "description": "Deserialize s to a Python object",
                    "relationships": [],
                },
            ],
        }

        file_path = tmp_path / "test_data.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        return file_path

    def test_init_with_valid_path(self, sample_data_file):
        """Test adapter initialization with valid data path."""
        adapter = PythonDocsAdapter(data_path=str(sample_data_file))
        assert adapter.data_path == Path(sample_data_file)

    def test_init_without_data_path_raises_error(self):
        """Test that missing data_path raises ValueError."""
        with pytest.raises(ValueError, match="data_path is required"):
            PythonDocsAdapter()

    def test_init_with_nonexistent_file_raises_error(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            PythonDocsAdapter(data_path="nonexistent.json")

    def test_load_documents(self, sample_data_file):
        """Test loading documents from JSON file."""
        adapter = PythonDocsAdapter(data_path=str(sample_data_file))
        docs = adapter.load_documents()

        assert len(docs) == 2
        assert all(isinstance(doc, Document) for doc in docs)

    def test_document_conversion(self, sample_data_file):
        """Test that JSON entries are correctly converted to Documents."""
        adapter = PythonDocsAdapter(data_path=str(sample_data_file))
        docs = adapter.load_documents()

        doc = docs[0]
        assert doc.id == "stdlib.json.load"
        assert "load" in doc.content
        assert "Deserialize" in doc.content
        assert doc.metadata["module"] == "json"
        assert doc.metadata["type"] == "function"
        assert doc.metadata["signature"] == "(fp, *, cls=None)"
        assert doc.embedding is None

    def test_relationship_conversion(self, sample_data_file):
        """Test that relationships are correctly converted."""
        adapter = PythonDocsAdapter(data_path=str(sample_data_file))
        docs = adapter.load_documents()

        doc = docs[0]
        assert len(doc.relationships) == 1

        rel = doc.relationships[0]
        assert isinstance(rel, Relationship)
        assert rel.source_id == "stdlib.json.load"
        assert rel.target_id == "loads"
        assert rel.relationship_type == "calls"

    def test_self_loop_filtering(self, tmp_path):
        """Test that self-loops are filtered out."""
        data = {
            "metadata": {},
            "data": [
                {
                    "id": "stdlib.test.func",
                    "name": "func",
                    "module": "test",
                    "type": "function",
                    "signature": "()",
                    "description": "Test",
                    "relationships": [
                        {"target": "stdlib.test.func", "type": "calls"}  # Self-loop!
                    ],
                }
            ],
        }

        file_path = tmp_path / "self_loop.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        adapter = PythonDocsAdapter(data_path=str(file_path))
        docs = adapter.load_documents()

        # Self-loop should be filtered out
        assert len(docs[0].relationships) == 0

    def test_empty_description_handling(self, tmp_path):
        """Test that empty descriptions are handled gracefully."""
        data = {
            "metadata": {},
            "data": [
                {
                    "id": "stdlib.test.func",
                    "name": "func",
                    "module": "test",
                    "type": "function",
                    "signature": "()",
                    "description": "",
                    "relationships": [],
                }
            ],
        }

        file_path = tmp_path / "empty_desc.json"
        with open(file_path, "w") as f:
            json.dump(data, f)

        adapter = PythonDocsAdapter(data_path=str(file_path))
        docs = adapter.load_documents()

        assert len(docs) == 1
        assert docs[0].content  # Should still have name and signature
