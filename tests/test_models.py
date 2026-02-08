"""Unit tests for core data models."""

import pytest
from pydantic import ValidationError
from src.core import Document, Relationship


class TestRelationship:
    """Tests for the Relationship model."""
    
    def test_create_basic_relationship(self):
        """Test creating a relationship with required fields."""
        rel = Relationship(
            source_id="doc1",
            target_id="doc2",
            relationship_type="calls"
        )
        assert rel.source_id == "doc1"
        assert rel.target_id == "doc2"
        assert rel.relationship_type == "calls"
        assert rel.metadata == {}
    
    def test_relationship_with_metadata(self):
        """Test relationship with optional metadata."""
        rel = Relationship(
            source_id="doc1",
            target_id="doc2",
            relationship_type="calls",
            metadata={"param": "value"}
        )
        assert rel.metadata == {"param": "value"}
    
    def test_relationship_default_metadata(self):
        """Test that metadata defaults to empty dict."""
        rel = Relationship(
            source_id="doc1",
            target_id="doc2",
            relationship_type="calls"
        )
        assert rel.metadata == {}
        assert isinstance(rel.metadata, dict)
    
    def test_self_loop_raises_error(self):
        """Test that source_id == target_id raises ValueError."""
        with pytest.raises(ValueError, match="self-loop detected"):
            Relationship(
                source_id="same",
                target_id="same",
                relationship_type="calls"
            )
    
    def test_mutable_default_metadata(self):
        """Test that different relationships don't share metadata dict."""
        rel1 = Relationship(source_id="doc1", target_id="doc2", relationship_type="calls")
        rel2 = Relationship(source_id="doc3", target_id="doc4", relationship_type="imports")
        
        rel1.metadata["key"] = "value"
        assert "key" not in rel2.metadata


class TestDocument:
    """Tests for the Document model."""
    
    def test_create_minimal_document(self):
        """Test creating document with only required fields."""
        doc = Document(id="test", content="test content")
        assert doc.id == "test"
        assert doc.content == "test content"
        assert doc.metadata == {}
        assert doc.embedding is None
        assert doc.relationships == []
    
    def test_create_document_with_all_fields(self):
        """Test creating document with all optional fields."""
        rel = Relationship(source_id="test", target_id="other", relationship_type="calls")
        doc = Document(
            id="test",
            content="content",
            metadata={"key": "value"},
            embedding=[1.0, 2.0, 3.0],
            relationships=[rel]
        )
        assert doc.metadata == {"key": "value"}
        assert doc.embedding == [1.0, 2.0, 3.0]
        assert len(doc.relationships) == 1
        assert doc.relationships[0] == rel
    
    def test_document_with_valid_embedding(self):
        """Test that valid embeddings are accepted."""
        doc = Document(
            id="test",
            content="content",
            embedding=[1.0, 2.0, 3.0]
        )
        assert doc.embedding == [1.0, 2.0, 3.0]
    
    def test_empty_embedding_raises_error(self):
        """Test that empty embedding list raises ValueError."""
        with pytest.raises(ValueError, match="Embedding cannot be an empty list"):
            Document(id="test", content="content", embedding=[])
    
    def test_embedding_with_non_float_raises_error(self):
        """Test that non-float elements in embedding raise ValidationError."""
        with pytest.raises(ValidationError):
            Document(id="test", content="content", embedding=[1.0, "bad", 3.0])
    
    def test_none_embedding_is_valid(self):
        """Test that None embedding is allowed."""
        doc = Document(id="test", content="content", embedding=None)
        assert doc.embedding is None
    
    def test_mutable_default_relationships(self):
        """Test that different documents don't share relationships list."""
        doc1 = Document(id="1", content="content1")
        doc2 = Document(id="2", content="content2")
        
        rel = Relationship(source_id="1", target_id="other", relationship_type="calls")
        doc1.relationships.append(rel)
        
        assert len(doc1.relationships) == 1
        assert len(doc2.relationships) == 0
    
    def test_mutable_default_metadata(self):
        """Test that different documents don't share metadata dict."""
        doc1 = Document(id="1", content="content1")
        doc2 = Document(id="2", content="content2")
        
        doc1.metadata["key"] = "value"
        assert "key" not in doc2.metadata
    
    def test_validate_assignment_embedding(self):
        """Test that assignment validation catches invalid embeddings."""
        doc = Document(id="test", content="content")
        
        with pytest.raises(ValueError):
            doc.embedding = []  # Empty list should fail
    
    def test_serialization_roundtrip(self):
        """Test that document can be serialized and deserialized."""
        rel = Relationship(source_id="test", target_id="other", relationship_type="calls")
        original = Document(
            id="test",
            content="content",
            metadata={"key": "value"},
            embedding=[1.0, 2.0],
            relationships=[rel]
        )
        
        # Serialize
        dict_data = original.model_dump()
        
        # Deserialize
        recreated = Document.model_validate(dict_data)
        
        assert original.id == recreated.id
        assert original.content == recreated.content
        assert original.metadata == recreated.metadata
        assert original.embedding == recreated.embedding
        assert len(original.relationships) == len(recreated.relationships)


class TestDocumentRelationshipIntegration:
    """Tests for Document and Relationship working together."""
    
    def test_document_with_relationships(self):
        """Test creating document with relationship list."""
        rel = Relationship(source_id="test", target_id="other", relationship_type="calls")
        doc = Document(
            id="test",
            content="content",
            relationships=[rel]
        )
        assert len(doc.relationships) == 1
        assert doc.relationships[0] == rel
    
    def test_add_relationship_to_document(self):
        """Test adding relationship to document after creation."""
        doc = Document(id="test", content="content")
        rel = Relationship(source_id="test", target_id="other", relationship_type="calls")
        
        doc.relationships.append(rel)
        assert len(doc.relationships) == 1
        assert doc.relationships[0] == rel