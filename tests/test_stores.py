"""Unit tests for memory stores."""

import pytest
from src.stores import MemoryStore, VectorStore
from src.core import Document, Relationship


class TestMemoryStore:
    """Tests for MemoryStore ABC."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that MemoryStore cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MemoryStore()


class TestVectorStore:
    """Tests for VectorStore."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(
                id="doc1",
                content="This is about JSON parsing and serialization",
                metadata={"module": "json", "type": "function"}
            ),
            Document(
                id="doc2",
                content="This is about file operations and I/O",
                metadata={"module": "io", "type": "function"}
            ),
            Document(
                id="doc3",
                content="This is about regular expressions and pattern matching",
                metadata={"module": "re", "type": "function"}
            )
        ]
    
    @pytest.fixture
    def vector_store(self):
        """Create a VectorStore instance."""
        store = VectorStore(collection_name="test_collection")
        yield store
        # Cleanup
        store.clear()
    
    def test_init(self):
        """Test VectorStore initialization."""
        store = VectorStore(collection_name="test")
        assert store.collection_name == "test"
        assert store.size() == 0
        store.clear()
    
    def test_init_with_custom_model(self):
        """Test VectorStore with custom model name."""
        store = VectorStore(
            collection_name="test2",
            model_name="all-MiniLM-L6-v2"
        )
        assert store.collection_name == "test2"
        store.clear()
    
    def test_index_documents(self, vector_store, sample_documents):
        """Test indexing documents."""
        vector_store.index(sample_documents)
        assert vector_store.size() == 3
    
    def test_size_empty_store(self, vector_store):
        """Test size() on empty store."""
        assert vector_store.size() == 0
    
    def test_size_after_indexing(self, vector_store, sample_documents):
        """Test size() after indexing documents."""
        vector_store.index(sample_documents)
        assert vector_store.size() == len(sample_documents)
    
    def test_query_returns_documents(self, vector_store, sample_documents):
        """Test that query returns Document objects."""
        vector_store.index(sample_documents)
        results = vector_store.query("JSON parsing", n_results=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc in results)
    
    def test_query_semantic_similarity(self, vector_store, sample_documents):
        """Test that query finds semantically similar documents."""
        vector_store.index(sample_documents)
        results = vector_store.query("JSON serialization", n_results=1)
        
        # Should find doc1 which is about JSON
        assert len(results) > 0
        assert "json" in results[0].content.lower()
    
    def test_query_top_k(self, vector_store, sample_documents):
        """Test that query respects n_results parameter."""
        vector_store.index(sample_documents)
        
        results_1 = vector_store.query("test query", n_results=1)
        assert len(results_1) == 1
        
        results_2 = vector_store.query("test query", n_results=2)
        assert len(results_2) == 2
    
    def test_query_returns_correct_fields(self, vector_store, sample_documents):
        """Test that queried documents have correct fields."""
        vector_store.index(sample_documents)
        results = vector_store.query("test", n_results=1)
        
        doc = results[0]
        assert doc.id
        assert doc.content
        assert isinstance(doc.metadata, dict)
        assert doc.embedding is None  # Should not return embeddings
        assert doc.relationships == []  # Vector store doesn't preserve relationships
    
    def test_clear_removes_all_documents(self, vector_store, sample_documents):
        """Test that clear() removes all indexed documents."""
        vector_store.index(sample_documents)
        assert vector_store.size() == 3
        
        vector_store.clear()
        assert vector_store.size() == 0
    
    def test_clear_allows_reindexing(self, vector_store, sample_documents):
        """Test that documents can be reindexed after clear()."""
        vector_store.index(sample_documents)
        vector_store.clear()
        vector_store.index(sample_documents[:1])  # Index just one doc
        
        assert vector_store.size() == 1
    
    def test_multiple_index_calls(self, vector_store, sample_documents):
        """Test indexing in multiple batches."""
        vector_store.index(sample_documents[:2])
        assert vector_store.size() == 2
        
        vector_store.index(sample_documents[2:])
        assert vector_store.size() == 3
    
    def test_query_empty_store(self, vector_store):
        """Test querying an empty store returns empty list."""
        results = vector_store.query("test query", n_results=5)
        assert results == []