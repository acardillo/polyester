"""Unit tests for memory stores."""

import pytest
from src.stores import MemoryStore, VectorStore, GraphStore
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


class TestGraphStore:
    """Tests for GraphStore."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents with relationships for testing."""
        return [
            Document(
                id="json.load",
                content="Load JSON data from a file",
                metadata={"module": "json", "type": "function"},
                relationships=[
                    Relationship(
                        source_id="json.load",
                        target_id="json.loads",
                        relationship_type="calls"
                    )
                ]
            ),
            Document(
                id="json.loads",
                content="Parse JSON string into Python object",
                metadata={"module": "json", "type": "function"},
                relationships=[]
            ),
            Document(
                id="json.JSONDecoder",
                content="JSON decoder class for custom decoding",
                metadata={"module": "json", "type": "class"},
                relationships=[
                    Relationship(
                        source_id="json.JSONDecoder",
                        target_id="builtins.object",
                        relationship_type="base_class"
                    )
                ]
            )
        ]
    
    @pytest.fixture
    def graph_store(self):
        """Create a GraphStore instance."""
        from src.stores import GraphStore
        return GraphStore()
    
    # === INITIALIZATION TESTS ===
    
    def test_init(self, graph_store):
        """Test GraphStore initialization."""
        assert graph_store.size() == 0
        assert hasattr(graph_store, 'graph')
        assert hasattr(graph_store, 'inverted_index')
    
    # === INDEXING TESTS ===
    
    def test_index_documents(self, graph_store, sample_documents):
        """Test indexing documents creates nodes."""
        graph_store.index(sample_documents)
        assert graph_store.size() == 3
    
    def test_index_creates_edges(self, graph_store, sample_documents):
        """Test that relationships create edges in graph."""
        graph_store.index(sample_documents)
        
        # json.load should have edge to json.loads
        neighbors = graph_store.get_neighbors("json.load")
        assert len(neighbors) == 1
        assert neighbors[0].id == "json.loads"
    
    def test_index_filters_ghost_edges(self, graph_store):
        """Test that edges to non-existent nodes are not created."""
        docs = [
            Document(
                id="doc1",
                content="Test document",
                metadata={},
                relationships=[
                    Relationship(
                        source_id="doc1",
                        target_id="ghost_node",  # Doesn't exist!
                        relationship_type="calls"
                    )
                ]
            )
        ]
        
        graph_store.index(docs)
        
        # Should have 1 node but 0 edges (ghost edge filtered)
        assert graph_store.size() == 1
        neighbors = graph_store.get_neighbors("doc1")
        assert len(neighbors) == 0
    
    def test_reindex_replaces_data(self, graph_store, sample_documents):
        """Test that re-indexing clears old data."""
        graph_store.index(sample_documents)
        assert graph_store.size() == 3
        
        # Re-index with just one document
        graph_store.index([sample_documents[0]])
        assert graph_store.size() == 1
    
    # === QUERY TESTS ===
    
    def test_query_exact_id_match(self, graph_store, sample_documents):
        """Test querying by exact document ID."""
        graph_store.index(sample_documents)
        
        results = graph_store.query("json.loads", n_results=5)
        
        assert len(results) >= 1
        assert results[0].id == "json.loads"  # Exact match first
    
    def test_query_keyword_match(self, graph_store, sample_documents):
        """Test querying by content keywords."""
        graph_store.index(sample_documents)
        
        results = graph_store.query("parse JSON", n_results=5)
        
        # Should find json.loads which mentions "parse" and "JSON"
        assert len(results) > 0
        ids = [doc.id for doc in results]
        assert "json.loads" in ids
    
    def test_query_neighbor_expansion(self, graph_store, sample_documents):
        """Test that query expands to neighbors when needed."""
        graph_store.index(sample_documents)
        
        # Query for "load" should find json.load, then expand to json.loads
        results = graph_store.query("load file", n_results=5)
        
        ids = [doc.id for doc in results]
        assert "json.load" in ids
        # May also include json.loads via neighbor expansion
    
    def test_query_respects_n_results(self, graph_store, sample_documents):
        """Test that n_results parameter limits results."""
        graph_store.index(sample_documents)
        
        results_1 = graph_store.query("JSON", n_results=1)
        assert len(results_1) == 1
        
        results_2 = graph_store.query("JSON", n_results=2)
        assert len(results_2) == 2
    
    def test_query_returns_document_objects(self, graph_store, sample_documents):
        """Test that query returns proper Document objects."""
        graph_store.index(sample_documents)
        
        results = graph_store.query("JSON", n_results=2)
        
        for doc in results:
            assert isinstance(doc, Document)
            assert doc.id
            assert doc.content
            assert isinstance(doc.metadata, dict)
    
    def test_query_no_duplicates(self, graph_store, sample_documents):
        """Test that query doesn't return duplicate documents."""
        graph_store.index(sample_documents)
        
        results = graph_store.query("JSON", n_results=5)
        
        ids = [doc.id for doc in results]
        assert len(ids) == len(set(ids))  # No duplicates
    
    # === GET_NEIGHBORS TESTS ===
    
    def test_get_neighbors_returns_connected_docs(self, graph_store, sample_documents):
        """Test get_neighbors returns documents connected by edges."""
        graph_store.index(sample_documents)
        
        neighbors = graph_store.get_neighbors("json.load")
        
        assert len(neighbors) == 1
        assert neighbors[0].id == "json.loads"
    
    def test_get_neighbors_with_edge_type_filter(self, graph_store):
        """Test get_neighbors filters by relationship type."""
        docs = [
            Document(
                id="class1",
                content="A class",
                metadata={},
                relationships=[
                    Relationship(source_id="class1", target_id="class2", relationship_type="calls"),
                    Relationship(source_id="class1", target_id="class3", relationship_type="inherits")
                ]
            ),
            Document(id="class2", content="Called", metadata={}, relationships=[]),
            Document(id="class3", content="Parent", metadata={}, relationships=[])
        ]
        
        graph_store.index(docs)
        
        # Get only "calls" relationships
        calls_neighbors = graph_store.get_neighbors("class1", edge_type="calls")
        assert len(calls_neighbors) == 1
        assert calls_neighbors[0].id == "class2"
        
        # Get only "inherits" relationships
        inherits_neighbors = graph_store.get_neighbors("class1", edge_type="inherits")
        assert len(inherits_neighbors) == 1
        assert inherits_neighbors[0].id == "class3"
    
    def test_get_neighbors_nonexistent_node(self, graph_store, sample_documents):
        """Test get_neighbors for non-existent node returns empty list."""
        graph_store.index(sample_documents)
        
        neighbors = graph_store.get_neighbors("nonexistent.id")
        assert neighbors == []
    
    def test_get_neighbors_no_edges(self, graph_store, sample_documents):
        """Test get_neighbors for node with no outgoing edges."""
        graph_store.index(sample_documents)
        
        # json.loads has no outgoing edges
        neighbors = graph_store.get_neighbors("json.loads")
        assert neighbors == []
    
    # === CLEAR & SIZE TESTS ===
    
    def test_clear_removes_all_data(self, graph_store, sample_documents):
        """Test that clear() removes nodes, edges, and inverted index."""
        graph_store.index(sample_documents)
        assert graph_store.size() == 3
        
        graph_store.clear()
        
        assert graph_store.size() == 0
        assert len(graph_store.inverted_index) == 0
    
    def test_size_empty_store(self, graph_store):
        """Test size() on empty store."""
        assert graph_store.size() == 0
    
    def test_size_after_indexing(self, graph_store, sample_documents):
        """Test size() returns correct node count."""
        graph_store.index(sample_documents)
        assert graph_store.size() == len(sample_documents)
    
    # === EDGE CASE TESTS ===
    
    def test_query_empty_store(self, graph_store):
        """Test querying empty store returns empty list."""
        results = graph_store.query("test query", n_results=5)
        assert results == []
    
    def test_circular_relationships(self, graph_store):
        """Test that circular relationships work correctly."""
        docs = [
            Document(
                id="func1",
                content="Function 1",
                metadata={},
                relationships=[Relationship(source_id="func1", target_id="func2", relationship_type="calls")]
            ),
            Document(
                id="func2",
                content="Function 2",
                metadata={},
                relationships=[Relationship(source_id="func2", target_id="func1", relationship_type="calls")]
            )
        ]
        
        graph_store.index(docs)
        
        # Both should be neighbors of each other
        neighbors1 = graph_store.get_neighbors("func1")
        neighbors2 = graph_store.get_neighbors("func2")
        
        assert len(neighbors1) == 1
        assert neighbors1[0].id == "func2"
        assert len(neighbors2) == 1
        assert neighbors2[0].id == "func1"

class TestBM25Store:
    """Tests for BM25Store."""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing BM25 keyword matching."""
        return [
            Document(
                id="json.load",
                content="Load JSON data from a file and parse it into Python objects",
                metadata={"module": "json", "type": "function"}
            ),
            Document(
                id="json.dump",
                content="Serialize Python objects to JSON format and write to file",
                metadata={"module": "json", "type": "function"}
            ),
            Document(
                id="pickle.load",
                content="Load Python objects from a pickle file using deserialization",
                metadata={"module": "pickle", "type": "function"}
            ),
            Document(
                id="csv.reader",
                content="Read CSV data from a file line by line",
                metadata={"module": "csv", "type": "function"}
            ),
            Document(
                id="yaml.safe_load",
                content="Safely parse YAML documents into Python data structures",
                metadata={"module": "yaml", "type": "function"}
            )
        ]
    
    @pytest.fixture
    def bm25_store(self):
        """Create a BM25Store instance."""
        from src.stores import BM25Store
        return BM25Store()
    
    # === INITIALIZATION TESTS ===
    
    def test_init(self, bm25_store):
        """Test BM25Store initialization."""
        assert bm25_store.size() == 0
        assert bm25_store.bm25_index is None
        assert bm25_store.documents == []
    
    # === INDEXING TESTS ===
    
    def test_index_documents(self, bm25_store, sample_documents):
        """Test indexing documents."""
        bm25_store.index(sample_documents)
        assert bm25_store.size() == 5
        assert bm25_store.bm25_index is not None
    
    def test_reindex_replaces_data(self, bm25_store, sample_documents):
        """Test that re-indexing replaces old data."""
        bm25_store.index(sample_documents)
        assert bm25_store.size() == 5
        
        # Re-index with just one document
        bm25_store.index([sample_documents[0]])
        assert bm25_store.size() == 1
    
    # === QUERY TESTS ===
    
    def test_query_exact_keyword_match(self, bm25_store, sample_documents):
        """Test querying with exact keyword matching."""
        bm25_store.index(sample_documents)
        
        # Query for "JSON" should rank JSON documents highest
        results = bm25_store.query("JSON", n_results=3)
        
        assert len(results) > 0
        # json.load or json.dump should be first (both have "JSON")
        assert results[0].id in ["json.load", "json.dump"]
    
    def test_query_multiple_keywords(self, bm25_store, sample_documents):
        """Test querying with multiple keywords."""
        bm25_store.index(sample_documents)
        
        # Query for "load file" should rank documents with both words higher
        results = bm25_store.query("load file", n_results=5)
        
        assert len(results) > 0
        # Documents with both "load" and "file" should rank higher
        top_ids = [doc.id for doc in results[:2]]
        # json.load, pickle.load, or csv.reader likely at top
        assert any("load" in doc_id or "reader" in doc_id for doc_id in top_ids)
    
    def test_query_case_insensitive(self, bm25_store, sample_documents):
        """Test that queries are case-insensitive."""
        bm25_store.index(sample_documents)
        
        results_lower = bm25_store.query("json", n_results=2)
        results_upper = bm25_store.query("JSON", n_results=2)
        results_mixed = bm25_store.query("Json", n_results=2)
        
        # All should return same results
        assert [doc.id for doc in results_lower] == [doc.id for doc in results_upper]
        assert [doc.id for doc in results_lower] == [doc.id for doc in results_mixed]
    
    def test_query_respects_n_results(self, bm25_store, sample_documents):
        """Test that n_results parameter limits results."""
        bm25_store.index(sample_documents)
        
        results_1 = bm25_store.query("load", n_results=1)
        assert len(results_1) == 1
        
        results_3 = bm25_store.query("load", n_results=3)
        assert len(results_3) == 3
        
        results_all = bm25_store.query("load", n_results=10)
        assert len(results_all) == 5  # Only 5 docs total
    
    def test_query_returns_document_objects(self, bm25_store, sample_documents):
        """Test that query returns proper Document objects."""
        bm25_store.index(sample_documents)
        
        results = bm25_store.query("file", n_results=2)
        
        for doc in results:
            assert isinstance(doc, Document)
            assert doc.id
            assert doc.content
            assert isinstance(doc.metadata, dict)
    
    def test_query_ranks_by_relevance(self, bm25_store, sample_documents):
        """Test that BM25 ranks documents by relevance."""
        bm25_store.index(sample_documents)
        
        # Query for "JSON file" - json.load has both words prominently
        results = bm25_store.query("JSON file", n_results=5)
        
        # json.load or json.dump should be first (both have "JSON" and "file")
        assert results[0].id in ["json.load", "json.dump"]
    
    def test_query_no_matches(self, bm25_store, sample_documents):
        """Test query with no matching keywords."""
        bm25_store.index(sample_documents)
        
        # Query for term not in any document
        results = bm25_store.query("tensorflow machine learning", n_results=5)
        
        # Should still return documents (lowest BM25 scores)
        assert len(results) == 5
    
    # === CLEAR & SIZE TESTS ===
    
    def test_clear_removes_all_data(self, bm25_store, sample_documents):
        """Test that clear() removes all indexed documents."""
        bm25_store.index(sample_documents)
        assert bm25_store.size() == 5
        
        bm25_store.clear()
        
        assert bm25_store.size() == 0
        assert bm25_store.bm25_index is None
        assert bm25_store.documents == []
    
    def test_clear_allows_reindexing(self, bm25_store, sample_documents):
        """Test that documents can be reindexed after clear()."""
        bm25_store.index(sample_documents)
        bm25_store.clear()
        bm25_store.index(sample_documents[:2])  # Index just two docs
        
        assert bm25_store.size() == 2
    
    def test_size_empty_store(self, bm25_store):
        """Test size() on empty store."""
        assert bm25_store.size() == 0
    
    def test_size_after_indexing(self, bm25_store, sample_documents):
        """Test size() returns correct doc count."""
        bm25_store.index(sample_documents)
        assert bm25_store.size() == len(sample_documents)
    
    # === EDGE CASE TESTS ===
    
    def test_query_empty_store(self, bm25_store):
        """Test querying empty store returns empty list."""
        results = bm25_store.query("test query", n_results=5)
        assert results == []
    
    def test_query_empty_string(self, bm25_store, sample_documents):
        """Test query with empty string."""
        bm25_store.index(sample_documents)
        
        results = bm25_store.query("", n_results=5)
        
        # Should return documents (BM25 will give scores even for empty query)
        assert len(results) <= 5
    
    def test_query_single_document(self, bm25_store, sample_documents):
        """Test query with only one document indexed."""
        bm25_store.index([sample_documents[0]])
        
        results = bm25_store.query("JSON", n_results=5)
        
        assert len(results) == 1
        assert results[0].id == "json.load"
    
    def test_documents_with_special_characters(self, bm25_store):
        """Test indexing documents with special characters."""
        docs = [
            Document(
                id="doc1",
                content="This has special chars: @#$% and numbers 123",
                metadata={}
            ),
            Document(
                id="doc2",
                content="Normal text without special characters",
                metadata={}
            )
        ]
        
        bm25_store.index(docs)
        results = bm25_store.query("special chars", n_results=2)
        
        # Should find doc1
        assert len(results) >= 1
        assert results[0].id == "doc1"
    
    def test_tokenization_consistency(self, bm25_store):
        """Test that query and index use same tokenization."""
        docs = [
            Document(id="1", content="Python programming language", metadata={}),
            Document(id="2", content="Java programming language", metadata={})
        ]
        
        bm25_store.index(docs)
        
        # Query should use same tokenization (lowercase, split)
        results = bm25_store.query("Python Programming", n_results=1)
        
        assert len(results) == 1
        assert results[0].id == "1"