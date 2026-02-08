import chromadb
from chromadb.utils import embedding_functions
from .base import MemoryStore
from src.core import Document

class VectorStore(MemoryStore):
    """
    Vector store using ChromaDB with sentence transformers.
    
    Provides semantic search by converting documents to embeddings
    and finding nearest neighbors via cosine similarity.
    """

    def _create_collection(self) -> None:
        """Create or recreate the ChromaDB collection."""
        self.collection = self.client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn
        )
    
    def __init__(self, collection_name: str = "documents", model_name: str = "all-MiniLM-L6-v2"):
        self.collection_name = collection_name
        self.client = chromadb.Client()
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self._create_collection()
    
    def index(self, documents: list[Document]) -> None:
        """
        Index documents by generating embeddings and storing in ChromaDB.
        
        Args:
            documents: List of Document objects to index
        """
        self.collection.add(
            ids=[doc.id for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[doc.metadata for doc in documents]
        )
    
    def query(self, query_text: str, n_results: int = 5) -> list[Document]:
        """
        Query store using semantic similarity.
        
        Args:
            query_text: Natural language query
            n_results: Number of results to return
            
        Returns:
            List of Document objects, ordered by relevance
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )

        documents = []
        for i in range(len(results['ids'][0])):
            documents.append(Document(
                id=results['ids'][0][i],
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                embedding=None, # Embeddings are large and not needed out of the store
                relationships=[] # Relationships are only needed for graph stores
            ))
        return documents
    
    def clear(self) -> None:
        """Clear all indexed documents from the store."""
        self.client.delete_collection(name=self.collection_name)
        self._create_collection()

    def size(self) -> int:
        """Return number of indexed documents."""
        return self.collection.count()