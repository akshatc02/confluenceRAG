"""
Hybrid Vector Store for Confluence RAG
Combines vector similarity search with BM25 keyword search for better retrieval
"""
from typing import List, Dict, Any, Optional
import numpy as np
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import pickle
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Represents a search result with score"""
    document: Document
    score: float
    source: str  # 'vector', 'bm25', or 'hybrid'


class HybridVectorStore:
    """
    Hybrid search combining:
    1. Dense vector similarity (semantic search)
    2. BM25 sparse retrieval (keyword search)
    3. Metadata filtering
    """
    
    def __init__(self, 
                 embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
                 persist_directory: str = "./chroma_db"):
        """
        Initialize hybrid vector store
        
        Args:
            embedding_model_name: HuggingFace model for embeddings
            persist_directory: Directory to persist vector store
        """
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.persist_directory = persist_directory
        self.vector_store = None
        self.bm25 = None
        self.documents = []
        self.bm25_corpus = []
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to both vector store and BM25 index
        
        Args:
            documents: List of LangChain Document objects
        """
        print(f"Adding {len(documents)} documents to hybrid store...")
        
        # Store documents for BM25
        self.documents = documents
        
        # Create Chroma vector store
        print("Building vector store...")
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            persist_directory=self.persist_directory
        )
        
        # Build BM25 index
        print("Building BM25 index...")
        self.bm25_corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.bm25_corpus)
        
        print("Hybrid store built successfully!")
    
    def save_bm25(self, filepath: str = "./bm25_index.pkl") -> None:
        """Save BM25 index to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'documents': self.documents,
                'corpus': self.bm25_corpus
            }, f)
        print(f"BM25 index saved to {filepath}")
    
    def load_bm25(self, filepath: str = "./bm25_index.pkl") -> None:
        """Load BM25 index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.documents = data['documents']
            self.bm25_corpus = data['corpus']
        print(f"BM25 index loaded from {filepath}")
    
    def load_vector_store(self) -> None:
        """Load existing Chroma vector store"""
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )
        print("Vector store loaded successfully!")
    
    def search_vector(self, query: str, k: int = 10, 
                     filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query
            k: Number of results to return
            filter_dict: Metadata filters (e.g., {'space_key': 'ENG'})
        
        Returns:
            List of SearchResult objects
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call add_documents() first.")
        
        # Perform similarity search with metadata filtering
        if filter_dict:
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=k,
                filter=filter_dict
            )
        else:
            results = self.vector_store.similarity_search_with_score(query, k=k)
        
        # Convert to SearchResult objects
        search_results = []
        for doc, score in results:
            # Chroma returns distance, convert to similarity (lower is better, so invert)
            similarity_score = 1 / (1 + score)
            search_results.append(SearchResult(
                document=doc,
                score=similarity_score,
                source='vector'
            ))
        
        return search_results
    
    def search_bm25(self, query: str, k: int = 10) -> List[SearchResult]:
        """
        Perform BM25 keyword search
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of SearchResult objects
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not initialized. Call add_documents() first.")
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k indices
        top_k_indices = np.argsort(scores)[-k:][::-1]
        
        # Build results
        search_results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Only include relevant results
                search_results.append(SearchResult(
                    document=self.documents[idx],
                    score=float(scores[idx]),
                    source='bm25'
                ))
        
        return search_results
    
    def search_hybrid(self, 
                     query: str, 
                     k: int = 10,
                     vector_weight: float = 0.7,
                     bm25_weight: float = 0.3,
                     filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25 results
        
        Args:
            query: Search query
            k: Number of results to return
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            filter_dict: Metadata filters for vector search
        
        Returns:
            List of SearchResult objects, ranked by combined score
        """
        # Get results from both methods
        vector_results = self.search_vector(query, k=k*2, filter_dict=filter_dict)
        bm25_results = self.search_bm25(query, k=k*2)
        
        # Normalize scores (0-1 range)
        if vector_results:
            max_vector_score = max(r.score for r in vector_results)
            for r in vector_results:
                r.score = r.score / max_vector_score if max_vector_score > 0 else 0
        
        if bm25_results:
            max_bm25_score = max(r.score for r in bm25_results)
            for r in bm25_results:
                r.score = r.score / max_bm25_score if max_bm25_score > 0 else 0
        
        # Combine results using document page_content as key
        combined_scores = {}
        
        # Add vector scores
        for result in vector_results:
            doc_key = result.document.page_content
            combined_scores[doc_key] = {
                'document': result.document,
                'score': result.score * vector_weight,
                'vector_score': result.score,
                'bm25_score': 0
            }
        
        # Add BM25 scores
        for result in bm25_results:
            doc_key = result.document.page_content
            if doc_key in combined_scores:
                combined_scores[doc_key]['score'] += result.score * bm25_weight
                combined_scores[doc_key]['bm25_score'] = result.score
            else:
                combined_scores[doc_key] = {
                    'document': result.document,
                    'score': result.score * bm25_weight,
                    'vector_score': 0,
                    'bm25_score': result.score
                }
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:k]
        
        # Convert to SearchResult objects
        hybrid_results = []
        for item in sorted_results:
            result = SearchResult(
                document=item['document'],
                score=item['score'],
                source='hybrid'
            )
            # Add score breakdown to metadata
            result.document.metadata['vector_score'] = item['vector_score']
            result.document.metadata['bm25_score'] = item['bm25_score']
            hybrid_results.append(result)
        
        return hybrid_results
    
    def search(self, 
              query: str, 
              k: int = 5,
              method: str = 'hybrid',
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        Main search interface
        
        Args:
            query: Search query
            k: Number of results to return
            method: 'vector', 'bm25', or 'hybrid'
            filter_dict: Metadata filters
        
        Returns:
            List of Document objects
        """
        if method == 'vector':
            results = self.search_vector(query, k, filter_dict)
        elif method == 'bm25':
            results = self.search_bm25(query, k)
        elif method == 'hybrid':
            results = self.search_hybrid(query, k, filter_dict=filter_dict)
        else:
            raise ValueError(f"Unknown search method: {method}")
        
        return [r.document for r in results]


# Alternative: Production-ready implementation using Pinecone
class PineconeHybridStore:
    """
    Hybrid store using Pinecone for production use.
    Pinecone supports metadata filtering and hybrid search natively.
    
    Note: Requires pinecone-client library and API key
    """
    
    def __init__(self, api_key: str, environment: str, index_name: str):
        """
        Initialize Pinecone hybrid store
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., 'us-east1-gcp')
            index_name: Name of the Pinecone index
        """
        try:
            import pinecone
            from pinecone import Pinecone as PineconeClient
        except ImportError:
            raise ImportError("Please install pinecone-client: pip install pinecone-client")
        
        # Initialize Pinecone
        self.pc = PineconeClient(api_key=api_key)
        
        # Check if index exists
        if index_name not in self.pc.list_indexes().names():
            # Create index with hybrid search support
            self.pc.create_index(
                name=index_name,
                dimension=768,  # Dimension for all-mpnet-base-v2
                metric='cosine',
                spec=pinecone.ServerlessSpec(
                    cloud='aws',
                    region=environment
                )
            )
        
        self.index = self.pc.Index(index_name)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
    
    def add_documents(self, documents: List[Document], batch_size: int = 100):
        """Add documents to Pinecone index"""
        print(f"Adding {len(documents)} documents to Pinecone...")
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Prepare vectors
            vectors = []
            for j, doc in enumerate(batch):
                doc_id = f"doc_{i+j}"
                embedding = self.embedding_model.embed_query(doc.page_content)
                
                # Prepare metadata (Pinecone has size limits)
                metadata = {
                    'text': doc.page_content[:1000],  # Truncate for metadata
                    'page_id': doc.metadata.get('page_id', ''),
                    'page_title': doc.metadata.get('page_title', ''),
                    'space_key': doc.metadata.get('space_key', ''),
                    'url': doc.metadata.get('url', ''),
                    'chunk_type': doc.metadata.get('chunk_type', ''),
                }
                
                vectors.append({
                    'id': doc_id,
                    'values': embedding,
                    'metadata': metadata
                })
            
            # Upsert batch
            self.index.upsert(vectors=vectors)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}")
    
    def search(self, query: str, k: int = 5, 
              filter_dict: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Search Pinecone index with optional metadata filtering"""
        # Get query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Search with filter
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            filter=filter_dict
        )
        
        # Convert to Documents
        documents = []
        for match in results['matches']:
            doc = Document(
                page_content=match['metadata']['text'],
                metadata=match['metadata']
            )
            documents.append(doc)
        
        return documents


if __name__ == "__main__":
    # Example usage
    import json
    from document_splitter_improved import split_documents
    
    # Load and split documents
    with open('confluence_pages.json', 'r') as f:
        pages = json.load(f)
    
    documents = split_documents(pages)
    
    # Initialize hybrid store
    hybrid_store = HybridVectorStore(
        embedding_model_name="sentence-transformers/all-mpnet-base-v2",
        persist_directory="./chroma_confluence"
    )
    
    # Add documents
    hybrid_store.add_documents(documents)
    hybrid_store.save_bm25("./bm25_confluence.pkl")
    
    # Test search
    query = "What is our PTO policy?"
    
    print("\n=== Vector Search ===")
    results = hybrid_store.search(query, k=3, method='vector')
    for i, doc in enumerate(results):
        print(f"\n{i+1}. {doc.metadata.get('page_title')}")
        print(f"   Section: {' > '.join(doc.metadata.get('section_hierarchy', []))}")
        print(f"   Preview: {doc.page_content[:150]}...")
    
    print("\n=== BM25 Search ===")
    results = hybrid_store.search(query, k=3, method='bm25')
    for i, doc in enumerate(results):
        print(f"\n{i+1}. {doc.metadata.get('page_title')}")
        print(f"   Preview: {doc.page_content[:150]}...")
    
    print("\n=== Hybrid Search ===")
    results = hybrid_store.search(query, k=3, method='hybrid')
    for i, doc in enumerate(results):
        print(f"\n{i+1}. {doc.metadata.get('page_title')}")
        print(f"   Vector: {doc.metadata.get('vector_score', 0):.3f}, BM25: {doc.metadata.get('bm25_score', 0):.3f}")
        print(f"   Preview: {doc.page_content[:150]}...")
