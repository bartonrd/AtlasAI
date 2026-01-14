"""
Embedding service - handles document embeddings with caching
"""

import pickle
from pathlib import Path
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class EmbeddingService:
    """Service for creating and managing document embeddings"""
    
    def __init__(self, model_name: str, cache_dir: Optional[Path] = None):
        """
        Initialize embedding service
        
        Args:
            model_name: Name or path of the embedding model
            cache_dir: Optional directory for caching embeddings
        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._embeddings = None
        self._vectorstore = None
        
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create embeddings model (lazy loading with caching)"""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            # Verify model works
            try:
                _ = self._embeddings.embed_query("test")
            except Exception as e:
                raise RuntimeError(f"Embedding model failed to initialize: {e}")
        return self._embeddings
    
    def create_vectorstore(
        self, 
        documents: List[Document],
        use_cache: bool = True
    ) -> FAISS:
        """
        Create vector store from documents with optional caching
        
        Args:
            documents: List of document chunks to embed
            use_cache: Whether to use cached embeddings if available
            
        Returns:
            FAISS vector store
            
        Raises:
            ValueError: If documents list is empty
            IndexError: If embedding fails
        """
        if not documents:
            raise ValueError("Cannot create vectorstore from empty document list")
        
        # Generate cache key based on document content
        cache_key = None
        if use_cache and self.cache_dir:
            cache_key = self._generate_cache_key(documents)
            cached_store = self._load_cached_vectorstore(cache_key)
            if cached_store:
                self._vectorstore = cached_store
                return cached_store
        
        # Create new vectorstore
        try:
            vectorstore = FAISS.from_documents(documents, self.embeddings)
        except IndexError:
            raise IndexError("Failed to create embeddings - ensure documents contain extractable text")
        
        # Cache the vectorstore
        if cache_key and self.cache_dir:
            self._save_vectorstore_cache(vectorstore, cache_key)
        
        self._vectorstore = vectorstore
        return vectorstore
    
    def get_retriever(
        self, 
        vectorstore: Optional[FAISS] = None,
        top_k: int = 4
    ) -> VectorStoreRetriever:
        """
        Get a retriever from the vector store
        
        Args:
            vectorstore: Optional vector store (uses cached if not provided)
            top_k: Number of documents to retrieve
            
        Returns:
            Vector store retriever
            
        Raises:
            RuntimeError: If no vector store is available
        """
        store = vectorstore or self._vectorstore
        if store is None:
            raise RuntimeError("No vector store available. Create one first with create_vectorstore()")
        
        return store.as_retriever(search_kwargs={"k": top_k})
    
    def _generate_cache_key(self, documents: List[Document]) -> str:
        """
        Generate a cache key based on document content
        
        Args:
            documents: List of documents
            
        Returns:
            Cache key string
        """
        # Create hash from first few documents' content
        # This is a simple approach; for production, consider using content hashing
        sample_size = min(5, len(documents))
        sample_content = "".join([doc.page_content[:100] for doc in documents[:sample_size]])
        # Simple hash - in production, use hashlib
        hash_val = hash(sample_content + str(len(documents)))
        return f"vectorstore_{abs(hash_val)}.pkl"
    
    def _load_cached_vectorstore(self, cache_key: str) -> Optional[FAISS]:
        """
        Load cached vector store if available
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached FAISS store or None
        """
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / cache_key
        if not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            # If cache is corrupted, ignore it
            return None
    
    def _save_vectorstore_cache(self, vectorstore: FAISS, cache_key: str):
        """
        Save vector store to cache
        
        Args:
            vectorstore: FAISS store to cache
            cache_key: Cache key
        """
        if not self.cache_dir:
            return
        
        cache_path = self.cache_dir / cache_key
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(vectorstore, f)
        except Exception:
            # If caching fails, continue without it
            pass
    
    def clear_cache(self):
        """Clear all cached vector stores"""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("vectorstore_*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass
