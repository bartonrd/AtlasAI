"""
Vector store management with caching support for AtlasAI.

Handles embeddings and FAISS vector store with persistent caching.
"""

import pickle
from pathlib import Path
from typing import List, Optional
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStoreManager:
    """Manages vector store with caching capabilities."""
    
    def __init__(
        self,
        embedding_model_path: str,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True
    ):
        """
        Initialize vector store manager.
        
        Args:
            embedding_model_path: Path to embedding model
            cache_dir: Directory for caching vector store
            use_cache: Whether to use persistent caching
        """
        self.embedding_model_path = embedding_model_path
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vectorstore: Optional[FAISS] = None
        
        if self.use_cache and cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, doc_hash: str) -> Path:
        """Get cache file path for given document hash."""
        if not self.cache_dir:
            raise ValueError("Cache directory not configured")
        return self.cache_dir / f"vectorstore_{doc_hash}.pkl"
    
    def _load_from_cache(self, doc_hash: str) -> Optional[FAISS]:
        """
        Load vector store from cache.
        
        Args:
            doc_hash: Hash of documents for cache key
            
        Returns:
            Cached FAISS vector store or None if not found
        """
        if not self.use_cache:
            return None
        
        cache_path = self._get_cache_path(doc_hash)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    self.vectorstore = pickle.load(f)
                return self.vectorstore
            except Exception as e:
                st.warning(f"Failed to load vector store from cache: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, doc_hash: str):
        """
        Save vector store to cache.
        
        Args:
            doc_hash: Hash of documents for cache key
        """
        if not self.use_cache or not self.vectorstore:
            return
        
        cache_path = self._get_cache_path(doc_hash)
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(self.vectorstore, f)
        except Exception as e:
            st.warning(f"Failed to save vector store to cache: {e}")
    
    def initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Initialize embedding model.
        
        Returns:
            Initialized HuggingFaceEmbeddings instance
        """
        if self.embeddings is not None:
            return self.embeddings
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_path
            )
            # Test embedding
            _ = self.embeddings.embed_query("test")
            return self.embeddings
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize embedding model from '{self.embedding_model_path}': {e}"
            )
    
    def create_vector_store(
        self,
        documents: List[Document],
        doc_hash: Optional[str] = None
    ) -> FAISS:
        """
        Create or load vector store from documents.
        
        Args:
            documents: List of document chunks
            doc_hash: Hash of documents for caching (optional)
            
        Returns:
            FAISS vector store
        """
        # Try loading from cache first
        if doc_hash and self.use_cache:
            cached_store = self._load_from_cache(doc_hash)
            if cached_store is not None:
                self.vectorstore = cached_store
                return self.vectorstore
        
        # Initialize embeddings if not already done
        if self.embeddings is None:
            self.initialize_embeddings()
        
        # Create new vector store
        try:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        except IndexError as e:
            raise RuntimeError(
                "Failed to create vector store - ensure documents have extractable text"
            ) from e
        
        # Save to cache
        if doc_hash:
            self._save_to_cache(doc_hash)
        
        return self.vectorstore
    
    def get_retriever(self, k: int = 4) -> VectorStoreRetriever:
        """
        Get retriever from vector store.
        
        Args:
            k: Number of documents to retrieve
            
        Returns:
            VectorStoreRetriever instance
        """
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized")
        
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
    
    def clear_cache(self):
        """Clear all cached vector stores."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("vectorstore_*.pkl"):
            try:
                cache_file.unlink()
            except Exception as e:
                st.warning(f"Failed to delete cache file {cache_file}: {e}")
