"""
Retrieval abstraction module.

Provides an abstract interface for hybrid search (keyword + vector)
without tying to a specific backend. Maps intents to document filters.
"""

from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass


@dataclass
class RetrievedDoc:
    """A retrieved document snippet with metadata."""
    title: str
    url: str
    content: str
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


class SearchBackend(Protocol):
    """Protocol for search backend implementations."""
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None
    ) -> List[RetrievedDoc]:
        """
        Search for documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters
        
        Returns:
            List of retrieved documents with scores
        """
        ...


class Retriever:
    """
    Abstract retrieval layer for hybrid search.
    
    Maps intents to appropriate document filters and retrieves
    relevant chunks from the search backend.
    """
    
    def __init__(
        self,
        search_backend: SearchBackend,
        intent_filters: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        """
        Initialize the retriever.
        
        Args:
            search_backend: The search backend implementation
            intent_filters: Mapping from intent to metadata filters
        """
        self.search_backend = search_backend
        self.intent_filters = intent_filters or {}
    
    def retrieve(
        self,
        query: str,
        intent: str,
        top_k: int = 5,
        additional_filters: Optional[Dict[str, str]] = None,
    ) -> List[RetrievedDoc]:
        """
        Retrieve documents based on query and intent.
        
        Args:
            query: The search query
            intent: The classified intent
            top_k: Number of documents to retrieve
            additional_filters: Additional metadata filters to apply
        
        Returns:
            List of retrieved documents sorted by relevance score
        """
        # Get intent-specific filters
        filters = self.intent_filters.get(intent, {}).copy()
        
        # Merge with additional filters
        if additional_filters:
            filters.update(additional_filters)
        
        # Perform search
        results = self.search_backend.search(
            query=query,
            top_k=top_k,
            filters=filters if filters else None
        )
        
        return results


class FAISSSearchBackend:
    """
    Adapter for FAISS vector store to match SearchBackend protocol.
    
    This bridges the existing RAG engine's FAISS retriever with the
    new abstraction layer.
    """
    
    def __init__(self, faiss_retriever):
        """
        Initialize with a FAISS retriever.
        
        Args:
            faiss_retriever: LangChain FAISS retriever instance
        """
        self.faiss_retriever = faiss_retriever
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None
    ) -> List[RetrievedDoc]:
        """
        Search using FAISS retriever.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters (applied post-retrieval for FAISS)
        
        Returns:
            List of retrieved documents
        """
        # Update retriever's k parameter
        original_k = self.faiss_retriever.search_kwargs.get("k", 4)
        self.faiss_retriever.search_kwargs["k"] = top_k
        
        try:
            # Retrieve documents
            docs = self.faiss_retriever.get_relevant_documents(query)
            
            # Convert to RetrievedDoc format
            results = []
            for doc in docs:
                metadata = doc.metadata or {}
                
                # Apply filters if specified (post-retrieval filtering for FAISS)
                if filters:
                    matches = all(
                        metadata.get(key) == value
                        for key, value in filters.items()
                    )
                    if not matches:
                        continue
                
                # Extract source information
                source = metadata.get("source", "unknown")
                page = metadata.get("page", "unknown")
                
                # Create title from source filename and page
                import os
                source_name = os.path.basename(source)
                title = f"{source_name} (page {page})"
                
                # For FAISS, we don't have direct access to similarity scores in the retriever
                # We use a placeholder score based on position (higher is better)
                score = 1.0 / (len(results) + 1)
                
                results.append(RetrievedDoc(
                    title=title,
                    url=source,  # Use source path as URL
                    content=doc.page_content,
                    score=score,
                    metadata=metadata,
                ))
            
            return results
        finally:
            # Restore original k
            self.faiss_retriever.search_kwargs["k"] = original_k


class MockSearchBackend:
    """
    Mock search backend for testing without dependencies.
    
    Returns predefined results based on query patterns.
    """
    
    def __init__(self):
        """Initialize mock backend with sample data."""
        self.documents = [
            RetrievedDoc(
                title="Installation Guide",
                url="/docs/install.pdf",
                content="To install the software, follow these steps: 1. Download the installer...",
                score=0.9,
                metadata={"doc_type": "procedure", "page": "1"}
            ),
            RetrievedDoc(
                title="Error Code Reference",
                url="/docs/errors.pdf",
                content="Error errno 13: Permission denied. This occurs when...",
                score=0.85,
                metadata={"doc_type": "incident", "page": "42"}
            ),
            RetrievedDoc(
                title="API Concepts",
                url="/docs/api.pdf",
                content="The API module provides a REST interface for external systems...",
                score=0.8,
                metadata={"doc_type": "concept", "page": "5"}
            ),
        ]
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, str]] = None
    ) -> List[RetrievedDoc]:
        """
        Mock search implementation.
        
        Returns filtered documents based on query and filters.
        """
        results = []
        
        # Apply filters
        for doc in self.documents:
            if filters:
                matches = all(
                    doc.metadata.get(key) == value
                    for key, value in filters.items()
                )
                if not matches:
                    continue
            
            # Simple keyword matching for demo
            query_lower = query.lower()
            if any(word in doc.content.lower() for word in query_lower.split()):
                results.append(doc)
        
        # Return top-k results
        return results[:top_k]
