"""
Retriever Interface Module

Provides abstraction for hybrid search (keyword + vector) with intent-based filtering.
"""

from typing import List, Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from .intent_classifier import IntentType


class DocumentSnippet:
    """A retrieved document snippet with metadata."""
    
    def __init__(
        self,
        title: str,
        url: str,
        content: str,
        score: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize document snippet.
        
        Args:
            title: Document title
            url: Document URL or path
            content: Snippet content
            score: Relevance score (0.0 to 1.0)
            metadata: Additional metadata (doc_type, version, etc.)
        """
        self.title = title
        self.url = url
        self.content = content
        self.score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata
        }


class RetrieverInterface(ABC):
    """
    Abstract interface for document retrieval.
    
    Supports hybrid search (keyword + vector) with filtering capabilities.
    """
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentSnippet]:
        """
        Retrieve top-k relevant document snippets.
        
        Args:
            query: Search query (can be rewritten query)
            top_k: Number of snippets to retrieve
            filters: Optional filters (doc_type, version, etc.)
            
        Returns:
            List of DocumentSnippet objects sorted by relevance
        """
        pass
    
    @abstractmethod
    def supports_hybrid_search(self) -> bool:
        """Return True if retriever supports hybrid keyword + vector search."""
        pass


class IntentBasedRetriever:
    """
    Retriever that maps intent to document type filters.
    
    Wraps a base retriever and adds intent-based filtering logic.
    """
    
    # Intent to document type mapping
    INTENT_DOC_TYPE_MAP = {
        IntentType.HOW_TO: "procedure",
        IntentType.BUG_RESOLUTION: "incident",
        IntentType.TOOL_EXPLANATION: "concept",
        IntentType.ESCALATE_OR_TICKET: None,  # No specific filter
        IntentType.CHITCHAT: None,
        IntentType.OTHER: None
    }
    
    def __init__(self, base_retriever: RetrieverInterface):
        """
        Initialize intent-based retriever.
        
        Args:
            base_retriever: Underlying retriever implementation
        """
        self.base_retriever = base_retriever
    
    def retrieve_by_intent(
        self,
        query: str,
        intent: IntentType,
        top_k: int = 5,
        additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentSnippet]:
        """
        Retrieve documents with intent-based filtering.
        
        Args:
            query: Search query
            intent: User intent
            top_k: Number of snippets to retrieve
            additional_filters: Additional custom filters
            
        Returns:
            List of DocumentSnippet objects
        """
        # Build filters based on intent
        filters = additional_filters.copy() if additional_filters else {}
        
        # Add doc_type filter based on intent
        doc_type = self.INTENT_DOC_TYPE_MAP.get(intent)
        if doc_type:
            filters["doc_type"] = doc_type
        
        # Retrieve with filters
        return self.base_retriever.retrieve(
            query=query,
            top_k=top_k,
            filters=filters
        )


class FakeRetriever(RetrieverInterface):
    """
    Fake retriever for testing purposes.
    
    Returns mock results without external dependencies.
    """
    
    def __init__(self, mock_results: Optional[List[DocumentSnippet]] = None):
        """
        Initialize fake retriever.
        
        Args:
            mock_results: Pre-configured results to return
        """
        self.mock_results = mock_results or []
        self.last_query = None
        self.last_filters = None
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentSnippet]:
        """Retrieve mock results."""
        # Store for inspection
        self.last_query = query
        self.last_filters = filters
        
        # Apply filters if specified
        results = self.mock_results
        
        if filters:
            filtered = []
            for snippet in results:
                # Check if snippet matches all filters
                matches = True
                for key, value in filters.items():
                    if key not in snippet.metadata or snippet.metadata[key] != value:
                        matches = False
                        break
                
                if matches:
                    filtered.append(snippet)
            
            results = filtered
        
        # Return top-k
        return results[:top_k]
    
    def supports_hybrid_search(self) -> bool:
        """Fake retriever supports hybrid search."""
        return True


class VectorStoreRetriever(RetrieverInterface):
    """
    Real retriever implementation using LangChain vector store.
    
    Adapts existing RAG engine retriever to the interface.
    """
    
    def __init__(self, vectorstore, embedding_model=None):
        """
        Initialize vector store retriever.
        
        Args:
            vectorstore: LangChain vector store instance (e.g., FAISS)
            embedding_model: Optional embedding model for hybrid search
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentSnippet]:
        """
        Retrieve from vector store.
        
        Args:
            query: Search query
            top_k: Number of results
            filters: Metadata filters
            
        Returns:
            List of DocumentSnippet objects
        """
        # Build search kwargs
        search_kwargs = {"k": top_k}
        
        # Add filters if supported
        if filters:
            search_kwargs["filter"] = filters
        
        # Search vector store
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, **search_kwargs
            )
        except TypeError:
            # Fallback if similarity_search_with_score doesn't support filters
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=top_k
            )
        
        # Convert to DocumentSnippet objects
        snippets = []
        for doc, score in docs_with_scores:
            # Convert score to normalized 0-1 range (FAISS uses distance, lower is better)
            # For FAISS L2 distance, typical range is 0-2
            normalized_score = max(0.0, 1.0 - (score / 2.0))
            
            # Extract metadata
            metadata = doc.metadata or {}
            title = metadata.get("source", "Unknown")
            url = metadata.get("source", "")
            
            snippet = DocumentSnippet(
                title=title,
                url=url,
                content=doc.page_content,
                score=normalized_score,
                metadata=metadata
            )
            snippets.append(snippet)
        
        return snippets
    
    def supports_hybrid_search(self) -> bool:
        """Vector store supports semantic search; keyword search can be added."""
        return False  # Pure vector for now; hybrid requires additional implementation
