"""
Unit tests for retriever module.
"""

import pytest
from atlasai_runtime.retriever import (
    Retriever,
    RetrievedDoc,
    MockSearchBackend,
)
from atlasai_runtime.intent_classifier import (
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
)


class TestRetrievedDoc:
    """Test suite for RetrievedDoc."""
    
    def test_retrieved_doc_creation(self):
        """Test creating a RetrievedDoc."""
        doc = RetrievedDoc(
            title="Test Doc",
            url="/test.pdf",
            content="Test content",
            score=0.85,
            metadata={"page": "1", "doc_type": "procedure"}
        )
        
        assert doc.title == "Test Doc"
        assert doc.url == "/test.pdf"
        assert doc.content == "Test content"
        assert doc.score == 0.85
        assert doc.metadata["page"] == "1"
    
    def test_retrieved_doc_to_dict(self):
        """Test RetrievedDoc serialization."""
        doc = RetrievedDoc(
            title="Test",
            url="/test",
            content="Content",
            score=0.9,
            metadata={"key": "value"}
        )
        
        d = doc.to_dict()
        assert d["title"] == "Test"
        assert d["url"] == "/test"
        assert d["content"] == "Content"
        assert d["score"] == 0.9
        assert d["metadata"]["key"] == "value"


class TestMockSearchBackend:
    """Test suite for MockSearchBackend."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockSearchBackend()
    
    def test_search_returns_results(self):
        """Test that search returns results."""
        results = self.backend.search("install", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_search_with_filters(self):
        """Test search with metadata filters."""
        # Search for procedure documents
        results = self.backend.search(
            "install",
            top_k=5,
            filters={"doc_type": "procedure"}
        )
        
        # All results should match filter
        for doc in results:
            assert doc.metadata.get("doc_type") == "procedure"
    
    def test_search_respects_top_k(self):
        """Test that search respects top_k parameter."""
        results = self.backend.search("test", top_k=2)
        
        assert len(results) <= 2


class TestRetriever:
    """Test suite for Retriever."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockSearchBackend()
        
        self.intent_filters = {
            INTENT_HOW_TO: {"doc_type": "procedure"},
            INTENT_BUG_RESOLUTION: {"doc_type": "incident"},
            INTENT_TOOL_EXPLANATION: {"doc_type": "concept"},
        }
        
        self.retriever = Retriever(
            search_backend=self.backend,
            intent_filters=self.intent_filters,
        )
    
    def test_retrieve_with_intent(self):
        """Test retrieval with intent-specific filters."""
        results = self.retriever.retrieve(
            query="install software",
            intent=INTENT_HOW_TO,
            top_k=5,
        )
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_retrieve_applies_intent_filters(self):
        """Test that intent-specific filters are applied."""
        # This test verifies the integration, though with mock backend
        # the actual filtering logic is in the backend
        results = self.retriever.retrieve(
            query="install",
            intent=INTENT_HOW_TO,
            top_k=5,
        )
        
        # With the mock backend and how_to intent, should get procedure docs
        if results:
            assert all(doc.metadata.get("doc_type") == "procedure" for doc in results)
    
    def test_retrieve_with_additional_filters(self):
        """Test retrieval with additional filters."""
        results = self.retriever.retrieve(
            query="error",
            intent=INTENT_BUG_RESOLUTION,
            top_k=5,
            additional_filters={"page": "42"},
        )
        
        # Additional filters should be merged with intent filters
        assert isinstance(results, list)
    
    def test_retrieve_unknown_intent(self):
        """Test retrieval with unknown intent."""
        results = self.retriever.retrieve(
            query="test",
            intent="unknown_intent",
            top_k=5,
        )
        
        # Should still return results without filters
        assert isinstance(results, list)


class TestRetrieverEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockSearchBackend()
        self.retriever = Retriever(self.backend)
    
    def test_empty_query(self):
        """Test retrieval with empty query."""
        results = self.retriever.retrieve(
            query="",
            intent=INTENT_HOW_TO,
            top_k=5,
        )
        
        assert isinstance(results, list)
    
    def test_very_high_top_k(self):
        """Test retrieval with very high top_k."""
        results = self.retriever.retrieve(
            query="test",
            intent=INTENT_HOW_TO,
            top_k=1000,
        )
        
        # Should return available results (limited by backend)
        assert isinstance(results, list)
    
    def test_zero_top_k(self):
        """Test retrieval with zero top_k."""
        results = self.retriever.retrieve(
            query="test",
            intent=INTENT_HOW_TO,
            top_k=0,
        )
        
        # Should return empty list
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
