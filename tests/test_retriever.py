"""
Unit tests for Retriever interfaces.
"""

import unittest
from atlasai_runtime.retriever import (
    DocumentSnippet,
    FakeRetriever,
    IntentBasedRetriever,
    RetrieverInterface
)
from atlasai_runtime.intent_classifier import IntentType


class TestDocumentSnippet(unittest.TestCase):
    """Test cases for DocumentSnippet."""
    
    def test_creation(self):
        """Test creating a document snippet."""
        snippet = DocumentSnippet(
            title="Test Doc",
            url="/docs/test",
            content="Test content",
            score=0.85,
            metadata={"doc_type": "procedure"}
        )
        
        self.assertEqual(snippet.title, "Test Doc")
        self.assertEqual(snippet.score, 0.85)
        self.assertEqual(snippet.metadata["doc_type"], "procedure")
    
    def test_score_clamping(self):
        """Test that scores are clamped to [0, 1]."""
        snippet1 = DocumentSnippet("Test", "/url", "content", 1.5)
        self.assertEqual(snippet1.score, 1.0)
        
        snippet2 = DocumentSnippet("Test", "/url", "content", -0.5)
        self.assertEqual(snippet2.score, 0.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        snippet = DocumentSnippet(
            title="Test",
            url="/test",
            content="Content",
            score=0.75,
            metadata={"key": "value"}
        )
        
        result = snippet.to_dict()
        self.assertEqual(result["title"], "Test")
        self.assertEqual(result["score"], 0.75)
        self.assertEqual(result["metadata"]["key"], "value")


class TestFakeRetriever(unittest.TestCase):
    """Test cases for FakeRetriever."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.snippets = [
            DocumentSnippet("Doc1", "/1", "Content 1", 0.9, {"doc_type": "procedure"}),
            DocumentSnippet("Doc2", "/2", "Content 2", 0.8, {"doc_type": "incident"}),
            DocumentSnippet("Doc3", "/3", "Content 3", 0.7, {"doc_type": "concept"}),
        ]
        self.retriever = FakeRetriever(self.snippets)
    
    def test_retrieve_without_filters(self):
        """Test retrieval without filters."""
        results = self.retriever.retrieve("test query", top_k=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].title, "Doc1")
        self.assertEqual(self.retriever.last_query, "test query")
    
    def test_retrieve_with_filters(self):
        """Test retrieval with metadata filters."""
        results = self.retriever.retrieve(
            "test query",
            top_k=5,
            filters={"doc_type": "procedure"}
        )
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].title, "Doc1")
        self.assertEqual(results[0].metadata["doc_type"], "procedure")
    
    def test_supports_hybrid_search(self):
        """Test hybrid search support."""
        self.assertTrue(self.retriever.supports_hybrid_search())
    
    def test_empty_results(self):
        """Test retriever with no results."""
        empty_retriever = FakeRetriever([])
        results = empty_retriever.retrieve("test", top_k=5)
        self.assertEqual(len(results), 0)


class TestIntentBasedRetriever(unittest.TestCase):
    """Test cases for IntentBasedRetriever."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.snippets = [
            DocumentSnippet("How-to Doc", "/1", "Steps", 0.9, {"doc_type": "procedure"}),
            DocumentSnippet("Bug Doc", "/2", "Error fix", 0.8, {"doc_type": "incident"}),
            DocumentSnippet("Concept Doc", "/3", "Overview", 0.7, {"doc_type": "concept"}),
        ]
        self.base_retriever = FakeRetriever(self.snippets)
        self.intent_retriever = IntentBasedRetriever(self.base_retriever)
    
    def test_retrieve_by_how_to_intent(self):
        """Test retrieval with how_to intent."""
        results = self.intent_retriever.retrieve_by_intent(
            query="configure system",
            intent=IntentType.HOW_TO,
            top_k=5
        )
        
        # Should filter to procedure docs
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["doc_type"], "procedure")
        
        # Check that filter was applied
        self.assertEqual(self.base_retriever.last_filters["doc_type"], "procedure")
    
    def test_retrieve_by_bug_resolution_intent(self):
        """Test retrieval with bug_resolution intent."""
        results = self.intent_retriever.retrieve_by_intent(
            query="fix error",
            intent=IntentType.BUG_RESOLUTION,
            top_k=5
        )
        
        # Should filter to incident docs
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["doc_type"], "incident")
    
    def test_retrieve_by_tool_explanation_intent(self):
        """Test retrieval with tool_explanation intent."""
        results = self.intent_retriever.retrieve_by_intent(
            query="what is",
            intent=IntentType.TOOL_EXPLANATION,
            top_k=5
        )
        
        # Should filter to concept docs
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["doc_type"], "concept")
    
    def test_retrieve_with_no_doc_type_filter(self):
        """Test retrieval for intents without doc_type filter."""
        results = self.intent_retriever.retrieve_by_intent(
            query="hello",
            intent=IntentType.CHITCHAT,
            top_k=5
        )
        
        # Should not apply doc_type filter
        self.assertEqual(len(results), 3)  # All docs returned
    
    def test_retrieve_with_additional_filters(self):
        """Test retrieval with additional custom filters."""
        results = self.intent_retriever.retrieve_by_intent(
            query="test",
            intent=IntentType.HOW_TO,
            top_k=5,
            additional_filters={"version": "2.0"}
        )
        
        # Should combine intent filter with additional filters
        filters = self.base_retriever.last_filters
        self.assertEqual(filters["doc_type"], "procedure")
        self.assertEqual(filters["version"], "2.0")


if __name__ == "__main__":
    unittest.main()
