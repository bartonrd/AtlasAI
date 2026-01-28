"""
Unit tests for Process Query (integration).
"""

import unittest
from atlasai_runtime.process_query import process_query, ProcessQueryResult
from atlasai_runtime.retriever import FakeRetriever, DocumentSnippet
from atlasai_runtime.config import ChatbotConfig
from atlasai_runtime.telemetry import TelemetryCollector


class TestProcessQuery(unittest.TestCase):
    """Test cases for process_query integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock retriever with diverse content
        self.snippets = [
            DocumentSnippet(
                "Installation Guide",
                "/docs/install",
                "Step 1: Download package. Step 2: Run installer. Step 3: Configure.",
                0.9,
                {"doc_type": "procedure"}
            ),
            DocumentSnippet(
                "Error 404 Resolution",
                "/docs/errors/404",
                "Error 404 occurs when resource not found. Solution: Check URL path.",
                0.85,
                {"doc_type": "incident"}
            ),
            DocumentSnippet(
                "API Overview",
                "/docs/api",
                "The API provides programmatic access to system features.",
                0.8,
                {"doc_type": "concept"}
            ),
        ]
        self.retriever = FakeRetriever(self.snippets)
        self.config = ChatbotConfig()
        self.telemetry = TelemetryCollector()
    
    def test_how_to_query_with_answer(self):
        """Test how-to query that produces an answer."""
        result = process_query(
            user_query="How do I install the application?",
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        self.assertIsInstance(result, ProcessQueryResult)
        self.assertEqual(result.intent, "how_to")
        self.assertGreater(result.confidence, 0.5)
        self.assertIsNone(result.question)  # Should provide answer, not question
        self.assertGreater(len(result.answer), 0)
        self.assertGreater(len(result.citations), 0)
        
        # Check telemetry
        events = self.telemetry.get_events()
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].had_clarification)
    
    def test_bug_resolution_query_with_answer(self):
        """Test bug resolution query."""
        result = process_query(
            user_query="I'm getting error 404 when accessing the page",
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        self.assertEqual(result.intent, "bug_resolution")
        self.assertGreaterEqual(result.confidence, 0.90)  # Rule-based detection should be high
        self.assertIsNone(result.question)
        self.assertGreater(len(result.answer), 0)
        self.assertIn("**Problem:**", result.answer)
    
    def test_tool_explanation_query(self):
        """Test tool explanation query."""
        result = process_query(
            user_query="What is the API module?",  # More specific query
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        self.assertEqual(result.intent, "tool_explanation")
        self.assertGreater(result.confidence, 0.4)
        # Check if we got answer or clarification
        if result.question:
            # If it asks clarification, answer should be empty
            self.assertEqual(result.answer, "")
        else:
            self.assertGreater(len(result.answer), 0)
            self.assertIn("**Overview:**", result.answer)
    
    def test_low_confidence_triggers_clarification(self):
        """Test that low confidence triggers clarification."""
        result = process_query(
            user_query="tell me",  # Very vague query
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        self.assertLess(result.confidence, 0.55)
        self.assertIsNotNone(result.question)
        self.assertEqual(result.answer, "")
        
        # Check telemetry
        events = self.telemetry.get_events()
        self.assertTrue(events[0].had_clarification)
    
    def test_weak_retrieval_triggers_clarification(self):
        """Test that weak retrieval triggers clarification."""
        # Use retriever with low-scoring snippets
        low_score_snippets = [
            DocumentSnippet("Doc", "/doc", "Content", 0.1, {})
        ]
        weak_retriever = FakeRetriever(low_score_snippets)
        
        result = process_query(
            user_query="How do I configure the system?",
            retriever=weak_retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        self.assertIsNotNone(result.question)
        self.assertEqual(result.answer, "")
    
    def test_chitchat_query(self):
        """Test chitchat query."""
        result = process_query(
            user_query="Hello!",
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        self.assertEqual(result.intent, "chitchat")
        self.assertGreater(len(result.answer), 0)
        self.assertIn("help", result.answer.lower())
    
    def test_custom_config(self):
        """Test using custom configuration."""
        custom_config = ChatbotConfig(
            confidence_threshold=0.7,
            top_k=3,
            min_retrieval_score=0.4
        )
        
        result = process_query(
            user_query="How do I install?",
            retriever=self.retriever,
            config=custom_config,
            telemetry_collector=self.telemetry
        )
        
        self.assertIsInstance(result, ProcessQueryResult)
    
    def test_telemetry_collection(self):
        """Test that telemetry is collected properly."""
        telemetry = TelemetryCollector()
        
        # Process multiple queries
        process_query("How do I install?", self.retriever, self.config, telemetry)
        process_query("What is API?", self.retriever, self.config, telemetry)
        process_query("error 404", self.retriever, self.config, telemetry)
        
        events = telemetry.get_events()
        self.assertEqual(len(events), 3)
        
        # Check stats
        stats = telemetry.get_summary_stats()
        self.assertEqual(stats["total_queries"], 3)
        self.assertGreater(stats["avg_elapsed_ms"], 0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = process_query(
            user_query="How do I configure?",
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        result_dict = result.to_dict()
        
        self.assertIn("answer", result_dict)
        self.assertIn("intent", result_dict)
        self.assertIn("confidence", result_dict)
        self.assertIn("telemetry", result_dict)
    
    def test_intent_based_filtering(self):
        """Test that intent-based filtering is applied."""
        result = process_query(
            user_query="How do I install?",
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        # Check that retriever was called with doc_type filter
        self.assertEqual(self.retriever.last_filters["doc_type"], "procedure")
    
    def test_query_rewriting(self):
        """Test that query rewriting occurs."""
        result = process_query(
            user_query="How to configure API?",
            retriever=self.retriever,
            config=self.config,
            telemetry_collector=self.telemetry
        )
        
        # Check telemetry for rewritten query
        events = self.telemetry.get_events()
        self.assertGreater(len(events[0].rewritten_query), 0)
        self.assertNotEqual(events[0].rewritten_query, events[0].user_query)


if __name__ == "__main__":
    unittest.main()
