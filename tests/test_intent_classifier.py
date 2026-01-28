"""
Unit tests for Intent Classifier.
"""

import unittest
from atlasai_runtime.intent_classifier import (
    IntentClassifier,
    IntentType,
    IntentClassificationResult
)


class TestIntentClassifier(unittest.TestCase):
    """Test cases for IntentClassifier."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = IntentClassifier(confidence_threshold=0.55)
    
    def test_empty_query(self):
        """Test classification of empty query."""
        result = self.classifier.classify("")
        self.assertEqual(result.intent, IntentType.OTHER)
        self.assertEqual(result.confidence, 1.0)
    
    def test_how_to_intent(self):
        """Test classification of how-to queries."""
        queries = [
            "How do I configure the database?",
            "Steps to install the application",
            "Guide for setting up authentication"
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result.intent, IntentType.HOW_TO, f"Failed for: {query}")
            self.assertGreater(result.confidence, 0.4)
    
    def test_bug_resolution_intent_via_rules(self):
        """Test bug resolution with rule-based detection."""
        queries = [
            "I got a NullPointerException in the API",
            "Error code 500 when calling the service",
            "Application crashed with stack trace",
            "HTTP 404 error not working"
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result.intent, IntentType.BUG_RESOLUTION, f"Failed for: {query}")
            self.assertEqual(result.confidence, 0.95)  # Rule-based = high confidence
            self.assertIn("Rule-based", result.rationale)
    
    def test_bug_resolution_intent_via_keywords(self):
        """Test bug resolution with keyword matching."""
        queries = [
            "The feature is not working properly",
            "I found a bug in the system",
            "Need to fix this issue"
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result.intent, IntentType.BUG_RESOLUTION, f"Failed for: {query}")
            self.assertGreater(result.confidence, 0.4)
    
    def test_tool_explanation_intent(self):
        """Test classification of explanation queries."""
        queries = [
            "What is the authentication module?",
            "Explain the database connector",
            "What does the API gateway do?"
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result.intent, IntentType.TOOL_EXPLANATION, f"Failed for: {query}")
            self.assertGreater(result.confidence, 0.4)
    
    def test_escalate_intent(self):
        """Test classification of escalation queries."""
        queries = [
            "I need to speak to support",
            "Need help from a human",
            "This is urgent, escalate please"
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result.intent, IntentType.ESCALATE_OR_TICKET, f"Failed for: {query}")
            self.assertGreater(result.confidence, 0.4)
    
    def test_chitchat_intent(self):
        """Test classification of chitchat queries."""
        queries = [
            "Hello there!",
            "Thank you for your help",
            "Goodbye"
        ]
        
        for query in queries:
            result = self.classifier.classify(query)
            self.assertEqual(result.intent, IntentType.CHITCHAT, f"Failed for: {query}")
            self.assertGreater(result.confidence, 0.4)
    
    def test_low_confidence_query(self):
        """Test query with unclear intent."""
        result = self.classifier.classify("stuff things")
        self.assertLess(result.confidence, 0.55)
    
    def test_needs_clarification(self):
        """Test clarification detection."""
        # High confidence - no clarification
        high_conf = IntentClassificationResult(IntentType.HOW_TO, 0.8, "test")
        self.assertFalse(self.classifier.needs_clarification(high_conf))
        
        # Low confidence - needs clarification
        low_conf = IntentClassificationResult(IntentType.OTHER, 0.3, "test")
        self.assertTrue(self.classifier.needs_clarification(low_conf))
    
    def test_confidence_clamping(self):
        """Test that confidence is clamped to [0, 1]."""
        result = IntentClassificationResult(IntentType.HOW_TO, 1.5, "test")
        self.assertEqual(result.confidence, 1.0)
        
        result = IntentClassificationResult(IntentType.HOW_TO, -0.5, "test")
        self.assertEqual(result.confidence, 0.0)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = IntentClassificationResult(IntentType.HOW_TO, 0.75, "Test rationale")
        result_dict = result.to_dict()
        
        self.assertEqual(result_dict["intent"], "how_to")
        self.assertEqual(result_dict["confidence"], 0.75)
        self.assertEqual(result_dict["rationale"], "Test rationale")


if __name__ == "__main__":
    unittest.main()
