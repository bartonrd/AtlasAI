"""
Unit tests for Answer Synthesizer.
"""

import unittest
from atlasai_runtime.answer_synthesizer import AnswerSynthesizer, AnswerSynthesisResult
from atlasai_runtime.retriever import DocumentSnippet
from atlasai_runtime.intent_classifier import IntentType


class TestAnswerSynthesizer(unittest.TestCase):
    """Test cases for AnswerSynthesizer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.synthesizer = AnswerSynthesizer(
            min_retrieval_score=0.25,
            max_answer_tokens=500
        )
    
    def test_weak_retrieval_triggers_clarification(self):
        """Test that weak retrieval triggers clarification."""
        # Snippets with low scores
        snippets = [
            DocumentSnippet("Doc", "/1", "Content", 0.1, {})
        ]
        
        result = self.synthesizer.synthesize(
            query="test query",
            intent=IntentType.HOW_TO,
            snippets=snippets
        )
        
        self.assertTrue(result.should_ask_clarification)
        self.assertIsNotNone(result.clarification_question)
        self.assertEqual(result.answer, "")
    
    def test_no_snippets_triggers_clarification(self):
        """Test that empty retrieval triggers clarification."""
        result = self.synthesizer.synthesize(
            query="test query",
            intent=IntentType.HOW_TO,
            snippets=[]
        )
        
        self.assertTrue(result.should_ask_clarification)
        self.assertIsNotNone(result.clarification_question)
    
    def test_how_to_synthesis(self):
        """Test how-to answer synthesis."""
        snippets = [
            DocumentSnippet(
                "Installation Guide",
                "/docs/install",
                "1. Download the package. 2. Run installer. 3. Configure settings.",
                0.85,
                {}
            )
        ]
        
        result = self.synthesizer.synthesize(
            query="How do I install?",
            intent=IntentType.HOW_TO,
            snippets=snippets
        )
        
        self.assertFalse(result.should_ask_clarification)
        self.assertIn("**", result.answer)  # Markdown formatting
        self.assertTrue(len(result.citations) > 0)
        self.assertEqual(result.citations[0]["title"], "Installation Guide")
    
    def test_bug_resolution_synthesis(self):
        """Test bug resolution answer synthesis."""
        snippets = [
            DocumentSnippet(
                "Error 404 Fix",
                "/docs/errors",
                "This error occurs when resource not found. To resolve, check the URL path.",
                0.8,
                {}
            )
        ]
        
        result = self.synthesizer.synthesize(
            query="Getting error 404",
            intent=IntentType.BUG_RESOLUTION,
            snippets=snippets
        )
        
        self.assertFalse(result.should_ask_clarification)
        self.assertIn("**Problem:**", result.answer)
        self.assertIn("**Resolution:**", result.answer)
        self.assertTrue(len(result.citations) > 0)
    
    def test_tool_explanation_synthesis(self):
        """Test tool explanation answer synthesis."""
        snippets = [
            DocumentSnippet(
                "API Overview",
                "/docs/api",
                "The API provides programmatic access. Use cases include automation and integration.",
                0.9,
                {}
            )
        ]
        
        result = self.synthesizer.synthesize(
            query="What is the API?",
            intent=IntentType.TOOL_EXPLANATION,
            snippets=snippets
        )
        
        self.assertFalse(result.should_ask_clarification)
        self.assertIn("**Overview:**", result.answer)
        self.assertTrue(len(result.citations) > 0)
    
    def test_chitchat_synthesis(self):
        """Test chitchat answer synthesis."""
        result = self.synthesizer.synthesize(
            query="Hello!",
            intent=IntentType.CHITCHAT,
            snippets=[]  # No snippets needed for chitchat
        )
        
        # Chitchat should not trigger clarification
        self.assertFalse(result.should_ask_clarification)
        self.assertIn("help", result.answer.lower())
    
    def test_citation_extraction(self):
        """Test citation extraction from multiple snippets."""
        snippets = [
            DocumentSnippet("Doc1", "/1", "Content 1", 0.9, {}),
            DocumentSnippet("Doc2", "/2", "Content 2", 0.8, {}),
            DocumentSnippet("Doc3", "/3", "Content 3", 0.7, {}),
            DocumentSnippet("Doc1", "/1", "More content", 0.6, {}),  # Duplicate
        ]
        
        result = self.synthesizer.synthesize(
            query="test",
            intent=IntentType.OTHER,
            snippets=snippets
        )
        
        # Should extract unique citations (max 3)
        self.assertEqual(len(result.citations), 3)
        titles = [c["title"] for c in result.citations]
        self.assertEqual(len(set(titles)), 3)  # All unique
    
    def test_pre_generated_clarification(self):
        """Test using pre-generated clarification question."""
        snippets = [
            DocumentSnippet("Doc", "/1", "Content", 0.1, {})  # Low score
        ]
        
        custom_question = "What specific version are you using?"
        
        result = self.synthesizer.synthesize(
            query="test",
            intent=IntentType.BUG_RESOLUTION,
            snippets=snippets,
            clarification_question=custom_question
        )
        
        self.assertTrue(result.should_ask_clarification)
        self.assertEqual(result.clarification_question, custom_question)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = AnswerSynthesisResult(
            answer="Test answer",
            citations=[{"title": "Doc", "url": "/doc"}],
            should_ask_clarification=False,
            clarification_question=None
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["answer"], "Test answer")
        self.assertEqual(len(result_dict["citations"]), 1)
        self.assertFalse(result_dict["should_ask_clarification"])


if __name__ == "__main__":
    unittest.main()
