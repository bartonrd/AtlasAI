"""
Unit tests for Clarifying Question Generator.
"""

import unittest
from atlasai_runtime.clarifying_question_generator import ClarifyingQuestionGenerator
from atlasai_runtime.intent_classifier import IntentType


class TestClarifyingQuestionGenerator(unittest.TestCase):
    """Test cases for ClarifyingQuestionGenerator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = ClarifyingQuestionGenerator()
    
    def test_high_confidence_with_entities_no_question(self):
        """Test that high confidence with entities doesn't generate question."""
        result = self.generator.generate(
            query="How do I configure ADMS?",
            intent=IntentType.HOW_TO,
            confidence=0.85,
            entities=["ADMS"],
            constraints=[]
        )
        
        self.assertIsNone(result)
    
    def test_low_confidence_generates_question(self):
        """Test that low confidence generates clarifying question."""
        result = self.generator.generate(
            query="configure something",
            intent=IntentType.HOW_TO,
            confidence=0.4,
            entities=[],
            constraints=[]
        )
        
        self.assertIsNotNone(result)
        self.assertIn("?", result)
    
    def test_how_to_missing_component(self):
        """Test how-to with missing component."""
        result = self.generator.generate(
            query="how do I configure",
            intent=IntentType.HOW_TO,
            confidence=0.45,
            entities=[],
            constraints=[]
        )
        
        self.assertIsNotNone(result)
        self.assertIn("component", result.lower())
    
    def test_bug_resolution_missing_error_code(self):
        """Test bug resolution with missing error code."""
        result = self.generator.generate(
            query="something is broken",
            intent=IntentType.BUG_RESOLUTION,
            confidence=0.4,
            entities=[],
            constraints=[]
        )
        
        self.assertIsNotNone(result)
        # Should ask for error details
        self.assertTrue(
            "error" in result.lower() or "message" in result.lower()
        )
    
    def test_bug_resolution_missing_version(self):
        """Test bug resolution with missing version."""
        result = self.generator.generate(
            query="I got an error exception",
            intent=IntentType.BUG_RESOLUTION,
            confidence=0.45,
            entities=[],
            constraints=[]  # No version constraint
        )
        
        self.assertIsNotNone(result)
    
    def test_tool_explanation_missing_tool_name(self):
        """Test tool explanation with missing tool name."""
        result = self.generator.generate(
            query="what is this",
            intent=IntentType.TOOL_EXPLANATION,
            confidence=0.4,
            entities=[],
            constraints=[]
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(
            "tool" in result.lower() or "feature" in result.lower()
        )
    
    def test_chitchat_no_question(self):
        """Test that chitchat with high confidence doesn't need clarification."""
        result = self.generator.generate(
            query="hello",
            intent=IntentType.CHITCHAT,
            confidence=0.8,
            entities=[],
            constraints=[]
        )
        
        self.assertIsNone(result)
    
    def test_slots_filled_no_question(self):
        """Test that filled slots don't generate questions."""
        result = self.generator.generate(
            query="How do I configure ADMS using CLI?",
            intent=IntentType.HOW_TO,
            confidence=0.7,
            entities=["ADMS"],
            constraints=[]
        )
        
        self.assertIsNone(result)
    
    def test_version_constraint_recognized(self):
        """Test that version constraint is recognized as filled slot."""
        # With version constraint, should have fewer missing slots
        result_with_version = self.generator.generate(
            query="error in version 2.0",
            intent=IntentType.BUG_RESOLUTION,
            confidence=0.45,
            entities=[],
            constraints=["version:2.0"]
        )
        
        # Version slot is filled, so question should be about other slots
        self.assertIsNotNone(result_with_version)
    
    def test_os_constraint_recognized(self):
        """Test that OS constraint is recognized as filled slot."""
        result = self.generator.generate(
            query="error on Windows",
            intent=IntentType.BUG_RESOLUTION,
            confidence=0.45,
            entities=[],
            constraints=["os:windows"]
        )
        
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
