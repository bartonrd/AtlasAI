"""
Unit tests for Query Rewriter.
"""

import unittest
from atlasai_runtime.query_rewriter import QueryRewriter, QueryRewriteResult
from atlasai_runtime.intent_classifier import IntentType


class TestQueryRewriter(unittest.TestCase):
    """Test cases for QueryRewriter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()
    
    def test_empty_query(self):
        """Test rewriting of empty query."""
        result = self.rewriter.rewrite("", IntentType.HOW_TO)
        self.assertEqual(result.rewritten_query, "")
        self.assertEqual(result.entities, [])
        self.assertEqual(result.constraints, [])
    
    def test_how_to_rewrite(self):
        """Test how-to query rewriting."""
        query = "How do I configure ADMS?"
        result = self.rewriter.rewrite(query, IntentType.HOW_TO)
        
        # Should expand with action verbs and procedure terms
        self.assertIn("procedure", result.rewritten_query.lower())
        self.assertIn("configure", result.rewritten_query.lower())
        
        # Should extract ADMS as entity
        self.assertIn("ADMS", result.entities)
    
    def test_bug_resolution_rewrite_with_error_code(self):
        """Test bug resolution query with error code."""
        query = "Getting error 404 in Windows 10 version 2.1.0"
        result = self.rewriter.rewrite(query, IntentType.BUG_RESOLUTION)
        
        # Should add diagnostic terms
        self.assertIn("troubleshoot", result.rewritten_query.lower())
        
        # Should extract constraints
        error_constraints = [c for c in result.constraints if c.startswith("error:")]
        self.assertTrue(len(error_constraints) > 0)
        
        version_constraints = [c for c in result.constraints if c.startswith("version:")]
        self.assertTrue(len(version_constraints) > 0)
        
        os_constraints = [c for c in result.constraints if c.startswith("os:")]
        self.assertTrue(len(os_constraints) > 0)
    
    def test_bug_resolution_with_exception(self):
        """Test bug resolution with exception."""
        query = "NullPointerException in API module"
        result = self.rewriter.rewrite(query, IntentType.BUG_RESOLUTION)
        
        # Should extract exception as entity
        self.assertIn("NullPointerException", result.entities)
        self.assertIn("API", result.entities)
    
    def test_tool_explanation_rewrite(self):
        """Test tool explanation query rewriting."""
        query = "What is the AuthManager component?"
        result = self.rewriter.rewrite(query, IntentType.TOOL_EXPLANATION)
        
        # Should add conceptual terms
        self.assertIn("overview", result.rewritten_query.lower())
        self.assertIn("concept", result.rewritten_query.lower())
        
        # Should extract entity
        self.assertIn("AuthManager", result.entities)
    
    def test_entity_extraction(self):
        """Test entity extraction."""
        queries = [
            ("Using ADMS API", ["ADMS", "API"]),
            ("Configure ModelManager", ["ModelManager"]),
            ("HTTP GET request", ["HTTP", "GET"])
        ]
        
        for query, expected_entities in queries:
            result = self.rewriter.rewrite(query, IntentType.OTHER)
            for entity in expected_entities:
                self.assertIn(entity, result.entities, f"Missing {entity} in {query}")
    
    def test_version_extraction(self):
        """Test version constraint extraction."""
        queries = [
            "Using version 2.1.0",
            "Running v3.5",
            "On version 10"
        ]
        
        for query in queries:
            result = self.rewriter.rewrite(query, IntentType.OTHER)
            version_constraints = [c for c in result.constraints if c.startswith("version:")]
            self.assertTrue(len(version_constraints) > 0, f"No version in {query}")
    
    def test_os_extraction(self):
        """Test OS constraint extraction."""
        queries = [
            "Running on Windows",
            "Linux environment",
            "macOS system"
        ]
        
        for query in queries:
            result = self.rewriter.rewrite(query, IntentType.OTHER)
            os_constraints = [c for c in result.constraints if c.startswith("os:")]
            self.assertTrue(len(os_constraints) > 0, f"No OS in {query}")
    
    def test_abbreviation_expansion(self):
        """Test abbreviation expansion."""
        query = "How to use the API with CLI?"
        result = self.rewriter.rewrite(query, IntentType.OTHER)
        
        # Should expand abbreviations
        expanded = result.rewritten_query
        self.assertIn("application programming interface", expanded.lower())
        self.assertIn("command line interface", expanded.lower())
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = QueryRewriteResult(
            rewritten_query="expanded query",
            entities=["Entity1", "Entity2"],
            constraints=["version:1.0", "os:linux"]
        )
        
        result_dict = result.to_dict()
        self.assertEqual(result_dict["rewritten_query"], "expanded query")
        self.assertEqual(len(result_dict["entities"]), 2)
        self.assertEqual(len(result_dict["constraints"]), 2)


if __name__ == "__main__":
    unittest.main()
