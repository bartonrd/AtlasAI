"""
Unit tests for query rewriter module.
"""

import pytest
from atlasai_runtime.query_rewriter import QueryRewriter, RewrittenQuery
from atlasai_runtime.intent_classifier import (
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
    INTENT_OTHER,
)


class TestQueryRewriter:
    """Test suite for QueryRewriter."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()
    
    def test_how_to_rewriting(self):
        """Test rewriting of how-to queries."""
        query = "How to install the module"
        result = self.rewriter.rewrite(query, INTENT_HOW_TO)
        
        assert isinstance(result, RewrittenQuery)
        # Should expand verbs
        assert "install" in result.rewritten_query.lower()
        # Should add procedural keywords
        assert any(kw in result.rewritten_query.lower() for kw in ["step", "procedure"])
    
    def test_bug_resolution_rewriting(self):
        """Test rewriting of bug resolution queries."""
        query = "Exception: NullPointer at line 42 in v1.2.3"
        result = self.rewriter.rewrite(query, INTENT_BUG_RESOLUTION)
        
        assert isinstance(result, RewrittenQuery)
        # Should extract entities
        assert len(result.entities) > 0
        # Should extract version
        assert any("1.2" in str(e) for e in result.constraints)
    
    def test_tool_explanation_rewriting(self):
        """Test rewriting of tool explanation queries."""
        query = "What is the API"
        result = self.rewriter.rewrite(query, INTENT_TOOL_EXPLANATION)
        
        assert isinstance(result, RewrittenQuery)
        # Should add synonyms
        assert "api" in result.rewritten_query.lower()
    
    def test_entity_extraction_camelcase(self):
        """Test extraction of CamelCase entities."""
        query = "What is ModelManager doing"
        result = self.rewriter.rewrite(query, INTENT_TOOL_EXPLANATION)
        
        assert "ModelManager" in result.entities
    
    def test_entity_extraction_caps(self):
        """Test extraction of ALL_CAPS entities."""
        query = "How does ADMS work"
        result = self.rewriter.rewrite(query, INTENT_TOOL_EXPLANATION)
        
        assert "ADMS" in result.entities
    
    def test_entity_extraction_quoted(self):
        """Test extraction of quoted entities."""
        query = 'What is the "API Gateway" module'
        result = self.rewriter.rewrite(query, INTENT_TOOL_EXPLANATION)
        
        assert "API Gateway" in result.entities
    
    def test_error_code_extraction(self):
        """Test extraction of error codes."""
        test_cases = [
            ("errno 13 permission denied", ["errno 13"]),
            ("exit code 1 occurred", ["exit code 1"]),
            ("Error: FileNotFound", ["FileNotFound"]),
        ]
        
        for query, expected_codes in test_cases:
            result = self.rewriter.rewrite(query, INTENT_BUG_RESOLUTION)
            # Check that at least one expected code is found
            found = any(any(code in entity for entity in result.entities) for code in expected_codes)
            assert found, f"Expected {expected_codes} in {result.entities}"
    
    def test_version_extraction(self):
        """Test extraction of version numbers."""
        test_cases = [
            "version 1.2.3 issue",
            "v2.0 bug",
            "running 3.14 release",
        ]
        
        for query in test_cases:
            result = self.rewriter.rewrite(query, INTENT_BUG_RESOLUTION)
            assert len(result.constraints) > 0
            assert any("v" in str(c) or "." in str(c) for c in result.constraints)
    
    def test_os_platform_extraction(self):
        """Test extraction of OS/platform information."""
        test_cases = [
            ("Error on Windows 10", ["windows"]),
            ("Bug in Linux environment", ["linux"]),
            ("Issue with macOS", ["macos"]),
        ]
        
        for query, expected_os in test_cases:
            result = self.rewriter.rewrite(query, INTENT_BUG_RESOLUTION)
            found = any(os in result.entities for os in expected_os)
            assert found, f"Expected {expected_os} in {result.entities}"
    
    def test_empty_query(self):
        """Test handling of empty queries."""
        result = self.rewriter.rewrite("", INTENT_OTHER)
        
        assert result.rewritten_query == ""
        assert result.entities == []
        assert result.constraints == []
    
    def test_rewritten_query_to_dict(self):
        """Test RewrittenQuery serialization."""
        result = RewrittenQuery(
            rewritten_query="test query",
            entities=["Entity1", "Entity2"],
            constraints=["v1.0"],
        )
        
        d = result.to_dict()
        assert d["rewritten_query"] == "test query"
        assert d["entities"] == ["Entity1", "Entity2"]
        assert d["constraints"] == ["v1.0"]


class TestQueryRewriterEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.rewriter = QueryRewriter()
    
    def test_no_expandable_verbs(self):
        """Test query with no expandable verbs."""
        query = "Where is the file?"
        result = self.rewriter.rewrite(query, INTENT_HOW_TO)
        
        # Should still add procedural keywords
        assert any(kw in result.rewritten_query.lower() for kw in ["step", "procedure"])
    
    def test_multiple_versions(self):
        """Test query with multiple version numbers."""
        query = "Upgrade from v1.0 to v2.5"
        result = self.rewriter.rewrite(query, INTENT_BUG_RESOLUTION)
        
        # Should extract both versions
        versions = [c for c in result.constraints if "v" in str(c)]
        assert len(versions) >= 2
    
    def test_complex_error_pattern(self):
        """Test complex error pattern extraction."""
        query = "Getting Exception: IOError with errno=13 on exit code 1"
        result = self.rewriter.rewrite(query, INTENT_BUG_RESOLUTION)
        
        # Should extract multiple error indicators
        assert len(result.entities) > 0 or len(result.constraints) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
