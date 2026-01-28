"""
Unit tests for answer synthesizer module.
"""

import pytest
from atlasai_runtime.answer_synthesizer import AnswerSynthesizer, SynthesizedAnswer
from atlasai_runtime.retriever import RetrievedDoc
from atlasai_runtime.intent_classifier import (
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
)


class TestAnswerSynthesizer:
    """Test suite for AnswerSynthesizer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.synthesizer = AnswerSynthesizer(llm_provider=None)
        
        # Sample retrieved documents
        self.sample_docs = [
            RetrievedDoc(
                title="Installation Guide",
                url="/docs/install.pdf",
                content="To install the software, download the installer and run setup.exe. Follow the on-screen instructions.",
                score=0.9,
                metadata={"doc_type": "procedure", "page": "1"}
            ),
            RetrievedDoc(
                title="Configuration Guide",
                url="/docs/config.pdf",
                content="Configure the system by editing config.yml. Set the port to 8080 and enable SSL.",
                score=0.85,
                metadata={"doc_type": "procedure", "page": "5"}
            ),
            RetrievedDoc(
                title="Troubleshooting",
                url="/docs/troubleshoot.pdf",
                content="If you encounter errors, check the logs. Common issues include permission denied.",
                score=0.8,
                metadata={"doc_type": "incident", "page": "10"}
            ),
        ]
    
    def test_synthesize_with_docs(self):
        """Test synthesis with retrieved documents."""
        result = self.synthesizer.synthesize(
            user_query="How to install?",
            intent=INTENT_HOW_TO,
            retrieved_docs=self.sample_docs,
        )
        
        assert isinstance(result, SynthesizedAnswer)
        assert result.answer is not None
        assert len(result.answer) > 0
        assert len(result.citations) > 0
        assert result.intent_formatting_applied is True
    
    def test_synthesize_no_docs(self):
        """Test synthesis with no documents."""
        result = self.synthesizer.synthesize(
            user_query="How to install?",
            intent=INTENT_HOW_TO,
            retrieved_docs=[],
        )
        
        assert isinstance(result, SynthesizedAnswer)
        assert "couldn't find" in result.answer.lower()
        assert len(result.citations) == 0
    
    def test_citations_include_top_sources(self):
        """Test that citations include top 2-3 sources."""
        result = self.synthesizer.synthesize(
            user_query="How to install?",
            intent=INTENT_HOW_TO,
            retrieved_docs=self.sample_docs,
        )
        
        # Should have up to 3 citations
        assert 1 <= len(result.citations) <= 3
        
        # Each citation should have required fields
        for citation in result.citations:
            assert "index" in citation
            assert "title" in citation
            assert "url" in citation
    
    def test_how_to_formatting(self):
        """Test how-to specific formatting."""
        result = self.synthesizer.synthesize(
            user_query="How to install?",
            intent=INTENT_HOW_TO,
            retrieved_docs=self.sample_docs,
        )
        
        # Answer should have some structure (bullets or numbers)
        assert "-" in result.answer or any(c.isdigit() for c in result.answer)
    
    def test_bug_resolution_formatting(self):
        """Test bug resolution specific formatting."""
        result = self.synthesizer.synthesize(
            user_query="Error: permission denied",
            intent=INTENT_BUG_RESOLUTION,
            retrieved_docs=self.sample_docs,
        )
        
        # Answer should have some structure
        assert len(result.answer) > 0
    
    def test_tool_explanation_formatting(self):
        """Test tool explanation specific formatting."""
        result = self.synthesizer.synthesize(
            user_query="What is the API?",
            intent=INTENT_TOOL_EXPLANATION,
            retrieved_docs=self.sample_docs,
        )
        
        # Answer should have some structure
        assert len(result.answer) > 0
    
    def test_synthesized_answer_to_dict(self):
        """Test SynthesizedAnswer serialization."""
        answer = SynthesizedAnswer(
            answer="Test answer",
            citations=[{"index": "1", "title": "Doc", "url": "/doc"}],
            intent_formatting_applied=True,
        )
        
        d = answer.to_dict()
        assert d["answer"] == "Test answer"
        assert len(d["citations"]) == 1
        assert d["intent_formatting_applied"] is True


class TestAnswerSynthesizerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.synthesizer = AnswerSynthesizer()
    
    def test_single_doc(self):
        """Test synthesis with single document."""
        doc = RetrievedDoc(
            title="Guide",
            url="/guide",
            content="Simple guide content.",
            score=0.9,
            metadata={}
        )
        
        result = self.synthesizer.synthesize(
            user_query="How to?",
            intent=INTENT_HOW_TO,
            retrieved_docs=[doc],
        )
        
        assert len(result.citations) == 1
    
    def test_very_short_content(self):
        """Test synthesis with very short content."""
        doc = RetrievedDoc(
            title="Short",
            url="/short",
            content="Brief.",
            score=0.5,
            metadata={}
        )
        
        result = self.synthesizer.synthesize(
            user_query="What?",
            intent=INTENT_TOOL_EXPLANATION,
            retrieved_docs=[doc],
        )
        
        # Should still produce an answer
        assert len(result.answer) > 0
    
    def test_very_long_content(self):
        """Test synthesis with very long content."""
        long_content = "This is a very long document. " * 100
        doc = RetrievedDoc(
            title="Long",
            url="/long",
            content=long_content,
            score=0.9,
            metadata={}
        )
        
        result = self.synthesizer.synthesize(
            user_query="Explain this",
            intent=INTENT_TOOL_EXPLANATION,
            retrieved_docs=[doc],
            max_tokens=100,  # Limit tokens
        )
        
        # Should still produce an answer
        assert len(result.answer) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
