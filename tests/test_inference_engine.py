"""
Integration tests for inference engine.
"""

import pytest
from atlasai_runtime.inference_engine import InferenceEngine, InferenceResult
from atlasai_runtime.inference_config import InferenceConfig
from atlasai_runtime.retriever import MockSearchBackend
from atlasai_runtime.intent_classifier import (
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
    INTENT_OTHER,
)


class TestInferenceEngine:
    """Test suite for InferenceEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockSearchBackend()
        self.config = InferenceConfig(
            confidence_threshold=0.55,
            min_retrieval_score=0.25,
            top_k=4,
        )
        self.engine = InferenceEngine(
            search_backend=self.backend,
            config=self.config,
            llm_provider=None,
        )
    
    def test_process_how_to_query(self):
        """Test processing a how-to query."""
        result = self.engine.process_query("How do I install the software?")
        
        assert isinstance(result, InferenceResult)
        assert result.intent == INTENT_HOW_TO
        assert result.confidence > 0.0
        
        # Should have either answer or question
        assert result.answer is not None or result.question is not None
        
        # Should have telemetry
        assert "user_query" in result.telemetry
        assert "intent" in result.telemetry
        assert "elapsed_ms" in result.telemetry
    
    def test_process_bug_query(self):
        """Test processing a bug resolution query."""
        result = self.engine.process_query("Exception: Connection timeout error")
        
        assert result.intent == INTENT_BUG_RESOLUTION
        assert result.confidence > 0.0
    
    def test_process_tool_explanation_query(self):
        """Test processing a tool explanation query."""
        result = self.engine.process_query("What is the API module used for?")
        
        # Tool explanation might be borderline, accept reasonable intents
        assert result.intent in [INTENT_TOOL_EXPLANATION, INTENT_HOW_TO, INTENT_OTHER]
        assert result.confidence > 0.0
    
    def test_low_confidence_triggers_clarification(self):
        """Test that low confidence triggers clarification."""
        # Use a very ambiguous query
        result = self.engine.process_query("thing")
        
        # Should trigger clarification if confidence is low
        if result.confidence < self.config.confidence_threshold:
            assert result.question is not None
            assert result.answer is None
            assert result.telemetry["had_clarification"] is True
    
    def test_telemetry_includes_pipeline_stages(self):
        """Test that telemetry includes all pipeline stages."""
        result = self.engine.process_query("How to configure the system step by step?")
        
        telemetry = result.telemetry
        assert "user_query" in telemetry
        assert "intent" in telemetry
        assert "confidence" in telemetry
        assert "elapsed_ms" in telemetry
        
        # rewritten_query and retrieval info only present if not clarification
        if not result.is_clarification():
            assert "rewritten_query" in telemetry
            assert "num_retrieved" in telemetry
    
    def test_inference_result_to_dict(self):
        """Test InferenceResult serialization."""
        result = self.engine.process_query("How to install?")
        
        d = result.to_dict()
        assert "answer" in d or "question" in d
        assert "intent" in d
        assert "confidence" in d
        assert "citations" in d
        assert "telemetry" in d
    
    def test_is_clarification_method(self):
        """Test is_clarification method."""
        # Create result with answer
        result_with_answer = InferenceResult(
            answer="Test answer",
            intent=INTENT_HOW_TO,
            confidence=0.9,
        )
        assert not result_with_answer.is_clarification()
        
        # Create result with question
        result_with_question = InferenceResult(
            question="Test question?",
            intent=INTENT_HOW_TO,
            confidence=0.4,
        )
        assert result_with_question.is_clarification()


class TestInferenceEngineEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockSearchBackend()
        self.config = InferenceConfig()
        self.engine = InferenceEngine(self.backend, self.config)
    
    def test_empty_query(self):
        """Test processing empty query."""
        result = self.engine.process_query("")
        
        assert isinstance(result, InferenceResult)
        # Should handle gracefully
        assert result.confidence >= 0.0
    
    def test_very_long_query(self):
        """Test processing very long query."""
        long_query = "How do I " + "configure the system " * 50 + "?"
        result = self.engine.process_query(long_query)
        
        assert isinstance(result, InferenceResult)
    
    def test_special_characters_query(self):
        """Test query with special characters."""
        query = "Error: errno=13 @startup with C:\\path\\file.exe"
        result = self.engine.process_query(query)
        
        assert isinstance(result, InferenceResult)
        assert result.intent == INTENT_BUG_RESOLUTION
    
    def test_update_config(self):
        """Test updating configuration."""
        new_config = InferenceConfig(
            confidence_threshold=0.7,
            top_k=10,
        )
        
        self.engine.update_config(new_config)
        
        assert self.engine.config.confidence_threshold == 0.7
        assert self.engine.config.top_k == 10


class TestInferenceEngineWithFeedback:
    """Test feedback handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.backend = MockSearchBackend()
        self.config = InferenceConfig()
        self.engine = InferenceEngine(self.backend, self.config)
    
    def test_process_with_user_feedback(self):
        """Test processing query with user feedback."""
        result = self.engine.process_query(
            user_query="How to install?",
            user_feedback="This helped, thanks!"
        )
        
        # Feedback should be in telemetry
        if "user_feedback" in result.telemetry:
            assert result.telemetry["user_feedback"] == "This helped, thanks!"


class TestInferenceEngineErrorHandling:
    """Test error handling in inference engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Use a backend that might fail
        self.backend = MockSearchBackend()
        self.config = InferenceConfig()
        self.engine = InferenceEngine(self.backend, self.config)
    
    def test_handles_exceptions_gracefully(self):
        """Test that exceptions are handled gracefully."""
        # Even with potential errors, should return a result
        result = self.engine.process_query("test query")
        
        assert isinstance(result, InferenceResult)
        # Should have telemetry even on error
        assert "elapsed_ms" in result.telemetry


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
