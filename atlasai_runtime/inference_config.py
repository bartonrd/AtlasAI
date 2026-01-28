"""
Configuration for the inference engine.

Provides configurable parameters for intent classification, retrieval, and answer synthesis.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class InferenceConfig:
    """
    Configuration for the inference engine.
    
    All thresholds and parameters can be overridden via environment variables
    or programmatically at runtime.
    """
    
    # Intent classification
    confidence_threshold: float = 0.55
    """Minimum confidence to proceed without clarification (0.0 to 1.0)"""
    
    # Retrieval
    top_k: int = 4
    """Number of document chunks to retrieve"""
    
    min_retrieval_score: float = 0.25
    """Minimum retrieval score to generate an answer (0.0 to 1.0)"""
    
    # Answer synthesis
    max_answer_tokens: int = 384
    """Maximum tokens in generated answer"""
    
    # Default document filters by intent
    default_filters: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "how_to": {"doc_type": "procedure"},
        "bug_resolution": {"doc_type": "incident"},
        "tool_explanation": {"doc_type": "concept"},
        "escalate_or_ticket": {},
        "chitchat": {},
        "other": {},
    })
    """Intent-specific document type filters"""
    
    # LLM provider configuration
    llm_provider: str = "huggingface"
    """LLM provider type (pluggable: huggingface, openai, anthropic, etc.)"""
    
    llm_model_path: Optional[str] = None
    """Path or identifier for the LLM model"""
    
    embedding_model_path: Optional[str] = None
    """Path or identifier for the embedding model"""
    
    # Feature toggle
    enable_inference: bool = True
    """Enable or disable the inference pipeline (for gradual rollout)"""
    
    # Bug detection rules
    bug_signal_keywords: list = field(default_factory=lambda: [
        "exception", "error", "failed", "crash", "bug", "issue",
        "traceback", "stack trace", "errno", "exit code"
    ])
    """Keywords that boost bug_resolution intent confidence"""
    
    def __post_init__(self):
        """Validate configuration values."""
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.min_retrieval_score <= 1.0:
            raise ValueError("min_retrieval_score must be between 0.0 and 1.0")
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.max_answer_tokens < 1:
            raise ValueError("max_answer_tokens must be at least 1")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for telemetry/logging."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "top_k": self.top_k,
            "min_retrieval_score": self.min_retrieval_score,
            "max_answer_tokens": self.max_answer_tokens,
            "llm_provider": self.llm_provider,
            "enable_inference": self.enable_inference,
        }
