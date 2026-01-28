"""
Configuration Module

Manages configuration parameters for the intent-aware chatbot.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ChatbotConfig:
    """Configuration parameters for the chatbot."""
    
    # Intent classification
    confidence_threshold: float = 0.55
    
    # Retrieval
    top_k: int = 5
    min_retrieval_score: float = 0.25
    
    # Answer synthesis
    max_answer_tokens: int = 500
    
    # Document filters
    filter_by_product: Optional[str] = None
    filter_by_version: Optional[str] = None
    filter_by_owner: Optional[str] = None
    filter_by_updated_after: Optional[str] = None  # ISO date
    
    # Providers (pluggable)
    embedding_provider: str = "huggingface"  # or "openai", "custom"
    llm_provider: str = "huggingface"  # or "openai", "custom"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confidence_threshold": self.confidence_threshold,
            "top_k": self.top_k,
            "min_retrieval_score": self.min_retrieval_score,
            "max_answer_tokens": self.max_answer_tokens,
            "filter_by_product": self.filter_by_product,
            "filter_by_version": self.filter_by_version,
            "filter_by_owner": self.filter_by_owner,
            "filter_by_updated_after": self.filter_by_updated_after,
            "embedding_provider": self.embedding_provider,
            "llm_provider": self.llm_provider
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChatbotConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})
    
    def get_document_filters(self) -> Dict[str, Any]:
        """Get active document filters."""
        filters = {}
        
        if self.filter_by_product:
            filters["product"] = self.filter_by_product
        
        if self.filter_by_version:
            filters["version"] = self.filter_by_version
        
        if self.filter_by_owner:
            filters["owner"] = self.filter_by_owner
        
        if self.filter_by_updated_after:
            filters["updated_at"] = {"$gte": self.filter_by_updated_after}
        
        return filters
