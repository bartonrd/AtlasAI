"""
Configuration management for AtlasAI.

Supports environment variables and provides sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional


class Config:
    """Application configuration with environment variable support."""
    
    def __init__(self):
        # Base directories
        self.script_dir = Path(__file__).parent.parent
        self.documents_dir = self.script_dir / "documents"
        self.cache_dir = self.script_dir / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Model paths - support environment variables for flexibility
        self.embedding_model = os.getenv(
            "ATLAS_EMBEDDING_MODEL",
            r"C:\models\bge-small-en-v1.5"  # Better than MiniLM
        )
        self.text_gen_model = os.getenv(
            "ATLAS_TEXT_GEN_MODEL",
            r"C:\models\flan-t5-large"  # Larger model for better quality
        )
        
        # Fallback models (if primary models not found, will try these)
        self.fallback_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.fallback_text_gen_model = "google/flan-t5-base"
        
        # RAG settings
        self.top_k = int(os.getenv("ATLAS_TOP_K", "4"))
        self.chunk_size = int(os.getenv("ATLAS_CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("ATLAS_CHUNK_OVERLAP", "150"))
        self.max_chunk_overlap = 1000
        self.max_overlap_percentage = 0.5
        
        # LLM generation settings
        self.max_new_tokens = int(os.getenv("ATLAS_MAX_NEW_TOKENS", "384"))
        self.use_sampling = os.getenv("ATLAS_USE_SAMPLING", "false").lower() == "true"
        self.temperature = float(os.getenv("ATLAS_TEMPERATURE", "0.2"))
        
        # Document settings
        self.default_pdf_paths = [
            self.documents_dir / "distribution_model_manager_user_guide.pdf",
            self.documents_dir / "adms-16-20-0-modeling-overview-and-converter-user-guide.pdf",
        ]
        
        # Chat settings
        self.default_chat_name = "New Chat"
        self.max_chat_name_length = 30
        
        # Vector store settings
        self.use_persistent_cache = os.getenv("ATLAS_USE_CACHE", "true").lower() == "true"
        self.vector_store_path = self.cache_dir / "vector_store"
        
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation messages (empty if all valid)
        """
        messages = []
        
        if self.chunk_overlap >= self.chunk_size:
            messages.append("chunk_overlap must be less than chunk_size")
        
        if self.chunk_overlap > self.chunk_size * self.max_overlap_percentage:
            max_percent = int(self.max_overlap_percentage * 100)
            messages.append(
                f"chunk_overlap should not exceed {max_percent}% of chunk_size"
            )
        
        if not self.documents_dir.exists():
            messages.append(f"Documents directory not found: {self.documents_dir}")
        
        return messages


# Global configuration instance
config = Config()
