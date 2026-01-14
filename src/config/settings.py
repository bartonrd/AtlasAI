"""
Centralized configuration management for AtlasAI
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    # Embedding model - smaller, faster model for semantic search
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Text generation models (in priority order)
    # For better quality RAG, we recommend:
    # 1. microsoft/Phi-2 (2.7B parameters, optimized for Q&A)
    # 2. mistralai/Mistral-7B-Instruct-v0.2 (better than FLAN-T5)
    # 3. google/flan-t5-large (for compatibility)
    text_gen_model: str = "google/flan-t5-base"
    
    # Local model paths (override above if provided)
    local_embedding_model: Optional[str] = None
    local_text_gen_model: Optional[str] = None
    
    # Model generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    
    def get_embedding_model_path(self) -> str:
        """Get the embedding model path (local or HuggingFace)"""
        if self.local_embedding_model and os.path.exists(self.local_embedding_model):
            return self.local_embedding_model
        return self.embedding_model
    
    def get_text_gen_model_path(self) -> str:
        """Get the text generation model path (local or HuggingFace)"""
        if self.local_text_gen_model and os.path.exists(self.local_text_gen_model):
            return self.local_text_gen_model
        return self.text_gen_model


@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    # Retrieval settings
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 150
    
    # Constraints
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    max_overlap_percentage: float = 0.5
    max_top_k: int = 20
    
    def validate(self) -> list[str]:
        """Validate RAG configuration settings"""
        errors = []
        
        if self.top_k < 1:
            errors.append("Top K must be at least 1")
        elif self.top_k > self.max_top_k:
            errors.append(f"Top K should not exceed {self.max_top_k}")
        
        if self.chunk_size < self.min_chunk_size:
            errors.append(f"Chunk size must be at least {self.min_chunk_size}")
        elif self.chunk_size > self.max_chunk_size:
            errors.append(f"Chunk size should not exceed {self.max_chunk_size}")
        
        if self.chunk_overlap < 0:
            errors.append("Chunk overlap must be non-negative")
        elif self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be less than chunk size")
        elif self.chunk_overlap > self.chunk_size * self.max_overlap_percentage:
            max_percent = int(self.max_overlap_percentage * 100)
            errors.append(f"Chunk overlap should not exceed {max_percent}% of chunk size")
        
        return errors


@dataclass
class PathConfig:
    """Configuration for file paths"""
    script_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent.absolute())
    documents_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "documents")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / ".cache")
    temp_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent / "tmp_docs")
    
    def __post_init__(self):
        """Ensure directories exist"""
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.temp_dir.mkdir(exist_ok=True, parents=True)
    
    def get_default_documents(self) -> list[Path]:
        """Get list of default document paths"""
        default_docs = [
            self.documents_dir / "distribution_model_manager_user_guide.pdf",
            self.documents_dir / "adms-16-20-0-modeling-overview-and-converter-user-guide.pdf",
        ]
        return [doc for doc in default_docs if doc.exists()]


@dataclass
class UIConfig:
    """Configuration for UI settings"""
    page_title: str = "AtlasAI Chat"
    layout: str = "wide"
    default_chat_name: str = "New Chat"
    max_chat_name_length: int = 30


@dataclass
class Settings:
    """Main settings container"""
    model: ModelConfig = field(default_factory=ModelConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    @classmethod
    def load_from_env(cls) -> "Settings":
        """Load settings from environment variables"""
        settings = cls()
        
        # Override with environment variables if present
        if embedding_model := os.getenv("ATLAS_EMBEDDING_MODEL"):
            settings.model.embedding_model = embedding_model
        
        if text_gen_model := os.getenv("ATLAS_TEXT_GEN_MODEL"):
            settings.model.text_gen_model = text_gen_model
        
        if local_embedding := os.getenv("ATLAS_LOCAL_EMBEDDING_MODEL"):
            settings.model.local_embedding_model = local_embedding
        
        if local_text_gen := os.getenv("ATLAS_LOCAL_TEXT_GEN_MODEL"):
            settings.model.local_text_gen_model = local_text_gen
        
        # RAG settings
        if top_k := os.getenv("ATLAS_TOP_K"):
            settings.rag.top_k = int(top_k)
        
        if chunk_size := os.getenv("ATLAS_CHUNK_SIZE"):
            settings.rag.chunk_size = int(chunk_size)
        
        if chunk_overlap := os.getenv("ATLAS_CHUNK_OVERLAP"):
            settings.rag.chunk_overlap = int(chunk_overlap)
        
        return settings
