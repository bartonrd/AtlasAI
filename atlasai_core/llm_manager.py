"""
LLM initialization and management for AtlasAI.

Handles loading and configuring language models with fallback support.
"""

from pathlib import Path
from typing import Optional
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


class LLMManager:
    """Manages language model initialization and configuration."""
    
    def __init__(
        self,
        model_path: str,
        fallback_model: Optional[str] = None,
        max_new_tokens: int = 384,
        use_sampling: bool = False,
        temperature: float = 0.2
    ):
        """
        Initialize LLM manager.
        
        Args:
            model_path: Path to local model or Hugging Face model name
            fallback_model: Fallback model if primary fails
            max_new_tokens: Maximum tokens to generate
            use_sampling: Whether to use sampling (vs greedy decoding)
            temperature: Temperature for sampling (if enabled)
        """
        self.model_path = model_path
        self.fallback_model = fallback_model
        self.max_new_tokens = max_new_tokens
        self.use_sampling = use_sampling
        self.temperature = temperature
        self.llm: Optional[HuggingFacePipeline] = None
    
    def _try_load_model(self, model_path: str) -> Optional[HuggingFacePipeline]:
        """
        Attempt to load a model.
        
        Args:
            model_path: Path to model
            
        Returns:
            Loaded model or None if failed
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            # Configure generation pipeline
            gen_config = {
                "max_new_tokens": self.max_new_tokens,
                "truncation": True,
                "do_sample": self.use_sampling,
            }
            
            if self.use_sampling:
                gen_config.update({
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "top_k": 50
                })
            
            gen_pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                **gen_config
            )
            
            return HuggingFacePipeline(pipeline=gen_pipe)
            
        except Exception as e:
            st.warning(f"Failed to load model from '{model_path}': {e}")
            return None
    
    def initialize_llm(self) -> HuggingFacePipeline:
        """
        Initialize language model with fallback support.
        
        Returns:
            Initialized HuggingFacePipeline instance
            
        Raises:
            RuntimeError: If both primary and fallback models fail to load
        """
        if self.llm is not None:
            return self.llm
        
        # Try primary model
        self.llm = self._try_load_model(self.model_path)
        
        # Try fallback if primary failed
        if self.llm is None and self.fallback_model:
            st.info(f"Trying fallback model: {self.fallback_model}")
            self.llm = self._try_load_model(self.fallback_model)
        
        if self.llm is None:
            raise RuntimeError(
                f"Failed to load any language model. "
                f"Tried: {self.model_path}"
                f"{f', {self.fallback_model}' if self.fallback_model else ''}"
            )
        
        return self.llm
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": self.model_path,
            "fallback_model": self.fallback_model,
            "max_new_tokens": self.max_new_tokens,
            "use_sampling": self.use_sampling,
            "temperature": self.temperature if self.use_sampling else None,
        }
