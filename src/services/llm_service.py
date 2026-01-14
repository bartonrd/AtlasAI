"""
LLM service - handles language model loading and inference with caching
"""

from typing import Optional
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


class LLMService:
    """Service for managing language models with caching"""
    
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True
    ):
        """
        Initialize LLM service
        
        Args:
            model_name: Name or path of the language model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
        """
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        
        # Cached model components
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._llm = None
    
    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get or load tokenizer (lazy loading with caching)"""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    @property
    def model(self):
        """Get or load model (lazy loading with caching)"""
        if self._model is None:
            # Try loading as Seq2Seq model (FLAN-T5, T5)
            try:
                self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                self._model_type = "seq2seq"
            except Exception:
                # Try loading as Causal LM (GPT-style, Mistral, Llama, Phi)
                try:
                    self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    self._model_type = "causal"
                except Exception as e:
                    raise RuntimeError(f"Failed to load model from '{self.model_name}': {e}")
        return self._model
    
    @property
    def generation_pipeline(self):
        """Get or create generation pipeline (lazy loading with caching)"""
        if self._pipeline is None:
            # Ensure model is loaded first to set _model_type
            _ = self.model
            
            # Determine task type based on model
            task = "text2text-generation" if hasattr(self, '_model_type') and self._model_type == "seq2seq" else "text-generation"
            
            # Build pipeline config
            pipeline_config = {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "max_new_tokens": self.max_new_tokens,
                "truncation": True,
            }
            
            # Add sampling parameters if enabled
            if self.do_sample:
                pipeline_config.update({
                    "do_sample": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "top_k": self.top_k,
                })
            else:
                pipeline_config["do_sample"] = False
            
            self._pipeline = pipeline(task, **pipeline_config)
        return self._pipeline
    
    @property
    def llm(self) -> HuggingFacePipeline:
        """Get or create LangChain LLM wrapper (lazy loading with caching)"""
        if self._llm is None:
            self._llm = HuggingFacePipeline(pipeline=self.generation_pipeline)
        return self._llm
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from a prompt
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Generated text
        """
        return self.llm.invoke(prompt)
    
    def update_generation_params(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        do_sample: Optional[bool] = None
    ):
        """
        Update generation parameters (requires reloading pipeline)
        
        Args:
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            do_sample: Whether to use sampling
        """
        if max_new_tokens is not None:
            self.max_new_tokens = max_new_tokens
        if temperature is not None:
            self.temperature = temperature
        if top_p is not None:
            self.top_p = top_p
        if top_k is not None:
            self.top_k = top_k
        if do_sample is not None:
            self.do_sample = do_sample
        
        # Clear cached pipeline to force reload with new params
        self._pipeline = None
        self._llm = None
    
    def unload_model(self):
        """Unload model from memory"""
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self._llm = None
        
        # Force garbage collection
        import gc
        gc.collect()
