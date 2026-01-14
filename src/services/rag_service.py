"""
RAG service - orchestrates document retrieval and generation
"""

from typing import Dict, List, Any
from pathlib import Path
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

from ..config import Settings
from .document_service import DocumentService
from .embedding_service import EmbeddingService
from .llm_service import LLMService


class RAGService:
    """Service that orchestrates the entire RAG pipeline"""
    
    def __init__(self, settings: Settings):
        """
        Initialize RAG service
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Initialize services
        self.document_service = DocumentService(
            chunk_size=settings.rag.chunk_size,
            chunk_overlap=settings.rag.chunk_overlap
        )
        
        self.embedding_service = EmbeddingService(
            model_name=settings.model.get_embedding_model_path(),
            cache_dir=settings.paths.cache_dir
        )
        
        self.llm_service = LLMService(
            model_name=settings.model.get_text_gen_model_path(),
            max_new_tokens=settings.model.max_new_tokens,
            temperature=settings.model.temperature,
            top_p=settings.model.top_p,
            top_k=settings.model.top_k,
            do_sample=settings.model.do_sample
        )
        
        # Prompt template for RAG
        self.prompt_template = """You are a concise, helpful assistant for a RAG system.

Rules:
- If the question is unrelated to the context, reply briefly to the user without using the context.
- Otherwise, answer using ONLY the retrieved context.
- FORMAT your final answer as 3–7 Markdown bullet points.
- Each bullet MUST start with "- " and be followed by a newline.
- Keep each bullet to one sentence. No preface, no closing remarks—bullets only.

Question:
{question}

Context:
{context}

Answer (markdown bullets only):
"""
        
        self._qa_chain = None
        self._current_document_paths = None
    
    def _build_qa_chain(self, document_paths: List[Path]) -> RetrievalQA:
        """
        Build QA chain from documents
        
        Args:
            document_paths: List of document paths to process
            
        Returns:
            RetrievalQA chain
            
        Raises:
            ValueError: If documents cannot be loaded or processed
        """
        # Load and process documents
        splits, errors = self.document_service.process_documents(document_paths)
        
        if errors:
            error_msg = "Errors loading documents:\n- " + "\n- ".join(errors)
            if not splits:
                raise ValueError(error_msg)
        
        if not splits:
            raise ValueError("No documents were successfully processed")
        
        # Create vector store
        vectorstore = self.embedding_service.create_vectorstore(splits)
        retriever = self.embedding_service.get_retriever(
            vectorstore, 
            top_k=self.settings.rag.top_k
        )
        
        # Create prompt
        qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=self.prompt_template,
        )
        
        # Build QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm_service.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )
        
        return qa_chain
    
    def query(
        self, 
        question: str, 
        document_paths: List[Path]
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            document_paths: List of document paths to query
            
        Returns:
            Dictionary with 'answer' and 'sources' keys
            
        Raises:
            ValueError: If query fails
        """
        # Rebuild chain if documents changed or chain doesn't exist
        if self._qa_chain is None or self._current_document_paths != document_paths:
            self._qa_chain = self._build_qa_chain(document_paths)
            self._current_document_paths = document_paths
        
        # Execute query
        try:
            result = self._qa_chain.invoke({"query": question})
        except Exception as e:
            raise ValueError(f"Query failed: {e}")
        
        # Extract and format results
        answer = result.get("result", "")
        sources = result.get("source_documents", [])
        
        # Format sources
        formatted_sources = []
        for i, doc in enumerate(sources, start=1):
            meta = doc.metadata or {}
            page = meta.get("page", "unknown")
            source = meta.get("source", "unknown")
            # Extract filename from source path
            source_name = Path(source).name if source != "unknown" else "unknown"
            formatted_sources.append(f"{i}. {source_name} (page {page})")
        
        return {
            "answer": answer,
            "sources": formatted_sources,
            "source_documents": sources
        }
    
    def update_rag_settings(self, top_k: int, chunk_size: int, chunk_overlap: int):
        """
        Update RAG settings and invalidate cache
        
        Args:
            top_k: Number of chunks to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.settings.rag.top_k = top_k
        self.settings.rag.chunk_size = chunk_size
        self.settings.rag.chunk_overlap = chunk_overlap
        
        # Update document service
        self.document_service.update_settings(chunk_size, chunk_overlap)
        
        # Invalidate chain cache
        self._qa_chain = None
        self._current_document_paths = None
    
    def clear_cache(self):
        """Clear all caches"""
        self.embedding_service.clear_cache()
        self._qa_chain = None
        self._current_document_paths = None
