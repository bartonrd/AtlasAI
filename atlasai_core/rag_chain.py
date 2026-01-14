"""
RAG chain implementation for AtlasAI.

Provides retrieval-augmented generation with customizable prompts.
"""

from typing import Optional
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_huggingface import HuggingFacePipeline


class RAGChain:
    """Manages RAG chain for question answering."""
    
    # Default prompt template optimized for bullet-point responses
    DEFAULT_TEMPLATE = """You are a concise, helpful assistant for a RAG system.

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
    
    def __init__(
        self,
        llm: HuggingFacePipeline,
        retriever: VectorStoreRetriever,
        prompt_template: Optional[str] = None
    ):
        """
        Initialize RAG chain.
        
        Args:
            llm: Language model
            retriever: Vector store retriever
            prompt_template: Custom prompt template (optional)
        """
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE
        self.chain: Optional[RetrievalQA] = None
    
    def build_chain(self) -> RetrievalQA:
        """
        Build the RAG chain.
        
        Returns:
            RetrievalQA chain instance
        """
        qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=self.prompt_template,
        )
        
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # For larger corpora, consider "map_reduce"
            retriever=self.retriever,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )
        
        return self.chain
    
    def query(self, question: str) -> dict:
        """
        Query the RAG chain.
        
        Args:
            question: User question
            
        Returns:
            Dictionary with 'result' and 'source_documents'
        """
        if self.chain is None:
            self.build_chain()
        
        return self.chain.invoke({"query": question})
    
    @staticmethod
    def format_sources(source_documents: list) -> list[str]:
        """
        Format source documents for display.
        
        Args:
            source_documents: List of source documents
            
        Returns:
            List of formatted source strings
        """
        source_list = []
        
        for i, doc in enumerate(source_documents, start=1):
            meta = doc.metadata or {}
            page = meta.get("page", "unknown")
            source = meta.get("source", "unknown")
            
            # Extract filename from full path
            if "/" in str(source):
                source = str(source).split("/")[-1]
            elif "\\" in str(source):
                source = str(source).split("\\")[-1]
            
            source_list.append(f"{i}. {source} (page {page})")
        
        return source_list
