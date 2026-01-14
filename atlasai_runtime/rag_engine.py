"""
RAG Engine - Core logic for document retrieval and question answering.
"""

import os
import re
from typing import List, Dict, Any, Optional

# Loaders: PDF + DOCX
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# Splitter (v1 package)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (v1 package)
from langchain_huggingface import HuggingFaceEmbeddings

# Vector store
from langchain_community.vectorstores import FAISS

# Legacy chain (still supported via langchain_classic)
from langchain_classic.chains import RetrievalQA

# Prompt (v1 package)
from langchain_core.prompts import PromptTemplate

# Local HF LLM
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for document-based question answering.
    """

    def __init__(
        self,
        documents_dir: str,
        embedding_model: str,
        text_gen_model: str,
        top_k: int = 4,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        """
        Initialize the RAG engine.

        Args:
            documents_dir: Path to directory containing documents
            embedding_model: Path to HuggingFace embedding model
            text_gen_model: Path to HuggingFace text generation model
            top_k: Number of chunks to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.documents_dir = documents_dir
        self.embedding_model_path = embedding_model
        self.text_gen_model_path = text_gen_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Lazy initialization
        self._embeddings = None
        self._llm = None
        self._qa_chain = None

    def _load_documents(self, additional_paths: Optional[List[str]] = None) -> List[Any]:
        """
        Load documents from the documents directory and additional paths.

        Args:
            additional_paths: Additional file paths to load

        Returns:
            List of loaded documents
        """
        docs = []
        missing = []

        # Load default documents from documents directory
        if os.path.exists(self.documents_dir):
            for filename in os.listdir(self.documents_dir):
                filepath = os.path.join(self.documents_dir, filename)
                if not os.path.isfile(filepath):
                    continue

                ext = os.path.splitext(filepath)[1].lower()
                try:
                    if ext == ".pdf":
                        docs.extend(PyPDFLoader(filepath).load())
                    elif ext == ".docx":
                        docs.extend(Docx2txtLoader(filepath).load())
                except Exception as e:
                    print(f"Warning: Failed to read {filepath}: {e}")

        # Load additional documents
        if additional_paths:
            for path in additional_paths:
                if not os.path.exists(path):
                    missing.append(path)
                    continue

                ext = os.path.splitext(path)[1].lower()
                try:
                    if ext == ".pdf":
                        docs.extend(PyPDFLoader(path).load())
                    elif ext == ".docx":
                        docs.extend(Docx2txtLoader(path).load())
                except Exception as e:
                    print(f"Warning: Failed to read {path}: {e}")

        if missing:
            raise FileNotFoundError(f"Missing files: {', '.join(missing)}")

        if not docs:
            raise ValueError("No documents loaded")

        return docs

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create embeddings model."""
        if self._embeddings is None:
            self._embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model_path)
            # Quick check
            _ = self._embeddings.embed_query("probe")
        return self._embeddings

    def _get_llm(self) -> HuggingFacePipeline:
        """Get or create LLM pipeline."""
        if self._llm is None:
            tokenizer = AutoTokenizer.from_pretrained(self.text_gen_model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.text_gen_model_path)
            gen_pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=384,
                do_sample=False,
                truncation=True,
            )
            self._llm = HuggingFacePipeline(pipeline=gen_pipe)
        return self._llm

    def _create_qa_chain(self, additional_paths: Optional[List[str]] = None) -> RetrievalQA:
        """
        Create or recreate the QA chain with current settings.

        Args:
            additional_paths: Additional document paths to include

        Returns:
            RetrievalQA chain
        """
        # Load documents
        docs = self._load_documents(additional_paths)

        # Split documents
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
        splits = splitter.split_documents(docs)

        if not splits:
            raise ValueError("No text chunks produced from documents")

        # Create vector store
        embeddings = self._get_embeddings()
        vectorstore = FAISS.from_documents(splits, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": self.top_k})

        # Get LLM
        llm = self._get_llm()

        # Create prompt
        template = """You are a concise, helpful assistant for a RAG system.

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
        qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template=template,
        )

        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
        )

        return qa

    def query(self, question: str, additional_documents: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Query the RAG system with a question.

        Args:
            question: The question to ask
            additional_documents: Optional additional document paths

        Returns:
            Dictionary with 'answer' and 'sources' keys
        """
        # Create QA chain (recreates if documents changed)
        qa_chain = self._create_qa_chain(additional_documents)

        # Run the chain
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        # Format answer as bullets
        formatted_answer = self._to_bullets(answer)

        # Extract source information
        source_list = []
        if sources:
            for i, doc in enumerate(sources, start=1):
                meta = doc.metadata or {}
                page = meta.get("page", "unknown")
                source = meta.get("source", "unknown")
                # Extract filename from path
                source_name = os.path.basename(source)
                source_list.append({
                    "index": i,
                    "source": source_name,
                    "page": page,
                })

        return {
            "answer": formatted_answer,
            "sources": source_list,
        }

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        """Remove common boilerplate from text."""
        if not text:
            return text
        patterns = [
            r"\bProprietary\s*-\s*See\s*Copyright\s*Page\b",
            r"\bContents\b",
            r"\bADMS\s*[\d\.]+\s*Modeling\s*Overview\s*and\s*Converter\s*User\s*Guide\b",
            r"\bDistribution\s*Model\s*Manager\s*User\s*Guide\b",
            r"^\s*Page\s*\d+\s*$",
        ]
        cleaned = text
        for pat in patterns:
            cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _to_bullets(text: str, min_items: int = 3, max_items: int = 10) -> str:
        """
        Convert text to Markdown bullet points.

        Args:
            text: The text to convert
            min_items: Minimum number of bullets
            max_items: Maximum number of bullets

        Returns:
            Formatted bullet points
        """
        if not text:
            return ""

        # Remove boilerplate
        text = RAGEngine._strip_boilerplate(text)

        # Normalize whitespace
        normalized = re.sub(r"\s+", " ", text).strip()

        # Split on bullet glyphs or line-start hyphens
        parts = re.split(
            r"(?:\n+|[•▪●·]\s+|(?:^|\s)-\s+)",
            text,
            flags=re.UNICODE
        )
        parts = [p.strip() for p in parts if p and p.strip()]

        # Fallback: sentence splitting
        if len(parts) <= 1:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
            parts = [p.strip() for p in parts if p and p.strip()]

        # Clean up leading bullets or numbering
        cleaned = []
        for p in parts:
            p = re.sub(r"^[•\-–\*\u2022\u25AA\u25CF\u00B7]+\s*", "", p)
            p = re.sub(r"^(?:\d+|[A-Za-z])[\.\)\:]\s*", "", p)
            p = p.strip()
            if p:
                cleaned.append(p)

        # Try splitting by '•' or semicolons if too few
        if len(cleaned) < min_items:
            if "•" in text:
                cleaned = [re.sub(r"^[•\-\*\s]+", "", p).strip() for p in text.split("•") if p.strip()]
            elif ";" in text:
                cleaned = [re.sub(r"^[•\-\*\s]+", "", p).strip() for p in text.split(";") if p.strip()]

        # Format as Markdown bullets
        bullets = [f"- {p}" for p in cleaned[:max_items]]
        return "\n".join(bullets)
