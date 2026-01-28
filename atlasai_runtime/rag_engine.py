"""
RAG Engine - Core logic for document retrieval and question answering.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

# Loaders: PDF + DOCX
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# OneNote converter
from .onenote_converter import convert_onenote_directory

# Intent classifier
from .intent_classifier import IntentClassifier

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

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILES_TO_DISPLAY = 5  # Maximum number of files to show in logging output


class RAGEngine:
    """
    Retrieval-Augmented Generation engine for document-based question answering.
    """

    def __init__(
        self,
        documents_dir: str,
        onenote_runbook_path: str,
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
            onenote_runbook_path: Path to directory containing OneNote runbook files
            embedding_model: Path to HuggingFace embedding model
            text_gen_model: Path to HuggingFace text generation model
            top_k: Number of chunks to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.documents_dir = documents_dir
        self.onenote_runbook_path = onenote_runbook_path
        self.embedding_model_path = embedding_model
        self.text_gen_model_path = text_gen_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Lazy initialization for models
        self._embeddings = None
        self._llm = None
        self._retriever = None  # Cache the retriever (expensive to create)
        self._intent_qa_chains = {}  # Cache QA chains per intent
        self._intent_classifier = None
        
        # Convert OneNote files to PDF on initialization
        self._convert_onenote_runbook()
        
        # Eagerly initialize the retriever with all documents
        try:
            print("Initializing RAG corpus...")
            self._retriever = self._create_retriever()
            print("RAG corpus initialized and ready for queries")
        except Exception as e:
            print(f"Warning: Failed to initialize RAG corpus during startup: {e}")
            print("RAG corpus will be initialized on first query")
    
    def _convert_onenote_runbook(self):
        """
        Convert OneNote files from the runbook path to PDFs.
        Saves converted PDFs in documents/runbook/ directory.
        Uses non-destructive mode with local copies.
        """
        # Create runbook directory path
        runbook_output_dir = os.path.join(self.documents_dir, "runbook")
        
        # Create local copy directory for non-destructive processing
        local_copy_dir = os.path.join(self.documents_dir, "onenote_copies")
        
        print("=" * 70)
        print("OneNote to PDF Conversion (Non-Destructive Mode)")
        print("=" * 70)
        print(f"Source directory: {self.onenote_runbook_path}")
        print(f"Local copies directory: {local_copy_dir}")
        print(f"Output directory: {runbook_output_dir}")
        print(f"Source exists: {os.path.exists(self.onenote_runbook_path)}")
        print(f"\nNOTE: Original OneNote files will NOT be modified.")
        print(f"      Local copies will be created for conversion processing.")
        
        # Convert all .one files to PDF using non-destructive mode
        converted_count = convert_onenote_directory(
            source_dir=self.onenote_runbook_path,
            output_dir=runbook_output_dir,
            overwrite=True,
            use_local_copies=True,
            local_copy_dir=local_copy_dir
        )
        
        if converted_count > 0:
            print(f"[SUCCESS] Successfully converted {converted_count} OneNote file(s) to PDF")
            print(f"[INFO] Original files remain untouched in: {self.onenote_runbook_path}")
            print(f"[INFO] Local copies stored in: {local_copy_dir}")
            # List the converted files
            if os.path.exists(runbook_output_dir):
                pdf_files = [f for f in os.listdir(runbook_output_dir) if f.lower().endswith('.pdf')]
                print(f"[INFO] PDF files in runbook directory: {len(pdf_files)}")
                for pdf_file in pdf_files[:MAX_FILES_TO_DISPLAY]:
                    pdf_path = os.path.join(runbook_output_dir, pdf_file)
                    try:
                        size = os.path.getsize(pdf_path)
                        print(f"  - {pdf_file} ({size:,} bytes)")
                    except OSError as e:
                        print(f"  - {pdf_file} (error reading size: {e})")
                if len(pdf_files) > MAX_FILES_TO_DISPLAY:
                    print(f"  ... and {len(pdf_files) - MAX_FILES_TO_DISPLAY} more")
        else:
            print("[WARNING] No OneNote files were converted")
            if not os.path.exists(self.onenote_runbook_path):
                print(f"  Reason: Source directory does not exist")
                print(f"  Tip: Set ATLASAI_ONENOTE_RUNBOOK_PATH environment variable to your OneNote files location")
        print("=" * 70)

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
        
        print(f"Loading documents from: {self.documents_dir}")

        # Load default documents from documents directory (including subdirectories)
        if os.path.exists(self.documents_dir):
            # Walk through documents directory and subdirectories
            for root, dirs, files in os.walk(self.documents_dir):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    ext = os.path.splitext(filepath)[1].lower()
                    
                    try:
                        if ext == ".pdf":
                            loaded_docs = PyPDFLoader(filepath).load()
                            docs.extend(loaded_docs)
                            print(f"Loaded PDF: {os.path.relpath(filepath, self.documents_dir)} ({len(loaded_docs)} pages)")
                        elif ext == ".docx":
                            loaded_docs = Docx2txtLoader(filepath).load()
                            docs.extend(loaded_docs)
                            print(f"Loaded DOCX: {os.path.relpath(filepath, self.documents_dir)} ({len(loaded_docs)} pages)")
                    except Exception as e:
                        print(f"Warning: Failed to read {filepath}: {e}")
        else:
            print(f"Warning: Documents directory does not exist: {self.documents_dir}")

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
        
        print(f"Total documents loaded: {len(docs)} pages/chunks")

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

    def _get_intent_classifier(self) -> IntentClassifier:
        """Get or create intent classifier."""
        if self._intent_classifier is None:
            logger.info("Initializing intent classifier...")
            self._intent_classifier = IntentClassifier()
        return self._intent_classifier

    def _get_intent_specific_prompt(self, intent: str) -> PromptTemplate:
        """
        Get an intent-specific prompt template.
        
        Args:
            intent: Detected intent category
            
        Returns:
            PromptTemplate configured for the intent
        """
        if intent == "error_log_resolution":
            template = """You are a technical support assistant specializing in troubleshooting.

The user is experiencing an error or issue. Using the retrieved context, help them resolve it.

Rules:
- Focus on identifying the root cause and providing solutions
- If context contains error codes or error messages, explain them
- Provide step-by-step troubleshooting steps
- If the context doesn't contain relevant error information, acknowledge this and provide general guidance
- FORMAT your answer as 3–7 Markdown bullet points
- Each bullet MUST start with "- " and be followed by a newline
- Keep each bullet to 1-2 sentences focused on actionable solutions

Question:
{question}

Context:
{context}

Answer (troubleshooting steps as markdown bullets):
"""
        elif intent == "how_to":
            template = """You are a helpful technical guide providing step-by-step instructions.

The user wants to know how to perform a specific task. Using the retrieved context, provide clear instructions.

Rules:
- Provide step-by-step instructions in a logical order
- Include any prerequisites or requirements
- Mention important warnings or considerations
- If the context doesn't contain specific instructions, provide general guidance based on what's available
- FORMAT your answer as 3–7 Markdown bullet points
- Each bullet MUST start with "- " and be followed by a newline
- Each bullet should be a clear, actionable step

Question:
{question}

Context:
{context}

Answer (step-by-step instructions as markdown bullets):
"""
        elif intent == "chit_chat":
            template = """You are a friendly and professional assistant.

The user is engaging in casual conversation. Respond appropriately and naturally.

Rules:
- Be conversational and friendly but professional
- Keep responses brief and relevant
- If appropriate, offer to help with technical questions
- Don't force the use of context if it's not relevant to casual conversation
- FORMAT your answer as 1–3 Markdown bullet points
- Each bullet MUST start with "- " and be followed by a newline

Question:
{question}

Context (may not be relevant for casual conversation):
{context}

Answer (brief, friendly response as markdown bullets):
"""
        elif intent == "concept_explanation":
            template = """You are a technical expert providing clear explanations of concepts.

The user wants to understand a concept or technical detail. Using the retrieved context, explain it clearly.

Rules:
- Define key terms and concepts clearly
- Provide context about why it's important
- Include relevant technical details from the context
- Use examples if available in the context
- If context is limited, explain what you can and note what's not covered
- FORMAT your answer as 3–7 Markdown bullet points
- Each bullet MUST start with "- " and be followed by a newline
- Build explanation progressively from basic to more detailed

Question:
{question}

Context:
{context}

Answer (concept explanation as markdown bullets):
"""
        else:
            # Use default template
            template = self.DEFAULT_PROMPT_TEMPLATE
        
        return PromptTemplate(
            input_variables=["question", "context"],
            template=template,
        )

    # Default prompt template as a constant to avoid duplication
    DEFAULT_PROMPT_TEMPLATE = """You are a concise, helpful assistant for a RAG system.

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

    def _create_retriever(self, additional_paths: Optional[List[str]] = None):
        """
        Create or recreate the retriever with current settings.
        This is the expensive operation that loads and indexes documents.

        Args:
            additional_paths: Additional document paths to include

        Returns:
            FAISS retriever
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

        return retriever

    def _create_qa_chain_with_intent(self, retriever, intent: Optional[str] = None) -> RetrievalQA:
        """
        Create a QA chain with the given retriever and intent.
        This is lightweight as it only creates the chain wrapper.

        Args:
            retriever: Document retriever to use
            intent: Intent category for customizing the prompt

        Returns:
            RetrievalQA chain
        """
        # Get LLM
        llm = self._get_llm()

        # Get intent-specific prompt if intent is provided
        if intent:
            qa_prompt = self._get_intent_specific_prompt(intent)
            logger.info(f"Using intent-specific prompt for: {intent}")
        else:
            # Use default prompt template
            qa_prompt = PromptTemplate(
                input_variables=["question", "context"],
                template=self.DEFAULT_PROMPT_TEMPLATE,
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
            Dictionary with 'answer', 'sources', 'intent', 'intent_confidence' keys
        """
        # Classify intent
        intent_classifier = self._get_intent_classifier()
        intent_result = intent_classifier.classify(question)
        detected_intent = intent_result["intent"]
        intent_confidence = intent_result["confidence"]
        
        logger.info(f"Detected intent: {detected_intent} (confidence: {intent_confidence:.2f})")
        logger.info(f"Classification method: {intent_result.get('method', 'unknown')}")
        
        # Get or create retriever (expensive operation, done once or when docs change)
        if additional_documents or self._retriever is None:
            retriever = self._create_retriever(additional_documents)
            # Cache the retriever if it wasn't initialized before and no additional docs
            if self._retriever is None and not additional_documents:
                self._retriever = retriever
        else:
            retriever = self._retriever
        
        # Get or create intent-specific QA chain (lightweight operation)
        cache_key = detected_intent if not additional_documents else f"{detected_intent}_custom"
        if cache_key not in self._intent_qa_chains:
            self._intent_qa_chains[cache_key] = self._create_qa_chain_with_intent(retriever, detected_intent)
            logger.info(f"Created new QA chain for intent: {detected_intent}")
        
        qa_chain = self._intent_qa_chains[cache_key]

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
                # Convert page to string (PDF loaders return int, but API expects string)
                page_str = str(page)
                source_list.append({
                    "index": i,
                    "source": source_name,
                    "page": page_str,
                })

        return {
            "answer": formatted_answer,
            "sources": source_list,
            "intent": detected_intent,
            "intent_confidence": intent_confidence,
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
