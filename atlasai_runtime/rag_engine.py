"""
RAG Engine - Core logic for document retrieval and question answering.
"""

import os
import re
from typing import List, Dict, Any, Optional

# Loaders: PDF + DOCX
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# OneNote converter
from .onenote_converter import convert_onenote_directory

# Intent classifier
from .intent_classifier import (
    create_intent_classifier,
    INTENT_ERROR_LOG,
    INTENT_HOW_TO,
    INTENT_CHIT_CHAT,
    INTENT_CONCEPT_EXPLANATION,
)

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


# Configuration constants
MAX_FILES_TO_DISPLAY = 5  # Maximum number of files to show in logging output
CHIT_CHAT_BYPASS_CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for bypassing RAG on chit-chat


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
        use_intent_classifier: bool = True,
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
            use_intent_classifier: Whether to use intent classification for better responses
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
        self._qa_chain = None
        self._intent_classifier = None
        self.use_intent_classifier = use_intent_classifier
        
        # Initialize intent classifier if enabled
        if use_intent_classifier:
            try:
                print("Initializing intent classifier...")
                self._intent_classifier = create_intent_classifier(use_model=True)
                print("Intent classifier initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize intent classifier: {e}")
                print("Falling back to basic heuristic classification")
                self._intent_classifier = create_intent_classifier(use_model=False)
        
        # Convert OneNote files to PDF on initialization
        self._convert_onenote_runbook()
        
        # Eagerly initialize the QA chain with all documents
        try:
            print("Initializing RAG corpus...")
            self._qa_chain = self._create_qa_chain()
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

    def _get_prompt_for_intent(self, intent: str) -> PromptTemplate:
        """
        Get the appropriate prompt template based on detected intent.
        
        Args:
            intent: The detected intent type
            
        Returns:
            PromptTemplate configured for the intent
        """
        if intent == INTENT_ERROR_LOG:
            # Prompt optimized for error resolution
            template = """You are a technical troubleshooting assistant specializing in error resolution.

Rules:
- Focus on identifying the root cause of errors from the context
- Provide step-by-step troubleshooting guidance
- Include specific error codes, messages, or symptoms if mentioned
- If the error isn't covered in the context, provide general troubleshooting advice
- FORMAT your answer as 3–7 clear, actionable Markdown bullet points
- Each bullet MUST start with "- " and be followed by a newline
- Keep each bullet focused and actionable

Question (Error/Issue):
{question}

Context (Documentation):
{context}

Troubleshooting Steps (markdown bullets only):
"""
        elif intent == INTENT_HOW_TO:
            # Prompt optimized for procedural instructions
            template = """You are a step-by-step instruction guide for technical tasks.

Rules:
- Provide clear, sequential steps from the context
- Number steps when appropriate for clarity
- Include prerequisites or requirements if mentioned
- If the procedure isn't in the context, provide general guidance
- Be specific about actions to take
- FORMAT your answer as 3–7 clear Markdown bullet points
- Each bullet MUST start with "- " and be followed by a newline
- Each bullet should be a distinct step or action

Question (How-To):
{question}

Context (Documentation):
{context}

Steps (markdown bullets only):
"""
        elif intent == INTENT_CHIT_CHAT:
            # Prompt optimized for conversational responses
            template = """You are a friendly, conversational assistant.

Rules:
- Respond naturally and concisely to greetings or casual conversation
- Keep the tone warm and professional
- If it's a greeting, respond appropriately
- If it's thanks, acknowledge politely
- If it's a casual question, answer briefly
- Context may not be relevant for casual chat, use your judgment
- Keep responses SHORT - 1 to 3 sentences maximum
- No bullet points needed for casual chat

User Message:
{question}

Context (may not be relevant):
{context}

Response (conversational, brief):
"""
        elif intent == INTENT_CONCEPT_EXPLANATION:
            # Prompt optimized for explaining concepts
            template = """You are an educational assistant explaining technical concepts clearly.

Rules:
- Define the concept using information from the context
- Explain the purpose, function, or importance
- Use clear, accessible language
- Provide examples if available in the context
- If the concept isn't in the context, provide a general explanation
- FORMAT your answer as 3–7 informative Markdown bullet points
- Each bullet MUST start with "- " and be followed by a newline
- Build from basic to more detailed information

Question (Concept):
{question}

Context (Documentation):
{context}

Explanation (markdown bullets only):
"""
        else:
            # Default prompt (same as original)
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
        
        return PromptTemplate(
            input_variables=["question", "context"],
            template=template,
        )

    def _create_qa_chain(self, additional_paths: Optional[List[str]] = None, intent: Optional[str] = None) -> RetrievalQA:
        """
        Create or recreate the QA chain with current settings.

        Args:
            additional_paths: Additional document paths to include
            intent: Optional intent type to customize the prompt

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

        # Get prompt based on intent
        if intent:
            qa_prompt = self._get_prompt_for_intent(intent)
        else:
            # Use default prompt
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
            Dictionary with 'answer', 'sources', 'intent', and 'intent_confidence' keys
        """
        # Classify intent if enabled
        intent = None
        intent_confidence = 0.0
        
        if self.use_intent_classifier and self._intent_classifier:
            try:
                intent, intent_confidence = self._intent_classifier.classify(question)
                print(f"Detected intent: {intent} (confidence: {intent_confidence:.2f})")
            except Exception as e:
                print(f"Warning: Intent classification failed: {e}")
                intent = None
        
        # For chit-chat with high confidence, we can provide a quick response without heavy RAG
        if intent == INTENT_CHIT_CHAT and intent_confidence > CHIT_CHAT_BYPASS_CONFIDENCE_THRESHOLD:
            # Handle simple chit-chat without full RAG processing
            formatted_answer = self._handle_chit_chat(question)
            return {
                "answer": formatted_answer,
                "sources": [],
                "intent": intent,
                "intent_confidence": intent_confidence,
            }
        
        # Use existing QA chain, or recreate if additional documents are provided or if chain not initialized
        # For intent-based queries, we create a new chain with the appropriate prompt
        if additional_documents or self._qa_chain is None or (intent and self.use_intent_classifier):
            qa_chain = self._create_qa_chain(additional_documents, intent=intent)
            # Only cache the chain if it wasn't initialized before and no intent-specific prompt
            if self._qa_chain is None and not intent:
                self._qa_chain = qa_chain
        else:
            qa_chain = self._qa_chain

        # Run the chain
        result = qa_chain.invoke({"query": question})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])

        # Format answer appropriately based on intent
        if intent == INTENT_CHIT_CHAT:
            # For chit-chat, don't force bullets
            formatted_answer = answer.strip()
        else:
            # For other intents, format as bullets
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
            "intent": intent or "unknown",
            "intent_confidence": intent_confidence,
        }

    def _handle_chit_chat(self, message: str) -> str:
        """
        Handle simple chit-chat messages without full RAG processing.
        
        Args:
            message: The user's message
            
        Returns:
            A friendly response
        """
        message_lower = message.lower().strip()
        
        # Greetings - use word boundaries to avoid false matches
        if re.search(r'\b(hello|hi|hey|greetings?)\b', message_lower):
            return "Hello! I'm AtlasAI, your technical documentation assistant. How can I help you today?"
        
        # Thanks - use word boundaries
        if re.search(r'\b(thanks?|thank\s+you)\b', message_lower):
            return "You're welcome! Feel free to ask if you have any other questions."
        
        # Goodbye - use word boundaries
        if re.search(r'\b(bye|goodbye)\b', message_lower):
            return "Goodbye! Have a great day!"
        
        # How are you
        if "how are you" in message_lower:
            return "I'm functioning well, thank you! I'm here to help you with technical documentation and questions."
        
        # What's up
        if re.search(r"\bwhat'?s\s+up\b", message_lower):
            return "I'm here and ready to help! Do you have any questions about your documentation?"
        
        # Default friendly response
        return "I'm AtlasAI, here to help you with technical documentation. Feel free to ask me anything!"

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
