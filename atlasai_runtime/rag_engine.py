"""
RAG Engine - Core logic for document retrieval and question answering using Ollama and ChromaDB.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional

# Document loaders
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# OneNote converter
from .onenote_converter import convert_onenote_directory

# Intent classifier
from .intent_classifier import IntentClassifier

# Text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ChromaDB for vector storage
import chromadb
from chromadb.config import Settings

# Ollama
import ollama

# Configure logging
logger = logging.getLogger(__name__)

# Configuration constants
MAX_FILES_TO_DISPLAY = 5


class RAGEngine:
    """
    Retrieval-Augmented Generation engine using Ollama and ChromaDB.
    """

    def __init__(
        self,
        documents_dir: str,
        onenote_runbook_path: str,
        ollama_model: str = "llama3.1:8b",
        embedding_model: str = "bge-base-en",
        top_k: int = 6,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        chroma_persist_dir: str = None,
    ):
        """
        Initialize the RAG engine.

        Args:
            documents_dir: Path to directory containing documents
            onenote_runbook_path: Path to directory containing OneNote runbook files
            ollama_model: Ollama model name for text generation
            embedding_model: Ollama embedding model name
            top_k: Number of chunks to retrieve
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            chroma_persist_dir: Directory for ChromaDB persistence
        """
        self.documents_dir = documents_dir
        self.onenote_runbook_path = onenote_runbook_path
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Set up ChromaDB persistence directory
        if chroma_persist_dir is None:
            chroma_persist_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                "chroma_db"
            )
        self.chroma_persist_dir = chroma_persist_dir
        
        # Lazy initialization
        self._chroma_client = None
        self._collection = None
        self._intent_classifier = None
        
        # Convert OneNote files to PDF on initialization
        self._convert_onenote_runbook()
        
        # Eagerly initialize the vector store
        try:
            print("Initializing RAG corpus with ChromaDB...")
            self._initialize_vector_store()
            print("RAG corpus initialized and ready for queries")
        except Exception as e:
            print(f"Warning: Failed to initialize RAG corpus during startup: {e}")
            print("RAG corpus will be initialized on first query")

    def _convert_onenote_runbook(self):
        """Convert OneNote files from the runbook path to PDFs."""
        runbook_output_dir = os.path.join(self.documents_dir, "runbook")
        local_copy_dir = os.path.join(self.documents_dir, "onenote_copies")
        
        print("=" * 70)
        print("OneNote to PDF Conversion (Non-Destructive Mode)")
        print("=" * 70)
        print(f"Source directory: {self.onenote_runbook_path}")
        print(f"Local copies directory: {local_copy_dir}")
        print(f"Output directory: {runbook_output_dir}")
        print(f"Source exists: {os.path.exists(self.onenote_runbook_path)}")
        print(f"\nNOTE: Original OneNote files will NOT be modified.")
        
        converted_count = convert_onenote_directory(
            source_dir=self.onenote_runbook_path,
            output_dir=runbook_output_dir,
            overwrite=True,
            use_local_copies=True,
            local_copy_dir=local_copy_dir
        )
        
        if converted_count > 0:
            print(f"[SUCCESS] Successfully converted {converted_count} OneNote file(s) to PDF")
        else:
            print("[WARNING] No OneNote files were converted")
        print("=" * 70)

    def _load_documents(self, additional_paths: Optional[List[str]] = None) -> List[Any]:
        """Load documents from the documents directory and additional paths."""
        docs = []
        missing = []
        
        print(f"Loading documents from: {self.documents_dir}")

        # Load default documents from documents directory (including subdirectories)
        if os.path.exists(self.documents_dir):
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

    def _get_chroma_client(self):
        """Get or create ChromaDB client."""
        if self._chroma_client is None:
            # Ensure persistence directory exists
            os.makedirs(self.chroma_persist_dir, exist_ok=True)
            
            self._chroma_client = chromadb.PersistentClient(
                path=self.chroma_persist_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        return self._chroma_client

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Ollama."""
        try:
            response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
            return response["embedding"]
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise

    def _initialize_vector_store(self, additional_paths: Optional[List[str]] = None):
        """Initialize or update the ChromaDB vector store."""
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

        print(f"Created {len(splits)} text chunks")

        # Get or create ChromaDB collection
        client = self._get_chroma_client()
        
        # Delete existing collection if it exists (for fresh rebuild)
        try:
            client.delete_collection("atlasai_docs")
        except Exception:
            pass
        
        # Create new collection
        self._collection = client.create_collection(
            name="atlasai_docs",
            metadata={"description": "AtlasAI document collection"}
        )

        # Add documents to collection in batches
        batch_size = 100
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            
            # Prepare data for ChromaDB
            documents = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            ids = [f"doc_{i+j}" for j in range(len(batch))]
            
            # Get embeddings for batch
            embeddings = []
            for doc_text in documents:
                emb = self._get_embedding(doc_text)
                embeddings.append(emb)
            
            # Add to collection
            self._collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"Indexed {min(i+batch_size, len(splits))}/{len(splits)} chunks")

        print("Vector store initialization complete")

    def _get_intent_classifier(self) -> IntentClassifier:
        """Get or create intent classifier."""
        if self._intent_classifier is None:
            logger.info("Initializing intent classifier...")
            self._intent_classifier = IntentClassifier()
        return self._intent_classifier

    def _retrieve_relevant_chunks(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve relevant document chunks for a query."""
        if k is None:
            k = self.top_k
        
        # Ensure collection is initialized
        if self._collection is None:
            self._initialize_vector_store()
        
        # Get query embedding
        query_embedding = self._get_embedding(query)
        
        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        chunks = []
        if results and results['documents']:
            for i in range(len(results['documents'][0])):
                chunks.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        return chunks

    def _generate_answer(self, query: str, context: str, intent: str) -> str:
        """Generate answer using Ollama."""
        # Get intent-specific prompt
        prompt = self._get_intent_specific_prompt(intent, query, context)
        
        try:
            # Generate response using Ollama
            response = ollama.generate(
                model=self.ollama_model,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            
            return response["response"]
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"

    def _get_intent_specific_prompt(self, intent: str, question: str, context: str) -> str:
        """Get an intent-specific prompt."""
        if intent == "error_log_resolution":
            template = f"""You are a technical support assistant specializing in troubleshooting electrical grid systems.

The user has encountered an error or issue. Using the retrieved context, help them understand and resolve it.

Rules:
- FIRST explain what the error means and why it occurs
- Then provide step-by-step troubleshooting and resolution steps
- Include relevant technical details and examples from the context
- FORMAT your answer as 5–10 Markdown bullet points
- Each bullet MUST start with "- "

Question:
{question}

Context:
{context}

Answer (comprehensive error explanation and troubleshooting as markdown bullets):
"""
        elif intent == "how_to":
            template = f"""You are a helpful technical guide providing step-by-step instructions.

The user wants to know how to perform a specific task. Using the retrieved context, provide clear instructions.

Rules:
- Provide step-by-step instructions in a logical order
- Include any prerequisites or requirements
- Mention important warnings or considerations
- FORMAT your answer as 3–7 Markdown bullet points
- Each bullet MUST start with "- "

Question:
{question}

Context:
{context}

Answer (step-by-step instructions as markdown bullets):
"""
        elif intent == "chit_chat":
            template = f"""You are a friendly and professional assistant.

The user is engaging in casual conversation. Respond appropriately and naturally.

Rules:
- Be conversational and friendly but professional
- Keep responses brief and relevant
- FORMAT your answer as 1–3 Markdown bullet points
- Each bullet MUST start with "- "

Question:
{question}

Context (may not be relevant for casual conversation):
{context}

Answer (brief, friendly response as markdown bullets):
"""
        else:  # concept_explanation or default
            template = f"""You are a technical expert providing clear explanations.

The user wants to understand a concept, technical detail, or system component. Using the retrieved context, explain it comprehensively.

Rules:
- Define what the concept/component is and its purpose
- Explain how it works or how it's used
- Include technical details and parameters from the context
- FORMAT your answer as 3–7 Markdown bullet points
- Each bullet MUST start with "- "

Question:
{question}

Context:
{context}

Answer (comprehensive explanation as markdown bullets):
"""
        
        return template

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

        # Retrieve relevant chunks
        chunks = self._retrieve_relevant_chunks(question)
        
        # Build context from chunks
        context = "\n\n".join([chunk['content'] for chunk in chunks])
        
        # Generate answer
        answer = self._generate_answer(question, context, detected_intent)
        
        # Format answer as bullets if needed
        formatted_answer = self._to_bullets(answer)
        
        # Extract source information
        source_list = []
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.get('metadata', {})
            page = meta.get('page', 'unknown')
            source = meta.get('source', 'unknown')
            source_name = os.path.basename(source) if source != 'unknown' else 'unknown'
            source_list.append({
                "index": i,
                "source": source_name,
                "page": str(page),
            })

        return {
            "answer": formatted_answer,
            "sources": source_list,
            "intent": detected_intent,
            "intent_confidence": intent_confidence,
        }

    @staticmethod
    def _to_bullets(text: str, min_items: int = 3, max_items: int = 10) -> str:
        """Convert text to Markdown bullet points."""
        if not text:
            return ""

        # If already formatted with bullets, clean up and return
        if text.strip().startswith("-") or "\n-" in text:
            lines = text.strip().split('\n')
            bullets = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith("-"):
                    bullets.append(f"- {line}")
                elif line:
                    bullets.append(line)
            return "\n".join(bullets[:max_items])

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Format as bullets
        bullets = [f"- {s}" for s in sentences[:max_items]]
        
        # Ensure minimum number of bullets
        if len(bullets) < min_items and len(bullets) > 0:
            # Just return what we have
            pass
        
        return "\n".join(bullets)
