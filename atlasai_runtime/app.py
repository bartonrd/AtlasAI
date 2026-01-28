"""
FastAPI application for AtlasAI Runtime.
Uses Ollama, Chroma, and bge-base-en embeddings.
"""

import os
import logging
import subprocess
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag_engine_ollama import RAGEngineOllama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
DOCUMENTS_DIR = os.getenv("ATLASAI_DOCUMENTS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents"))
ONENOTE_RUNBOOK_PATH = os.getenv("ATLASAI_ONENOTE_RUNBOOK_PATH", r"\\sce\workgroup\TDBU2\TD-PSC\PSC-DMS-ADV-APP\ADMS Operation & Maintenance Docs\Model Manager Runbook")
OLLAMA_MODEL = os.getenv("ATLASAI_OLLAMA_MODEL", "llama3.1:8b-instruct-q4_0")
OLLAMA_BASE_URL = os.getenv("ATLASAI_OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("ATLASAI_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
CHROMA_PERSIST_DIR = os.getenv("ATLASAI_CHROMA_PERSIST_DIR", None)
TOP_K = int(os.getenv("ATLASAI_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("ATLASAI_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("ATLASAI_CHUNK_OVERLAP", "150"))

# Global RAG engine instance
rag_engine: Optional[RAGEngineOllama] = None


def detect_available_ollama_model():
    """
    Detect which Ollama model is available on the system.
    Returns the first available model from the supported list.
    """
    supported_models = [
        "llama3.1:8b-instruct-q4_0",
        "llama3.1:8b",
        "qwen2.5:7b-instruct",
        "qwen2.5:7b",
        "mistral:7b-instruct",
        "mistral:7b",
        "llama3:8b",  # Fallback to llama3
        "mistral:latest",  # Fallback options
        "llama2:7b",
    ]
    
    try:
        # Get list of installed models
        result = subprocess.run(["ollama", "list"], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode != 0:
            logger.warning("Could not list Ollama models")
            return OLLAMA_MODEL  # Use configured default
        
        installed_models = result.stdout.lower()
        
        # Check if any supported model is installed
        for model in supported_models:
            model_base = model.split(':')[0]
            if model_base in installed_models or model in installed_models:
                logger.info(f"Detected available Ollama model: {model}")
                return model
        
        logger.warning("No preferred models found, using configured model")
        return OLLAMA_MODEL
        
    except FileNotFoundError:
        logger.warning("Ollama command not found")
        return OLLAMA_MODEL
    except subprocess.TimeoutExpired:
        logger.warning("Ollama command timed out")
        return OLLAMA_MODEL
    except Exception as e:
        logger.warning(f"Error detecting Ollama model: {e}")
        return OLLAMA_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global rag_engine
    
    # Startup
    logger.info("Starting AtlasAI Runtime with Ollama...")
    logger.info(f"Documents directory: {DOCUMENTS_DIR}")
    logger.info(f"OneNote runbook path: {ONENOTE_RUNBOOK_PATH}")
    logger.info(f"Ollama base URL: {OLLAMA_BASE_URL}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    
    # Detect available Ollama model
    detected_model = detect_available_ollama_model()
    logger.info(f"Using Ollama model: {detected_model}")
    
    logger.info(f"Settings - TOP_K: {TOP_K}, CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    
    try:
        rag_engine = RAGEngineOllama(
            documents_dir=DOCUMENTS_DIR,
            onenote_runbook_path=ONENOTE_RUNBOOK_PATH,
            ollama_model=detected_model,
            ollama_base_url=OLLAMA_BASE_URL,
            embedding_model=EMBEDDING_MODEL,
            chroma_persist_dir=CHROMA_PERSIST_DIR,
            top_k=TOP_K,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info("RAG engine initialized successfully with Ollama")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        # Don't fail startup - allow health checks to report the issue
    
    yield
    
    # Shutdown
    logger.info("Shutting down AtlasAI Runtime...")


# Create FastAPI app
app = FastAPI(
    title="AtlasAI Runtime",
    description="Python runtime service for RAG-based chat completion with Ollama",
    version="0.2.0",
    lifespan=lifespan,
)


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat completion."""
    message: str = Field(..., description="The user's question or message")
    additional_documents: Optional[List[str]] = Field(None, description="Optional additional document paths")


class Source(BaseModel):
    """Source reference model."""
    index: int = Field(..., description="Source index")
    source: str = Field(..., description="Source document name")
    page: str = Field(..., description="Page number or identifier")


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source references")
    intent: Optional[str] = Field(None, description="Detected user intent")
    intent_confidence: Optional[float] = Field(None, description="Confidence score for intent detection")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    config: dict = Field(..., description="Current configuration")


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the service status and configuration.
    """
    if rag_engine is None:
        return HealthResponse(
            status="unhealthy",
            message="RAG engine not initialized",
            config={
                "documents_dir": DOCUMENTS_DIR,
                "onenote_runbook_path": ONENOTE_RUNBOOK_PATH,
                "ollama_model": OLLAMA_MODEL,
                "ollama_base_url": OLLAMA_BASE_URL,
                "embedding_model": EMBEDDING_MODEL,
                "top_k": TOP_K,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
        )
    
    return HealthResponse(
        status="healthy",
        message="AtlasAI Runtime is ready with Ollama",
        config={
            "documents_dir": DOCUMENTS_DIR,
            "onenote_runbook_path": ONENOTE_RUNBOOK_PATH,
            "ollama_model": OLLAMA_MODEL,
            "ollama_base_url": OLLAMA_BASE_URL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint.
    
    Accepts a chat message and returns an answer with source references and detected intent.
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        logger.info(f"Processing chat request: {request.message[:100]}...")
        result = rag_engine.query(request.message, request.additional_documents)
        logger.info(f"Generated answer with {len(result['sources'])} sources")
        logger.info(f"Detected intent: {result.get('intent', 'unknown')} (confidence: {result.get('intent_confidence', 0):.2f})")
        
        return ChatResponse(
            answer=result["answer"],
            sources=[Source(**src) for src in result["sources"]],
            intent=result.get("intent"),
            intent_confidence=result.get("intent_confidence")
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - provides basic information."""
    return {
        "name": "AtlasAI Runtime",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "docs": "/docs",
        }
    }
