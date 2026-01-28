"""
FastAPI application for AtlasAI Runtime.
"""

import os
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag_engine import RAGEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
DOCUMENTS_DIR = os.getenv("ATLASAI_DOCUMENTS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents"))
ONENOTE_RUNBOOK_PATH = os.getenv("ATLASAI_ONENOTE_RUNBOOK_PATH", r"\\sce\workgroup\TDBU2\TD-PSC\PSC-DMS-ADV-APP\ADMS Operation & Maintenance Docs\Model Manager Runbook")
EMBEDDING_MODEL = os.getenv("ATLASAI_EMBEDDING_MODEL", r"C:\models\all-MiniLM-L6-v2")
TEXT_GEN_MODEL = os.getenv("ATLASAI_TEXT_GEN_MODEL", r"C:\models\flan-t5-base")
TOP_K = int(os.getenv("ATLASAI_TOP_K", "4"))
CHUNK_SIZE = int(os.getenv("ATLASAI_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("ATLASAI_CHUNK_OVERLAP", "150"))

# Global RAG engine instance
rag_engine: Optional[RAGEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global rag_engine
    
    # Startup
    logger.info("Starting AtlasAI Runtime...")
    logger.info(f"Documents directory: {DOCUMENTS_DIR}")
    logger.info(f"OneNote runbook path: {ONENOTE_RUNBOOK_PATH}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Text generation model: {TEXT_GEN_MODEL}")
    logger.info(f"Settings - TOP_K: {TOP_K}, CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    
    try:
        rag_engine = RAGEngine(
            documents_dir=DOCUMENTS_DIR,
            onenote_runbook_path=ONENOTE_RUNBOOK_PATH,
            embedding_model=EMBEDDING_MODEL,
            text_gen_model=TEXT_GEN_MODEL,
            top_k=TOP_K,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")
        # Don't fail startup - allow health checks to report the issue
    
    yield
    
    # Shutdown
    logger.info("Shutting down AtlasAI Runtime...")


# Create FastAPI app
app = FastAPI(
    title="AtlasAI Runtime",
    description="Python runtime service for RAG-based chat completion",
    version="0.1.0",
    lifespan=lifespan,
)


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat completion."""
    message: str = Field(..., description="The user's question or message")
    additional_documents: Optional[List[str]] = Field(None, description="Optional additional document paths")
    use_inference: bool = Field(False, description="Use inference pipeline (intent classification, query rewriting, etc.)")
    user_feedback: Optional[str] = Field(None, description="Optional user feedback on previous answer")


class Citation(BaseModel):
    """Citation reference model."""
    index: str = Field(..., description="Citation index")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Document URL or path")


class InferenceResponse(BaseModel):
    """Response model for inference-enhanced chat completion."""
    answer: Optional[str] = Field(None, description="The generated answer (if available)")
    question: Optional[str] = Field(None, description="Clarifying question (if answer not available)")
    intent: str = Field(..., description="Detected intent")
    confidence: float = Field(..., description="Intent confidence score")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    telemetry: dict = Field(default_factory=dict, description="Telemetry data")


class Source(BaseModel):
    """Source reference model."""
    index: int = Field(..., description="Source index")
    source: str = Field(..., description="Source document name")
    page: str = Field(..., description="Page number or identifier")


class ChatResponse(BaseModel):
    """Response model for chat completion."""
    answer: str = Field(..., description="The generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source references")


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
                "embedding_model": EMBEDDING_MODEL,
                "text_gen_model": TEXT_GEN_MODEL,
                "top_k": TOP_K,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
        )
    
    return HealthResponse(
        status="healthy",
        message="AtlasAI Runtime is ready",
        config={
            "documents_dir": DOCUMENTS_DIR,
            "onenote_runbook_path": ONENOTE_RUNBOOK_PATH,
            "embedding_model": EMBEDDING_MODEL,
            "text_gen_model": TEXT_GEN_MODEL,
            "top_k": TOP_K,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Chat completion endpoint (legacy mode).
    
    Accepts a chat message and returns an answer with source references.
    Uses the original RAG pipeline without inference enhancements.
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        logger.info(f"Processing chat request (legacy): {request.message[:100]}...")
        result = rag_engine.query(request.message, request.additional_documents)
        logger.info(f"Generated answer with {len(result['sources'])} sources")
        
        return ChatResponse(
            answer=result["answer"],
            sources=[Source(**src) for src in result["sources"]]
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


@app.post("/chat/inference", response_model=InferenceResponse)
async def chat_inference(request: ChatRequest):
    """
    Chat completion endpoint with inference pipeline.
    
    Accepts a chat message and returns an answer or clarifying question with:
    - Intent classification
    - Query rewriting
    - Retrieval routing
    - Intent-specific answer formatting
    - Citations and telemetry
    
    To toggle this feature on/off, set request.use_inference or use environment variable.
    """
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    try:
        logger.info(f"Processing inference request: {request.message[:100]}...")
        
        # Process with inference pipeline
        result = rag_engine.query_with_inference(
            question=request.message,
            use_inference=request.use_inference,
            user_feedback=request.user_feedback,
        )
        
        logger.info(
            f"Inference result - Intent: {result['intent']}, "
            f"Confidence: {result['confidence']:.2f}, "
            f"Answer: {result['answer'] is not None}"
        )
        
        # Convert citations
        citations = [Citation(**c) for c in result.get("citations", [])]
        
        return InferenceResponse(
            answer=result.get("answer"),
            question=result.get("question"),
            intent=result["intent"],
            confidence=result["confidence"],
            citations=citations,
            telemetry=result.get("telemetry", {}),
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing inference request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint - provides basic information."""
    return {
        "name": "AtlasAI Runtime",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (legacy mode)",
            "chat_inference": "/chat/inference (with intent classification & query rewriting)",
            "docs": "/docs",
        },
        "features": {
            "inference_pipeline": "Intent classification, query rewriting, retrieval routing, grounded synthesis",
            "intents_supported": [
                "how_to", "bug_resolution", "tool_explanation",
                "escalate_or_ticket", "chitchat", "other"
            ],
        }
    }
