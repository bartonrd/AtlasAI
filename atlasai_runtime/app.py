"""
FastAPI application for AtlasAI Runtime.
Uses Ollama for LLM, ChromaDB for vector storage, and supports local task execution.
"""

import os
import logging
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .rag_engine_new import RAGEngine
from .task_executor import LocalTaskExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
DOCUMENTS_DIR = os.getenv("ATLASAI_DOCUMENTS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents"))
ONENOTE_RUNBOOK_PATH = os.getenv("ATLASAI_ONENOTE_RUNBOOK_PATH", r"\\sce\workgroup\TDBU2\TD-PSC\PSC-DMS-ADV-APP\ADMS Operation & Maintenance Docs\Model Manager Runbook")
OLLAMA_MODEL = os.getenv("ATLASAI_OLLAMA_MODEL", "llama3.1:8b")
EMBEDDING_MODEL = os.getenv("ATLASAI_EMBEDDING_MODEL", "mxbai-embed-large")
TOP_K = int(os.getenv("ATLASAI_TOP_K", "6"))
CHUNK_SIZE = int(os.getenv("ATLASAI_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("ATLASAI_CHUNK_OVERLAP", "150"))
CHROMA_PERSIST_DIR = os.getenv("ATLASAI_CHROMA_PERSIST_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db"))

# Global instances
rag_engine: Optional[RAGEngine] = None
task_executor: Optional[LocalTaskExecutor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    global rag_engine, task_executor
    
    # Startup
    logger.info("Starting AtlasAI Runtime...")
    logger.info(f"Documents directory: {DOCUMENTS_DIR}")
    logger.info(f"OneNote runbook path: {ONENOTE_RUNBOOK_PATH}")
    logger.info(f"Ollama model: {OLLAMA_MODEL}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"ChromaDB persist directory: {CHROMA_PERSIST_DIR}")
    logger.info(f"Settings - TOP_K: {TOP_K}, CHUNK_SIZE: {CHUNK_SIZE}, CHUNK_OVERLAP: {CHUNK_OVERLAP}")
    
    try:
        # Initialize task executor
        task_executor = LocalTaskExecutor()
        logger.info("Task executor initialized successfully")
        
        # Initialize RAG engine
        rag_engine = RAGEngine(
            documents_dir=DOCUMENTS_DIR,
            onenote_runbook_path=ONENOTE_RUNBOOK_PATH,
            ollama_model=OLLAMA_MODEL,
            embedding_model=EMBEDDING_MODEL,
            top_k=TOP_K,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            chroma_persist_dir=CHROMA_PERSIST_DIR,
        )
        logger.info("RAG engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        # Don't fail startup - allow health checks to report the issue
    
    yield
    
    # Shutdown
    logger.info("Shutting down AtlasAI Runtime...")


# Create FastAPI app
app = FastAPI(
    title="AtlasAI Runtime",
    description="Python runtime service for RAG-based chat completion with local task execution",
    version="2.0.0",
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


class TaskRequest(BaseModel):
    """Request model for task execution."""
    command: str = Field(..., description="Command to execute")
    working_dir: Optional[str] = Field(None, description="Working directory for command execution")
    timeout: int = Field(30, description="Timeout in seconds")


class TaskResponse(BaseModel):
    """Response model for task execution."""
    success: bool = Field(..., description="Whether the task succeeded")
    stdout: str = Field(default="", description="Standard output")
    stderr: str = Field(default="", description="Standard error")
    return_code: int = Field(default=0, description="Return code")
    error: Optional[str] = Field(None, description="Error message if any")


class SystemInfoResponse(BaseModel):
    """Response model for system information."""
    system: str
    platform: str
    machine: str
    processor: str
    python_version: str
    cwd: str


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
                "embedding_model": EMBEDDING_MODEL,
                "top_k": TOP_K,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chroma_persist_dir": CHROMA_PERSIST_DIR,
            }
        )
    
    return HealthResponse(
        status="healthy",
        message="AtlasAI Runtime is ready",
        config={
            "documents_dir": DOCUMENTS_DIR,
            "onenote_runbook_path": ONENOTE_RUNBOOK_PATH,
            "ollama_model": OLLAMA_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "top_k": TOP_K,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chroma_persist_dir": CHROMA_PERSIST_DIR,
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
        "version": "2.0.0",
        "stack": {
            "llm": "Ollama",
            "vector_store": "ChromaDB",
            "orchestration": "LangGraph-ready",
            "embeddings": "mxbai-embed-large (via Ollama)"
        },
        "endpoints": {
            "health": "/health",
            "chat": "/chat",
            "task": "/task/execute",
            "system_info": "/task/system-info",
            "docs": "/docs",
        }
    }


@app.post("/task/execute", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """
    Execute a local system task.
    
    Accepts a command and executes it on the local machine.
    """
    if task_executor is None:
        raise HTTPException(status_code=503, detail="Task executor not initialized")
    
    try:
        logger.info(f"Executing task: {request.command}")
        result = task_executor.execute_command(
            command=request.command,
            working_dir=request.working_dir,
            timeout=request.timeout
        )
        
        return TaskResponse(**result)
    except Exception as e:
        logger.error(f"Error executing task: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/task/system-info", response_model=SystemInfoResponse)
async def get_system_info():
    """
    Get system information.
    
    Returns information about the system where the runtime is running.
    """
    if task_executor is None:
        raise HTTPException(status_code=503, detail="Task executor not initialized")
    
    try:
        info = task_executor.get_system_info()
        return SystemInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting system info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
