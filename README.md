# AtlasAI

A Retrieval-Augmented Generation (RAG) chatbot with local task execution capabilities using Ollama, Chroma, and LangGraph.

## Overview

AtlasAI is a C# application that communicates with a Python-based RAG runtime service. The system uses **fully local** Ollama models for LLM inference, Chroma for vector storage, and LangGraph for agent-based task execution - no external API dependencies required.

## Features

- **Local LLM Processing**: Uses Ollama with Llama 3.1 8B, Qwen2.5-7B, or Mistral-7B models
- **Local Task Execution**: LangGraph-based agent system can execute commands and perform tasks on your machine
- **Intelligent Intent Classification**: Automatically detects query intent and tailors responses
- **RAG System with Chroma**: Fast vector search using Chroma database with bge-base-en embeddings
- **OneNote Conversion**: Automatically converts .one files to PDF for RAG corpus
- **Automatic Dependency Installation**: Checks and installs all requirements on startup
- **HTTP API**: Clean boundary between C# host and Python runtime
- **Offline-First**: All models and tools run locally, no internet required after setup

## Technology Stack

### Model Runtime
- **Ollama** - Cross-platform local model runtime (CPU/GPU support)
  - Recommended models:
    - `llama3.1:8b-instruct-q4_0` (Recommended, ~4.7GB)
    - `qwen2.5:7b-instruct` (Alternative, ~4.7GB)
    - `mistral:7b-instruct` (Alternative, ~4.1GB)

### Embeddings
- **bge-base-en-v1.5** - High-quality embeddings for document retrieval

### Vector Store
- **Chroma** - Zero-config local vector database with persistence

### Agent Orchestration
- **LangGraph** - Graph-based agent framework for task execution
- **Available Tools**:
  - Execute shell commands (with safety checks)
  - List directories
  - Read/write files
  - Get system information

### UI
- **FastAPI** backend with REST API
- **Optional Streamlit** web interface

## Architecture

The application consists of two main components:

1. **C# Host Application** - Product host that:
   - Checks and installs Python dependencies automatically
   - Starts and manages the Python runtime as a child process
   - Provides an interactive console interface for chat
   - Communicates with the Python runtime over HTTP (localhost)

2. **Python Runtime Service** - FastAPI-based service that:
   - Exposes REST endpoints for health checks and chat completion
   - Handles document loading, embedding, and retrieval with Chroma
   - Routes queries to either RAG engine or task agent
   - Runs local Ollama LLM for answer generation

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│   C# Host       │ ◄───────────────────► │  Python Runtime  │
│  (AtlasAI.exe)  │  localhost:8000       │   (FastAPI)      │
└─────────────────┘                       └──────────────────┘
         │                                          │
         │ Auto-installs dependencies               ├─ RAG Engine (Ollama + Chroma)
         │ Manages process lifecycle                └─ Task Agent (LangGraph)
```

## Prerequisites

### 1. .NET SDK
- .NET 10.0 or later
- Download from: https://dotnet.microsoft.com/download

### 2. Python
- Python 3.9 or later
- Ensure Python is in your system PATH

### 3. Ollama
**Ollama is required for the chatbot to function.** The C# application will check for Ollama during startup and provide installation instructions if not found.

**Installation:**

- **Windows**: Download from https://ollama.com/download/windows
- **macOS**: Download from https://ollama.com/download/mac
- **Linux**: Run `curl -fsSL https://ollama.com/install.sh | sh`

**Install a Model:**

After installing Ollama, pull one of the recommended models:

```bash
# Recommended (best balance of quality and speed)
ollama pull llama3.1:8b-instruct-q4_0

# Alternatives
ollama pull qwen2.5:7b-instruct
ollama pull mistral:7b-instruct
```

### 4. Python Dependencies

**Dependencies are automatically installed when you first run the application!** The C# host will:
- Check Python version
- Install all required packages from `requirements.txt`
- Verify Ollama installation
- Check for available models

You can also install manually:
```bash
pip install -r requirements.txt
```

## Project Structure

```
AtlasAI/
├── AtlasAI/                     # C# console application
│   ├── Program.cs               # Main entry point
│   ├── AppConfiguration.cs      # Configuration management
│   ├── PythonRuntimeManager.cs  # Python process lifecycle
│   ├── RuntimeClient.cs         # HTTP client for runtime API
│   └── AtlasAI.csproj           # C# project file
├── atlasai_runtime/             # Python runtime service
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point for runtime
│   ├── app.py                   # FastAPI application
│   └── rag_engine.py            # RAG logic (LangChain)
├── documents/                   # PDF/DOCX files for RAG
│   ├── distribution_model_manager_user_guide.pdf
│   └── adms-16-20-0-modeling-overview-and-converter-user-guide.pdf
├── chatapp.py                   # Legacy Streamlit UI (optional)
├── requirements.txt             # Python dependencies
├── start_runtime.sh             # Unix script to start runtime
├── start_runtime.ps1            # PowerShell script to start runtime
└── README.md                    # This file
```

## Building and Running

### Quick Start (Recommended)

The application will automatically install dependencies on first run:

1. **Build the C# application:**
   ```bash
   cd AtlasAI
   dotnet build
   ```

2. **Run the application:**
   ```bash
   dotnet run
   ```

The application will:
- ✅ Check Python and pip
- ✅ Install Python dependencies automatically
- ✅ Check Ollama installation and available models
- ✅ Start the Python runtime service
- ✅ Provide an interactive chat interface

If dependencies are missing, you'll be prompted with instructions.

### Using the Application

**Console Interface:**
- Type your questions at the `You:` prompt
- The assistant will answer using RAG or execute tasks locally
- Type `exit` or `quit` to stop

**Task Execution Examples:**
```
You: list files in the current directory
You: create a file called test.txt with some content
You: show system information
You: run the command "echo Hello World"
```

**Document Q&A Examples:**
```
You: What is ADMS?
You: How do I configure the database?
You: Explain the model manager
```

**Streamlit UI:**
- Type `ui` to launch the web interface
- Opens at http://localhost:8501

### Advanced: Run Python Runtime Standalone

For development or testing, you can run the Python runtime independently:

**Requirements:**
- Ollama installed and running
- Python dependencies installed
- At least one Ollama model available

**Using the startup script (Unix/Linux/macOS):**
```bash
./start_runtime.sh
# Or with custom host/port:
./start_runtime.sh --host 0.0.0.0 --port 9000
```

**Using the startup script (Windows PowerShell):**
```powershell
.\start_runtime.ps1
# Or with custom host/port:
.\start_runtime.ps1 -Host 0.0.0.0 -Port 9000
```

**Using Python directly:**
```bash
python -m atlasai_runtime --host 127.0.0.1 --port 8000
```

Once running, you can:
- Check health: `curl http://localhost:8000/health`
- View API docs: Open `http://localhost:8000/docs` in a browser
- Send chat requests via HTTP POST to `/chat`

**Example API Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is ADMS?"}'
```

### Alternative: Run Streamlit UI Standalone

You can run the Streamlit UI independently if the Python runtime is already running:

```bash
streamlit run streamlit_ui.py
```

The UI will connect to the Python runtime at `localhost:8000`.

## Configuration

### C# Host Configuration

Configure the C# host using environment variables:

- `ATLASAI_PYTHON_PATH` - Path to Python executable (default: `python`)
- `ATLASAI_RUNTIME_HOST` - Runtime host (default: `127.0.0.1`)
- `ATLASAI_RUNTIME_PORT` - Runtime port (default: `8000`)

Example:
```bash
# Windows (PowerShell)
$env:ATLASAI_PYTHON_PATH="C:\Python39\python.exe"
$env:ATLASAI_RUNTIME_PORT="9000"
dotnet run

# Unix/Linux/macOS
export ATLASAI_PYTHON_PATH="/usr/bin/python3"
export ATLASAI_RUNTIME_PORT="9000"
dotnet run
```

### Python Runtime Configuration

Configure the Python runtime using environment variables:

- `ATLASAI_DOCUMENTS_DIR` - Path to documents folder (default: `./documents`)
- `ATLASAI_ONENOTE_RUNBOOK_PATH` - Path to OneNote runbook directory
- `ATLASAI_OLLAMA_MODEL` - Ollama model name (default: `llama3.1:8b-instruct-q4_0`)
- `ATLASAI_OLLAMA_BASE_URL` - Ollama server URL (default: `http://localhost:11434`)
- `ATLASAI_EMBEDDING_MODEL` - Embedding model name (default: `BAAI/bge-base-en-v1.5`)
- `ATLASAI_CHROMA_PERSIST_DIR` - Chroma database directory (default: `./documents/.chroma_db`)
- `ATLASAI_TOP_K` - Number of document chunks to retrieve (default: `4`)
- `ATLASAI_CHUNK_SIZE` - Size of text chunks (default: `800`)
- `ATLASAI_CHUNK_OVERLAP` - Overlap between chunks (default: `150`)
- `ATLASAI_ENABLE_AGENT` - Enable task execution agent (default: `true`)

Example:
```bash
# Windows (PowerShell)
$env:ATLASAI_OLLAMA_MODEL="qwen2.5:7b-instruct"
$env:ATLASAI_ENABLE_AGENT="false"
dotnet run

# Unix/Linux/macOS
export ATLASAI_OLLAMA_MODEL="mistral:7b-instruct"
export ATLASAI_ENABLE_AGENT="true"
dotnet run
```

## Usage

### Document Q&A

Place PDF or DOCX files in the `documents/` folder. The runtime will automatically:
1. Load and chunk the documents
2. Create embeddings using bge-base-en
3. Store in Chroma vector database
4. Use for RAG-based question answering

### Local Task Execution

The LangGraph agent can execute tasks on your local machine:

**Available Capabilities:**
- Execute shell commands (with safety checks to prevent destructive operations)
- List directory contents
- Read file contents
- Write files
- Get system information

**Safety Features:**
- Dangerous commands (rm -rf, format, etc.) are blocked
- 30-second timeout on command execution
- Clear error messages and logging

**Example Tasks:**
```
You: list files in my documents folder
You: create a file called notes.txt with "Hello World"
You: show me system information
You: read the contents of config.json
```

**Disable Agent:**
Set `ATLASAI_ENABLE_AGENT=false` to disable task execution and use only document Q&A.

### OneNote Document Support

AtlasAI provides **pure Python** conversion of OneNote (.one) files to PDF format:

**How it works:**
- On startup, scans configured OneNote path for .one files
- Creates local copies (non-destructive, originals untouched)
- Converts to PDF using pyOneNote library
- Adds PDFs to RAG corpus automatically

**Configuration:**
Set `ATLASAI_ONENOTE_RUNBOOK_PATH` to your OneNote files directory.

**Known Limitations:**
- Text content extracted successfully
- Images/screenshots not extracted (pyOneNote limitation)
- Complex formatting may not be preserved

## Troubleshooting

### "Ollama is not installed"
- Install Ollama from https://ollama.com/download
- Verify installation: `ollama --version`
- Start Ollama service (it may auto-start on some platforms)
- Pull a model: `ollama pull llama3.1:8b-instruct-q4_0`

### "No compatible models found"
Pull a recommended model:
```bash
ollama pull llama3.1:8b-instruct-q4_0
# or
ollama pull qwen2.5:7b-instruct
# or
ollama pull mistral:7b-instruct
```

### "Python runtime failed to start"
- Ensure Python 3.9+ is installed: `python --version`
- Verify Python is in PATH
- Check that pip is available: `python -m pip --version`
- Try manual install: `pip install -r requirements.txt`

### "Ollama service is not running"
- **Windows/Mac**: Start the Ollama application
- **Linux**: Run `ollama serve` in a separate terminal
- Verify: Check http://localhost:11434/api/tags in browser

### "No documents loaded"
- Verify PDF/DOCX files exist in `documents/` folder
- Check file permissions
- Ensure PDFs contain extractable text (not scanned images without OCR)

### Port already in use
- Change port: `$env:ATLASAI_RUNTIME_PORT="9000"` (PowerShell)
- Or: `export ATLASAI_RUNTIME_PORT=9000` (Linux/Mac)
- Kill existing process using port 8000

### Agent not working
- Check `ATLASAI_ENABLE_AGENT=true`
- Verify Ollama is running
- Check logs for error messages

## License

[Add your license here]

## Contact

[Add contact information here]
