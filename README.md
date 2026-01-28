# AtlasAI

A Retrieval-Augmented Generation (RAG) chatbot for querying technical documentation using **fully local AI models** with **Ollama** and **ChromaDB**.

## Overview

AtlasAI is a C# application that manages a Python-based RAG runtime service. The system uses **Ollama** for local LLM inference, **ChromaDB** for vector storage, and supports executing local tasks on your machine. All processing happens locally with no external API dependencies.

## Architecture

The application uses a modern local AI stack:

**Technology Stack:**
- 🤖 **Model Runtime**: Ollama (CPU/GPU, cross-platform)
- 🧠 **LLM Models**: Llama 3.1 8B Instruct (instruction-tuned, local)
- 📊 **Embeddings**: mxbai-embed-large (via Ollama)
- 🗄️ **Vector Store**: ChromaDB (zero-config, persistent)
- 🔄 **Orchestration**: LangGraph-ready architecture
- 🌐 **Backend**: FastAPI (Python)
- 💻 **Host**: C# console application
- 🎨 **UI**: Streamlit (optional web interface)

**System Components:**

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│   C# Host       │ ◄───────────────────► │  Python Runtime  │
│  (AtlasAI.exe)  │  localhost:8000       │   (FastAPI)      │
└─────────────────┘                       └──────────────────┘
        │                                          │
        │ • Auto-installs requirements             │ • RAG Engine (Ollama)
        │ • Manages process lifecycle              │ • ChromaDB vector store
        │ • Interactive console                    │ • Task executor
                                                   │ • OneNote/PDF loaders
```

## Features

- 🏠 **Fully Local**: All AI processing happens on your machine using Ollama
- 📚 **RAG System**: Retrieves relevant context from PDF/DOCX documents before answering
- 🧠 **Intelligent Intent Classification**: Automatically detects query intent and tailors responses
- 📝 **OneNote Support**: Automatically converts .one files to PDF for processing
- 🔧 **Local Task Execution**: Execute system commands and tasks on your machine
- ⚡ **Auto-Setup**: Automatically installs Python requirements on first run
- 🎯 **Zero External APIs**: No cloud dependencies or API keys needed
- 💾 **Persistent Vector Store**: ChromaDB stores embeddings for fast retrieval
- 🌐 **REST API**: Clean FastAPI interface with automatic documentation

## Prerequisites

### 1. .NET SDK
- .NET 10.0 or later
- Download from: https://dotnet.microsoft.com/download

### 2. Python
- Python 3.9 or later
- Ensure Python is in your system PATH

### 3. Ollama
**AtlasAI requires Ollama to be installed and running.**

Install Ollama:
- **Windows**: Download from [ollama.ai/download/windows](https://ollama.ai/download/windows)
- **macOS**: `brew install ollama` or download from [ollama.ai/download/mac](https://ollama.ai/download/mac)
- **Linux**: `curl -fsSL https://ollama.ai/install.sh | sh`

After installation, start the Ollama service:
```bash
ollama serve
```

Pull the default model (will be done automatically on first run):
```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

### 4. Python Dependencies
**Python dependencies are automatically installed when you run the C# application.**

The setup script will install:
- FastAPI and Uvicorn (web framework)
- Ollama Python client
- ChromaDB (vector database)
- LangChain components
- Document loaders (PDF, DOCX, OneNote)

You can also manually install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
AtlasAI/
├── AtlasAI/                     # C# console application
│   ├── Program.cs               # Main entry point
│   ├── AppConfiguration.cs      # Configuration management
│   ├── PythonRuntimeManager.cs  # Python process lifecycle (with auto-setup)
│   ├── RuntimeClient.cs         # HTTP client for runtime API
│   └── AtlasAI.csproj           # C# project file
├── atlasai_runtime/             # Python runtime service
│   ├── __init__.py              # Package initialization
│   ├── __main__.py              # Entry point for runtime
│   ├── app.py                   # FastAPI application
│   ├── rag_engine.py            # RAG logic (Ollama + ChromaDB)
│   ├── task_executor.py         # Local task execution
│   ├── intent_classifier.py     # Intent detection
│   └── onenote_converter.py     # OneNote to PDF conversion
├── documents/                   # PDF/DOCX files for RAG corpus
├── chroma_db/                   # ChromaDB persistent storage (auto-created)
├── setup_env.py                 # Environment setup script (auto-run by C#)
├── requirements.txt             # Python dependencies
├── streamlit_ui.py              # Optional Streamlit web UI
└── README.md                    # This file
```
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

The C# application handles everything automatically:

1. **Build the C# application:**
   ```bash
   cd AtlasAI
   dotnet build
   ```

2. **Run the application:**
   ```bash
   dotnet run
   ```

   On first run, the application will:
   - ✓ Check Python version
   - ✓ Install all required Python packages automatically
   - ✓ Check if Ollama is installed and running
   - ✓ Pull the default Llama 3.1 8B model (if needed)
   - ✓ Initialize ChromaDB vector store
   - ✓ Convert OneNote files to PDF
   - ✓ Start the FastAPI runtime service
   - ✓ Provide an interactive chat interface

3. **Start chatting!**
   - Type your questions at the `You:` prompt
   - View the assistant's answer with source citations
   - Type `ui` to launch the Streamlit web interface
   - Type `exit` or press Ctrl+C to quit

### Manual Setup (Optional)

If you prefer to set up manually:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install and start Ollama
ollama serve

# Pull models
ollama pull llama3.1:8b
ollama pull mxbai-embed-large

# Run the C# application
cd AtlasAI
dotnet run
```

#### Using the Streamlit UI

While the console is running, type `ui` and press Enter to launch the Streamlit web UI. The UI provides:
- Multiple chat sessions with history
- Graphical interface with source citations
- Document-based Q&A in a user-friendly format

The Streamlit UI will open in your default web browser at `http://localhost:8501`

### Option 2: Run Python Runtime Standalone

You can run the Python runtime service independently for development or testing.

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

### Option 3: Run Streamlit UI Standalone

You can run the Streamlit UI independently if the Python runtime is already running:

```bash
streamlit run streamlit_ui.py
```

The UI will connect to the Python runtime at `localhost:8000` (configurable via `ATLASAI_RUNTIME_HOST` and `ATLASAI_RUNTIME_PORT` environment variables).

### Option 4: Run Legacy Streamlit UI (Deprecated)

The original Streamlit UI with embedded RAG logic is still available but deprecated:

```bash
streamlit run chatapp.py
```

## Configuration

### Environment Variables

Configure AtlasAI using environment variables:

**C# Host Configuration:**
- `ATLASAI_PYTHON_PATH` - Path to Python executable (default: `python`)
- `ATLASAI_RUNTIME_HOST` - Runtime host (default: `127.0.0.1`)
- `ATLASAI_RUNTIME_PORT` - Runtime port (default: `8000`)

**Python Runtime Configuration:**
- `ATLASAI_DOCUMENTS_DIR` - Path to documents folder (default: `./documents`)
- `ATLASAI_ONENOTE_RUNBOOK_PATH` - Path to OneNote runbook directory
- `ATLASAI_OLLAMA_MODEL` - Ollama model name (default: `llama3.1:8b`)
- `ATLASAI_EMBEDDING_MODEL` - Embedding model (default: `mxbai-embed-large`)
- `ATLASAI_CHROMA_PERSIST_DIR` - ChromaDB storage path (default: `./chroma_db`)
- `ATLASAI_TOP_K` - Number of document chunks to retrieve (default: `6`)
- `ATLASAI_CHUNK_SIZE` - Size of text chunks (default: `800`)
- `ATLASAI_CHUNK_OVERLAP` - Overlap between chunks (default: `150`)

**Example:**
```bash
# Windows (PowerShell)
$env:ATLASAI_OLLAMA_MODEL="mistral:7b"
$env:ATLASAI_TOP_K="10"
dotnet run

# Unix/Linux/macOS
export ATLASAI_OLLAMA_MODEL="mistral:7b"
export ATLASAI_TOP_K="10"
dotnet run
```

### Using Different Ollama Models

AtlasAI supports any Ollama model. Recommended options:

**8-14B models (best quality/speed balance):**
- `llama3.1:8b` (default) - Meta's Llama 3.1, instruction-tuned
- `qwen2.5:7b` - Alibaba's Qwen 2.5, excellent for technical content
- `mistral:7b` - Mistral AI's flagship model

**Smaller models (faster on CPU):**
- `llama3.1:3b` - Faster Llama variant
- `phi3:mini` - Microsoft's compact model

**Larger models (better quality, needs GPU):**
- `llama3.1:13b` - Larger Llama variant
- `qwen2.5:14b` - Larger Qwen variant

To use a different model:
```bash
ollama pull qwen2.5:7b
export ATLASAI_OLLAMA_MODEL="qwen2.5:7b"
dotnet run
```

## Usage

### Interactive Console Chat

1. Start the application with `dotnet run`
2. Wait for the runtime to be ready (you'll see "AtlasAI is ready!")
3. Type your questions at the `You:` prompt
4. View the assistant's answer with intent detection and source citations
5. Type `ui` to launch the Streamlit web interface
6. Type `exit` or `quit` to stop, or press Ctrl+C

**Intent Detection:**
AtlasAI automatically detects your query intent and provides tailored responses:
- 🔧 **Error Resolution**: Troubleshooting and fixing issues
- 📝 **How-To**: Step-by-step instructions
- 💬 **Chit-Chat**: Casual conversation
- 📚 **Concept Explanation**: Technical details and definitions

### Local Task Execution

AtlasAI can execute system commands on your machine via the REST API:

**Get System Information:**
```bash
curl http://localhost:8000/task/system-info
```

**Execute a Command:**
```bash
curl -X POST http://localhost:8000/task/execute \
  -H "Content-Type: application/json" \
  -d '{
    "command": "python --version",
    "timeout": 30
  }'
```

**Security Note:** For safety, only whitelisted commands are allowed. The default whitelist includes common developer tools (python, git, npm, etc.). Dangerous commands like `rm -rf` require explicit permission.

### REST API Usage

When running the Python runtime, you can interact via HTTP:

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Chat Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I configure the database?",
    "additional_documents": []
  }'
```

**Response Format:**
```json
{
  "answer": "- Step 1: Configure connection string\n- Step 2: Run migrations\n- Step 3: Verify connection",
  "sources": [
    {"index": 1, "source": "database_guide.pdf", "page": "5"},
    {"index": 2, "source": "config_manual.pdf", "page": "12"}
  ],
  "intent": "how_to",
  "intent_confidence": 0.87
}
```

**Interactive API Documentation:**
Open `http://localhost:8000/docs` in your browser for full Swagger UI documentation.

### Streamlit Web UI

Launch the web interface by typing `ui` in the console, or run standalone:

```bash
streamlit run streamlit_ui.py
```

The UI provides:
- Multiple chat sessions with history
- Graphical interface with source citations
- Document-based Q&A in a user-friendly format
- Real-time intent detection display

## Intent Classification

AtlasAI includes an intelligent intent classification system that automatically detects the type of user query and provides tailored responses. See [INTENT_CLASSIFICATION.md](INTENT_CLASSIFICATION.md) for detailed documentation.

### Intent Categories

- **error_log_resolution**: Troubleshooting and resolving errors
- **how_to**: Step-by-step instructions for tasks  
- **chit_chat**: Casual conversation
- **concept_explanation**: Technical concepts and definitions

### How It Works

The system automatically:
1. Classifies each query into one of four intent categories
2. Selects an intent-specific prompt template
3. **Displays the intent and confidence score before the response**
4. Returns responses optimized for that intent type
5. Includes intent metadata in API responses

### Display Example

```
You: How do I configure the database?
🎯 Intent: How To | Confidence: 87.3%

The system uses zero-shot classification when available, with automatic fallback to keyword-based classification (91.7% accuracy) for offline scenarios.

## Adding Documents

Place PDF or DOCX files in the `documents/` folder. The runtime will automatically load and index them on startup or first query.

**Supported Formats:**
- PDF (`.pdf`) - Automatically extracted text content
- Word Documents (`.docx`) - Full text extraction
- OneNote (`.one`) - Converted to PDF automatically (see below)

### OneNote Document Support

AtlasAI provides **pure Python** OneNote to PDF conversion **without requiring Windows COM or OneNote installation**. This makes it cross-platform compatible.

**How it works:**
- On startup, scans the configured OneNote runbook path for `.one` files
- Creates local copies in `documents/onenote_copies/` (non-destructive)
- Converts all `.one` files to PDF using Python libraries
- Saves PDFs in `documents/runbook/` for indexing
- Original files remain completely untouched

**Configuration:**
Set the OneNote directory path:
```bash
export ATLASAI_ONENOTE_RUNBOOK_PATH="/path/to/onenote/files"
```

**Manual Conversion:**
```bash
# Convert a single file
python convert_onenote.py input.one output.pdf

# Convert an entire directory
python convert_onenote.py input_dir/ output_dir/ --directory --use-local-copies
```

**Known Limitations:**
- Special characters may not render perfectly
- Screenshots/images are not extracted
- Complex formatting may not be preserved
- Text content and structure are fully extracted

## Troubleshooting

### "Ollama not found" or "Ollama service not running"
- Install Ollama from [ollama.ai](https://ollama.ai)
- Start the service: `ollama serve`
- Verify it's running: `curl http://localhost:11434/api/tags`

### "Failed to pull model"
- Ensure Ollama is running: `ollama serve`
- Manually pull the model: `ollama pull llama3.1:8b`
- Check disk space (models are 4-8GB)
- Check internet connection

### "Python runtime failed to start"
- Ensure Python 3.9+ is installed and in PATH
- Check Python version: `python --version`
- Manually run setup: `python setup_env.py`
- Check logs for specific errors

### "Failed to initialize RAG engine"
- Ensure Ollama is running and models are pulled
- Check `documents/` folder exists and contains files
- Verify ChromaDB can write to `chroma_db/` directory
- Check logs for specific errors

### "No documents loaded"
- Verify PDF/DOCX files exist in `documents/` folder
- Ensure files contain extractable text (not scanned images)
- Check file permissions
- Check OneNote conversion logs if using `.one` files

### Port already in use (8000)
- Change port: `export ATLASAI_RUNTIME_PORT=9000`
- Kill existing process: `lsof -ti:8000 | xargs kill` (Unix) or Task Manager (Windows)

### ChromaDB errors
- Delete `chroma_db/` folder to reset vector store
- Ensure write permissions for the directory
- Check disk space

### Slow performance
- Use a smaller model: `export ATLASAI_OLLAMA_MODEL="llama3.1:3b"`
- Reduce chunks retrieved: `export ATLASAI_TOP_K="3"`
- Ensure Ollama has GPU support if available
- Close other applications to free RAM

## License

[Add your license here]

## Contact

[Add contact information here]
