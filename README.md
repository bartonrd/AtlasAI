# AtlasAI

A Retrieval-Augmented Generation (RAG) chatbot for querying technical documentation using local language models.

## Overview

AtlasAI is a C# application that communicates with a Python-based RAG runtime service over localhost. The system uses local Hugging Face models to answer questions about technical documents stored in the `documents` folder, with no external dependencies required.

## Architecture

The application consists of two main components:

1. **C# Host Application** - Product host that:
   - Starts and manages the Python runtime as a child process
   - Provides an interactive console interface for chat
   - Communicates with the Python runtime over HTTP (localhost)

2. **Python Runtime Service** - FastAPI-based service that:
   - Exposes REST endpoints for health checks and chat completion
   - Handles document loading, embedding, and retrieval
   - Runs the local LLM for answer generation

```
┌─────────────────┐         HTTP          ┌──────────────────┐
│   C# Host       │ ◄───────────────────► │  Python Runtime  │
│  (AtlasAI.exe)  │  localhost:8000       │   (FastAPI)      │
└─────────────────┘                       └──────────────────┘
        │                                          │
        │ Manages process lifecycle                │ RAG Engine
        │ (start/stop/monitor)                     │ (LangChain)
```

## Features

- **Local LLM Processing**: Uses offline Hugging Face models (FLAN-T5) for text generation
- **RAG System**: Retrieves relevant context from PDF/DOCX documents before answering
- **OneNote Integration**: Ingest Microsoft OneNote documents (Windows only) using COM API
- **HTTP API**: Clean boundary between C# host and Python runtime
- **Offline-First**: No required external SaaS dependencies
- **Interactive Console**: Simple chat interface in the C# application
- **Process Management**: C# host automatically starts/stops the Python runtime
- **Graceful Degradation**: Continues operating if optional features (like OneNote) are unavailable

## Prerequisites

### 1. .NET SDK
- .NET 10.0 or later
- Download from: https://dotnet.microsoft.com/download

### 2. Python
- Python 3.9 or later
- Ensure Python is in your system PATH

### 3. Python Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

**Note for Windows users:** The `requirements.txt` includes `pywin32` which is needed for OneNote COM API integration. On non-Windows platforms, this package will be installed but the OneNote feature will be gracefully disabled at runtime.

### 4. Hugging Face Models

**Note**: Models are required for actual chat functionality. The runtime service will start and respond to `/health` checks without models, but `/chat` requests will fail.

Download the following models to your local machine:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Text Generation Model**: `google/flan-t5-base` (or `google/flan-t5-small` for faster CPU runs)

Update the model paths via environment variables (see Configuration section) or use the defaults in the code:

- Default embedding model path: `C:\models\all-MiniLM-L6-v2`
- Default text generation model path: `C:\models\flan-t5-base`

You can download models using Python:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Download and save embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model.save(r'C:\models\all-MiniLM-L6-v2')

# Download and save text generation model
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
tokenizer.save_pretrained(r'C:\models\flan-t5-base')
model.save_pretrained(r'C:\models\flan-t5-base')
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

### Option 1: Run through C# Host (Recommended)

This is the recommended way to use AtlasAI. The C# host will automatically start the Python runtime.

1. **Install Python dependencies** (required before first run):
   ```bash
   pip install -r requirements.txt
   ```

2. Build the C# application:
   ```bash
   cd AtlasAI
   dotnet build
   ```

3. Run the application:
   ```bash
   dotnet run
   ```

The C# application will:
- Start the Python runtime service on localhost:8000
- Wait for the runtime to be healthy
- Provide an interactive chat interface in the console
- **Type 'ui' to launch the Streamlit UI** for a graphical web interface
- Automatically shut down the runtime when you exit

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
- `ATLASAI_EMBEDDING_MODEL` - Path to embedding model (default: `C:\models\all-MiniLM-L6-v2`)
- `ATLASAI_TEXT_GEN_MODEL` - Path to text generation model (default: `C:\models\flan-t5-base`)
- `ATLASAI_TOP_K` - Number of document chunks to retrieve (default: `4`)
- `ATLASAI_CHUNK_SIZE` - Size of text chunks (default: `800`)
- `ATLASAI_CHUNK_OVERLAP` - Overlap between chunks (default: `150`)

#### OneNote Integration (Windows Only)

AtlasAI can ingest Microsoft OneNote documents using the OneNote COM API:

- `ENABLE_ONENOTE` - Enable OneNote document ingestion (default: `true`, set to `false` to disable)
- `ONENOTE_RUNBOOK_PATH` - UNC or local path to OneNote files (default: `\\sce\workgroup\TDBU2\TD-PSC\PSC-DMS-ADV-APP\ADMS Operation & Maintenance Docs\Model Manager Runbook`)

**Requirements:**
- Windows OS with OneNote 2016 or Office 365 installed
- Python package `pywin32` (included in requirements.txt)
- Network access to UNC paths if using remote OneNote files

**Features:**
- Recursively processes all `.one` files in the specified directory
- Exports each OneNote page to HTML and extracts clean text
- Preserves metadata (notebook, section, page title, source path)
- Automatically deduplicates pages by page ID
- Gracefully degrades if OneNote is not installed (logs warning and continues)
- Handles locked files and access errors without crashing

Example:
```bash
# Windows (PowerShell) - with custom OneNote path
$env:ATLASAI_EMBEDDING_MODEL="D:\models\embeddings"
$env:ATLASAI_TEXT_GEN_MODEL="D:\models\flan-t5-small"
$env:ONENOTE_RUNBOOK_PATH="\\server\share\OneNote\Runbook"
python -m atlasai_runtime

# Unix/Linux/macOS (OneNote not available on these platforms)
export ATLASAI_EMBEDDING_MODEL="/home/user/models/embeddings"
export ATLASAI_TEXT_GEN_MODEL="/home/user/models/flan-t5-small"
python -m atlasai_runtime
```

## Usage

### Using the C# Console Interface

1. Start the application with `dotnet run`
2. Wait for the runtime to be ready (you'll see "AtlasAI is ready!")
3. Type your questions at the `You:` prompt
4. View the assistant's answer and source citations
5. Type `exit` or `quit` to stop, or press Ctrl+C

### Using the Python Runtime API

When running the Python runtime standalone, you can interact with it via HTTP:

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Chat Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is ADMS?"}'
```

**Interactive API Documentation:**

Open `http://localhost:8000/docs` in your browser to access the interactive FastAPI documentation (Swagger UI).

## Adding Documents

Place PDF or DOCX files in the `documents/` folder. The runtime will automatically load them when processing queries.

### Adding OneNote Documents (Windows Only)

**Important:** OneNote ingestion is **enabled by default** on Windows systems with the default UNC path configured.

To customize OneNote document ingestion:

1. **Disable the feature** (optional): Set the environment variable `ENABLE_ONENOTE=false` to disable
2. **Configure the path** (optional): Set `ONENOTE_RUNBOOK_PATH` to point to your OneNote files (defaults to the Model Manager Runbook UNC path)
3. **Ensure OneNote is installed**: OneNote 2016 or Office 365 must be installed on your Windows machine
4. **Start/restart the runtime**: The ingestion happens when the runtime starts

Example to customize (Windows PowerShell):
```powershell
# To use a custom path (feature is enabled by default)
$env:ONENOTE_RUNBOOK_PATH="\\server\share\path\to\onenote"
python -m atlasai_runtime

# To disable OneNote ingestion
$env:ENABLE_ONENOTE="false"
python -m atlasai_runtime
```

**Verification:**
Check the runtime logs during startup. You should see:
- `OneNote ingestion enabled. Loading from: <path>`
- `Successfully loaded X OneNote pages from <path>`
- `Total documents loaded: Y (including OneNote pages if enabled)`

The system will:
- Recursively find all `.one` files in the specified path
- Export each page to HTML and extract clean text
- Add the content to the RAG corpus alongside PDF/DOCX files
- Store metadata including notebook, section, and page titles

**Note:** 
- OneNote ingestion happens during runtime initialization (when loading documents for a query)
- If the OneNote COM API is unavailable or files are inaccessible, the system will log warnings and continue without OneNote documents
- To disable OneNote ingestion, set `ENABLE_ONENOTE=false`

## Troubleshooting

### "Python runtime failed to start"
- Ensure Python is installed and in your PATH
- Verify Python packages are installed: `pip install -r requirements.txt`
- Check that ML models are downloaded to the configured paths

### "Failed to load local HF model"
- Verify model paths are correct (check environment variables)
- Ensure models are downloaded to the specified locations
- Check that the models are compatible (FLAN-T5 for text generation, sentence-transformers for embeddings)

### "No documents loaded"
- Verify PDF/DOCX files exist in the `documents/` folder
- Check that file paths are correct
- Ensure PDFs contain extractable text (not scanned images without OCR)
- If using OneNote, verify `ONENOTE_RUNBOOK_PATH` is set correctly

### OneNote documents not being ingested
**Note: OneNote is enabled by default, but may fail if prerequisites aren't met.**

1. **Check if disabled**: Verify `ENABLE_ONENOTE` environment variable is not set to `false`
2. **Check the logs**: Look for these messages during runtime startup or when making a query:
   - `OneNote ingestion enabled. Loading from: <path>` - means it's trying to load
   - `Successfully loaded X OneNote pages` - means it worked!
   - `No OneNote documents found at <path>` - means path is empty or no .one files
   - `OneNote ingestion is disabled` - means feature flag is off
   - `OneNote enabled but ONENOTE_RUNBOOK_PATH is not set` - means you need to set the path
3. **Verify Windows & OneNote**: Ensure you're running on Windows with OneNote 2016 or Office 365 installed
4. **Check the path**: Verify `ONENOTE_RUNBOOK_PATH` is accessible (test with Windows Explorer)
5. **For UNC paths**: Ensure you have network access and proper permissions
6. **Check application logs**: Look for error messages with stack traces for more details

Example to verify settings:
```powershell
# Check if environment variable is set (should be empty or "true" for enabled)
echo $env:ENABLE_ONENOTE

# Check health endpoint
curl http://localhost:8000/health
# Look for "enable_onenote": true in the response
```

### Port already in use
- Change the runtime port using environment variables or command-line arguments
- Kill any existing processes using port 8000

## License

[Add your license here]

## Contact

[Add contact information here]
