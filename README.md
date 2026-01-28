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

- **Intent Inferring**: Automatically detects user intent (error resolution, how-to, chit-chat, concept explanation) and provides contextually appropriate responses
- **Local LLM Processing**: Uses offline Hugging Face models (FLAN-T5) for text generation
- **RAG System**: Retrieves relevant context from PDF/DOCX documents before answering
- **OneNote Conversion**: Automatically converts .one files to PDF on startup for better text extraction
- **HTTP API**: Clean boundary between C# host and Python runtime
- **Offline-First**: No required external SaaS dependencies
- **Interactive Console**: Simple chat interface in the C# application
- **Process Management**: C# host automatically starts/stops the Python runtime

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
- `ATLASAI_ONENOTE_RUNBOOK_PATH` - Path to OneNote runbook directory (default: `\\sce\workgroup\TDBU2\TD-PSC\PSC-DMS-ADV-APP\ADMS Operation & Maintenance Docs\Model Manager Runbook`)
- `ATLASAI_EMBEDDING_MODEL` - Path to embedding model (default: `C:\models\all-MiniLM-L6-v2`)
- `ATLASAI_TEXT_GEN_MODEL` - Path to text generation model (default: `C:\models\flan-t5-base`)
- `ATLASAI_TOP_K` - Number of document chunks to retrieve (default: `4`)
- `ATLASAI_CHUNK_SIZE` - Size of text chunks (default: `800`)
- `ATLASAI_CHUNK_OVERLAP` - Overlap between chunks (default: `150`)

Example:
```bash
# Windows (PowerShell)
$env:ATLASAI_EMBEDDING_MODEL="D:\models\embeddings"
$env:ATLASAI_TEXT_GEN_MODEL="D:\models\flan-t5-small"
$env:ATLASAI_ONENOTE_RUNBOOK_PATH="\\server\path\to\runbook"
python -m atlasai_runtime

# Unix/Linux/macOS
export ATLASAI_EMBEDDING_MODEL="/home/user/models/embeddings"
export ATLASAI_TEXT_GEN_MODEL="/home/user/models/flan-t5-small"
export ATLASAI_ONENOTE_RUNBOOK_PATH="/mnt/network/path/to/runbook"
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

### OneNote Document Support

AtlasAI provides a **pure Python solution** for converting OneNote (.one) files to PDF format **without requiring Windows COM automation or OneNote installation**. This makes it cross-platform compatible and fully automated.

**Non-Destructive Conversion:**
- The system now uses **non-destructive mode** by default
- Creates local copies of OneNote files before processing
- Original .one files remain completely untouched
- Local copies are stored in `documents/onenote_copies/`
- All conversion processing happens on the local copies

**How it works:**
- On startup, the application scans the configured OneNote runbook path for .one files
- Local copies are created in the `documents/onenote_copies/` directory
- All .one files are converted to PDF format using the built-in Python converter
- Converted PDFs are saved in the `documents/runbook/` folder
- The PDFs are automatically loaded into the RAG context
- The runbook folder is cleared and regenerated on each startup to ensure fresh conversions

**Pure Python Conversion (No OneNote Required):**
The conversion uses:
- `pyOneNote` library to parse the OneNote file structure
- `reportlab` library to generate searchable PDF documents
- Works on Windows, Linux, and macOS
- No COM automation or OneNote installation needed

**Known Limitations:**
- Special characters may not render perfectly (pyOneNote limitation)
- Screenshots/images are not extracted (pyOneNote limitation)
- Complex formatting may not be preserved
- Text content, metadata, and structure are extracted for RAG processing

**Manual Conversion Options:**

You can also convert OneNote files manually using the provided script:

```bash
# Convert a single file
python convert_onenote.py input.one output.pdf

# Convert an entire directory (standard mode)
python convert_onenote.py input_dir/ output_dir/ --directory

# Non-destructive conversion (creates local copies first)
python convert_onenote.py input_dir/ output_dir/ --directory --use-local-copies

# Convert with verbose logging
python convert_onenote.py input.one output.pdf --verbose

# Show conversion capabilities
python convert_onenote.py --info
```

**Programmatic Usage:**

```python
from atlasai_runtime.onenote_converter import (
    convert_onenote_to_pdf,
    batch_convert_onenote_to_pdf,
    convert_onenote_directory,
    copy_onenote_files_locally
)

# Convert single file
convert_onenote_to_pdf("notes.one", "notes.pdf", verbose=True)

# Batch convert multiple files
files = ["notes1.one", "notes2.one", "notes3.one"]
results = batch_convert_onenote_to_pdf(files, "output_dir/")

# Convert entire directory (non-destructive mode)
count = convert_onenote_directory(
    "onenote_files/", 
    "pdf_output/",
    use_local_copies=True,
    local_copy_dir="local_copies/"
)

# Create local copies only
copy_mapping = copy_onenote_files_locally(files, "local_copies/")
```

**Configuration:**
- Set `ATLASAI_ONENOTE_RUNBOOK_PATH` environment variable to point to your OneNote files directory
- Default path: `\\sce\workgroup\TDBU2\TD-PSC\PSC-DMS-ADV-APP\ADMS Operation & Maintenance Docs\Model Manager Runbook`

**Safety**: The non-destructive mode ensures your original OneNote files are never modified or damaged during conversion. All processing happens on local copies stored in the documents folder.

## Intent Inferring

AtlasAI includes an intelligent intent detection system that automatically identifies the type of question you're asking and provides contextually appropriate responses. See [INTENT_INFERRING.md](INTENT_INFERRING.md) for detailed documentation.

### Supported Intent Types:

1. **Error Log Resolution** - Troubleshooting and fixing errors
2. **How-To** - Step-by-step instructions for tasks
3. **Chit-Chat** - Casual conversation and greetings
4. **Concept Explanation** - Understanding concepts and definitions

The system uses a combination of keyword matching and optional zero-shot classification to determine intent, then applies specialized prompt templates to generate more relevant and helpful responses.

**Benefits:**
- Better responses for short or ambiguous queries
- Faster responses for casual conversation
- Specialized formatting for different question types
- Intent metadata in API responses for analytics

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

### Port already in use
- Change the runtime port using environment variables or command-line arguments
- Kill any existing processes using port 8000

## License

[Add your license here]

## Contact

[Add contact information here]
