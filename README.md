# AtlasAI

A Retrieval-Augmented Generation (RAG) chatbot for querying technical documentation using local language models.

## Overview

AtlasAI is a C# application that integrates a Python-based LLM chatbot with a Streamlit UI. The chatbot uses local Hugging Face models to answer questions about technical documents stored in the `documents` folder.

**NEW in v2.0**: Completely refactored with a modular, production-ready architecture! See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

## Features

- **Modular Architecture**: Clean separation of concerns with dedicated layers for config, services, UI, and utilities
- **Performance Optimizations**: Model caching, vector store caching, and lazy loading for 10-50x faster response times
- **Better Model Support**: Easily switch to better models like Mistral-7B-Instruct or Phi-2 (see [ARCHITECTURE.md](ARCHITECTURE.md))
- **Local LLM Processing**: Uses offline Hugging Face models for text generation
- **RAG System**: Retrieves relevant context from PDF/DOCX documents before answering
- **Streamlit UI**: Interactive web-based chat interface with multi-chat support
- **Configuration Management**: Environment variable support and centralized settings
- **C# Wrapper**: Launches the chatbot through a .NET console application

## Prerequisites

### 1. .NET SDK
- .NET 10.0 or later
- Download from: https://dotnet.microsoft.com/download

### 2. Python
- **Python 3.9 or later** (required for type annotations)
- Ensure Python is in your system PATH

### 3. Python Dependencies
Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Hugging Face Models

**Option 1: Use Models from HuggingFace Hub (Recommended for v2.0)**

The new modular version (`app.py`) will automatically download models from HuggingFace on first use. No manual setup required!

Recommended models (in priority order):
- **microsoft/Phi-2** - Best for Q&A, 2.7B parameters
- **mistralai/Mistral-7B-Instruct-v0.2** - Superior quality, 7B parameters
- **google/flan-t5-large** - Good quality, 780M parameters
- **google/flan-t5-base** - Default, 250M parameters (faster)

Set your preferred model via environment variable:
```bash
export ATLAS_TEXT_GEN_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
```

**Option 2: Use Local Models**

Download models to your local machine for offline use:

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Text Generation Model**: Your choice from above

Set paths via environment variables:
```bash
export ATLAS_LOCAL_EMBEDDING_MODEL="/path/to/all-MiniLM-L6-v2"
export ATLAS_LOCAL_TEXT_GEN_MODEL="/path/to/mistral-7b-instruct"
```

Or update in `chatapp.py` (legacy version):
```python
EMBEDDING_MODEL = r"C:\models\all-MiniLM-L6-v2"
LOCAL_TEXT_GEN_MODEL = r"C:\models\flan-t5-base"
```

Download models using Python:

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
├── AtlasAI/                 # C# console application
│   ├── Program.cs          # Main C# entry point
│   └── AtlasAI.csproj      # C# project file
├── src/                    # Modular Python application (NEW)
│   ├── config/            # Configuration management
│   │   └── settings.py   # Centralized settings
│   ├── services/         # Business logic layer
│   │   ├── document_service.py   # Document processing
│   │   ├── embedding_service.py  # Embeddings with caching
│   │   ├── llm_service.py        # LLM with model caching
│   │   └── rag_service.py        # RAG orchestration
│   ├── utils/            # Utility functions
│   │   └── formatting.py
│   └── ui/               # Presentation layer
│       └── chat_interface.py     # Streamlit UI
├── documents/              # PDF/DOCX files for RAG
│   ├── distribution_model_manager_user_guide.pdf
│   └── adms-16-20-0-modeling-overview-and-converter-user-guide.pdf
├── app.py                  # Main entry point (NEW modular version)
├── chatapp.py              # Legacy monolithic version (preserved)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── ARCHITECTURE.md        # Architecture documentation (NEW)
```

## Building and Running

### Option 1: Run through C# Application (Recommended)

1. Build the C# application:
   ```bash
   cd AtlasAI
   dotnet build
   ```

2. Run the application:
   ```bash
   dotnet run
   ```

The C# application will automatically launch the Streamlit chatbot and open it in your default web browser.

### Option 2: Run Python Script Directly

**NEW Modular Version (Recommended)**:
```bash
streamlit run app.py
```

**Legacy Version** (for backward compatibility):
```bash
streamlit run chatapp.py
```

## Usage

1. Once the Streamlit UI opens in your browser, you'll see a chat interface
2. Optionally upload additional PDF or DOCX files using the Documents tab in the sidebar
3. Configure RAG settings (Top K, Chunk Size, Overlap) in the Settings tab
4. Type your question in the chat input at the bottom
5. The chatbot will retrieve relevant context from the documents and generate an answer
6. View source citations in the expandable "Sources" section
7. Create multiple chat sessions using the Chats tab

## Configuration

### Using Environment Variables (NEW in v2.0)

```bash
# Model configuration
export ATLAS_TEXT_GEN_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export ATLAS_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Local model paths (optional)
export ATLAS_LOCAL_TEXT_GEN_MODEL="/path/to/local/model"
export ATLAS_LOCAL_EMBEDDING_MODEL="/path/to/local/embedding"

# RAG configuration
export ATLAS_TOP_K=4
export ATLAS_CHUNK_SIZE=800
export ATLAS_CHUNK_OVERLAP=150

# Run the app
streamlit run app.py
```

### Using Settings UI

Use the Settings tab in the sidebar to adjust:
- **Top K**: Number of document chunks to retrieve (1-20, default: 4)
- **Chunk Size**: Size of text chunks for splitting (100-2000, default: 800)
- **Chunk Overlap**: Overlap between chunks (0-50% of chunk size, default: 150)

### Legacy Configuration (chatapp.py)

Edit `chatapp.py` directly to customize hardcoded settings.

## Adding Documents

Place PDF or DOCX files in the `documents/` folder, or upload them via the Documents tab in the UI. The chatbot will automatically load them when you ask questions.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed information about:
- Modular architecture design
- Performance optimizations
- Model recommendations
- Testing strategies
- Migration guide
- Best practices

## Troubleshooting

### "streamlit: command not found"
- Ensure Streamlit is installed: `pip install streamlit`
- Verify Python's Scripts folder is in your PATH

### "Failed to load local HF model"
- Verify model paths are correct (use environment variables or check `src/config/settings.py`)
- Ensure models are downloaded to the specified locations
- Check that the models are compatible
- Try using HuggingFace Hub models instead (no local download needed)

### "No documents loaded"
- Verify PDF files exist in the `documents/` folder
- Check that file paths are correct
- Ensure PDFs contain extractable text (not scanned images without OCR)

## License

[Add your license here]

## Contact

[Add contact information here]