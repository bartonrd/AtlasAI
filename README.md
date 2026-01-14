# AtlasAI

A Retrieval-Augmented Generation (RAG) chatbot for querying technical documentation using local language models.

## Overview

AtlasAI is a Python-based RAG chatbot with a Streamlit UI that uses local Hugging Face models to answer questions about technical documents. Version 2.0 features a modular, maintainable architecture with persistent caching for improved performance.

## Features

- **Modular Architecture**: Clean separation of concerns with reusable components
- **Persistent Caching**: Vector store caching for 10-100x faster repeated queries
- **Flexible Configuration**: Environment variable support for easy deployment
- **Local LLM Processing**: Uses offline Hugging Face models (FLAN-T5, Mistral, Llama)
- **RAG System**: Retrieves relevant context from PDF/DOCX documents before answering
- **Multi-Chat Sessions**: Manage multiple conversation threads
- **Streamlit UI**: Interactive web-based chat interface
- **Model Fallback**: Automatic fallback to alternative models if primary fails

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
Download the following models to your local machine:

**Recommended Models** (Better quality):
- **Embedding Model**: `BAAI/bge-small-en-v1.5`
- **Text Generation Model**: `google/flan-t5-large` (CPU) or `mistralai/Mistral-7B-Instruct-v0.2` (GPU)

**Default Models** (Smaller, faster):
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Text Generation Model**: `google/flan-t5-base`

Configure model paths using environment variables (recommended):

```bash
export ATLAS_EMBEDDING_MODEL="C:/models/bge-small-en-v1.5"
export ATLAS_TEXT_GEN_MODEL="C:/models/flan-t5-large"
```

Or edit `atlasai_core/config.py`:

```python
self.embedding_model = r"C:\models\bge-small-en-v1.5"
self.text_gen_model = r"C:\models\flan-t5-large"
```

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
├── AtlasAI/                 # C# console application (optional wrapper)
│   ├── Program.cs          # Main C# entry point
│   └── AtlasAI.csproj      # C# project file
├── atlasai_core/           # Modular Python components (v2.0)
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration management
│   ├── document_processor.py  # Document loading and processing
│   ├── vector_store.py     # Vector store with caching
│   ├── llm_manager.py      # LLM initialization and management
│   ├── rag_chain.py        # RAG chain implementation
│   └── utils.py            # Utility functions
├── documents/              # PDF/DOCX files for RAG
│   ├── distribution_model_manager_user_guide.pdf
│   └── adms-16-20-0-modeling-overview-and-converter-user-guide.pdf
├── .cache/                 # Vector store cache (auto-created)
├── app.py                  # Main Streamlit app (v2.0 - recommended)
├── chatapp.py              # Legacy monolithic app (v1.0)
├── requirements.txt        # Python dependencies
├── ARCHITECTURE.md         # Architecture documentation
└── README.md              # This file
```

## Building and Running

### Option 1: Run New Modular App (Recommended)

```bash
streamlit run app.py
```

Benefits:
- 10-100x faster for repeated queries (vector store caching)
- Modular, maintainable codebase
- Better model support and configuration
- Improved error handling

### Option 2: Run through C# Application

1. Build the C# application:
   ```bash
   cd AtlasAI
   dotnet build
   ```

2. Run the application:
   ```bash
   dotnet run
   ```

The C# application will launch the Streamlit chatbot. To use the new modular app, update `Program.cs` to launch `app.py` instead of `chatapp.py`.

### Option 3: Run Legacy Monolithic App

```bash
streamlit run chatapp.py
```

The legacy app is still available but the new modular architecture is recommended.

## Usage

1. Once the Streamlit UI opens in your browser, you'll see a chat interface
2. Optionally upload additional PDF or DOCX files using the file uploader
3. Type your question in the chat input at the bottom
4. The chatbot will retrieve relevant context from the documents and generate an answer
5. View source citations in the expandable "Sources" section

## Configuration

### Environment Variables (Recommended)

```bash
export ATLAS_EMBEDDING_MODEL="C:/models/bge-small-en-v1.5"
export ATLAS_TEXT_GEN_MODEL="C:/models/flan-t5-large"
export ATLAS_TOP_K=4
export ATLAS_CHUNK_SIZE=800
export ATLAS_CHUNK_OVERLAP=150
export ATLAS_USE_CACHE=true
export ATLAS_MAX_NEW_TOKENS=384
export ATLAS_USE_SAMPLING=false
export ATLAS_TEMPERATURE=0.2
```

### UI Settings

You can also adjust settings in the Streamlit UI:
- **Top K**: Number of document chunks to retrieve (1-20)
- **Chunk Size**: Size of text chunks in characters (100-2000)
- **Chunk Overlap**: Overlap between chunks (0-chunk_size)

### Advanced Configuration

Edit `atlasai_core/config.py` for advanced customization.

## Adding Documents

Place PDF or DOCX files in the `documents/` folder. The chatbot will automatically load them when you ask questions.

## Architecture

AtlasAI v2.0 features a modular architecture with:
- **Separation of Concerns**: Each module has a single responsibility
- **Persistent Caching**: Vector stores cached to disk for fast reloading
- **Flexible Configuration**: Environment variables and config files
- **Fallback Support**: Automatic model fallbacks on errors

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed documentation.

## Troubleshooting

### "streamlit: command not found"
- Ensure Streamlit is installed: `pip install streamlit`
- Verify Python's Scripts folder is in your PATH

### "Failed to load model"
- Verify model paths in environment variables or `config.py`
- Check models are downloaded to specified locations
- **New**: Fallback models will be tried automatically

### "No documents loaded"
- Verify PDF files exist in the `documents/` folder
- Check that file paths are correct
- Ensure PDFs contain extractable text (not scanned images without OCR)

### Slow first query, fast subsequent queries
- **Expected behavior**: First query loads models and creates embeddings
- **Subsequent queries**: Use cached vector store (10-100x faster)
- Clear cache by deleting `.cache/` folder if needed

### Out of memory
- Use smaller models: `flan-t5-small` instead of `flan-t5-large`
- Reduce `ATLAS_CHUNK_SIZE` and `ATLAS_TOP_K`
- Close other applications

## Model Recommendations

### Best for CPU (Balanced)
- Embedding: `BAAI/bge-small-en-v1.5` or `sentence-transformers/all-MiniLM-L6-v2`
- Text Gen: `google/flan-t5-large` (best) or `google/flan-t5-base` (faster)

### Best for GPU (Highest Quality)
- Embedding: `BAAI/bge-small-en-v1.5`
- Text Gen: `mistralai/Mistral-7B-Instruct-v0.2` or `meta-llama/Llama-2-7b-chat-hf`

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed model comparisons and download instructions.

## What's New in v2.0

### Major Improvements
✅ **Modular Architecture**: 7 focused modules instead of 1 monolithic file  
✅ **10-100x Faster**: Persistent vector store caching  
✅ **Better Configuration**: Environment variables, fallback models  
✅ **Improved Error Handling**: Automatic fallbacks and clear error messages  
✅ **Better Models**: Easy to use higher-quality models  
✅ **Cache Management**: Smart cache invalidation based on document changes  

### Migration from v1.0
Both versions can coexist. Simply start using `app.py` instead of `chatapp.py`. No breaking changes to documents or settings.

## License

[Add your license here]

## Contact

[Add contact information here]