# Migration Guide: AtlasAI 1.0 to 2.0

This guide helps you migrate from the old AtlasAI stack (HuggingFace + FAISS) to the new stack (Ollama + ChromaDB).

## What's Changed

### Old Stack (v1.x)
- **LLM**: Local HuggingFace models (FLAN-T5)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (in-memory)
- **Manual Setup**: Required downloading and configuring models

### New Stack (v2.0)
- **LLM**: Ollama (Llama 3.1, Qwen, Mistral)
- **Embeddings**: mxbai-embed-large (via Ollama)
- **Vector Store**: ChromaDB (persistent)
- **Orchestration**: LangGraph-ready architecture
- **Task Execution**: Can run local system commands
- **Auto Setup**: Automated installation and configuration

## Benefits of Upgrading

1. **Better Performance**: Ollama provides faster inference and better quality outputs
2. **Easier Setup**: No manual model downloads, everything automated
3. **Persistent Storage**: ChromaDB saves embeddings between runs
4. **More Flexibility**: Easy model switching with Ollama
5. **Local Task Execution**: New capability to run system commands
6. **Cross-Platform**: Better Linux/macOS support

## Migration Steps

### Step 1: Install Ollama

**Windows:**
```powershell
# Download and install from https://ollama.ai/download/windows
# Or use winget
winget install Ollama.Ollama
```

**macOS:**
```bash
brew install ollama
```

**Linux:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### Step 2: Start Ollama Service

```bash
ollama serve
```

Keep this running in a separate terminal.

### Step 3: Pull Default Model

```bash
ollama pull llama3.1:8b
ollama pull mxbai-embed-large
```

This will download:
- Llama 3.1 8B Instruct (~4.7GB)
- mxbai-embed-large embeddings (~670MB)

### Step 4: Update Your Environment

**Old Environment Variables (no longer needed):**
```bash
# Remove these
ATLASAI_EMBEDDING_MODEL=C:\models\all-MiniLM-L6-v2
ATLASAI_TEXT_GEN_MODEL=C:\models\flan-t5-base
```

**New Environment Variables (optional):**
```bash
# Only set these if you want to customize
ATLASAI_OLLAMA_MODEL=llama3.1:8b  # Default model
ATLASAI_CHROMA_PERSIST_DIR=./chroma_db  # Vector store location
```

### Step 5: Pull Latest Code

```bash
git pull origin main
```

### Step 6: Run AtlasAI

The new version auto-installs dependencies:

```bash
cd AtlasAI
dotnet run
```

The first run will:
1. Install Python dependencies automatically
2. Check Ollama installation
3. Initialize ChromaDB vector store
4. Process and index your documents

## Comparing Configurations

### Old Configuration (v1.x)

```bash
# Environment variables
export ATLASAI_EMBEDDING_MODEL="C:\models\all-MiniLM-L6-v2"
export ATLASAI_TEXT_GEN_MODEL="C:\models\flan-t5-base"
export ATLASAI_TOP_K="4"

# Manual setup
pip install -r requirements.txt
python download_models.py  # Manual model download

# Run
cd AtlasAI
dotnet run
```

### New Configuration (v2.0)

```bash
# Minimal configuration (uses defaults)
cd AtlasAI
dotnet run  # That's it!

# Advanced configuration (optional)
export ATLASAI_OLLAMA_MODEL="qwen2.5:7b"
export ATLASAI_TOP_K="6"
cd AtlasAI
dotnet run
```

## Migrating Custom Code

### RAG Engine API Changes

**Old API (v1.x):**
```python
from atlasai_runtime.rag_engine import RAGEngine

engine = RAGEngine(
    documents_dir="./documents",
    onenote_runbook_path="path/to/onenote",
    embedding_model="C:/models/embeddings",
    text_gen_model="C:/models/flan-t5",
    top_k=4
)

result = engine.query("What is ADMS?")
print(result["answer"])
```

**New API (v2.0):**
```python
from atlasai_runtime.rag_engine import RAGEngine

engine = RAGEngine(
    documents_dir="./documents",
    onenote_runbook_path="path/to/onenote",
    ollama_model="llama3.1:8b",  # Changed
    embedding_model="mxbai-embed-large",  # Changed
    chroma_persist_dir="./chroma_db",  # New
    top_k=6  # Default increased
)

result = engine.query("What is ADMS?")
print(result["answer"])
# API response format unchanged
```

### FastAPI Endpoints

**Unchanged Endpoints:**
- `GET /health` - Health check (response format updated)
- `POST /chat` - Chat completion (same API)
- `GET /` - API info

**New Endpoints:**
- `POST /task/execute` - Execute local system commands
- `GET /task/system-info` - Get system information

## Testing Your Migration

### 1. Verify Ollama

```bash
curl http://localhost:11434/api/tags
```

Should return list of installed models.

### 2. Test AtlasAI

```bash
cd AtlasAI
dotnet run
```

Try a simple query:
```
You: What is Python?
```

Should see response with intent detection and sources.

### 3. Test API

```bash
curl http://localhost:8000/health
```

Should return:
```json
{
  "status": "healthy",
  "message": "AtlasAI Runtime is ready",
  "config": {
    "ollama_model": "llama3.1:8b",
    ...
  }
}
```

## Troubleshooting Migration

### "Ollama not found"
- Install Ollama from [ollama.ai](https://ollama.ai)
- Ensure `ollama` is in your PATH
- Run `ollama serve` before starting AtlasAI

### "Failed to pull model"
- Check internet connection
- Verify Ollama is running: `ollama serve`
- Manually pull: `ollama pull llama3.1:8b`

### "ChromaDB initialization failed"
- Check write permissions for `chroma_db/` directory
- Delete `chroma_db/` folder to reset
- Check disk space

### "Old models still referenced"
- Remove old environment variables
- Restart terminal/IDE
- Check no `.env` files with old config

### "Performance slower than before"
- Try a smaller model: `ollama pull llama3.1:3b`
- Reduce TOP_K: `export ATLASAI_TOP_K=3`
- Check Ollama GPU support: `ollama show llama3.1:8b`

## Rollback Plan

If you need to rollback to the old version:

```bash
# 1. Stop Ollama
pkill ollama

# 2. Checkout old version
git checkout v1.x

# 3. Reinstall old dependencies
pip install -r requirements.txt

# 4. Set old environment variables
export ATLASAI_EMBEDDING_MODEL="C:\models\all-MiniLM-L6-v2"
export ATLASAI_TEXT_GEN_MODEL="C:\models\flan-t5-base"

# 5. Run old version
cd AtlasAI
dotnet run
```

## FAQ

**Q: Can I keep my old models?**
A: Yes, but they won't be used by v2.0. You can delete them to free space.

**Q: Do I need to re-index my documents?**
A: Yes, the first run will rebuild the vector store using ChromaDB.

**Q: Can I use both versions?**
A: Not simultaneously. You can switch between versions by changing branches.

**Q: How much disk space does v2.0 need?**
A: ~5-10GB for models, plus space for ChromaDB index.

**Q: Can I use custom Ollama models?**
A: Yes! Any Ollama model works: `export ATLASAI_OLLAMA_MODEL="your-model"`

**Q: Is the API compatible?**
A: Yes, the `/chat` endpoint has the same request/response format.

## Getting Help

- Check the updated [README.md](README.md)
- Review [troubleshooting section](README.md#troubleshooting)
- Open an issue on GitHub
- Check Ollama docs: [ollama.ai/docs](https://ollama.ai/docs)
