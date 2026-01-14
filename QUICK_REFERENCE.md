# AtlasAI Quick Reference Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         app.py                              │
│                    (Entry Point)                            │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼─────────┐      ┌────────▼──────────┐
│   ChatInterface │      │    RAGService     │
│   (UI Layer)    │◄─────┤  (Orchestration)  │
└─────────────────┘      └────────┬──────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
           ┌────────▼──────┐ ┌───▼────────┐ ┌─▼──────────┐
           │ DocumentService│ │ Embedding  │ │   LLM     │
           │               │ │  Service   │ │  Service  │
           └───────────────┘ └────────────┘ └───────────┘
                                  │             │
                                  │  (Cache)    │ (Cache)
                                  ▼             ▼
                             [Disk Cache]  [Memory Cache]
```

## Quick Start

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run with new architecture (recommended)
streamlit run app.py

# Run with legacy version
streamlit run chatapp.py
```

### Configuration via Environment Variables

```bash
# Model configuration
export ATLAS_TEXT_GEN_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export ATLAS_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Local models (optional)
export ATLAS_LOCAL_TEXT_GEN_MODEL="/path/to/model"
export ATLAS_LOCAL_EMBEDDING_MODEL="/path/to/embedding"

# RAG settings
export ATLAS_TOP_K=4
export ATLAS_CHUNK_SIZE=800
export ATLAS_CHUNK_OVERLAP=150

streamlit run app.py
```

## Module Reference

### src/config/settings.py

**Purpose**: Centralized configuration management

**Key Classes**:
- `Settings`: Main container
- `ModelConfig`: Model settings
- `RAGConfig`: RAG parameters
- `PathConfig`: File paths
- `UIConfig`: UI settings

**Usage**:
```python
from src.config import Settings

settings = Settings.load_from_env()
print(settings.model.text_gen_model)
print(settings.rag.chunk_size)
```

### src/services/document_service.py

**Purpose**: Load and process documents

**Key Methods**:
- `load_pdf(path)`: Load PDF file
- `load_docx(path)`: Load DOCX file
- `split_documents(docs)`: Split into chunks
- `process_documents(paths)`: Load and split

**Usage**:
```python
from src.services import DocumentService

service = DocumentService(chunk_size=800)
splits, errors = service.process_documents([path1, path2])
```

### src/services/embedding_service.py

**Purpose**: Create and cache embeddings

**Key Methods**:
- `create_vectorstore(docs)`: Create FAISS store
- `get_retriever(vectorstore, top_k)`: Get retriever
- `clear_cache()`: Clear disk cache

**Usage**:
```python
from src.services import EmbeddingService

service = EmbeddingService(model_name="all-MiniLM-L6-v2")
vectorstore = service.create_vectorstore(documents)
retriever = service.get_retriever(vectorstore, top_k=4)
```

### src/services/llm_service.py

**Purpose**: Manage language models with caching

**Key Properties**:
- `llm`: LangChain LLM (cached)
- `model`: Transformers model (cached)
- `tokenizer`: Tokenizer (cached)

**Key Methods**:
- `generate(prompt)`: Generate text
- `update_generation_params()`: Update settings
- `unload_model()`: Free memory

**Usage**:
```python
from src.services import LLMService

service = LLMService(model_name="google/flan-t5-base")
response = service.generate("What is RAG?")
```

### src/services/rag_service.py

**Purpose**: Orchestrate RAG pipeline

**Key Methods**:
- `query(question, doc_paths)`: Execute query
- `update_rag_settings()`: Update and invalidate cache
- `clear_cache()`: Clear all caches

**Usage**:
```python
from src.config import Settings
from src.services import RAGService

settings = Settings()
service = RAGService(settings)
result = service.query("What is X?", [doc1, doc2])
print(result["answer"])
print(result["sources"])
```

### src/ui/chat_interface.py

**Purpose**: Streamlit UI management

**Key Methods**:
- `render()`: Render complete UI
- `_render_sidebar()`: Sidebar with tabs
- `_render_main_chat()`: Main chat area

**Usage**:
```python
from src.config import Settings
from src.services import RAGService
from src.ui import ChatInterface

settings = Settings()
rag_service = RAGService(settings)
interface = ChatInterface(settings, rag_service)
interface.render()
```

### src/utils/formatting.py

**Purpose**: Text formatting utilities

**Functions**:
- `format_answer_as_bullets(text)`: Convert to bullets
- `thinking_message(text)`: Format status message

**Usage**:
```python
from src.utils import format_answer_as_bullets

text = "First. Second. Third."
bullets = format_answer_as_bullets(text)
# Returns: "- First\n- Second\n- Third"
```

## Common Tasks

### Change Model

```python
# Option 1: Environment variable
export ATLAS_TEXT_GEN_MODEL="mistralai/Mistral-7B-Instruct-v0.2"

# Option 2: Edit src/config/settings.py
@dataclass
class ModelConfig:
    text_gen_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
```

### Adjust RAG Settings

```python
# Option 1: Via UI (Settings tab)
# - Adjust sliders
# - Click "Apply Settings"

# Option 2: Environment variables
export ATLAS_TOP_K=6
export ATLAS_CHUNK_SIZE=1000
export ATLAS_CHUNK_OVERLAP=200

# Option 3: Programmatically
settings.rag.top_k = 6
settings.rag.chunk_size = 1000
rag_service.update_rag_settings(6, 1000, 200)
```

### Clear Cache

```bash
# Option 1: Delete cache directory
rm -rf .cache/

# Option 2: Programmatically
rag_service.clear_cache()
embedding_service.clear_cache()
```

### Add New Document Loader

1. Edit `src/services/document_service.py`
2. Add new method:
```python
def load_txt(self, path: Path) -> List[Document]:
    """Load a TXT document"""
    if not path.exists():
        raise FileNotFoundError(f"TXT not found: {path}")
    
    with open(path, 'r') as f:
        content = f.read()
    
    return [Document(page_content=content, metadata={"source": str(path)})]
```
3. Update `load_document()` to handle `.txt` extension

### Add Custom Prompt Template

Edit `src/services/rag_service.py`:
```python
self.prompt_template = """Your custom prompt here.

Question: {question}
Context: {context}

Answer:
"""
```

## Troubleshooting

### Import Errors

```bash
# Ensure you're in the correct directory
cd /home/runner/work/AtlasAI/AtlasAI

# Check Python path
python -c "import sys; print(sys.path)"

# Install dependencies
pip install -r requirements.txt
```

### Model Not Found

```bash
# Check model path
echo $ATLAS_TEXT_GEN_MODEL

# Try downloading manually
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/flan-t5-base')"
```

### Cache Issues

```bash
# Clear cache
rm -rf .cache/
rm -rf tmp_docs/

# Restart application
streamlit run app.py
```

### Memory Issues

```python
# Use smaller model
export ATLAS_TEXT_GEN_MODEL="google/flan-t5-small"

# Reduce chunk size
export ATLAS_CHUNK_SIZE=500

# Unload model when done
llm_service.unload_model()
```

## Performance Tips

1. **Use Model Caching**: Models stay loaded between queries (10-50x speedup)
2. **Enable Vector Store Caching**: Embeddings cached to disk
3. **Choose Right Model Size**: Balance quality vs. speed
4. **Optimize Chunk Size**: 800 chars works well for most documents
5. **Limit Top K**: 4-6 chunks usually sufficient

## Development Workflow

### Adding a New Feature

1. Identify appropriate module
2. Add method with docstring
3. Write unit test
4. Test in isolation
5. Integrate with rest of system
6. Document in README

### Testing Changes

```bash
# Run validation
python validate_architecture.py

# Run with test data
streamlit run app.py

# Check logs for errors
```

## File Structure Quick Reference

```
AtlasAI/
├── app.py                    # Entry point (12 lines)
├── chatapp.py                # Legacy version (661 lines)
├── validate_architecture.py  # Validation script
├── requirements.txt          # Dependencies
├── README.md                 # Main documentation
├── ARCHITECTURE.md           # Architecture details
├── COMPARISON.md             # Before/After comparison
├── QUICK_REFERENCE.md        # This file
├── src/
│   ├── config/
│   │   └── settings.py       # Configuration (155 lines)
│   ├── services/
│   │   ├── document_service.py   # Documents (187 lines)
│   │   ├── embedding_service.py  # Embeddings (165 lines)
│   │   ├── llm_service.py        # LLM (159 lines)
│   │   └── rag_service.py        # RAG (179 lines)
│   ├── utils/
│   │   └── formatting.py     # Utils (75 lines)
│   └── ui/
│       └── chat_interface.py # UI (395 lines)
├── documents/                # Your PDFs/DOCX here
├── .cache/                   # Cached embeddings (gitignored)
└── tmp_docs/                 # Uploaded files (gitignored)
```

## Best Practices

1. **Always use environment variables** for deployment-specific config
2. **Keep services focused** on single responsibility
3. **Cache expensive operations** (models, embeddings)
4. **Handle errors gracefully** with try/except and error messages
5. **Document with docstrings** all public methods
6. **Test in isolation** before integration
7. **Use type hints** for clarity

## Getting Help

- See `ARCHITECTURE.md` for detailed architecture information
- See `COMPARISON.md` for before/after analysis
- See `README.md` for setup instructions
- Run `python validate_architecture.py` to check setup
- Check code docstrings for method documentation

## Version Information

- **Version**: 2.0.0
- **Architecture**: Modular, layered
- **Backward Compatible**: Yes (via chatapp.py)
- **Python**: 3.9+
- **Key Dependencies**: streamlit, langchain, transformers, torch
