# AtlasAI Architecture Documentation

## Overview

AtlasAI v2.0 features a modular, maintainable architecture that separates concerns and provides better performance, configurability, and extensibility.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit UI (app.py)                  │
│                    User Interface Layer                      │
└────────────────┬──────────────────────────────────┬─────────┘
                 │                                  │
                 ▼                                  ▼
┌────────────────────────────┐     ┌──────────────────────────┐
│     Config Manager         │     │   Document Processor     │
│   (config.py)              │     │   (document_processor.py)│
│                            │     │                          │
│ - Environment variables    │     │ - PDF/DOCX loading       │
│ - Default settings         │     │ - Text chunking          │
│ - Validation               │     │ - Hash computation       │
└────────────────────────────┘     └────────────┬─────────────┘
                                                 │
                 ┌───────────────────────────────┴─────────────┐
                 ▼                                             ▼
┌────────────────────────────┐     ┌──────────────────────────┐
│   Vector Store Manager     │     │      LLM Manager         │
│   (vector_store.py)        │     │   (llm_manager.py)       │
│                            │     │                          │
│ - Embeddings               │     │ - Model loading          │
│ - FAISS vector store       │     │ - Fallback support       │
│ - Persistent caching       │     │ - Pipeline config        │
└────────────┬───────────────┘     └─────────────┬────────────┘
             │                                   │
             └───────────────┬───────────────────┘
                             ▼
                ┌────────────────────────┐
                │      RAG Chain         │
                │    (rag_chain.py)      │
                │                        │
                │ - RetrievalQA chain    │
                │ - Prompt templates     │
                │ - Source formatting    │
                └────────────────────────┘
```

## Module Structure

### 1. Configuration (`config.py`)
**Purpose**: Centralized configuration management

**Features**:
- Environment variable support for all settings
- Sensible defaults
- Model path configuration with fallbacks
- Validation of settings

**Key Configuration**:
```python
# Environment variables (optional)
ATLAS_EMBEDDING_MODEL=path/to/model
ATLAS_TEXT_GEN_MODEL=path/to/model
ATLAS_TOP_K=4
ATLAS_CHUNK_SIZE=800
ATLAS_USE_CACHE=true
```

### 2. Document Processor (`document_processor.py`)
**Purpose**: Document loading, cleaning, and chunking

**Features**:
- PDF and DOCX loading
- Text cleaning and boilerplate removal
- Configurable chunking with RecursiveCharacterTextSplitter
- Document hash computation for cache invalidation

**Key Methods**:
- `load_documents()`: Load from file paths
- `split_documents()`: Split into chunks
- `compute_document_hash()`: Generate cache key

### 3. Vector Store Manager (`vector_store.py`)
**Purpose**: Embedding and vector store management with caching

**Features**:
- HuggingFace embeddings initialization
- FAISS vector store creation
- Persistent caching to disk
- Cache invalidation based on document hash
- Lazy loading of embeddings

**Performance Benefits**:
- Avoids re-embedding documents on every query
- Dramatically reduces startup time for repeated queries
- Cache automatically invalidates when documents change

### 4. LLM Manager (`llm_manager.py`)
**Purpose**: Language model initialization and configuration

**Features**:
- Automatic fallback to alternative models
- Configurable generation parameters
- Support for sampling vs greedy decoding
- Pipeline configuration for optimal performance

**Supported Parameters**:
- `max_new_tokens`: Control response length
- `temperature`: Control randomness (when sampling)
- `do_sample`: Enable/disable sampling
- `truncation`: Handle long inputs

### 5. RAG Chain (`rag_chain.py`)
**Purpose**: Retrieval-augmented generation pipeline

**Features**:
- RetrievalQA chain construction
- Customizable prompt templates
- Source document formatting
- Clean separation of retrieval and generation

**Default Prompt**: Optimized for bullet-point responses

### 6. Utilities (`utils.py`)
**Purpose**: Common utility functions

**Features**:
- Text formatting (bullet points)
- Boilerplate removal
- Chat name generation
- Status message formatting

### 7. Main Application (`app.py`)
**Purpose**: Streamlit UI and orchestration

**Features**:
- Multi-chat session management
- Document upload handling
- Settings management
- Error handling and user feedback

## Key Improvements Over v1.0

### 1. **Modularity**
- **Before**: Single 661-line monolithic file
- **After**: 7 focused modules with clear responsibilities
- **Benefit**: Easier maintenance, testing, and extension

### 2. **Performance**
- **Before**: Re-loaded and re-embedded documents on every query
- **After**: Persistent vector store caching
- **Benefit**: 10-100x faster for repeated queries

### 3. **Configuration**
- **Before**: Hardcoded paths in script
- **After**: Environment variable support, fallback models
- **Benefit**: Easier deployment and configuration

### 4. **Model Support**
- **Before**: Only FLAN-T5 base/small
- **After**: Easy to swap models, fallback support
- **Benefit**: Can use better models without code changes

### 5. **Error Handling**
- **Before**: Limited error handling
- **After**: Comprehensive error handling with fallbacks
- **Benefit**: More robust and user-friendly

### 6. **Scalability**
- **Before**: All logic in UI layer
- **After**: Separated concerns, reusable components
- **Benefit**: Can add API layer, CLI, or other interfaces

## Model Recommendations

### Embedding Models (Ranked)

1. **BAAI/bge-small-en-v1.5** (Recommended)
   - Size: ~133MB
   - Performance: Excellent for retrieval
   - Better than MiniLM for RAG tasks

2. **thenlper/gte-small**
   - Size: ~133MB
   - Performance: Similar to bge-small

3. **sentence-transformers/all-MiniLM-L6-v2** (Current default)
   - Size: ~90MB
   - Performance: Good, but older

### Text Generation Models (Ranked)

1. **mistralai/Mistral-7B-Instruct-v0.2** (Best quality)
   - Size: ~14GB
   - Performance: Excellent instruction following
   - Requires: GPU or powerful CPU

2. **meta-llama/Llama-2-7b-chat-hf** (High quality)
   - Size: ~13GB
   - Performance: Very good for chat
   - Requires: GPU or powerful CPU

3. **google/flan-t5-large** (Recommended for CPU)
   - Size: ~2.8GB
   - Performance: Good balance for CPU
   - Better than base/small

4. **google/flan-t5-base** (Current default)
   - Size: ~890MB
   - Performance: Acceptable for CPU
   - Fast but lower quality

### Setting Model Paths

#### Option 1: Environment Variables (Recommended)
```bash
export ATLAS_EMBEDDING_MODEL="C:/models/bge-small-en-v1.5"
export ATLAS_TEXT_GEN_MODEL="C:/models/flan-t5-large"
```

#### Option 2: Edit `config.py`
```python
self.embedding_model = r"C:\models\bge-small-en-v1.5"
self.text_gen_model = r"C:\models\flan-t5-large"
```

## Usage

### Running the Application

#### Option 1: New modular app (Recommended)
```bash
streamlit run app.py
```

#### Option 2: C# wrapper (unchanged)
```bash
cd AtlasAI
dotnet run
```

#### Option 3: Legacy monolithic app
```bash
streamlit run chatapp.py
```

### Configuration Examples

#### Fast CPU setup:
```bash
export ATLAS_TEXT_GEN_MODEL="google/flan-t5-base"
export ATLAS_EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export ATLAS_CHUNK_SIZE=600
export ATLAS_TOP_K=3
```

#### Quality setup (GPU recommended):
```bash
export ATLAS_TEXT_GEN_MODEL="mistralai/Mistral-7B-Instruct-v0.2"
export ATLAS_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
export ATLAS_CHUNK_SIZE=1000
export ATLAS_TOP_K=5
export ATLAS_USE_SAMPLING=true
export ATLAS_TEMPERATURE=0.3
```

## Future Enhancements

### Potential Improvements

1. **API Layer**: Add FastAPI for programmatic access
2. **Database Backend**: Store chat history in SQLite/PostgreSQL
3. **User Management**: Multi-user support with authentication
4. **Advanced RAG**: 
   - Re-ranking for better retrieval
   - Hybrid search (dense + sparse)
   - Multi-query retrieval
5. **Model Options**:
   - Support for OpenAI API
   - Support for Ollama
   - Support for other inference engines
6. **Monitoring**: Add logging and metrics
7. **Testing**: Unit tests for all modules
8. **CI/CD**: Automated testing and deployment

## Migration Guide

### Migrating from v1.0 (chatapp.py)

1. **Install new app**: Both versions can coexist
2. **Configure models**: Set environment variables or edit `config.py`
3. **Test with sample query**: Verify everything works
4. **Switch**: Use `app.py` instead of `chatapp.py`

### No Breaking Changes
- Documents folder structure unchanged
- C# wrapper still works (update to launch `app.py` if desired)
- All existing functionality preserved
- Settings maintained

## Troubleshooting

### Issue: Models not found
**Solution**: 
1. Check model paths in config
2. Verify models are downloaded
3. Set fallback models will auto-load

### Issue: Slow first query
**Cause**: Model loading and initial embedding
**Solution**: Subsequent queries use cache (much faster)

### Issue: Out of memory
**Solution**: 
1. Use smaller models (flan-t5-small)
2. Reduce chunk_size and top_k
3. Close other applications

### Issue: Cache not working
**Solution**:
1. Check `.cache` directory exists
2. Verify `ATLAS_USE_CACHE=true`
3. Clear cache and rebuild: delete `.cache` folder
