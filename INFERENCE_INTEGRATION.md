# Inference Pipeline Integration Guide

## Overview

The AtlasAI inference pipeline enhances the existing RAG system with:
- **Intent Classification**: Automatically detects user intent (how-to, bug resolution, tool explanation, etc.)
- **Query Rewriting**: Expands queries based on intent for better retrieval
- **Retrieval Routing**: Applies intent-specific filters to document search
- **Grounded Answer Synthesis**: Generates structured answers with citations
- **Clarification Handling**: Asks precise questions when confidence is low

## Architecture

The inference pipeline is **fully additive** and preserves backward compatibility:

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  /chat (Legacy)          /chat/inference (Enhanced)        │
│      │                          │                           │
│      │                          │                           │
│      ▼                          ▼                           │
│  RAGEngine.query()    RAGEngine.query_with_inference()     │
│      │                          │                           │
│      │                          │                           │
│      │                    ┌─────▼────────┐                 │
│      │                    │   Inference   │                 │
│      │                    │    Engine     │                 │
│      │                    └─────┬────────┘                 │
│      │                          │                           │
│      │         ┌────────────────┼────────────────┐         │
│      │         ▼                ▼                ▼          │
│      │    Classify          Rewrite         Retrieve       │
│      │      Intent           Query          Documents      │
│      │         │                │                │          │
│      │         └────────────────┴────────────────┘         │
│      │                          │                           │
│      │                          ▼                           │
│      │                    Synthesize/                       │
│      │                    Clarify                           │
│      │                          │                           │
│      ▼                          ▼                           │
│  Response                   Response                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. RAG Engine Integration

The `RAGEngine` class now has a new method `query_with_inference()` that can be toggled on/off:

```python
# Legacy mode (backward compatible)
result = rag_engine.query(question="How to install?")

# Inference mode (enhanced)
result = rag_engine.query_with_inference(
    question="How to install?",
    use_inference=True,  # Toggle feature on/off
    user_feedback=None   # Optional feedback
)
```

**Key characteristics:**
- ✅ Existing `query()` method unchanged
- ✅ New method is opt-in
- ✅ Feature toggle via `use_inference` parameter
- ✅ Graceful degradation if inference fails

### 2. FastAPI Endpoints

Two endpoints are now available:

#### `/chat` (Legacy - Unchanged)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I install the software?",
    "additional_documents": null
  }'
```

**Response:**
```json
{
  "answer": "- Download the installer\n- Run setup.exe\n- Follow instructions",
  "sources": [
    {"index": 1, "source": "install.pdf", "page": "1"}
  ]
}
```

#### `/chat/inference` (New - Enhanced)
```bash
curl -X POST http://localhost:8000/chat/inference \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I install the software?",
    "use_inference": true,
    "user_feedback": null
  }'
```

**Response:**
```json
{
  "answer": "**Steps:**\n1. Download installer\n2. Run setup\n...",
  "question": null,
  "intent": "how_to",
  "confidence": 0.85,
  "citations": [
    {"index": "1", "title": "Installation Guide", "url": "/docs/install.pdf"}
  ],
  "telemetry": {
    "user_query": "How do I install...",
    "intent": "how_to",
    "confidence": 0.85,
    "rewritten_query": "How do I install setup configure...",
    "num_retrieved": 4,
    "top_k_scores": [0.9, 0.85, 0.8, 0.75],
    "had_clarification": false,
    "elapsed_ms": 234
  }
}
```

**Clarification Response (low confidence):**
```json
{
  "answer": null,
  "question": "I understand you're looking for instructions. Could you clarify what specific task you want to accomplish?",
  "intent": "how_to",
  "confidence": 0.45,
  "citations": [],
  "telemetry": {
    "had_clarification": true,
    "clarification_reason": "Low confidence (0.45) on procedural intent"
  }
}
```

## Configuration

### Environment Variables

Configure the inference pipeline via environment variables:

```bash
# Confidence threshold for triggering clarification
export ATLASAI_CONFIDENCE_THRESHOLD=0.55

# Top-K documents to retrieve
export ATLASAI_TOP_K=4

# Minimum retrieval score threshold
export ATLASAI_MIN_RETRIEVAL_SCORE=0.25

# Maximum answer tokens
export ATLASAI_MAX_ANSWER_TOKENS=384

# Enable/disable inference pipeline
export ATLASAI_ENABLE_INFERENCE=true
```

### Programmatic Configuration

```python
from atlasai_runtime.inference_config import InferenceConfig
from atlasai_runtime.inference_engine import InferenceEngine

config = InferenceConfig(
    confidence_threshold=0.55,
    top_k=4,
    min_retrieval_score=0.25,
    max_answer_tokens=384,
    enable_inference=True,
    bug_signal_keywords=["error", "exception", "crash", "bug"],
)

engine = InferenceEngine(
    search_backend=my_backend,
    config=config,
    llm_provider=my_llm,
)
```

## Feature Toggle Guide

### Option 1: Gradual Rollout (Recommended)

Start with inference disabled by default and enable for specific users:

```python
# In app.py
USE_INFERENCE_FOR_USERS = {"user123", "user456"}  # Beta testers

@app.post("/chat")
async def chat_completion(request: ChatRequest, user_id: str = None):
    use_inference = user_id in USE_INFERENCE_FOR_USERS
    
    result = rag_engine.query_with_inference(
        question=request.message,
        use_inference=use_inference,
    )
    # ...
```

### Option 2: A/B Testing

Use a percentage rollout:

```python
import random

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    use_inference = random.random() < 0.10  # 10% of requests
    
    result = rag_engine.query_with_inference(
        question=request.message,
        use_inference=use_inference,
    )
    # ...
```

### Option 3: Environment-Based Toggle

Enable in specific environments:

```python
import os

ENABLE_INFERENCE = os.getenv("ATLASAI_ENABLE_INFERENCE", "false").lower() == "true"

@app.post("/chat")
async def chat_completion(request: ChatRequest):
    result = rag_engine.query_with_inference(
        question=request.message,
        use_inference=ENABLE_INFERENCE,
    )
    # ...
```

### Option 4: Client-Controlled

Let clients opt-in via request parameter:

```python
@app.post("/chat")
async def chat_completion(request: ChatRequest):
    # Client sends use_inference: true/false in request
    result = rag_engine.query_with_inference(
        question=request.message,
        use_inference=request.use_inference,
    )
    # ...
```

## Intent Types and Behaviors

### 1. `how_to` - Procedural Instructions
**Triggers:**
- "How do I...", "Steps to...", "Guide to..."
- Keywords: install, configure, setup, run

**Query Rewriting:**
- Expands action verbs (install → setup, configure, deploy)
- Adds procedural keywords (step, procedure, guide)

**Document Filters:**
- `doc_type: "procedure"`

**Answer Format:**
```
**Steps:**
1. Prerequisite: Check system requirements
2. Download the installer from...
3. Run setup.exe with admin privileges
4. Configure settings in config.yml
5. Verify installation with test command

**Common Pitfalls:**
- Ensure firewall allows port 8080
- Run as administrator on Windows
```

### 2. `bug_resolution` - Error Troubleshooting
**Triggers:**
- "Exception:", "Error:", "errno", "exit code"
- Keywords: error, crash, failed, traceback
- **High priority bias** (boosted confidence)

**Query Rewriting:**
- Extracts error codes (errno 13, exit code 1)
- Extracts version numbers (v1.2.3)
- Extracts OS/platform (Windows, Linux)
- Adds diagnostic keywords

**Document Filters:**
- `doc_type: "incident"`

**Answer Format:**
```
**Error Information:**
- Error: errno 13 (Permission Denied)
- Signature: IOError on file access

**Diagnostics:**
- Check file permissions: ls -l /path/to/file
- Verify user has read access

**Resolution:**
- Run with elevated privileges: sudo command
- Or change file permissions: chmod +r file

**Verification:**
- Retry the command
- Check logs for confirmation
```

### 3. `tool_explanation` - Concept Understanding
**Triggers:**
- "What is...", "Explain...", "Tell me about..."
- Keywords: describe, define, purpose, meaning

**Query Rewriting:**
- Adds concept synonyms (API → interface, endpoint, service)
- Adds explanation keywords

**Document Filters:**
- `doc_type: "concept"`

**Answer Format:**
```
**Explanation:**
- The API module provides a REST interface for external systems
- Simple analogy: Like a waiter taking orders from customers

**Use Cases:**
- Integrate with third-party services
- Automate workflows via HTTP requests
- Build custom dashboards

**Related Features:**
- Authentication system for API security
- Rate limiting to prevent abuse
```

### 4. `escalate_or_ticket` - Human Support
**Triggers:**
- "contact support", "speak to someone", "file a ticket"

**Behavior:**
- Routes to human support queue
- Provides ticket filing instructions

### 5. `chitchat` - Conversational
**Triggers:**
- "Hello", "Thank you", "How are you"

**Behavior:**
- Responds conversationally
- Does not trigger retrieval

### 6. `other` - Catch-all
**Behavior:**
- Defaults to legacy RAG behavior
- May trigger clarification if ambiguous

## Telemetry

The inference pipeline collects rich telemetry for monitoring and debugging:

```python
telemetry = {
    "user_query": "Original query text",
    "intent": "how_to",
    "confidence": 0.85,
    "intent_rationale": "Procedural instruction patterns detected",
    "rewritten_query": "Expanded query with synonyms",
    "entities": ["ModelManager", "ADMS"],
    "constraints": ["v1.2.3", "Windows"],
    "num_retrieved": 4,
    "top_k_scores": [0.9, 0.85, 0.8, 0.75],
    "selected_docs": ["install.pdf", "config.pdf"],
    "had_clarification": false,
    "answer_length": 234,
    "num_citations": 3,
    "elapsed_ms": 156,
    "user_feedback": "helpful"  # Optional
}
```

**Use cases for telemetry:**
- Monitor intent distribution
- Track clarification rate
- Identify low-confidence queries
- Measure retrieval quality
- Analyze user feedback

## Testing

### Run All Tests
```bash
cd /home/runner/work/AtlasAI/AtlasAI
python -m pytest tests/ -v
```

### Run Specific Test Module
```bash
python -m pytest tests/test_intent_classifier.py -v
python -m pytest tests/test_query_rewriter.py -v
python -m pytest tests/test_retriever.py -v
python -m pytest tests/test_answer_synthesizer.py -v
python -m pytest tests/test_inference_engine.py -v
```

### Test Coverage Summary
- **Intent Classifier**: 14 tests (pattern matching, edge cases)
- **Query Rewriter**: 14 tests (entity extraction, rewriting strategies)
- **Retriever**: 13 tests (filtering, mock backend, FAISS integration)
- **Answer Synthesizer**: 10 tests (grounded synthesis, citations)
- **Inference Engine**: 13 tests (end-to-end pipeline, telemetry)
- **Total**: 62 tests, all passing ✓

## Safety and Grounding

### Grounding Guardrails
1. **Only use retrieved snippets**: No information invented by LLM
2. **Always include citations**: 2-3 sources with title + URL
3. **Respect explicit versions**: If user mentions v1.2, use v1.2 docs
4. **Refuse unsourced claims**: Return clarification if no good matches

### Low Confidence Handling
When `confidence < threshold` (default 0.55):
- Return a clarifying question instead of an answer
- Ask single, precise question
- Provide 2-3 suggestions to guide user

### Low Retrieval Score Handling
When `top_score < min_threshold` (default 0.25):
- Return a clarifying question about specificity
- Suggest user provide more context
- Do not fabricate an answer

## Migration Path

### Phase 1: Deploy (No Change)
```python
# Deploy code but keep inference disabled
result = rag_engine.query_with_inference(
    question=query,
    use_inference=False,  # Legacy mode
)
```

### Phase 2: Internal Testing
```python
# Enable for internal users only
use_inference = user.is_internal
result = rag_engine.query_with_inference(
    question=query,
    use_inference=use_inference,
)
```

### Phase 3: Beta Rollout
```python
# Enable for 10% of users
use_inference = random.random() < 0.10
result = rag_engine.query_with_inference(
    question=query,
    use_inference=use_inference,
)
# Monitor telemetry for issues
```

### Phase 4: Full Rollout
```python
# Enable for all users
result = rag_engine.query_with_inference(
    question=query,
    use_inference=True,
)
```

## Troubleshooting

### Issue: High clarification rate
**Solution**: Lower `confidence_threshold` from 0.55 to 0.45

### Issue: Too many low-quality answers
**Solution**: Raise `min_retrieval_score` from 0.25 to 0.35

### Issue: Inference too slow
**Solution**: 
- Reduce `top_k` from 4 to 3
- Reduce `max_answer_tokens` from 384 to 256

### Issue: Wrong intent detected
**Solution**: 
- Add domain-specific keywords to `bug_signal_keywords`
- Review `intent_classifier.py` patterns for your domain

## Extending the System

### Adding New Intents

1. Add intent constant in `intent_classifier.py`:
```python
INTENT_MY_NEW_INTENT = "my_new_intent"
```

2. Add detection patterns in `IntentClassifier._detect_my_intent()`:
```python
def _detect_my_intent(self, query_lower: str) -> float:
    score = 0.0
    if "my_pattern" in query_lower:
        score += 0.4
    return min(score, 1.0)
```

3. Add to classification logic in `classify()`:
```python
my_score = self._detect_my_intent(query_lower)
if my_score > 0.6:
    return IntentResult(intent=INTENT_MY_NEW_INTENT, ...)
```

4. Add rewriting logic in `query_rewriter.py`
5. Add filter mapping in `inference_config.py`
6. Add formatting logic in `answer_synthesizer.py`

### Plugging in Different LLM Providers

The system supports pluggable LLM providers via the Protocol pattern:

```python
from atlasai_runtime.intent_classifier import LLMProvider

class MyLLMProvider:
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        # Call your LLM API
        response = my_llm_api.complete(prompt, max_tokens=max_tokens)
        return response.text

# Use it
llm = MyLLMProvider()
classifier = IntentClassifier(llm_provider=llm)
synthesizer = AnswerSynthesizer(llm_provider=llm)
engine = InferenceEngine(backend, config, llm_provider=llm)
```

### Plugging in Different Search Backends

Implement the `SearchBackend` protocol:

```python
from atlasai_runtime.retriever import SearchBackend, RetrievedDoc

class MySearchBackend:
    def search(self, query: str, top_k: int = 5, 
               filters: dict = None) -> list[RetrievedDoc]:
        # Call your search API
        results = my_search_api.search(query, limit=top_k, filters=filters)
        
        # Convert to RetrievedDoc format
        return [
            RetrievedDoc(
                title=r.title,
                url=r.url,
                content=r.content,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ]

# Use it
backend = MySearchBackend()
engine = InferenceEngine(backend, config)
```

## Best Practices

1. **Start with low adoption**: Use gradual rollout
2. **Monitor telemetry**: Watch for high clarification rates
3. **Collect feedback**: Add user feedback to telemetry
4. **A/B test thresholds**: Find optimal confidence thresholds
5. **Domain tuning**: Adjust patterns for your documentation domain
6. **Keep legacy path**: Maintain `/chat` endpoint for fallback
7. **Version explicitly**: Prefer user-specified versions in queries
8. **Test edge cases**: Empty queries, special chars, long queries
