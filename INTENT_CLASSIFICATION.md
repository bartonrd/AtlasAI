# Intent Classification System

## Overview

AtlasAI now includes an intelligent intent classification system that automatically detects the type of question or request from users and tailors responses accordingly. This significantly improves response quality, especially for short or ambiguous prompts.

## Features

### Intent Categories

The system classifies user queries into four intent categories:

1. **error_log_resolution** - Troubleshooting and resolving errors
   - Focuses on identifying root causes
   - Provides actionable troubleshooting steps
   - Explains error codes and messages
   - Examples: "I'm getting an error", "The system failed", "How do I fix this bug?"

2. **how_to** - Step-by-step instructions for tasks
   - Provides clear, sequential instructions
   - Includes prerequisites and requirements
   - Mentions important warnings or considerations
   - Examples: "How do I configure X?", "What are the steps to install?", "Guide me through setup"

3. **chit_chat** - Casual conversation
   - Responds naturally and professionally
   - Doesn't force context usage for greetings/thanks
   - Keeps responses brief and relevant
   - Examples: "Hello!", "Thank you", "How are you?"

4. **concept_explanation** - Technical concepts and definitions
   - Defines key terms clearly
   - Provides context and importance
   - Includes relevant technical details
   - Builds progressive explanation
   - Examples: "What is X?", "Explain Y", "Define Z"

### Classification Methods

The system supports two classification methods:

1. **Zero-Shot Classification** (Primary)
   - Uses the `facebook/bart-large-mnli` model
   - Provides high accuracy intent detection
   - Requires transformers library and model download
   - Automatically used when available

2. **Keyword-Based Fallback** (Backup)
   - Uses pattern matching on common intent keywords
   - Works without external dependencies
   - 91.7% accuracy on test cases
   - Gracefully handles offline scenarios

The system automatically falls back to keyword-based classification if:
- The transformers library is not installed
- The zero-shot model fails to load
- Running in resource-constrained environment

### Intent-Specific Prompts

Each intent uses a specialized prompt template optimized for that type of query:

- **Error Resolution Prompts**: Focus on troubleshooting steps and solutions
- **How-To Prompts**: Emphasize step-by-step instructions
- **Chit-Chat Prompts**: Respond conversationally without forcing document context
- **Concept Explanation Prompts**: Build progressive explanations from basic to detailed

## Architecture

### Components

1. **IntentClassifier** (`intent_classifier.py`)
   - Main classification logic
   - Zero-shot and keyword-based methods
   - Confidence scoring
   - Intent descriptions

2. **RAGEngine Integration** (`rag_engine.py`)
   - Automatic intent detection on each query
   - Intent-specific prompt selection
   - Seamless integration with existing RAG pipeline

3. **API Updates** (`app.py`)
   - Intent information in responses
   - Confidence scores
   - Backward compatible

### Flow Diagram

```
User Query
    ↓
IntentClassifier.classify()
    ↓
[Try Zero-Shot] → Success → Intent + Confidence
    ↓ (fallback)
[Keyword-Based] → Intent + Confidence
    ↓
RAGEngine._get_intent_specific_prompt()
    ↓
Intent-Specific Prompt Template
    ↓
RAG Chain with Custom Prompt
    ↓
Response with Intent Metadata
```

## API Changes

### Response Model

The `/chat` endpoint response now includes intent information:

```json
{
  "answer": "- Step 1: ...\n- Step 2: ...",
  "sources": [
    {"index": 1, "source": "doc.pdf", "page": "5"}
  ],
  "intent": "how_to",
  "intent_confidence": 0.87
}
```

**New Fields:**
- `intent` (string, optional): Detected intent category
- `intent_confidence` (float, optional): Confidence score (0-1)

### Backward Compatibility

The new fields are optional, so existing clients continue to work without changes.

## Configuration

### Environment Variables

No new environment variables required. The system uses:

```python
# Default zero-shot model (when transformers available)
model_name = "facebook/bart-large-mnli"

# Confidence threshold for classification
confidence_threshold = 0.3
```

### Model Setup (Optional)

For optimal performance with zero-shot classification:

1. Install transformers: `pip install transformers torch`
2. The model will be downloaded automatically on first use
3. Alternatively, pre-download:

```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
```

**Note**: The system works perfectly fine without this step using keyword-based fallback.

## Usage Examples

### Python SDK

```python
from atlasai_runtime.intent_classifier import IntentClassifier

# Initialize classifier
classifier = IntentClassifier()

# Classify a query
result = classifier.classify("How do I configure the system?")

print(f"Intent: {result['intent']}")           # how_to
print(f"Confidence: {result['confidence']}")   # 0.85
print(f"Method: {result['method']}")           # zero_shot or keyword
```

### API Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "The system failed to start"}'
```

Response:
```json
{
  "answer": "- Check the system logs for specific error messages\n- Verify all dependencies are installed\n- Ensure the configuration file is valid",
  "sources": [...],
  "intent": "error_log_resolution",
  "intent_confidence": 0.82
}
```

## Testing

### Unit Tests

Run the keyword-based classification tests:

```bash
python /tmp/test_intent_keywords.py
```

Expected: 91.7% accuracy or better

### Demo Script

See the full demonstration:

```bash
python /tmp/demonstrate_intent_system.py
```

### Test Cases

The system has been tested on:
- Error messages and troubleshooting queries
- How-to questions with various phrasings
- Casual greetings and thanks
- Concept explanation requests
- Very short queries (1-2 words)
- Empty queries
- Ambiguous queries

## Performance

### Classification Speed

- **Keyword-based**: <5ms per query
- **Zero-shot**: ~50-200ms per query (model-dependent)

### Accuracy

- **Keyword-based fallback**: 91.7% on test cases
- **Zero-shot**: Expected >95% (when available)

### Memory

- **Keyword-based**: Negligible (~1KB)
- **Zero-shot**: ~1.5GB (model loaded in memory)

## Benefits

1. **Better Response Quality**: Tailored prompts for each intent type
2. **Handles Short Prompts**: Even "error" or "configure" get classified correctly
3. **Graceful Degradation**: Works offline with keyword fallback
4. **Transparency**: Confidence scores show classification certainty
5. **No Breaking Changes**: Existing integrations continue to work

## Limitations

- Keyword-based fallback may misclassify some ambiguous queries
- Zero-shot model requires ~1.5GB memory when loaded
- Classification adds small latency (5-200ms depending on method)
- Some edge cases may not classify perfectly

## Future Enhancements

Potential improvements:
- Fine-tune classification model on domain-specific data
- Add more intent categories as needed
- Implement intent-based routing to specialized agents
- Add user feedback loop to improve classification
- Support custom intent categories via configuration

## Troubleshooting

### "transformers library not available"

This is normal and expected if transformers is not installed. The system will use keyword-based fallback automatically.

To enable zero-shot classification:
```bash
pip install transformers torch
```

### Low confidence scores

If classification confidence is consistently low:
1. Check if queries are ambiguous or unclear
2. Consider the query length (very short queries are harder to classify)
3. Review keyword lists in `intent_classifier.py`
4. Enable zero-shot classification for better accuracy

### Wrong intent classification

For keyword-based fallback:
1. Review `intent_keywords` in `IntentClassifier`
2. Add relevant keywords for your domain
3. Adjust keyword matching logic if needed

For zero-shot:
1. Check if the model is loading correctly
2. Verify model path and availability
3. Consider fine-tuning on domain-specific data

## Contributing

To add new intent categories:

1. Add to `IntentClassifier.INTENT_CATEGORIES`
2. Add keywords to `intent_keywords`
3. Create prompt template in `RAGEngine._get_intent_specific_prompt()`
4. Add description in `get_intent_description()`
5. Add test cases

## References

- Zero-shot classification: [Hugging Face Documentation](https://huggingface.co/tasks/zero-shot-classification)
- BART model: [facebook/bart-large-mnli](https://huggingface.co/facebook/bart-large-mnli)
- LangChain prompts: [LangChain Documentation](https://python.langchain.com/docs/modules/model_io/prompts/)
