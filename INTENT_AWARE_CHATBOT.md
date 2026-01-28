# Intent-Aware Retrieval-Augmented Chatbot

## Overview

This module implements a production-ready intent-aware retrieval-augmented generation (RAG) chatbot system with clean interfaces, small pure functions, and comprehensive testing.

## Architecture

The system consists of the following components:

1. **Intent Classifier** - Identifies user intent with confidence scoring
2. **Query Rewriter** - Rewrites queries based on intent for better retrieval
3. **Retriever Interface** - Abstract interface supporting hybrid search
4. **Answer Synthesizer** - Generates intent-specific formatted answers with citations
5. **Clarifying Question Generator** - Creates targeted clarifying questions
6. **Telemetry Collector** - Logs metrics for evaluation
7. **Configuration** - Centralized configuration management
8. **Process Query** - Main orchestration function

## Supported Intents

- **how_to**: Procedural questions (steps, guides, procedures)
- **bug_resolution**: Error messages, crashes, issues
- **tool_explanation**: Conceptual questions about features/tools
- **escalate_or_ticket**: Requests for human help or ticket creation
- **chitchat**: Casual conversation
- **other**: Unclear or out-of-scope queries

## Features

### Intent Classification
- Rule-based detection for obvious bug signals (exceptions, error codes)
- Keyword-based classification with confidence scoring
- Confidence threshold: 0.55 (configurable)
- Returns: `{intent, confidence, rationale}`

### Query Rewriting
- **how_to**: Expands with action verbs and product/module names
- **bug_resolution**: Extracts error codes, versions, OS/log keywords
- **tool_explanation**: Adds synonyms and related module terms
- Returns: `{rewritten_query, entities[], constraints[]}`

### Retrieval
- Abstract interface for hybrid search (keyword + vector)
- Intent-based document filtering:
  - how_to → doc_type: "procedure"
  - bug_resolution → doc_type: "incident"
  - tool_explanation → doc_type: "concept"
- Returns top-k snippets with scores and metadata

### Answer Synthesis
- Intent-specific formatting:
  - **how_to**: Numbered steps, prerequisites, time estimates, pitfalls
  - **bug_resolution**: Problem signature, diagnostics, resolution, verification
  - **tool_explanation**: Concept, analogy, use cases, related features
- Includes 2-3 citations (title + URL)
- Asks clarification if top score < 0.25 (configurable)

### Clarification
- Minimal, targeted questions based on missing "slots" per intent
- Triggers when confidence < 0.55 or retrieval score < 0.25

### Telemetry
- Logs: user_query, intent, confidence, rewrite, top-k scores, etc.
- Supports offline evaluation (intent accuracy, recall@k)
- Exportable to JSON for analysis

## Usage

### Basic Example

```python
from atlasai_runtime.process_query import process_query
from atlasai_runtime.retriever import FakeRetriever, DocumentSnippet
from atlasai_runtime.config import ChatbotConfig

# Setup retriever with documents
snippets = [
    DocumentSnippet(
        title="Installation Guide",
        url="/docs/install",
        content="Step 1: Download. Step 2: Install.",
        score=0.9,
        metadata={"doc_type": "procedure"}
    )
]
retriever = FakeRetriever(snippets)

# Configure
config = ChatbotConfig(
    confidence_threshold=0.55,
    top_k=5,
    min_retrieval_score=0.25
)

# Process query
result = process_query(
    user_query="How do I install the application?",
    retriever=retriever,
    config=config
)

if result.question:
    # Ask clarification
    print(f"Clarification needed: {result.question}")
else:
    # Provide answer
    print(f"Answer: {result.answer}")
    for citation in result.citations:
        print(f"Source: {citation['title']} ({citation['url']})")
```

### Using with Vector Store

```python
from atlasai_runtime.retriever import VectorStoreRetriever

# Assuming you have a FAISS vectorstore
vectorstore = ...  # Your FAISS instance
retriever = VectorStoreRetriever(vectorstore)

result = process_query(
    user_query="What is the API?",
    retriever=retriever
)
```

### Configuration

```python
from atlasai_runtime.config import ChatbotConfig

config = ChatbotConfig(
    confidence_threshold=0.60,      # Higher threshold = more clarifications
    top_k=10,                       # More results
    min_retrieval_score=0.30,       # Lower threshold = fewer clarifications
    max_answer_tokens=500,
    filter_by_product="ADMS",       # Optional filters
    filter_by_version="2.0",
    embedding_provider="huggingface",
    llm_provider="huggingface"
)
```

### Telemetry and Evaluation

```python
from atlasai_runtime.telemetry import TelemetryCollector, EvaluationInterface

# Collect telemetry
telemetry = TelemetryCollector()

# Process queries
for query in queries:
    result = process_query(query, retriever, config, telemetry)
    # ... handle result

# Get summary stats
stats = telemetry.get_summary_stats()
print(f"Total queries: {stats['total_queries']}")
print(f"Avg confidence: {stats['avg_confidence']}")
print(f"Clarification rate: {stats['clarification_rate']}")

# Evaluate intent accuracy
evaluator = EvaluationInterface(telemetry)
ground_truth = {
    "How do I install?": "how_to",
    "Error 404": "bug_resolution",
    # ...
}
accuracy = evaluator.compute_intent_accuracy(ground_truth)
print(f"Intent accuracy: {accuracy:.2%}")

# Compute recall@k
doc_ground_truth = {
    "How do I install?": ["Installation Guide", "Setup Doc"],
    # ...
}
recall = evaluator.compute_recall_at_k(doc_ground_truth, k=5)
print(f"Recall@5: {recall:.2%}")

# Export for offline analysis
evaluator.export_for_analysis("telemetry.json")
```

## Testing

Run all tests:

```bash
cd /home/runner/work/AtlasAI/AtlasAI
python -m unittest discover tests -v
```

Run specific test module:

```bash
python -m unittest tests.test_intent_classifier -v
```

## Component APIs

### IntentClassifier

```python
from atlasai_runtime.intent_classifier import IntentClassifier, IntentType

classifier = IntentClassifier(confidence_threshold=0.55)
result = classifier.classify("How do I configure ADMS?")

print(result.intent)        # IntentType.HOW_TO
print(result.confidence)    # 0.75
print(result.rationale)     # "Keyword-based: matched 'how do i', 'configure'"
```

### QueryRewriter

```python
from atlasai_runtime.query_rewriter import QueryRewriter

rewriter = QueryRewriter()
result = rewriter.rewrite("error 404 in Windows", IntentType.BUG_RESOLUTION)

print(result.rewritten_query)  # Expanded query
print(result.entities)          # ["Windows"]
print(result.constraints)       # ["error:404", "os:windows"]
```

### AnswerSynthesizer

```python
from atlasai_runtime.answer_synthesizer import AnswerSynthesizer

synthesizer = AnswerSynthesizer(min_retrieval_score=0.25)
result = synthesizer.synthesize(
    query="How do I install?",
    intent=IntentType.HOW_TO,
    snippets=snippets
)

if result.should_ask_clarification:
    print(result.clarification_question)
else:
    print(result.answer)
    print(result.citations)
```

## Design Principles

1. **Clean Interfaces**: All components have clear, typed interfaces
2. **Pure Functions**: Small, testable functions with minimal side effects
3. **No Hard-coded Dependencies**: Abstract interfaces for LLMs, embeddings, and retrieval
4. **Comprehensive Testing**: 82 unit tests covering all components and edge cases
5. **Framework Agnostic**: No hard-coded language or framework assumptions
6. **Production Ready**: Proper error handling, validation, and telemetry

## Safety & UX

- All answers cite sources (2-3 citations minimum)
- Refuses unsourced claims
- Respects version filters when specified
- Prefers most recent "approved" documents
- Keeps answers concise with optional expand details
- Maximum one clarifying question per interaction

## Files

- `intent_classifier.py` - Intent classification with rule layer
- `query_rewriter.py` - Intent-specific query rewriting
- `retriever.py` - Retrieval interface and implementations
- `answer_synthesizer.py` - Intent-specific answer formatting
- `clarifying_question_generator.py` - Targeted clarification questions
- `telemetry.py` - Metrics collection and evaluation
- `config.py` - Configuration management
- `process_query.py` - Main orchestration function
- `tests/` - Comprehensive test suite (82 tests)

## License

See repository LICENSE file.
