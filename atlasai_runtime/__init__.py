"""
AtlasAI Runtime - Python service for RAG-based chat completion.

Intent-Aware Retrieval-Augmented Chatbot Components:
- IntentClassifier: Classifies user intent with confidence scoring
- QueryRewriter: Rewrites queries based on intent for better retrieval
- RetrieverInterface: Abstract interface for hybrid search
- AnswerSynthesizer: Synthesizes intent-specific answers with citations
- ClarifyingQuestionGenerator: Generates targeted clarifying questions
- TelemetryCollector: Logs metrics for evaluation
- process_query: Main orchestration function

Usage:
    from atlasai_runtime.process_query import process_query
    from atlasai_runtime.retriever import FakeRetriever, DocumentSnippet
    from atlasai_runtime.config import ChatbotConfig
    
    # Setup
    retriever = FakeRetriever([...])
    config = ChatbotConfig(confidence_threshold=0.55, top_k=5)
    
    # Process query
    result = process_query(
        user_query="How do I configure the system?",
        retriever=retriever,
        config=config
    )
    
    if result.question:
        # Ask clarification
        print(result.question)
    else:
        # Provide answer
        print(result.answer)
        for citation in result.citations:
            print(f"Source: {citation['title']}")
"""

# Export main interfaces
from .intent_classifier import IntentClassifier, IntentType, IntentClassificationResult
from .query_rewriter import QueryRewriter, QueryRewriteResult
from .retriever import (
    RetrieverInterface,
    DocumentSnippet,
    IntentBasedRetriever,
    FakeRetriever,
    VectorStoreRetriever
)
from .answer_synthesizer import AnswerSynthesizer, AnswerSynthesisResult
from .clarifying_question_generator import ClarifyingQuestionGenerator
from .telemetry import TelemetryCollector, TelemetryEvent, EvaluationInterface
from .config import ChatbotConfig
from .process_query import process_query, ProcessQueryResult

__all__ = [
    # Intent Classification
    "IntentClassifier",
    "IntentType",
    "IntentClassificationResult",
    
    # Query Rewriting
    "QueryRewriter",
    "QueryRewriteResult",
    
    # Retrieval
    "RetrieverInterface",
    "DocumentSnippet",
    "IntentBasedRetriever",
    "FakeRetriever",
    "VectorStoreRetriever",
    
    # Answer Synthesis
    "AnswerSynthesizer",
    "AnswerSynthesisResult",
    
    # Clarification
    "ClarifyingQuestionGenerator",
    
    # Telemetry
    "TelemetryCollector",
    "TelemetryEvent",
    "EvaluationInterface",
    
    # Configuration
    "ChatbotConfig",
    
    # Main Function
    "process_query",
    "ProcessQueryResult",
]

__version__ = "0.1.0"
