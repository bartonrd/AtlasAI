"""
Integration Example: Using Intent-Aware Components with Existing RAG Engine

This example shows how to integrate the new intent-aware components
with the existing RAGEngine from atlasai_runtime/rag_engine.py.
"""

from typing import Dict, Any
from atlasai_runtime.intent_classifier import IntentClassifier, IntentType
from atlasai_runtime.query_rewriter import QueryRewriter
from atlasai_runtime.retriever import RetrieverInterface, DocumentSnippet
from atlasai_runtime.answer_synthesizer import AnswerSynthesizer
from atlasai_runtime.config import ChatbotConfig
from atlasai_runtime.telemetry import TelemetryCollector
from atlasai_runtime.process_query import process_query


class RAGEngineRetrieverAdapter(RetrieverInterface):
    """
    Adapter to use existing RAGEngine as a retriever.
    
    This wraps the RAGEngine's query method to conform to the
    RetrieverInterface for use with intent-aware components.
    """
    
    def __init__(self, rag_engine):
        """
        Initialize adapter.
        
        Args:
            rag_engine: Instance of RAGEngine from rag_engine.py
        """
        self.rag_engine = rag_engine
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] = None
    ) -> list:
        """
        Retrieve documents using RAGEngine.
        
        Note: The current RAGEngine doesn't support metadata filtering,
        so filters are ignored. To support filters, RAGEngine would need
        to be enhanced to accept filter parameters.
        
        Args:
            query: Search query
            top_k: Number of results (uses RAGEngine's configured top_k)
            filters: Metadata filters (not currently supported)
            
        Returns:
            List of DocumentSnippet objects
        """
        # Query the RAG engine
        result = self.rag_engine.query(query)
        
        # Convert source documents to DocumentSnippet objects
        snippets = []
        sources = result.get("sources", [])
        
        for i, source in enumerate(sources):
            # Estimate score based on rank (first is highest)
            score = 0.9 - (i * 0.1)  # Simple decay: 0.9, 0.8, 0.7, ...
            score = max(0.3, score)  # Floor at 0.3
            
            snippet = DocumentSnippet(
                title=source.get("source", "Unknown"),
                url=source.get("source", ""),
                content="",  # RAGEngine doesn't return content in sources
                score=score,
                metadata={"page": source.get("page", "unknown")}
            )
            snippets.append(snippet)
        
        return snippets[:top_k]
    
    def supports_hybrid_search(self) -> bool:
        """RAGEngine uses vector search only."""
        return False


def enhanced_rag_query(rag_engine, user_query: str, config: ChatbotConfig = None) -> Dict[str, Any]:
    """
    Enhanced RAG query with intent awareness.
    
    This function demonstrates how to use intent-aware components
    alongside the existing RAG engine.
    
    Args:
        rag_engine: Existing RAGEngine instance
        user_query: User's query
        config: Optional configuration (uses defaults if None)
        
    Returns:
        Dictionary with enhanced results including intent, rewritten query, etc.
    """
    if config is None:
        config = ChatbotConfig()
    
    # Create adapter for RAGEngine
    retriever_adapter = RAGEngineRetrieverAdapter(rag_engine)
    
    # Use process_query for intent-aware processing
    result = process_query(
        user_query=user_query,
        retriever=retriever_adapter,
        config=config
    )
    
    # Return enhanced result
    return {
        "query": user_query,
        "intent": result.intent,
        "confidence": result.confidence,
        "answer": result.answer,
        "clarification_question": result.question,
        "citations": result.citations,
        "telemetry": result.telemetry
    }


def example_usage():
    """Example of how to use the integration."""
    # Note: This is a conceptual example
    # In practice, you would initialize RAGEngine with your actual parameters
    
    print("Integration Example: Intent-Aware RAG")
    print("="*60)
    
    # Conceptual example - would need actual RAGEngine instance
    print("\n1. Initialize RAGEngine (your existing code):")
    print("""
    from atlasai_runtime.rag_engine import RAGEngine
    
    rag_engine = RAGEngine(
        documents_dir="./documents",
        onenote_runbook_path="...",
        embedding_model="path/to/model",
        text_gen_model="path/to/model",
        top_k=4
    )
    """)
    
    print("\n2. Use enhanced query function:")
    print("""
    # Simple integration
    result = enhanced_rag_query(
        rag_engine=rag_engine,
        user_query="How do I configure ADMS?"
    )
    
    if result['clarification_question']:
        print(f"Need clarification: {result['clarification_question']}")
    else:
        print(f"Intent: {result['intent']}")
        print(f"Answer: {result['answer']}")
        for citation in result['citations']:
            print(f"  - {citation['title']}")
    """)
    
    print("\n3. Alternative: Use components directly:")
    print("""
    # More control over individual components
    classifier = IntentClassifier()
    rewriter = QueryRewriter()
    
    # Classify intent
    classification = classifier.classify(user_query)
    print(f"Intent: {classification.intent}, Confidence: {classification.confidence}")
    
    # Rewrite query
    rewrite = rewriter.rewrite(user_query, classification.intent)
    print(f"Rewritten: {rewrite.rewritten_query}")
    print(f"Entities: {rewrite.entities}")
    
    # Use rewritten query with RAGEngine
    result = rag_engine.query(rewrite.rewritten_query)
    print(f"Answer: {result['answer']}")
    """)
    
    print("\n4. Benefits of integration:")
    print("  ✓ Intent classification for better understanding")
    print("  ✓ Query rewriting for improved retrieval")
    print("  ✓ Intent-specific answer formatting")
    print("  ✓ Automatic clarification when needed")
    print("  ✓ Telemetry for monitoring and evaluation")
    print("  ✓ Configurable thresholds and parameters")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    example_usage()
