"""
Example usage of the inference pipeline.

This script demonstrates how to use the new intent + retrieval layer
with various types of queries.
"""

import json
from atlasai_runtime.intent_classifier import IntentClassifier
from atlasai_runtime.query_rewriter import QueryRewriter
from atlasai_runtime.retriever import MockSearchBackend, Retriever
from atlasai_runtime.answer_synthesizer import AnswerSynthesizer
from atlasai_runtime.inference_engine import InferenceEngine
from atlasai_runtime.inference_config import InferenceConfig


def print_result(query: str, result):
    """Pretty print inference result."""
    print("\n" + "="*70)
    print(f"Query: {query}")
    print("="*70)
    print(f"Intent: {result.intent} (confidence: {result.confidence:.2f})")
    
    if result.is_clarification():
        print(f"\n[CLARIFICATION NEEDED]")
        print(f"Question: {result.question}")
    else:
        print(f"\n[ANSWER]")
        print(result.answer)
        
        if result.citations:
            print(f"\n[CITATIONS]")
            for citation in result.citations:
                print(f"  [{citation['index']}] {citation['title']}")
    
    print(f"\n[TELEMETRY]")
    print(f"  Elapsed: {result.telemetry.get('elapsed_ms', 0)}ms")
    if 'rewritten_query' in result.telemetry:
        print(f"  Rewritten query: {result.telemetry['rewritten_query'][:60]}...")
    print()


def main():
    """Run example queries through the inference pipeline."""
    print("AtlasAI Inference Pipeline - Example Usage")
    print("=" * 70)
    
    # Set up the inference engine
    backend = MockSearchBackend()
    config = InferenceConfig(
        confidence_threshold=0.55,
        top_k=4,
        min_retrieval_score=0.25,
    )
    engine = InferenceEngine(backend, config)
    
    # Example queries covering different intents
    queries = [
        # How-to query
        "How do I install the software?",
        
        # Bug resolution query
        "Exception: Connection timeout error errno 13",
        
        # Tool explanation query
        "What is the API module used for?",
        
        # Ambiguous query (should trigger clarification)
        "The thing",
        
        # Chitchat
        "Hello, how are you?",
    ]
    
    print("\nProcessing example queries...\n")
    
    for query in queries:
        result = engine.process_query(query)
        print_result(query, result)
    
    # Example: Processing with user feedback
    print("\n" + "="*70)
    print("Example: Query with user feedback")
    print("="*70)
    result = engine.process_query(
        user_query="How to configure the database?",
        user_feedback="This was helpful, thanks!"
    )
    print(f"Query: How to configure the database?")
    print(f"Feedback included in telemetry: {result.telemetry.get('user_feedback')}")
    print()
    
    # Example: Module-by-module usage
    print("\n" + "="*70)
    print("Example: Using modules independently")
    print("="*70)
    
    # 1. Intent Classification
    classifier = IntentClassifier()
    query = "Error: File not found in v1.2.3"
    intent_result = classifier.classify(query)
    print(f"\n1. Intent Classification")
    print(f"   Query: {query}")
    print(f"   Intent: {intent_result.intent}")
    print(f"   Confidence: {intent_result.confidence:.2f}")
    print(f"   Rationale: {intent_result.rationale}")
    
    # 2. Query Rewriting
    rewriter = QueryRewriter()
    rewritten = rewriter.rewrite(query, intent_result.intent)
    print(f"\n2. Query Rewriting")
    print(f"   Original: {query}")
    print(f"   Rewritten: {rewritten.rewritten_query}")
    print(f"   Entities: {rewritten.entities}")
    print(f"   Constraints: {rewritten.constraints}")
    
    # 3. Retrieval
    retriever = Retriever(backend)
    docs = retriever.retrieve(rewritten.rewritten_query, intent_result.intent, top_k=3)
    print(f"\n3. Retrieval")
    print(f"   Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"   [{i}] {doc.title} (score: {doc.score:.2f})")
    
    # 4. Answer Synthesis
    synthesizer = AnswerSynthesizer()
    answer = synthesizer.synthesize(query, intent_result.intent, docs)
    print(f"\n4. Answer Synthesis")
    print(f"   Answer length: {len(answer.answer)} chars")
    print(f"   Citations: {len(answer.citations)}")
    print(f"   First 100 chars: {answer.answer[:100]}...")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
