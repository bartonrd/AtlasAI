#!/usr/bin/env python3
"""
Demonstration script for the intent-aware chatbot system.

This script shows how to use all the components together with example queries.
"""

from atlasai_runtime.process_query import process_query
from atlasai_runtime.retriever import FakeRetriever, DocumentSnippet
from atlasai_runtime.config import ChatbotConfig
from atlasai_runtime.telemetry import TelemetryCollector, EvaluationInterface


def create_sample_knowledge_base():
    """Create a sample knowledge base with diverse documents."""
    return [
        # Procedure documents
        DocumentSnippet(
            title="Installation Guide",
            url="/docs/install",
            content=(
                "Prerequisites: Python 3.9+ and admin access. "
                "Steps to install: "
                "1. Download the installer from the official website. "
                "2. Run the installer with admin privileges. "
                "3. Configure the environment variables. "
                "4. Verify installation with 'app --version'. "
                "Common pitfall: Don't skip the environment variable setup."
            ),
            score=0.95,
            metadata={"doc_type": "procedure", "version": "2.0", "updated": "2024-01-15"}
        ),
        DocumentSnippet(
            title="Configuration Guide",
            url="/docs/config",
            content=(
                "To configure the system: "
                "1. Edit config.yaml in the installation directory. "
                "2. Set database connection string. "
                "3. Configure logging level (DEBUG, INFO, WARN). "
                "4. Restart the service."
            ),
            score=0.92,
            metadata={"doc_type": "procedure", "version": "2.0"}
        ),
        
        # Incident/bug resolution documents
        DocumentSnippet(
            title="Error 404 Troubleshooting",
            url="/docs/errors/404",
            content=(
                "Error 404 occurs when a resource cannot be found. "
                "Diagnostics: Check if the URL path is correct and resource exists. "
                "Resolution: Verify the endpoint configuration in config.yaml. "
                "Ensure the resource is deployed and accessible. "
                "Verification: Test with curl to confirm the resource loads."
            ),
            score=0.90,
            metadata={"doc_type": "incident", "error_code": "404"}
        ),
        DocumentSnippet(
            title="Connection Timeout Issues",
            url="/docs/errors/timeout",
            content=(
                "Connection timeout errors indicate network connectivity problems. "
                "Check firewall settings and network connectivity. "
                "Increase timeout values in configuration if needed."
            ),
            score=0.88,
            metadata={"doc_type": "incident"}
        ),
        
        # Concept/explanation documents
        DocumentSnippet(
            title="API Overview",
            url="/docs/api/overview",
            content=(
                "The API provides programmatic access to system features. "
                "It supports RESTful endpoints for CRUD operations. "
                "Use cases include: automation, integration with third-party systems, "
                "and building custom applications. "
                "Related features: webhooks, authentication, rate limiting."
            ),
            score=0.93,
            metadata={"doc_type": "concept"}
        ),
        DocumentSnippet(
            title="Database Architecture",
            url="/docs/db/architecture",
            content=(
                "The database uses a relational model with PostgreSQL. "
                "Key tables include: users, sessions, and logs. "
                "Performance is optimized with indexes and connection pooling."
            ),
            score=0.89,
            metadata={"doc_type": "concept"}
        ),
    ]


def print_result(query: str, result):
    """Pretty print a query result."""
    print("\n" + "="*80)
    print(f"QUERY: {query}")
    print("-"*80)
    print(f"Intent: {result.intent} (confidence: {result.confidence:.2f})")
    print(f"Telemetry: {result.telemetry.get('elapsed_ms', 0):.2f}ms")
    
    if result.question:
        print("\n🔍 CLARIFICATION NEEDED:")
        print(f"   {result.question}")
    else:
        print("\n✅ ANSWER:")
        print(result.answer)
        
        if result.citations:
            print("\n📚 SOURCES:")
            for i, citation in enumerate(result.citations, 1):
                print(f"   {i}. {citation['title']}")
                print(f"      {citation['url']}")
    
    print("="*80)


def main():
    """Run demonstration."""
    print("=" * 80)
    print("INTENT-AWARE RAG CHATBOT DEMONSTRATION")
    print("=" * 80)
    
    # Setup
    print("\n🔧 Setting up knowledge base and configuration...")
    kb = create_sample_knowledge_base()
    retriever = FakeRetriever(kb)
    
    config = ChatbotConfig(
        confidence_threshold=0.55,
        top_k=5,
        min_retrieval_score=0.25,
        max_answer_tokens=500
    )
    
    telemetry = TelemetryCollector()
    
    # Test queries for different intents
    test_queries = [
        # How-to queries
        "How do I install the application?",
        "Steps to configure the database connection",
        
        # Bug resolution queries
        "I'm getting error 404 when accessing the API",
        "Connection timeout error on startup",
        
        # Tool explanation queries
        "What is the API used for?",
        "Explain the database architecture",
        
        # Low confidence / clarification queries
        "configure something",
        "error",
        
        # Chitchat
        "Hello!",
        "Thanks for your help",
    ]
    
    print(f"\n📝 Processing {len(test_queries)} test queries...\n")
    
    # Process each query
    for query in test_queries:
        result = process_query(
            user_query=query,
            retriever=retriever,
            config=config,
            telemetry_collector=telemetry
        )
        print_result(query, result)
    
    # Show telemetry summary
    print("\n" + "="*80)
    print("📊 TELEMETRY SUMMARY")
    print("="*80)
    
    stats = telemetry.get_summary_stats()
    print(f"Total queries: {stats['total_queries']}")
    print(f"Average confidence: {stats['avg_confidence']:.2f}")
    print(f"Average response time: {stats['avg_elapsed_ms']:.2f}ms")
    print(f"Clarification rate: {stats['clarification_rate']:.1%}")
    
    print("\nIntent distribution:")
    for intent, count in stats['intent_distribution'].items():
        print(f"  - {intent}: {count} ({count/stats['total_queries']:.1%})")
    
    # Evaluation example
    print("\n" + "="*80)
    print("🎯 EVALUATION EXAMPLE")
    print("="*80)
    
    # Ground truth for intent classification
    ground_truth_intents = {
        "How do I install the application?": "how_to",
        "Steps to configure the database connection": "how_to",
        "I'm getting error 404 when accessing the API": "bug_resolution",
        "Connection timeout error on startup": "bug_resolution",
        "What is the API used for?": "tool_explanation",
        "Explain the database architecture": "tool_explanation",
        "Hello!": "chitchat",
        "Thanks for your help": "chitchat",
    }
    
    evaluator = EvaluationInterface(telemetry)
    accuracy = evaluator.compute_intent_accuracy(ground_truth_intents)
    print(f"Intent classification accuracy: {accuracy:.1%}")
    
    # Ground truth for retrieval
    ground_truth_docs = {
        "How do I install the application?": ["Installation Guide"],
        "I'm getting error 404 when accessing the API": ["Error 404 Troubleshooting"],
        "What is the API used for?": ["API Overview"],
    }
    
    recall_at_3 = evaluator.compute_recall_at_k(ground_truth_docs, k=3)
    print(f"Recall@3: {recall_at_3:.1%}")
    
    print("\n✨ Demonstration complete!")
    print("="*80)


if __name__ == "__main__":
    main()
