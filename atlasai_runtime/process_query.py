"""
Process Query Module

Main wiring function that integrates all chatbot components.
"""

import time
from typing import Dict, Any, Optional
from .intent_classifier import IntentClassifier, IntentType
from .query_rewriter import QueryRewriter
from .retriever import RetrieverInterface, IntentBasedRetriever
from .answer_synthesizer import AnswerSynthesizer
from .clarifying_question_generator import ClarifyingQuestionGenerator
from .telemetry import TelemetryCollector
from .config import ChatbotConfig


class ProcessQueryResult:
    """Result of query processing."""
    
    def __init__(
        self,
        answer: str,
        question: Optional[str],
        citations: list,
        intent: str,
        confidence: float,
        telemetry: Dict[str, Any]
    ):
        """
        Initialize query result.
        
        Args:
            answer: Generated answer (empty if asking clarification)
            question: Clarifying question (None if providing answer)
            citations: List of source citations
            intent: Classified intent
            confidence: Classification confidence
            telemetry: Telemetry data for this query
        """
        self.answer = answer
        self.question = question
        self.citations = citations
        self.intent = intent
        self.confidence = confidence
        self.telemetry = telemetry
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "answer": self.answer,
            "question": self.question,
            "citations": self.citations,
            "intent": self.intent,
            "confidence": self.confidence,
            "telemetry": self.telemetry
        }


def process_query(
    user_query: str,
    retriever: RetrieverInterface,
    config: Optional[ChatbotConfig] = None,
    telemetry_collector: Optional[TelemetryCollector] = None,
    context: Optional[Dict[str, Any]] = None
) -> ProcessQueryResult:
    """
    Process a user query through the intent-aware RAG pipeline.
    
    This is the main entry point that orchestrates all components:
    1. Classify intent
    2. Rewrite query
    3. Retrieve documents
    4. Synthesize answer or ask clarification
    5. Log telemetry
    
    Args:
        user_query: The user's input query
        retriever: Retriever implementation
        config: Configuration parameters (uses defaults if None)
        telemetry_collector: Telemetry collector (creates new if None)
        context: Optional context (e.g., conversation history, user info)
        
    Returns:
        ProcessQueryResult with answer or question, citations, and telemetry
    """
    start_time = time.time()
    
    # Initialize defaults
    if config is None:
        config = ChatbotConfig()
    
    if telemetry_collector is None:
        telemetry_collector = TelemetryCollector()
    
    if context is None:
        context = {}
    
    # Initialize components
    intent_classifier = IntentClassifier(
        confidence_threshold=config.confidence_threshold
    )
    query_rewriter = QueryRewriter()
    answer_synthesizer = AnswerSynthesizer(
        min_retrieval_score=config.min_retrieval_score,
        max_answer_tokens=config.max_answer_tokens
    )
    clarifying_generator = ClarifyingQuestionGenerator()
    intent_based_retriever = IntentBasedRetriever(retriever)
    
    # Step 1: Classify intent
    classification = intent_classifier.classify(user_query)
    
    # Step 2: Rewrite query
    rewrite_result = query_rewriter.rewrite(
        user_query,
        classification.intent
    )
    
    # Step 3: Check if clarification needed due to low confidence
    needs_clarification = intent_classifier.needs_clarification(classification)
    
    if needs_clarification:
        # Generate clarifying question
        clarifying_question = clarifying_generator.generate(
            query=user_query,
            intent=classification.intent,
            confidence=classification.confidence,
            entities=rewrite_result.entities,
            constraints=rewrite_result.constraints
        )
        
        if clarifying_question:
            # Return clarification instead of answer
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Log telemetry
            telemetry_collector.log_event(
                user_query=user_query,
                intent=classification.intent.value,
                confidence=classification.confidence,
                rewritten_query=rewrite_result.rewritten_query,
                entities=rewrite_result.entities,
                constraints=rewrite_result.constraints,
                top_k_scores=[],
                selected_doc_titles=[],
                had_clarification=True,
                answer_length=0,
                elapsed_ms=elapsed_ms
            )
            
            return ProcessQueryResult(
                answer="",
                question=clarifying_question,
                citations=[],
                intent=classification.intent.value,
                confidence=classification.confidence,
                telemetry={
                    "elapsed_ms": elapsed_ms,
                    "rewritten_query": rewrite_result.rewritten_query,
                    "entities": rewrite_result.entities,
                    "constraints": rewrite_result.constraints,
                    "rationale": classification.rationale
                }
            )
    
    # Step 4: Retrieve documents
    doc_filters = config.get_document_filters()
    snippets = intent_based_retriever.retrieve_by_intent(
        query=rewrite_result.rewritten_query,
        intent=classification.intent,
        top_k=config.top_k,
        additional_filters=doc_filters
    )
    
    # Step 5: Check if we need clarification due to weak retrieval
    top_k_scores = [s.score for s in snippets]
    selected_doc_titles = [s.title for s in snippets]
    
    # Generate clarifying question if retrieval is weak
    clarifying_question_for_weak_retrieval = None
    if not snippets or (snippets and snippets[0].score < config.min_retrieval_score):
        clarifying_question_for_weak_retrieval = clarifying_generator.generate(
            query=user_query,
            intent=classification.intent,
            confidence=classification.confidence,
            entities=rewrite_result.entities,
            constraints=rewrite_result.constraints
        )
    
    # Step 6: Synthesize answer
    synthesis_result = answer_synthesizer.synthesize(
        query=user_query,
        intent=classification.intent,
        snippets=snippets,
        clarification_question=clarifying_question_for_weak_retrieval
    )
    
    # Step 7: Prepare result
    elapsed_ms = (time.time() - start_time) * 1000
    
    answer_length = len(synthesis_result.answer) if synthesis_result.answer else 0
    
    # Log telemetry
    telemetry_collector.log_event(
        user_query=user_query,
        intent=classification.intent.value,
        confidence=classification.confidence,
        rewritten_query=rewrite_result.rewritten_query,
        entities=rewrite_result.entities,
        constraints=rewrite_result.constraints,
        top_k_scores=top_k_scores,
        selected_doc_titles=selected_doc_titles,
        had_clarification=synthesis_result.should_ask_clarification,
        answer_length=answer_length,
        elapsed_ms=elapsed_ms
    )
    
    # Return result
    if synthesis_result.should_ask_clarification:
        return ProcessQueryResult(
            answer="",
            question=synthesis_result.clarification_question,
            citations=[],
            intent=classification.intent.value,
            confidence=classification.confidence,
            telemetry={
                "elapsed_ms": elapsed_ms,
                "rewritten_query": rewrite_result.rewritten_query,
                "entities": rewrite_result.entities,
                "constraints": rewrite_result.constraints,
                "rationale": classification.rationale,
                "top_scores": top_k_scores[:3]
            }
        )
    else:
        return ProcessQueryResult(
            answer=synthesis_result.answer,
            question=None,
            citations=synthesis_result.citations,
            intent=classification.intent.value,
            confidence=classification.confidence,
            telemetry={
                "elapsed_ms": elapsed_ms,
                "rewritten_query": rewrite_result.rewritten_query,
                "entities": rewrite_result.entities,
                "constraints": rewrite_result.constraints,
                "rationale": classification.rationale,
                "top_scores": top_k_scores[:3],
                "num_citations": len(synthesis_result.citations)
            }
        )
