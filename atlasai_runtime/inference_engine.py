"""
Inference engine - orchestrates the intent classification, query rewriting,
retrieval, and answer synthesis pipeline.

This is the main facade that integrates all components.
"""

import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field

from .intent_classifier import IntentClassifier, IntentResult
from .query_rewriter import QueryRewriter, RewrittenQuery
from .retriever import Retriever, RetrievedDoc, SearchBackend
from .answer_synthesizer import AnswerSynthesizer, SynthesizedAnswer
from .clarification import ClarificationGenerator, ClarificationQuestion
from .inference_config import InferenceConfig


@dataclass
class InferenceResult:
    """
    Complete result from the inference pipeline.
    
    Contains either an answer or a clarification question, plus telemetry.
    """
    # Primary response (one of these will be populated)
    answer: Optional[str] = None
    question: Optional[str] = None
    
    # Intent information
    intent: str = ""
    confidence: float = 0.0
    
    # Citations (if answer was generated)
    citations: list = field(default_factory=list)
    
    # Telemetry
    telemetry: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "question": self.question,
            "intent": self.intent,
            "confidence": self.confidence,
            "citations": self.citations,
            "telemetry": self.telemetry,
        }
    
    def is_clarification(self) -> bool:
        """Check if this is a clarification question rather than an answer."""
        return self.question is not None


class InferenceEngine:
    """
    Main inference engine that orchestrates the complete pipeline.
    
    Pipeline stages:
    1. Intent classification
    2. Query rewriting (if needed)
    3. Document retrieval
    4. Answer synthesis or clarification generation
    
    Provides a single entry point: process_query()
    """
    
    def __init__(
        self,
        search_backend: SearchBackend,
        config: Optional[InferenceConfig] = None,
        llm_provider: Optional[Any] = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            search_backend: Search backend for retrieval
            config: Configuration (uses defaults if None)
            llm_provider: Optional LLM provider for classification/synthesis
        """
        self.config = config or InferenceConfig()
        self.search_backend = search_backend
        self.llm_provider = llm_provider
        
        # Initialize pipeline components
        self.intent_classifier = IntentClassifier(
            llm_provider=llm_provider,
            bug_signal_keywords=self.config.bug_signal_keywords,
        )
        
        self.query_rewriter = QueryRewriter()
        
        self.retriever = Retriever(
            search_backend=search_backend,
            intent_filters=self.config.default_filters,
        )
        
        self.answer_synthesizer = AnswerSynthesizer(
            llm_provider=llm_provider,
        )
        
        self.clarification_generator = ClarificationGenerator()
    
    def process_query(
        self,
        user_query: str,
        context: Optional[Dict[str, Any]] = None,
        user_feedback: Optional[str] = None,
    ) -> InferenceResult:
        """
        Process a user query through the complete inference pipeline.
        
        Args:
            user_query: The user's question
            context: Optional context (previous messages, user info, etc.)
            user_feedback: Optional user feedback on previous answer
        
        Returns:
            InferenceResult with answer/question and telemetry
        """
        start_time = time.time()
        telemetry = {
            "user_query": user_query,
            "config": self.config.to_dict(),
        }
        
        try:
            # Stage 1: Intent Classification
            intent_result = self.intent_classifier.classify(user_query, context)
            telemetry["intent"] = intent_result.intent
            telemetry["confidence"] = intent_result.confidence
            telemetry["intent_rationale"] = intent_result.rationale
            
            # Check if confidence is too low
            if intent_result.confidence < self.config.confidence_threshold:
                clarification = self.clarification_generator.generate_low_confidence_question(
                    user_query=user_query,
                    detected_intent=intent_result.intent,
                    confidence=intent_result.confidence,
                )
                telemetry["had_clarification"] = True
                telemetry["clarification_reason"] = clarification.reason
                telemetry["elapsed_ms"] = int((time.time() - start_time) * 1000)
                
                return InferenceResult(
                    question=clarification.question,
                    intent=intent_result.intent,
                    confidence=intent_result.confidence,
                    citations=[],
                    telemetry=telemetry,
                )
            
            # Stage 2: Query Rewriting
            rewritten = self.query_rewriter.rewrite(user_query, intent_result.intent)
            telemetry["rewritten_query"] = rewritten.rewritten_query
            telemetry["entities"] = rewritten.entities
            telemetry["constraints"] = rewritten.constraints
            
            # Stage 3: Retrieval
            retrieved_docs = self.retriever.retrieve(
                query=rewritten.rewritten_query,
                intent=intent_result.intent,
                top_k=self.config.top_k,
            )
            
            telemetry["num_retrieved"] = len(retrieved_docs)
            if retrieved_docs:
                telemetry["top_k_scores"] = [doc.score for doc in retrieved_docs]
                telemetry["selected_docs"] = [doc.title for doc in retrieved_docs]
            
            # Check if retrieval scores are too low
            if not retrieved_docs or retrieved_docs[0].score < self.config.min_retrieval_score:
                top_score = retrieved_docs[0].score if retrieved_docs else 0.0
                
                if not retrieved_docs:
                    clarification = self.clarification_generator.generate_no_results_question(
                        user_query=user_query,
                        detected_intent=intent_result.intent,
                    )
                else:
                    clarification = self.clarification_generator.generate_low_retrieval_question(
                        user_query=user_query,
                        detected_intent=intent_result.intent,
                        top_score=top_score,
                    )
                
                telemetry["had_clarification"] = True
                telemetry["clarification_reason"] = clarification.reason
                telemetry["elapsed_ms"] = int((time.time() - start_time) * 1000)
                
                return InferenceResult(
                    question=clarification.question,
                    intent=intent_result.intent,
                    confidence=intent_result.confidence,
                    citations=[],
                    telemetry=telemetry,
                )
            
            # Stage 4: Answer Synthesis
            synthesized = self.answer_synthesizer.synthesize(
                user_query=user_query,
                intent=intent_result.intent,
                retrieved_docs=retrieved_docs,
                max_tokens=self.config.max_answer_tokens,
            )
            
            telemetry["had_clarification"] = False
            telemetry["answer_length"] = len(synthesized.answer)
            telemetry["num_citations"] = len(synthesized.citations)
            telemetry["elapsed_ms"] = int((time.time() - start_time) * 1000)
            
            if user_feedback:
                telemetry["user_feedback"] = user_feedback
            
            return InferenceResult(
                answer=synthesized.answer,
                intent=intent_result.intent,
                confidence=intent_result.confidence,
                citations=synthesized.citations,
                telemetry=telemetry,
            )
        
        except Exception as e:
            # Error handling - return error state with generic user message
            telemetry["error"] = str(e)
            telemetry["elapsed_ms"] = int((time.time() - start_time) * 1000)
            
            return InferenceResult(
                answer="I encountered an error processing your query. Please try rephrasing your question.",
                intent="error",
                confidence=0.0,
                citations=[],
                telemetry=telemetry,
            )
    
    def update_config(self, new_config: InferenceConfig):
        """
        Update the inference engine configuration.
        
        Args:
            new_config: New configuration to apply
        """
        self.config = new_config
        
        # Update component configurations
        self.intent_classifier.bug_keywords = new_config.bug_signal_keywords
        self.retriever.intent_filters = new_config.default_filters
