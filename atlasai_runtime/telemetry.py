"""
Telemetry Module

Logs and tracks metrics for evaluation and monitoring.
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    
    # Request info
    user_query: str
    intent: str
    confidence: float
    
    # Rewriting
    rewritten_query: str
    entities: List[str]
    constraints: List[str]
    
    # Retrieval
    top_k_scores: List[float]
    selected_doc_titles: List[str]
    
    # Answer
    had_clarification: bool
    answer_length: int
    
    # Performance
    elapsed_ms: float
    
    # Optional user feedback
    user_feedback: Optional[str] = None
    
    # Timestamp
    timestamp: str = ""
    
    def __post_init__(self):
        """Set timestamp after initialization."""
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class TelemetryCollector:
    """
    Collects and stores telemetry events.
    
    Provides hooks for offline evaluation.
    """
    
    def __init__(self):
        """Initialize telemetry collector."""
        self.events: List[TelemetryEvent] = []
    
    def log_event(
        self,
        user_query: str,
        intent: str,
        confidence: float,
        rewritten_query: str,
        entities: List[str],
        constraints: List[str],
        top_k_scores: List[float],
        selected_doc_titles: List[str],
        had_clarification: bool,
        answer_length: int,
        elapsed_ms: float,
        user_feedback: Optional[str] = None
    ) -> TelemetryEvent:
        """
        Log a telemetry event.
        
        Args:
            user_query: Original query
            intent: Classified intent
            confidence: Classification confidence
            rewritten_query: Rewritten query
            entities: Extracted entities
            constraints: Extracted constraints
            top_k_scores: Retrieval scores
            selected_doc_titles: Document titles used
            had_clarification: Whether clarification was asked
            answer_length: Length of answer
            elapsed_ms: Time elapsed
            user_feedback: Optional user feedback
            
        Returns:
            TelemetryEvent object
        """
        event = TelemetryEvent(
            user_query=user_query,
            intent=intent,
            confidence=confidence,
            rewritten_query=rewritten_query,
            entities=entities,
            constraints=constraints,
            top_k_scores=top_k_scores,
            selected_doc_titles=selected_doc_titles,
            had_clarification=had_clarification,
            answer_length=answer_length,
            elapsed_ms=elapsed_ms,
            user_feedback=user_feedback
        )
        
        self.events.append(event)
        return event
    
    def get_events(self) -> List[TelemetryEvent]:
        """Get all logged events."""
        return self.events.copy()
    
    def get_events_dict(self) -> List[Dict[str, Any]]:
        """Get all events as dictionaries."""
        return [event.to_dict() for event in self.events]
    
    def clear_events(self):
        """Clear all logged events."""
        self.events.clear()
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary with summary metrics
        """
        if not self.events:
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "avg_elapsed_ms": 0.0,
                "clarification_rate": 0.0,
                "intent_distribution": {}
            }
        
        total = len(self.events)
        avg_confidence = sum(e.confidence for e in self.events) / total
        avg_elapsed = sum(e.elapsed_ms for e in self.events) / total
        clarifications = sum(1 for e in self.events if e.had_clarification)
        clarification_rate = clarifications / total
        
        # Intent distribution
        intent_counts: Dict[str, int] = {}
        for event in self.events:
            intent_counts[event.intent] = intent_counts.get(event.intent, 0) + 1
        
        return {
            "total_queries": total,
            "avg_confidence": round(avg_confidence, 3),
            "avg_elapsed_ms": round(avg_elapsed, 2),
            "clarification_rate": round(clarification_rate, 3),
            "intent_distribution": intent_counts
        }


class EvaluationInterface:
    """
    Interface for offline evaluation.
    
    Provides methods to compute intent accuracy and recall@k.
    """
    
    def __init__(self, telemetry_collector: TelemetryCollector):
        """
        Initialize evaluation interface.
        
        Args:
            telemetry_collector: Telemetry collector instance
        """
        self.telemetry = telemetry_collector
    
    def compute_intent_accuracy(
        self,
        ground_truth: Dict[str, str]
    ) -> float:
        """
        Compute intent classification accuracy.
        
        Args:
            ground_truth: Mapping of query -> correct intent
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        events = self.telemetry.get_events()
        
        if not events:
            return 0.0
        
        correct = 0
        total = 0
        
        for event in events:
            if event.user_query in ground_truth:
                total += 1
                if event.intent == ground_truth[event.user_query]:
                    correct += 1
        
        if total == 0:
            return 0.0
        
        return correct / total
    
    def compute_recall_at_k(
        self,
        ground_truth: Dict[str, List[str]],
        k: int = 5
    ) -> float:
        """
        Compute recall@k for retrieval.
        
        Args:
            ground_truth: Mapping of query -> list of relevant doc titles
            k: Number of top results to consider
            
        Returns:
            Average recall@k score
        """
        events = self.telemetry.get_events()
        
        if not events:
            return 0.0
        
        recall_scores = []
        
        for event in events:
            if event.user_query in ground_truth:
                relevant_docs = set(ground_truth[event.user_query])
                retrieved_docs = set(event.selected_doc_titles[:k])
                
                if not relevant_docs:
                    continue
                
                # Recall = |relevant ∩ retrieved| / |relevant|
                intersect = relevant_docs & retrieved_docs
                recall = len(intersect) / len(relevant_docs)
                recall_scores.append(recall)
        
        if not recall_scores:
            return 0.0
        
        return sum(recall_scores) / len(recall_scores)
    
    def export_for_analysis(self, output_path: str):
        """
        Export telemetry data for offline analysis.
        
        Args:
            output_path: Path to output file (JSON)
        """
        import json
        
        data = self.telemetry.get_events_dict()
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
