"""
Unit tests for Telemetry.
"""

import unittest
from atlasai_runtime.telemetry import (
    TelemetryEvent,
    TelemetryCollector,
    EvaluationInterface
)


class TestTelemetryEvent(unittest.TestCase):
    """Test cases for TelemetryEvent."""
    
    def test_creation(self):
        """Test creating a telemetry event."""
        event = TelemetryEvent(
            user_query="test query",
            intent="how_to",
            confidence=0.8,
            rewritten_query="expanded test query",
            entities=["Entity1"],
            constraints=["version:1.0"],
            top_k_scores=[0.9, 0.8, 0.7],
            selected_doc_titles=["Doc1", "Doc2"],
            had_clarification=False,
            answer_length=150,
            elapsed_ms=123.45
        )
        
        self.assertEqual(event.user_query, "test query")
        self.assertEqual(event.confidence, 0.8)
        self.assertIsNotNone(event.timestamp)
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = TelemetryEvent(
            user_query="test",
            intent="how_to",
            confidence=0.8,
            rewritten_query="test expanded",
            entities=[],
            constraints=[],
            top_k_scores=[0.9],
            selected_doc_titles=["Doc"],
            had_clarification=False,
            answer_length=100,
            elapsed_ms=50.0
        )
        
        event_dict = event.to_dict()
        self.assertEqual(event_dict["user_query"], "test")
        self.assertEqual(event_dict["confidence"], 0.8)


class TestTelemetryCollector(unittest.TestCase):
    """Test cases for TelemetryCollector."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = TelemetryCollector()
    
    def test_log_event(self):
        """Test logging an event."""
        event = self.collector.log_event(
            user_query="test query",
            intent="how_to",
            confidence=0.8,
            rewritten_query="expanded query",
            entities=["Entity"],
            constraints=["version:1.0"],
            top_k_scores=[0.9, 0.8],
            selected_doc_titles=["Doc1", "Doc2"],
            had_clarification=False,
            answer_length=200,
            elapsed_ms=100.5
        )
        
        self.assertIsInstance(event, TelemetryEvent)
        self.assertEqual(len(self.collector.get_events()), 1)
    
    def test_multiple_events(self):
        """Test logging multiple events."""
        for i in range(5):
            self.collector.log_event(
                user_query=f"query {i}",
                intent="how_to",
                confidence=0.7 + i * 0.05,
                rewritten_query=f"expanded {i}",
                entities=[],
                constraints=[],
                top_k_scores=[0.8],
                selected_doc_titles=["Doc"],
                had_clarification=False,
                answer_length=100,
                elapsed_ms=50.0
            )
        
        self.assertEqual(len(self.collector.get_events()), 5)
    
    def test_clear_events(self):
        """Test clearing events."""
        self.collector.log_event(
            user_query="test",
            intent="how_to",
            confidence=0.8,
            rewritten_query="test",
            entities=[],
            constraints=[],
            top_k_scores=[],
            selected_doc_titles=[],
            had_clarification=False,
            answer_length=0,
            elapsed_ms=50.0
        )
        
        self.assertEqual(len(self.collector.get_events()), 1)
        
        self.collector.clear_events()
        self.assertEqual(len(self.collector.get_events()), 0)
    
    def test_summary_stats_empty(self):
        """Test summary stats with no events."""
        stats = self.collector.get_summary_stats()
        
        self.assertEqual(stats["total_queries"], 0)
        self.assertEqual(stats["avg_confidence"], 0.0)
    
    def test_summary_stats(self):
        """Test summary statistics."""
        # Log events with different intents
        self.collector.log_event(
            user_query="q1",
            intent="how_to",
            confidence=0.8,
            rewritten_query="q1",
            entities=[],
            constraints=[],
            top_k_scores=[],
            selected_doc_titles=[],
            had_clarification=False,
            answer_length=100,
            elapsed_ms=100.0
        )
        
        self.collector.log_event(
            user_query="q2",
            intent="bug_resolution",
            confidence=0.6,
            rewritten_query="q2",
            entities=[],
            constraints=[],
            top_k_scores=[],
            selected_doc_titles=[],
            had_clarification=True,
            answer_length=0,
            elapsed_ms=50.0
        )
        
        stats = self.collector.get_summary_stats()
        
        self.assertEqual(stats["total_queries"], 2)
        self.assertEqual(stats["avg_confidence"], 0.7)
        self.assertEqual(stats["avg_elapsed_ms"], 75.0)
        self.assertEqual(stats["clarification_rate"], 0.5)
        self.assertEqual(stats["intent_distribution"]["how_to"], 1)
        self.assertEqual(stats["intent_distribution"]["bug_resolution"], 1)


class TestEvaluationInterface(unittest.TestCase):
    """Test cases for EvaluationInterface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.collector = TelemetryCollector()
        self.evaluator = EvaluationInterface(self.collector)
    
    def test_intent_accuracy_empty(self):
        """Test intent accuracy with no events."""
        accuracy = self.evaluator.compute_intent_accuracy({})
        self.assertEqual(accuracy, 0.0)
    
    def test_intent_accuracy(self):
        """Test intent accuracy computation."""
        # Log events
        self.collector.log_event(
            user_query="how to configure",
            intent="how_to",
            confidence=0.8,
            rewritten_query="",
            entities=[],
            constraints=[],
            top_k_scores=[],
            selected_doc_titles=[],
            had_clarification=False,
            answer_length=0,
            elapsed_ms=50.0
        )
        
        self.collector.log_event(
            user_query="what is API",
            intent="tool_explanation",
            confidence=0.7,
            rewritten_query="",
            entities=[],
            constraints=[],
            top_k_scores=[],
            selected_doc_titles=[],
            had_clarification=False,
            answer_length=0,
            elapsed_ms=50.0
        )
        
        self.collector.log_event(
            user_query="error 404",
            intent="bug_resolution",
            confidence=0.9,
            rewritten_query="",
            entities=[],
            constraints=[],
            top_k_scores=[],
            selected_doc_titles=[],
            had_clarification=False,
            answer_length=0,
            elapsed_ms=50.0
        )
        
        # Ground truth (1 incorrect)
        ground_truth = {
            "how to configure": "how_to",  # Correct
            "what is API": "tool_explanation",  # Correct
            "error 404": "how_to"  # Incorrect (predicted bug_resolution)
        }
        
        accuracy = self.evaluator.compute_intent_accuracy(ground_truth)
        self.assertAlmostEqual(accuracy, 2/3, places=2)
    
    def test_recall_at_k_empty(self):
        """Test recall@k with no events."""
        recall = self.evaluator.compute_recall_at_k({}, k=3)
        self.assertEqual(recall, 0.0)
    
    def test_recall_at_k(self):
        """Test recall@k computation."""
        # Log events
        self.collector.log_event(
            user_query="query1",
            intent="how_to",
            confidence=0.8,
            rewritten_query="",
            entities=[],
            constraints=[],
            top_k_scores=[0.9, 0.8, 0.7],
            selected_doc_titles=["Doc1", "Doc2", "Doc3"],
            had_clarification=False,
            answer_length=100,
            elapsed_ms=50.0
        )
        
        self.collector.log_event(
            user_query="query2",
            intent="bug_resolution",
            confidence=0.7,
            rewritten_query="",
            entities=[],
            constraints=[],
            top_k_scores=[0.85, 0.75],
            selected_doc_titles=["Doc4", "Doc5"],
            had_clarification=False,
            answer_length=100,
            elapsed_ms=50.0
        )
        
        # Ground truth
        ground_truth = {
            "query1": ["Doc1", "Doc2", "Doc6"],  # 2/3 retrieved
            "query2": ["Doc4", "Doc7"]  # 1/2 retrieved
        }
        
        recall = self.evaluator.compute_recall_at_k(ground_truth, k=3)
        
        # Average: (2/3 + 1/2) / 2 = 7/12 ≈ 0.583
        self.assertAlmostEqual(recall, 7/12, places=2)


if __name__ == "__main__":
    unittest.main()
