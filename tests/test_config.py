"""
Unit tests for Configuration.
"""

import unittest
from atlasai_runtime.config import ChatbotConfig


class TestChatbotConfig(unittest.TestCase):
    """Test cases for ChatbotConfig."""
    
    def test_default_config(self):
        """Test creating config with defaults."""
        config = ChatbotConfig()
        
        self.assertEqual(config.confidence_threshold, 0.55)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.min_retrieval_score, 0.25)
        self.assertEqual(config.max_answer_tokens, 500)
        self.assertIsNone(config.filter_by_product)
    
    def test_custom_config(self):
        """Test creating config with custom values."""
        config = ChatbotConfig(
            confidence_threshold=0.7,
            top_k=10,
            min_retrieval_score=0.3,
            filter_by_product="ADMS"
        )
        
        self.assertEqual(config.confidence_threshold, 0.7)
        self.assertEqual(config.top_k, 10)
        self.assertEqual(config.filter_by_product, "ADMS")
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ChatbotConfig(
            confidence_threshold=0.6,
            top_k=7
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict["confidence_threshold"], 0.6)
        self.assertEqual(config_dict["top_k"], 7)
        self.assertIn("min_retrieval_score", config_dict)
    
    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "confidence_threshold": 0.65,
            "top_k": 8,
            "filter_by_version": "2.0"
        }
        
        config = ChatbotConfig.from_dict(config_dict)
        
        self.assertEqual(config.confidence_threshold, 0.65)
        self.assertEqual(config.top_k, 8)
        self.assertEqual(config.filter_by_version, "2.0")
    
    def test_get_document_filters_empty(self):
        """Test getting document filters with no filters set."""
        config = ChatbotConfig()
        filters = config.get_document_filters()
        
        self.assertEqual(filters, {})
    
    def test_get_document_filters_with_product(self):
        """Test getting document filters with product filter."""
        config = ChatbotConfig(filter_by_product="ADMS")
        filters = config.get_document_filters()
        
        self.assertEqual(filters["product"], "ADMS")
    
    def test_get_document_filters_multiple(self):
        """Test getting multiple document filters."""
        config = ChatbotConfig(
            filter_by_product="ADMS",
            filter_by_version="2.0",
            filter_by_owner="engineering"
        )
        
        filters = config.get_document_filters()
        
        self.assertEqual(filters["product"], "ADMS")
        self.assertEqual(filters["version"], "2.0")
        self.assertEqual(filters["owner"], "engineering")
    
    def test_get_document_filters_with_date(self):
        """Test getting document filters with date filter."""
        config = ChatbotConfig(filter_by_updated_after="2024-01-01")
        filters = config.get_document_filters()
        
        self.assertIn("updated_at", filters)
        self.assertEqual(filters["updated_at"]["$gte"], "2024-01-01")


if __name__ == "__main__":
    unittest.main()
