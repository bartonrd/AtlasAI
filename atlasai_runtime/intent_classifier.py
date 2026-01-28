"""
Intent Classification Module for AtlasAI.

Classifies user queries into predefined intent categories to improve response quality.
"""

import logging
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Try to import transformers, but make it optional
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("transformers library not available, will use keyword-based classification only")
    TRANSFORMERS_AVAILABLE = False
    torch = None


class IntentClassifier:
    """
    Classifies user queries into intent categories using zero-shot classification.
    
    Intent Categories:
    - error_log_resolution: User is trying to troubleshoot errors or resolve issues from logs
    - how_to: User is asking how to perform a specific task or action
    - chit_chat: User is engaging in casual conversation or greetings
    - concept_explanation: User wants to understand concepts, definitions, or technical details
    """
    
    # Intent categories and their descriptions
    INTENT_CATEGORIES = {
        "error_log_resolution": [
            "troubleshooting an error",
            "fixing a problem",
            "resolving an issue",
            "debugging",
            "error message",
            "failure",
            "not working"
        ],
        "how_to": [
            "how to do something",
            "steps to perform a task",
            "instructions",
            "tutorial",
            "guide",
            "procedure"
        ],
        "chit_chat": [
            "casual conversation",
            "greeting",
            "small talk",
            "thank you",
            "appreciation",
            "opinion"
        ],
        "concept_explanation": [
            "explain a concept",
            "what is something",
            "definition",
            "technical details",
            "understanding",
            "theory"
        ]
    }
    
    # Keyword-based classification constants
    # Confidence normalization: divide keyword match count by this value
    # 3 matches = high confidence (~1.0), 1 match = medium confidence (~0.33)
    KEYWORD_CONFIDENCE_DIVISOR = 3.0
    MAX_KEYWORD_CONFIDENCE = 0.9  # Cap for keyword-based confidence
    
    def __init__(self, model_name: str = "facebook/bart-large-mnli", confidence_threshold: float = 0.3):
        """
        Initialize the intent classifier.
        
        Args:
            model_name: HuggingFace model for zero-shot classification
            confidence_threshold: Minimum confidence to accept classification
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self._classifier = None
        
        # Intent keywords for quick classification of simple cases
        self.intent_keywords = {
            "error_log_resolution": ["error", "failed", "failure", "issue", "problem", "troubleshoot", 
                                    "debug", "fix", "broken", "not working", "exception", "crash"],
            "how_to": ["how to", "how do i", "how can i", "steps to", "guide", "tutorial", 
                      "procedure", "instructions", "configure", "setup", "install"],
            "chit_chat": ["hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye", 
                         "how are you", "what's up", "good morning", "good afternoon"],
            "concept_explanation": ["what is", "what are", "explain", "define", "definition", 
                                   "meaning of", "understand", "describe", "overview"]
        }
    
    def _get_classifier(self):
        """Lazy load the classifier pipeline."""
        if self._classifier is None:
            if not TRANSFORMERS_AVAILABLE:
                logger.info("transformers not available, using keyword-based fallback")
                self._classifier = "keyword_fallback"
                return self._classifier
                
            logger.info(f"Loading intent classifier model: {self.model_name}")
            try:
                # Check if CUDA is available (torch is guaranteed to be available here due to TRANSFORMERS_AVAILABLE check)
                device = 0 if (torch and torch.cuda.is_available()) else -1
                self._classifier = pipeline(
                    "zero-shot-classification",
                    model=self.model_name,
                    device=device
                )
                logger.info("Intent classifier loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load zero-shot classifier: {e}")
                logger.info("Falling back to keyword-based classification")
                self._classifier = "keyword_fallback"
        return self._classifier
    
    def _keyword_based_classification(self, query: str) -> Tuple[str, float]:
        """
        Fallback keyword-based classification.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (intent, confidence)
        """
        query_lower = query.lower()
        scores = {}
        
        # Count keyword matches for each intent
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for kw in keywords if kw in query_lower)
            if score > 0:
                scores[intent] = score
        
        if not scores:
            # Default to concept_explanation for technical queries
            return "concept_explanation", 0.5
        
        # Return intent with highest score
        max_intent = max(scores.items(), key=lambda x: x[1])
        # Normalize confidence: more keyword matches = higher confidence
        confidence = min(max_intent[1] / self.KEYWORD_CONFIDENCE_DIVISOR, self.MAX_KEYWORD_CONFIDENCE)
        return max_intent[0], confidence
    
    def classify(self, query: str) -> Dict[str, Any]:
        """
        Classify the user query into an intent category.
        
        Args:
            query: User query text
            
        Returns:
            Dictionary with 'intent', 'confidence', and 'all_scores' keys
        """
        if not query or len(query.strip()) == 0:
            return {
                "intent": "chit_chat",
                "confidence": 0.5,
                "all_scores": {}
            }
        
        classifier = self._get_classifier()
        
        # Use keyword-based fallback if model failed to load
        if classifier == "keyword_fallback":
            intent, confidence = self._keyword_based_classification(query)
            return {
                "intent": intent,
                "confidence": confidence,
                "all_scores": {intent: confidence},
                "method": "keyword"
            }
        
        try:
            # Use all intent labels for classification
            candidate_labels = list(self.INTENT_CATEGORIES.keys())
            
            # Perform zero-shot classification
            result = classifier(query, candidate_labels, multi_label=False)
            
            # Extract results
            intent = result["labels"][0]
            confidence = result["scores"][0]
            all_scores = dict(zip(result["labels"], result["scores"]))
            
            # If confidence is too low, try keyword-based as fallback
            if confidence < self.confidence_threshold:
                keyword_intent, keyword_conf = self._keyword_based_classification(query)
                if keyword_conf > confidence:
                    logger.info(f"Using keyword-based classification (conf={keyword_conf:.2f}) over model (conf={confidence:.2f})")
                    return {
                        "intent": keyword_intent,
                        "confidence": keyword_conf,
                        "all_scores": {keyword_intent: keyword_conf},
                        "method": "keyword_fallback"
                    }
            
            return {
                "intent": intent,
                "confidence": confidence,
                "all_scores": all_scores,
                "method": "zero_shot"
            }
            
        except Exception as e:
            logger.error(f"Error during intent classification: {e}", exc_info=True)
            # Fallback to keyword-based
            intent, confidence = self._keyword_based_classification(query)
            return {
                "intent": intent,
                "confidence": confidence,
                "all_scores": {intent: confidence},
                "method": "keyword_fallback_error"
            }
    
    def get_intent_description(self, intent: str) -> str:
        """
        Get a human-readable description of an intent.
        
        Args:
            intent: Intent category
            
        Returns:
            Description string
        """
        descriptions = {
            "error_log_resolution": "Troubleshooting and resolving errors",
            "how_to": "Step-by-step instructions for tasks",
            "chit_chat": "Casual conversation",
            "concept_explanation": "Technical concepts and definitions"
        }
        return descriptions.get(intent, "Unknown intent")
