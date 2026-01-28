"""
Intent Classifier - Determines user intent from chat messages.

This module provides intent classification to help the chatbot provide better,
more contextual responses. It classifies user queries into one of four intent types:
- error_log_resolution: User is asking about resolving errors or issues
- how_to: User is asking how to do something or perform a task
- chit_chat: User is engaging in casual conversation or greetings
- concept_explanation: User wants to understand a concept or learn about something

The classifier uses a combination of keyword matching and zero-shot classification
to determine intent, falling back to heuristics when the model is unavailable.
"""

import re
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Intent types
INTENT_ERROR_LOG = "error_log_resolution"
INTENT_HOW_TO = "how_to"
INTENT_CHIT_CHAT = "chit_chat"
INTENT_CONCEPT_EXPLANATION = "concept_explanation"

# Configuration constants for classification
MODEL_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to use model prediction
MIN_INTENT_CONFIDENCE_THRESHOLD = 0.4  # Minimum confidence to use detected intent (fall back to default if lower)
MAX_HEURISTIC_CONFIDENCE = 0.95   # Maximum confidence for heuristic matches
BASE_HEURISTIC_CONFIDENCE = 0.7   # Base confidence when keywords match
CONFIDENCE_INCREMENT = 0.1        # Increment per additional keyword match
SHORT_MESSAGE_CONFIDENCE = 0.6    # Confidence for very short messages (likely chit-chat)
QUESTION_CONFIDENCE = 0.7         # Confidence for question patterns (likely concept explanation)
DEFAULT_CONFIDENCE = 0.5          # Default confidence when no strong pattern matches
TECHNICAL_TERM_BOOST = 0.3        # Boost confidence when technical terms detected

# Keyword patterns for each intent type
ERROR_KEYWORDS = [
    r"\berror\b", r"\bfail(ed|ure|ing)?\b", r"\bissue\b", r"\bproblem\b",
    r"\bbug\b", r"\bcrash(ed|ing)?\b", r"\bexception\b", r"\btroubleshoot\b",
    r"\bfix\b", r"\bresolve\b", r"\bdebug\b", r"\bwrong\b", r"\bnot\s+work(ing)?\b",
    r"\bbroken\b", r"\bstack\s+trace\b", r"\berror\s+(code|message)\b",
]

HOW_TO_KEYWORDS = [
    r"\bhow\s+(do|can|to)\b", r"\bsteps?\s+to\b", r"\bguide\b",
    r"\btutorial\b", r"\bprocess\s+(of|for)\b", r"\bprocedure\b",
    r"\binstructions?\b", r"\bwalkthrough\b", r"\bsetup\b",
    r"\bconfigure\b", r"\binstall\b", r"\bcreate\b", r"\bimplement\b",
    r"\bperform\b", r"\bmake\b", r"\bdo\s+i\b",
]

CHIT_CHAT_KEYWORDS = [
    r"\bhello\b", r"\bhi\b", r"\bhey\b", r"\bgreetings?\b",
    r"\bthanks?\b", r"\bthank\s+you\b", r"\bgoodbye\b", r"\bbye\b",
    r"\bhow\s+are\s+you\b", r"\bwhat'?s\s+up\b", r"\bgood\s+(morning|afternoon|evening)\b",
    r"\bokay\b", r"\bok\b", r"\bcool\b", r"\bnice\b", r"\bawesome\b",
]

CONCEPT_KEYWORDS = [
    r"\bwhat\s+(is|are)\b", r"\bdefine\b", r"\bdefinition\b",
    r"\bexplain\b", r"\bdescribe\b", r"\btell\s+me\s+about\b",
    r"\bmeaning\s+of\b", r"\bunderstand\b", r"\blearn\s+about\b",
    r"\bconcept\b", r"\btheory\b", r"\bpurpose\s+of\b",
]

# Domain-specific technical terms for SCADA/Electrical Grid
# These indicate technical queries, not chit-chat, even if the query is short
TECHNICAL_DOMAIN_KEYWORDS = [
    # SCADA/Grid specific
    r"\bpoint\b", r"\bdevice\b", r"\bstation\b", r"\bmapped?\b", r"\bmapping\b",
    r"\btelemetry\b", r"\bscada\b", r"\bems\b", r"\badms\b",
    r"\bconverter\b", r"\binternal\b", r"\bexternal\b",
    r"\bxml\b", r"\bsifb\b", r"\baudit\s+trail\b",
    r"\bmeasurement\b", r"\btap\s+position\b", r"\bdisplay\s+file\b",
    r"\bstation\s+placement\b", r"\bdevxref\b", r"\bconnectivity\b",
    # General technical terms
    r"\bmodel\b", r"\bbuild\b", r"\brebuild\b", r"\bconfig\b",
    r"\bconnection\b", r"\bassociation\b", r"\bmismatch\b",
    r"\bunmapped?\b", r"\bmis-assigned\b", r"\brename\b",
]


class IntentClassifier:
    """
    Intent classifier using pattern matching and optional zero-shot classification.
    """
    
    def __init__(self, use_model: bool = True, model_name: str = "facebook/bart-large-mnli"):
        """
        Initialize the intent classifier.
        
        Args:
            use_model: Whether to use a zero-shot classification model
            model_name: Name of the Hugging Face model to use for classification
        """
        self.use_model = use_model
        self.model_name = model_name
        self._classifier = None
        
        if use_model:
            try:
                from transformers import pipeline
                logger.info(f"Loading zero-shot classification model: {model_name}")
                self._classifier = pipeline("zero-shot-classification", model=model_name)
                logger.info("Zero-shot classification model loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load zero-shot model, falling back to heuristics: {e}")
                self.use_model = False
    
    def classify(self, text: str) -> Tuple[str, float]:
        """
        Classify the intent of a text message.
        
        Args:
            text: The user's message
            
        Returns:
            Tuple of (intent_type, confidence_score)
        """
        if not text or not text.strip():
            return INTENT_CHIT_CHAT, 0.5
        
        text = text.lower().strip()
        
        # Try model-based classification first if available
        if self.use_model and self._classifier is not None:
            try:
                intent, confidence = self._classify_with_model(text)
                # If model is highly confident, use it
                if confidence > MODEL_CONFIDENCE_THRESHOLD:
                    return intent, confidence
                # Otherwise, fall through to heuristics for better accuracy
            except Exception as e:
                logger.warning(f"Model classification failed: {e}")
        
        # Fall back to heuristic-based classification
        return self._classify_with_heuristics(text)
    
    def _classify_with_model(self, text: str) -> Tuple[str, float]:
        """
        Classify intent using zero-shot classification model.
        
        Args:
            text: The user's message (lowercase)
            
        Returns:
            Tuple of (intent_type, confidence_score)
        """
        candidate_labels = [
            "resolving errors or troubleshooting issues",
            "learning how to do something or following instructions",
            "casual conversation or greeting",
            "understanding concepts or getting explanations",
        ]
        
        result = self._classifier(text, candidate_labels)
        
        # Map labels back to intent types
        label_to_intent = {
            "resolving errors or troubleshooting issues": INTENT_ERROR_LOG,
            "learning how to do something or following instructions": INTENT_HOW_TO,
            "casual conversation or greeting": INTENT_CHIT_CHAT,
            "understanding concepts or getting explanations": INTENT_CONCEPT_EXPLANATION,
        }
        
        top_label = result["labels"][0]
        confidence = result["scores"][0]
        intent = label_to_intent.get(top_label, INTENT_CONCEPT_EXPLANATION)
        
        logger.info(f"Model classified as '{intent}' with confidence {confidence:.2f}")
        return intent, confidence
    
    def _classify_with_heuristics(self, text: str) -> Tuple[str, float]:
        """
        Classify intent using keyword-based heuristics.
        
        Args:
            text: The user's message (lowercase)
            
        Returns:
            Tuple of (intent_type, confidence_score)
        """
        # Check for domain-specific technical terms first
        has_technical_terms = self._count_keyword_matches(text, TECHNICAL_DOMAIN_KEYWORDS) > 0
        
        # Count matches for each intent type
        error_score = self._count_keyword_matches(text, ERROR_KEYWORDS)
        how_to_score = self._count_keyword_matches(text, HOW_TO_KEYWORDS)
        chit_chat_score = self._count_keyword_matches(text, CHIT_CHAT_KEYWORDS)
        concept_score = self._count_keyword_matches(text, CONCEPT_KEYWORDS)
        
        # Determine the intent with highest score
        scores = [
            (INTENT_ERROR_LOG, error_score),
            (INTENT_HOW_TO, how_to_score),
            (INTENT_CHIT_CHAT, chit_chat_score),
            (INTENT_CONCEPT_EXPLANATION, concept_score),
        ]
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        intent, score = scores[0]
        
        # If no strong matches, default based on message characteristics
        if score == 0:
            # If message contains technical terms, it's likely a technical query, not chit-chat
            if has_technical_terms:
                # Technical queries without explicit intent keywords are likely asking for explanation
                intent = INTENT_CONCEPT_EXPLANATION
                confidence = BASE_HEURISTIC_CONFIDENCE
            # Very short messages without technical terms are likely chit-chat
            elif len(text.split()) <= 3:
                intent = INTENT_CHIT_CHAT
                confidence = SHORT_MESSAGE_CONFIDENCE
            # Questions are likely concept explanations
            elif "?" in text or text.startswith(("what", "why", "when", "where", "who")):
                intent = INTENT_CONCEPT_EXPLANATION
                confidence = QUESTION_CONFIDENCE
            else:
                intent = INTENT_CONCEPT_EXPLANATION
                confidence = DEFAULT_CONFIDENCE
        else:
            # Calculate confidence based on score magnitude
            # More matches = higher confidence
            confidence = min(MAX_HEURISTIC_CONFIDENCE, BASE_HEURISTIC_CONFIDENCE + (score * CONFIDENCE_INCREMENT))
            
            # Boost confidence if technical terms are present (indicates technical query, not casual chat)
            if has_technical_terms and intent != INTENT_CHIT_CHAT:
                confidence = min(MAX_HEURISTIC_CONFIDENCE, confidence + TECHNICAL_TERM_BOOST)
            
            # Special handling: if both error and concept keywords present with similar scores,
            # prefer concept explanation as it can encompass error information
            if error_score > 0 and concept_score > 0 and abs(error_score - concept_score) <= 1:
                intent = INTENT_CONCEPT_EXPLANATION
                confidence = min(MAX_HEURISTIC_CONFIDENCE, BASE_HEURISTIC_CONFIDENCE + (max(error_score, concept_score) * CONFIDENCE_INCREMENT))
        
        logger.info(f"Heuristic classified as '{intent}' with confidence {confidence:.2f}")
        return intent, confidence
        
        logger.info(f"Heuristic classified as '{intent}' with confidence {confidence:.2f}")
        return intent, confidence
    
    def _count_keyword_matches(self, text: str, patterns: list) -> int:
        """
        Count how many keyword patterns match in the text.
        
        Args:
            text: The text to search (lowercase)
            patterns: List of regex patterns
            
        Returns:
            Number of matching patterns
        """
        count = 0
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                count += 1
        return count


def create_intent_classifier(use_model: bool = True) -> IntentClassifier:
    """
    Factory function to create an intent classifier.
    
    Args:
        use_model: Whether to use a zero-shot classification model
        
    Returns:
        IntentClassifier instance
    """
    return IntentClassifier(use_model=use_model)
