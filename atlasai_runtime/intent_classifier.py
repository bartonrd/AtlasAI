"""
Intent Classification Module

Classifies user intent from input queries with confidence scoring and rule-based
enhancements for obvious signals like bug reports.
"""

import re
from typing import Dict, Any, List, Tuple
from enum import Enum


class IntentType(str, Enum):
    """Supported intent types."""
    HOW_TO = "how_to"
    BUG_RESOLUTION = "bug_resolution"
    TOOL_EXPLANATION = "tool_explanation"
    ESCALATE_OR_TICKET = "escalate_or_ticket"
    CHITCHAT = "chitchat"
    OTHER = "other"


class IntentClassificationResult:
    """Result of intent classification."""
    
    def __init__(self, intent: IntentType, confidence: float, rationale: str):
        """
        Initialize classification result.
        
        Args:
            intent: Classified intent type
            confidence: Confidence score (0.0 to 1.0)
            rationale: Human-readable explanation of classification
        """
        self.intent = intent
        self.confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]
        self.rationale = rationale
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "intent": self.intent.value,
            "confidence": self.confidence,
            "rationale": self.rationale
        }


class IntentClassifier:
    """
    Classifies user intent with rule-based enhancements and confidence scoring.
    
    Intents supported:
    - how_to: Procedural questions about performing tasks
    - bug_resolution: Error messages, crashes, issues
    - tool_explanation: Conceptual questions about features/tools
    - escalate_or_ticket: Requests for human help or ticket creation
    - chitchat: Casual conversation
    - other: Unclear or out-of-scope queries
    """
    
    # Rule patterns for bug signals (high confidence)
    BUG_PATTERNS = [
        r"\bexception\b",
        r"\berror\b.*\b\d{3,4}\b",  # error 404, error code 500
        r"\b\d{3,4}\b.*\berror\b",  # 404 error
        r"\bstack\s*trace\b",
        r"\bfailed\b",
        r"\bcrash(ed|ing)?\b",
        r"\bnot\s+working\b",
        r"\bbug\b",
        r"\bissue\b.*\b(with|in)\b",
        r"\b[A-Z][a-z]+Exception\b",  # Java-style exceptions
        r"\b[A-Z][a-z]+Error\b",      # JavaScript-style errors
        r"\bHTTP\s*[45]\d{2}\b",       # HTTP error codes
    ]
    
    # Keywords for each intent
    INTENT_KEYWORDS = {
        IntentType.HOW_TO: [
            "how to", "how do i", "how can i", "steps", "guide", "tutorial",
            "instructions", "procedure", "process", "way to", "method",
            "configure", "setup", "install", "create", "deploy"
        ],
        IntentType.BUG_RESOLUTION: [
            "error", "exception", "failed", "crash", "issue", "problem",
            "bug", "broken", "not working", "fix", "troubleshoot", "debug"
        ],
        IntentType.TOOL_EXPLANATION: [
            "what is", "what does", "explain", "describe", "definition",
            "meaning", "purpose", "why", "difference between", "overview",
            "documentation", "feature", "capability", "use case"
        ],
        IntentType.ESCALATE_OR_TICKET: [
            "speak to", "talk to", "human", "person", "support", "help desk",
            "ticket", "escalate", "urgent", "critical", "contact", "reach out"
        ],
        IntentType.CHITCHAT: [
            "hello", "hi", "hey", "thank you", "thanks", "bye", "goodbye",
            "how are you", "nice", "good", "great", "awesome", "cool"
        ]
    }
    
    def __init__(self, confidence_threshold: float = 0.55):
        """
        Initialize intent classifier.
        
        Args:
            confidence_threshold: Minimum confidence to accept classification (default 0.55)
        """
        self.confidence_threshold = confidence_threshold
        self._compile_bug_patterns()
    
    def _compile_bug_patterns(self):
        """Compile bug detection patterns for efficiency."""
        self._bug_regex_list = [
            re.compile(pattern, re.IGNORECASE) for pattern in self.BUG_PATTERNS
        ]
    
    def classify(self, user_query: str) -> IntentClassificationResult:
        """
        Classify user intent from query.
        
        Args:
            user_query: The user's input query
            
        Returns:
            IntentClassificationResult with intent, confidence, and rationale
        """
        if not user_query or not user_query.strip():
            return IntentClassificationResult(
                intent=IntentType.OTHER,
                confidence=1.0,
                rationale="Empty query"
            )
        
        query_lower = user_query.lower()
        
        # Rule-based bug detection (high confidence)
        bug_match, bug_pattern = self._check_bug_patterns(user_query)
        if bug_match:
            return IntentClassificationResult(
                intent=IntentType.BUG_RESOLUTION,
                confidence=0.95,
                rationale=f"Rule-based: matched bug pattern '{bug_pattern}'"
            )
        
        # Keyword-based classification
        scores = self._score_by_keywords(query_lower)
        
        # Get best match
        if not scores:
            return IntentClassificationResult(
                intent=IntentType.OTHER,
                confidence=0.3,
                rationale="No clear intent detected"
            )
        
        best_intent, raw_score, matched_keywords = max(
            scores, key=lambda x: x[1]
        )
        
        # Normalize confidence (raw score is count of matched keywords)
        # Use log scale to avoid over-confidence with many matches
        import math
        confidence = min(0.95, 0.4 + (0.15 * math.log1p(raw_score)))
        
        # Generate rationale
        keyword_list = ", ".join(f"'{kw}'" for kw in matched_keywords[:3])
        rationale = f"Keyword-based: matched {keyword_list}"
        if len(matched_keywords) > 3:
            rationale += f" and {len(matched_keywords) - 3} more"
        
        return IntentClassificationResult(
            intent=best_intent,
            confidence=confidence,
            rationale=rationale
        )
    
    def _check_bug_patterns(self, text: str) -> Tuple[bool, str]:
        """
        Check if text matches bug patterns.
        
        Returns:
            Tuple of (matched: bool, pattern: str)
        """
        for pattern_obj in self._bug_regex_list:
            match = pattern_obj.search(text)
            if match:
                return True, match.group(0)
        return False, ""
    
    def _score_by_keywords(self, query_lower: str) -> List[Tuple[IntentType, float, List[str]]]:
        """
        Score intents by keyword matching.
        
        Returns:
            List of (intent, score, matched_keywords)
        """
        results = []
        
        for intent, keywords in self.INTENT_KEYWORDS.items():
            matched = []
            for keyword in keywords:
                if keyword in query_lower:
                    matched.append(keyword)
            
            if matched:
                # Score is number of unique keywords matched
                score = len(matched)
                results.append((intent, score, matched))
        
        return results
    
    def needs_clarification(self, result: IntentClassificationResult) -> bool:
        """
        Determine if clarification is needed based on confidence.
        
        Args:
            result: Classification result
            
        Returns:
            True if confidence is below threshold
        """
        return result.confidence < self.confidence_threshold
