"""
Intent classification module.

Classifies user queries into predefined intents with confidence scores.
Supports rule-based bias for bug-related queries.
"""

import re
from typing import Dict, Any, Optional, Protocol
from dataclasses import dataclass


# Intent types
INTENT_HOW_TO = "how_to"
INTENT_BUG_RESOLUTION = "bug_resolution"
INTENT_TOOL_EXPLANATION = "tool_explanation"
INTENT_ESCALATE = "escalate_or_ticket"
INTENT_CHITCHAT = "chitchat"
INTENT_OTHER = "other"

ALL_INTENTS = [
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
    INTENT_ESCALATE,
    INTENT_CHITCHAT,
    INTENT_OTHER,
]


@dataclass
class IntentResult:
    """Result of intent classification."""
    intent: str
    confidence: float
    rationale: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "intent": self.intent,
            "confidence": self.confidence,
            "rationale": self.rationale,
        }


class LLMProvider(Protocol):
    """Protocol for LLM providers used in classification."""
    
    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate text from prompt."""
        ...


class IntentClassifier:
    """
    Classifies user queries into intents with confidence scores.
    
    Uses a combination of rule-based detection and optional LLM-based classification.
    Provides special bias for bug-related signals.
    """
    
    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        bug_signal_keywords: Optional[list] = None,
    ):
        """
        Initialize the intent classifier.
        
        Args:
            llm_provider: Optional LLM provider for advanced classification
            bug_signal_keywords: Keywords that indicate bug-related queries
        """
        self.llm_provider = llm_provider
        self.bug_keywords = bug_signal_keywords or [
            "exception", "error", "failed", "crash", "bug", "issue",
            "traceback", "stack trace", "errno", "exit code", "warning",
            "fault", "failure", "broken", "not working", "doesn't work"
        ]
    
    def classify(self, user_query: str, context: Optional[Dict[str, Any]] = None) -> IntentResult:
        """
        Classify user query into an intent.
        
        Args:
            user_query: The user's question or message
            context: Optional context (previous messages, user info, etc.)
        
        Returns:
            IntentResult with intent, confidence, and rationale
        """
        if not user_query or not user_query.strip():
            return IntentResult(
                intent=INTENT_OTHER,
                confidence=0.0,
                rationale="Empty query"
            )
        
        query_lower = user_query.lower()
        
        # Rule 1: Bug detection with high priority
        bug_score = self._detect_bug_signals(query_lower)
        if bug_score > 0.7:
            return IntentResult(
                intent=INTENT_BUG_RESOLUTION,
                confidence=bug_score,
                rationale=f"Bug signals detected in query (score: {bug_score:.2f})"
            )
        
        # Rule 2: How-to patterns
        how_to_score = self._detect_how_to(query_lower)
        if how_to_score > 0.6:
            return IntentResult(
                intent=INTENT_HOW_TO,
                confidence=how_to_score,
                rationale=f"Procedural instruction patterns detected (score: {how_to_score:.2f})"
            )
        
        # Rule 3: Tool explanation patterns
        tool_score = self._detect_tool_explanation(query_lower)
        if tool_score > 0.6:
            return IntentResult(
                intent=INTENT_TOOL_EXPLANATION,
                confidence=tool_score,
                rationale=f"Concept/tool inquiry patterns detected (score: {tool_score:.2f})"
            )
        
        # Rule 4: Escalation patterns
        escalate_score = self._detect_escalation(query_lower)
        if escalate_score > 0.6:
            return IntentResult(
                intent=INTENT_ESCALATE,
                confidence=escalate_score,
                rationale=f"Escalation/support request patterns detected (score: {escalate_score:.2f})"
            )
        
        # Rule 5: Chitchat patterns
        chitchat_score = self._detect_chitchat(query_lower)
        if chitchat_score > 0.6:
            return IntentResult(
                intent=INTENT_CHITCHAT,
                confidence=chitchat_score,
                rationale=f"Conversational/chitchat patterns detected (score: {chitchat_score:.2f})"
            )
        
        # If LLM provider available, use it for ambiguous cases
        if self.llm_provider and max(bug_score, how_to_score, tool_score, escalate_score, chitchat_score) < 0.6:
            return self._classify_with_llm(user_query)
        
        # Default: Select highest scoring intent or fallback to OTHER
        scores = {
            INTENT_BUG_RESOLUTION: bug_score,
            INTENT_HOW_TO: how_to_score,
            INTENT_TOOL_EXPLANATION: tool_score,
            INTENT_ESCALATE: escalate_score,
            INTENT_CHITCHAT: chitchat_score,
        }
        
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        if best_score < 0.3:
            return IntentResult(
                intent=INTENT_OTHER,
                confidence=0.4,
                rationale="No clear intent pattern detected"
            )
        
        return IntentResult(
            intent=best_intent,
            confidence=best_score,
            rationale=f"Best match based on pattern analysis (score: {best_score:.2f})"
        )
    
    def _detect_bug_signals(self, query_lower: str) -> float:
        """Detect bug-related signals with bias boost."""
        score = 0.0
        
        # Check for bug keywords
        keyword_count = sum(1 for kw in self.bug_keywords if kw in query_lower)
        if keyword_count > 0:
            score += min(0.4 + (keyword_count * 0.15), 0.9)
        
        # Check for error patterns (Exception:, Error:, errno, etc.)
        error_patterns = [
            r'\b(?:exception|error)\s*:\s*',
            r'\berrno\s*[-:=]?\s*\d+',
            r'\bexit\s+code\s*[-:=]?\s*\d+',
            r'\bstack\s+trace\b',
            r'\btraceback\b',
        ]
        for pattern in error_patterns:
            if re.search(pattern, query_lower):
                score += 0.3
                break
        
        return min(score, 1.0)
    
    def _detect_how_to(self, query_lower: str) -> float:
        """Detect how-to/procedural patterns."""
        score = 0.0
        
        # How-to keywords
        how_to_patterns = [
            r'\bhow\s+(do|can|to|should)\b',
            r'\bsteps?\s+to\b',
            r'\bguide\b',
            r'\btutorial\b',
            r'\bprocedure\b',
            r'\binstructions?\b',
            r'\bwalkthrough\b',
            r'\bsetup\b',
            r'\bconfigure\b',
            r'\binstall\b',
        ]
        
        for pattern in how_to_patterns:
            if re.search(pattern, query_lower):
                score += 0.25
        
        # Question words that suggest procedural queries
        if re.match(r'^(how|what)\s+(do|can|should|is\s+the\s+way)', query_lower):
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_tool_explanation(self, query_lower: str) -> float:
        """Detect tool/concept explanation patterns."""
        score = 0.0
        
        # Explanation patterns
        explanation_patterns = [
            r'\bwhat\s+is\b',
            r'\bwhat\s+does\b',
            r'\bexplain\b',
            r'\bdescribe\b',
            r'\bdefine\b',
            r'\btell\s+me\s+about\b',
            r'\bmeaning\s+of\b',
            r'\bpurpose\s+of\b',
            r'\bdifference\s+between\b',
            r'\bcompare\b',
        ]
        
        for pattern in explanation_patterns:
            if re.search(pattern, query_lower):
                score += 0.25
        
        # Check for technical terms or module names (basic heuristic)
        if re.search(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b', query_lower):
            score += 0.15
        
        return min(score, 1.0)
    
    def _detect_escalation(self, query_lower: str) -> float:
        """Detect escalation/ticket patterns."""
        score = 0.0
        
        escalation_patterns = [
            r'\bcontact\s+support\b',
            r'\bspeak\s+to\s+(someone|human|person)\b',
            r'\bfile\s+a\s+ticket\b',
            r'\bescalate\b',
            r'\bopen\s+a\s+case\b',
            r'\bneed\s+help\b',
            r'\bcan\'t\s+solve\b',
            r'\bnot\s+working\b',
        ]
        
        for pattern in escalation_patterns:
            if re.search(pattern, query_lower):
                score += 0.35
        
        return min(score, 1.0)
    
    def _detect_chitchat(self, query_lower: str) -> float:
        """Detect chitchat/conversational patterns."""
        score = 0.0
        
        chitchat_patterns = [
            r'^(hi|hello|hey|greetings)\b',
            r'\bhow\s+are\s+you\b',
            r'\bthank\s+you\b',
            r'\bthanks\b',
            r'\bbye\b',
            r'\bgoodbye\b',
            r'\bwhat\'s\s+up\b',
            r'\bgood\s+(morning|afternoon|evening)\b',
        ]
        
        for pattern in chitchat_patterns:
            if re.search(pattern, query_lower):
                score += 0.4
        
        # Very short queries might be chitchat
        if len(query_lower.split()) <= 3 and not any(kw in query_lower for kw in self.bug_keywords):
            score += 0.2
        
        return min(score, 1.0)
    
    def _classify_with_llm(self, user_query: str) -> IntentResult:
        """
        Use LLM for classification when rule-based approach is uncertain.
        
        This is a placeholder for LLM-based classification.
        In production, this would call the LLM provider with a classification prompt.
        """
        if not self.llm_provider:
            return IntentResult(
                intent=INTENT_OTHER,
                confidence=0.4,
                rationale="Ambiguous query, no LLM available"
            )
        
        # Construct classification prompt
        prompt = f"""Classify the following user query into one of these intents:
- how_to: User wants step-by-step instructions
- bug_resolution: User is reporting or troubleshooting an error
- tool_explanation: User wants to understand a concept or tool
- escalate_or_ticket: User wants human support
- chitchat: Casual conversation
- other: Doesn't fit above categories

Query: "{user_query}"

Respond with only: intent|confidence (e.g., "how_to|0.85")
"""
        
        try:
            response = self.llm_provider.generate(prompt, max_tokens=20)
            parts = response.strip().split("|")
            if len(parts) == 2:
                intent = parts[0].strip()
                confidence = float(parts[1].strip())
                if intent in ALL_INTENTS:
                    return IntentResult(
                        intent=intent,
                        confidence=confidence,
                        rationale=f"LLM classification (confidence: {confidence:.2f})"
                    )
        except Exception:
            pass
        
        return IntentResult(
            intent=INTENT_OTHER,
            confidence=0.4,
            rationale="LLM classification failed, using fallback"
        )
