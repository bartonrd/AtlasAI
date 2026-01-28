"""
Clarification generation module.

Generates precise clarifying questions when confidence or retrieval scores are low.
"""

from typing import Optional
from dataclasses import dataclass

from .intent_classifier import (
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
    INTENT_ESCALATE,
    INTENT_OTHER,
)


@dataclass
class ClarificationQuestion:
    """A clarifying question to ask the user."""
    question: str
    reason: str
    suggestions: list
    
    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "question": self.question,
            "reason": self.reason,
            "suggestions": self.suggestions,
        }


class ClarificationGenerator:
    """
    Generates clarifying questions when the system needs more information.
    
    Triggered by:
    - Low intent confidence (< threshold)
    - Low retrieval scores (no good matches)
    - Ambiguous queries
    """
    
    def generate_low_confidence_question(
        self,
        user_query: str,
        detected_intent: str,
        confidence: float,
    ) -> ClarificationQuestion:
        """
        Generate a clarifying question for low confidence intent.
        
        Args:
            user_query: Original user query
            detected_intent: The detected intent
            confidence: Confidence score
        
        Returns:
            ClarificationQuestion with targeted follow-up
        """
        if detected_intent == INTENT_HOW_TO:
            return ClarificationQuestion(
                question="I understand you're looking for instructions. Could you clarify what specific task you want to accomplish?",
                reason=f"Low confidence ({confidence:.2f}) on procedural intent",
                suggestions=[
                    "Are you trying to install, configure, or use a specific feature?",
                    "Which module or component are you working with?",
                ]
            )
        
        elif detected_intent == INTENT_BUG_RESOLUTION:
            return ClarificationQuestion(
                question="I see you might be encountering an issue. Can you provide more details about the error?",
                reason=f"Low confidence ({confidence:.2f}) on bug resolution intent",
                suggestions=[
                    "What error message or code are you seeing?",
                    "What were you trying to do when the issue occurred?",
                    "Which version of the software are you using?",
                ]
            )
        
        elif detected_intent == INTENT_TOOL_EXPLANATION:
            return ClarificationQuestion(
                question="I'd like to help explain that. Which specific concept or feature would you like to understand better?",
                reason=f"Low confidence ({confidence:.2f}) on explanation intent",
                suggestions=[
                    "Are you asking about a specific module, API, or component?",
                    "Would you like a general overview or details about a particular aspect?",
                ]
            )
        
        elif detected_intent == INTENT_ESCALATE:
            return ClarificationQuestion(
                question="I'm here to help! Could you describe what you need assistance with?",
                reason=f"Low confidence ({confidence:.2f}) on escalation intent",
                suggestions=[
                    "Do you need help with a technical issue?",
                    "Are you looking for documentation or guidance?",
                    "Would you like to file a support ticket?",
                ]
            )
        
        else:  # INTENT_OTHER or unknown
            return ClarificationQuestion(
                question="I'm not sure I understand your question. Could you rephrase it or provide more context?",
                reason=f"Unclear intent (detected: {detected_intent}, confidence: {confidence:.2f})",
                suggestions=[
                    "Are you looking for instructions on how to do something?",
                    "Are you troubleshooting an error or issue?",
                    "Would you like an explanation of a concept or feature?",
                ]
            )
    
    def generate_low_retrieval_question(
        self,
        user_query: str,
        detected_intent: str,
        top_score: float,
    ) -> ClarificationQuestion:
        """
        Generate a clarifying question when retrieval scores are too low.
        
        Args:
            user_query: Original user query
            detected_intent: The detected intent
            top_score: Highest retrieval score
        
        Returns:
            ClarificationQuestion to gather more specifics
        """
        if detected_intent == INTENT_HOW_TO:
            return ClarificationQuestion(
                question="I couldn't find specific instructions matching your request. Can you provide more details about what you're trying to do?",
                reason=f"Low retrieval score ({top_score:.2f}) for procedural query",
                suggestions=[
                    "Which specific feature or module are you working with?",
                    "What is the end goal you're trying to achieve?",
                    "Are there any specific constraints (version, platform, etc.)?",
                ]
            )
        
        elif detected_intent == INTENT_BUG_RESOLUTION:
            return ClarificationQuestion(
                question="I couldn't find matching error information. Can you provide more details about the problem?",
                reason=f"Low retrieval score ({top_score:.2f}) for error query",
                suggestions=[
                    "What is the exact error message or code?",
                    "When does this error occur (during startup, runtime, etc.)?",
                    "What steps can I take to reproduce the issue?",
                ]
            )
        
        elif detected_intent == INTENT_TOOL_EXPLANATION:
            return ClarificationQuestion(
                question="I couldn't find documentation matching your question. Could you specify what you'd like to learn about?",
                reason=f"Low retrieval score ({top_score:.2f}) for explanation query",
                suggestions=[
                    "What is the exact name of the module or feature?",
                    "Are you looking for conceptual understanding or usage examples?",
                    "Is there a related topic I could help explain instead?",
                ]
            )
        
        else:
            return ClarificationQuestion(
                question="I couldn't find relevant information in the documentation. Could you rephrase your question or ask something more specific?",
                reason=f"Low retrieval score ({top_score:.2f}) for query",
                suggestions=[
                    "Try using specific technical terms or module names",
                    "Break down your question into smaller, more specific parts",
                    "Check if your question relates to a documented feature",
                ]
            )
    
    def generate_no_results_question(
        self,
        user_query: str,
        detected_intent: str,
    ) -> ClarificationQuestion:
        """
        Generate a clarifying question when no results are found.
        
        Args:
            user_query: Original user query
            detected_intent: The detected intent
        
        Returns:
            ClarificationQuestion explaining the lack of results
        """
        return ClarificationQuestion(
            question="I couldn't find any documentation matching your query. Could you try rephrasing or asking about a related topic?",
            reason="No retrieval results found",
            suggestions=[
                "Verify the spelling of technical terms or module names",
                "Try asking a more general question first",
                "Check if this topic is covered in the available documentation",
            ]
        )
