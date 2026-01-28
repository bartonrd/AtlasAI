"""
Clarifying Question Generator Module

Generates minimal, targeted clarifying questions based on missing information.
"""

from typing import Optional, List, Dict, Any
from .intent_classifier import IntentType


class ClarifyingQuestionGenerator:
    """
    Generates clarifying questions when confidence is low or information is missing.
    
    Uses intent-specific "slots" to determine what information is needed.
    """
    
    # Required slots per intent
    INTENT_SLOTS = {
        IntentType.HOW_TO: [
            ("component", "Which component or feature?"),
            ("action", "What action do you want to perform?"),
            ("environment", "What environment (GUI or CLI)?")
        ],
        IntentType.BUG_RESOLUTION: [
            ("error_code", "What is the specific error message or code?"),
            ("version", "Which version are you using?"),
            ("os", "What operating system?"),
            ("steps", "What were you doing when the error occurred?")
        ],
        IntentType.TOOL_EXPLANATION: [
            ("tool_name", "Which specific tool or feature?"),
            ("context", "What aspect interests you (overview, configuration, usage)?")
        ],
        IntentType.ESCALATE_OR_TICKET: [
            ("issue_type", "What type of issue are you experiencing?"),
            ("urgency", "How urgent is this? (critical/high/normal)")
        ]
    }
    
    def __init__(self):
        """Initialize clarifying question generator."""
        pass
    
    def generate(
        self,
        query: str,
        intent: IntentType,
        confidence: float,
        entities: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Generate a clarifying question if needed.
        
        Args:
            query: Original user query
            intent: Classified intent
            confidence: Classification confidence
            entities: Extracted entities
            constraints: Extracted constraints
            
        Returns:
            Clarifying question or None if not needed
        """
        entities = entities or []
        constraints = constraints or []
        
        # Chitchat never needs clarification
        if intent == IntentType.CHITCHAT:
            return None
        
        # Check if we need clarification
        if confidence >= 0.55 and entities:
            # Sufficient confidence and some entities found
            return None
        
        # For very low confidence, always ask for clarification
        if confidence < 0.4:
            return "Could you provide more details about what you're looking for?"
        
        # Determine missing slots
        missing_slots = self._identify_missing_slots(
            query, intent, entities, constraints
        )
        
        if not missing_slots:
            # All slots filled but confidence is low
            if confidence < 0.55:
                return "Could you rephrase your question to be more specific?"
            return None
        
        # Generate question for first missing slot
        return self._format_question(missing_slots[0], intent)
    
    def _identify_missing_slots(
        self,
        query: str,
        intent: IntentType,
        entities: List[str],
        constraints: List[str]
    ) -> List[str]:
        """
        Identify missing required slots.
        
        Returns:
            List of missing slot names
        """
        missing = []
        
        if intent not in self.INTENT_SLOTS:
            return missing
        
        slots = self.INTENT_SLOTS[intent]
        query_lower = query.lower()
        
        for slot_name, _ in slots:
            if not self._is_slot_filled(slot_name, query_lower, entities, constraints):
                missing.append(slot_name)
        
        return missing
    
    def _is_slot_filled(
        self,
        slot_name: str,
        query_lower: str,
        entities: List[str],
        constraints: List[str]
    ) -> bool:
        """Check if a slot is filled based on query and extracted info."""
        
        if slot_name == "component" or slot_name == "tool_name":
            # Check if entities were extracted
            return len(entities) > 0
        
        elif slot_name == "action":
            # Check for action verbs
            action_verbs = [
                "configure", "install", "setup", "create", "deploy",
                "enable", "disable", "update", "delete", "run"
            ]
            return any(verb in query_lower for verb in action_verbs)
        
        elif slot_name == "error_code":
            # Check for error patterns in query or constraints
            error_patterns = ["error", "exception", "failed", "crash"]
            has_error_mention = any(p in query_lower for p in error_patterns)
            has_error_constraint = any(c.startswith("error:") for c in constraints)
            return has_error_mention or has_error_constraint
        
        elif slot_name == "version":
            # Check if version is in constraints
            return any(c.startswith("version:") for c in constraints)
        
        elif slot_name == "os":
            # Check if OS is in constraints
            return any(c.startswith("os:") for c in constraints)
        
        elif slot_name == "environment":
            # Check for GUI/CLI mentions
            return "gui" in query_lower or "cli" in query_lower or "command" in query_lower
        
        elif slot_name == "steps":
            # Check if query describes steps or context
            return len(query_lower.split()) > 15  # Longer queries likely have context
        
        elif slot_name == "context":
            # Check if specific aspect is mentioned
            aspects = ["overview", "configure", "use", "example", "setup"]
            return any(aspect in query_lower for aspect in aspects)
        
        elif slot_name == "issue_type":
            # Always consider filled for escalation
            return True
        
        elif slot_name == "urgency":
            # Check for urgency keywords
            urgency_words = ["urgent", "critical", "asap", "immediately", "emergency"]
            return any(word in query_lower for word in urgency_words)
        
        # Unknown slot type - assume filled
        return True
    
    def _format_question(self, slot_name: str, intent: IntentType) -> str:
        """
        Format clarifying question for a missing slot.
        
        Args:
            slot_name: Name of missing slot
            intent: User intent
            
        Returns:
            Formatted clarifying question
        """
        if intent not in self.INTENT_SLOTS:
            return "Could you provide more details about your question?"
        
        slots = dict(self.INTENT_SLOTS[intent])
        
        if slot_name in slots:
            return slots[slot_name]
        
        return "Could you provide more details?"
