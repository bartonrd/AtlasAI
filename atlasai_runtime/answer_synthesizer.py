"""
Answer Synthesizer Module

Synthesizes answers with intent-specific formatting and citations.
"""

import re
from typing import List, Dict, Any, Optional
from .intent_classifier import IntentType
from .retriever import DocumentSnippet


class AnswerSynthesisResult:
    """Result of answer synthesis."""
    
    def __init__(
        self,
        answer: str,
        citations: List[Dict[str, str]],
        should_ask_clarification: bool = False,
        clarification_question: Optional[str] = None
    ):
        """
        Initialize synthesis result.
        
        Args:
            answer: Synthesized answer (empty if clarification needed)
            citations: List of citations with title and url
            should_ask_clarification: Whether to ask for clarification
            clarification_question: The clarification question if needed
        """
        self.answer = answer
        self.citations = citations
        self.should_ask_clarification = should_ask_clarification
        self.clarification_question = clarification_question
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "should_ask_clarification": self.should_ask_clarification,
            "clarification_question": self.clarification_question
        }


class AnswerSynthesizer:
    """
    Synthesizes answers with intent-specific formatting.
    
    Grounds answers only in retrieved snippets and formats based on intent.
    """
    
    def __init__(self, min_retrieval_score: float = 0.25, max_answer_tokens: int = 500):
        """
        Initialize answer synthesizer.
        
        Args:
            min_retrieval_score: Minimum score to use retrieved content
            max_answer_tokens: Maximum tokens in generated answer
        """
        self.min_retrieval_score = min_retrieval_score
        self.max_answer_tokens = max_answer_tokens
    
    def synthesize(
        self,
        query: str,
        intent: IntentType,
        snippets: List[DocumentSnippet],
        clarification_question: Optional[str] = None
    ) -> AnswerSynthesisResult:
        """
        Synthesize answer from retrieved snippets.
        
        Args:
            query: Original user query
            intent: Classified intent
            snippets: Retrieved document snippets
            clarification_question: Pre-generated clarification question if needed
            
        Returns:
            AnswerSynthesisResult with answer and citations
        """
        # Chitchat doesn't need retrieval
        if intent == IntentType.CHITCHAT:
            answer = self._synthesize_chitchat(query)
            return AnswerSynthesisResult(
                answer=answer,
                citations=[],
                should_ask_clarification=False,
                clarification_question=None
            )
        
        # Check if retrieval is weak
        if not snippets or (snippets and snippets[0].score < self.min_retrieval_score):
            # Ask for clarification instead of answering
            if not clarification_question:
                clarification_question = self._generate_generic_clarification(query, intent)
            
            return AnswerSynthesisResult(
                answer="",
                citations=[],
                should_ask_clarification=True,
                clarification_question=clarification_question
            )
        
        # Filter snippets by score threshold
        relevant_snippets = [s for s in snippets if s.score >= self.min_retrieval_score]
        
        if not relevant_snippets:
            return AnswerSynthesisResult(
                answer="",
                citations=[],
                should_ask_clarification=True,
                clarification_question=self._generate_generic_clarification(query, intent)
            )
        
        # Synthesize answer based on intent
        if intent == IntentType.HOW_TO:
            answer = self._synthesize_how_to(query, relevant_snippets)
        elif intent == IntentType.BUG_RESOLUTION:
            answer = self._synthesize_bug_resolution(query, relevant_snippets)
        elif intent == IntentType.TOOL_EXPLANATION:
            answer = self._synthesize_tool_explanation(query, relevant_snippets)
        else:
            answer = self._synthesize_generic(query, relevant_snippets)
        
        # Extract citations (top 2-3 sources)
        citations = self._extract_citations(relevant_snippets[:3])
        
        return AnswerSynthesisResult(
            answer=answer,
            citations=citations,
            should_ask_clarification=False,
            clarification_question=None
        )
    
    def _synthesize_how_to(self, query: str, snippets: List[DocumentSnippet]) -> str:
        """
        Synthesize how-to answer with numbered steps.
        
        Format:
        - Prerequisites
        - Numbered steps
        - Time estimate
        - Common pitfalls
        """
        content_parts = [s.content for s in snippets[:3]]
        combined = "\n\n".join(content_parts)
        
        # Extract steps (look for numbered lists or bullet points)
        steps = self._extract_steps(combined)
        
        # Build answer
        answer_parts = []
        
        # Prerequisites (if mentioned)
        prereqs = self._extract_prerequisites(combined)
        if prereqs:
            answer_parts.append(f"**Prerequisites:**\n{prereqs}")
        
        # Steps
        if steps:
            answer_parts.append("**Steps:**")
            for i, step in enumerate(steps, 1):
                answer_parts.append(f"{i}. {step}")
        else:
            # Fallback: use first 3 sentences
            sentences = self._extract_sentences(combined)[:3]
            answer_parts.append("**Procedure:**")
            for i, sent in enumerate(sentences, 1):
                answer_parts.append(f"{i}. {sent}")
        
        # Pitfalls (if mentioned)
        pitfalls = self._extract_pitfalls(combined)
        if pitfalls:
            answer_parts.append(f"\n**Common Pitfalls:**\n{pitfalls}")
        
        return "\n".join(answer_parts)
    
    def _synthesize_bug_resolution(self, query: str, snippets: List[DocumentSnippet]) -> str:
        """
        Synthesize bug resolution answer.
        
        Format:
        - Problem signature
        - Diagnostics
        - Resolution
        - Verification
        """
        content_parts = [s.content for s in snippets[:3]]
        combined = "\n\n".join(content_parts)
        
        answer_parts = []
        
        # Problem signature
        answer_parts.append(f"**Problem:**\n{self._extract_problem_description(query, combined)}")
        
        # Diagnostics
        diagnostics = self._extract_diagnostics(combined)
        if diagnostics:
            answer_parts.append(f"\n**Diagnostics:**\n{diagnostics}")
        
        # Resolution
        resolution = self._extract_resolution(combined)
        answer_parts.append(f"\n**Resolution:**\n{resolution}")
        
        # Verification
        verification = self._extract_verification(combined)
        if verification:
            answer_parts.append(f"\n**Verification:**\n{verification}")
        
        return "\n".join(answer_parts)
    
    def _synthesize_tool_explanation(self, query: str, snippets: List[DocumentSnippet]) -> str:
        """
        Synthesize tool explanation answer.
        
        Format:
        - Concept definition
        - Analogy/example
        - Use cases
        - Related features
        """
        content_parts = [s.content for s in snippets[:3]]
        combined = "\n\n".join(content_parts)
        
        answer_parts = []
        
        # Concept
        concept = self._extract_concept(combined)
        answer_parts.append(f"**Overview:**\n{concept}")
        
        # Use cases
        use_cases = self._extract_use_cases(combined)
        if use_cases:
            answer_parts.append(f"\n**Use Cases:**\n{use_cases}")
        
        # Related features (extract from metadata if available)
        related = self._extract_related_features(snippets)
        if related:
            answer_parts.append(f"\n**Related:**\n{related}")
        
        return "\n".join(answer_parts)
    
    def _synthesize_chitchat(self, query: str) -> str:
        """Synthesize brief chitchat response."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm here to help you with technical questions about the documentation. What can I assist you with today?"
        elif any(word in query_lower for word in ["thank", "thanks"]):
            return "You're welcome! Let me know if you have any other questions."
        elif any(word in query_lower for word in ["bye", "goodbye"]):
            return "Goodbye! Feel free to return if you have more questions."
        else:
            return "I'm a technical documentation assistant. How can I help you today?"
    
    def _synthesize_generic(self, query: str, snippets: List[DocumentSnippet]) -> str:
        """Synthesize generic answer from snippets."""
        # Extract key sentences
        content_parts = [s.content for s in snippets[:3]]
        combined = "\n\n".join(content_parts)
        
        sentences = self._extract_sentences(combined)[:5]
        
        return "\n\n".join(f"- {s}" for s in sentences)
    
    def _extract_citations(self, snippets: List[DocumentSnippet]) -> List[Dict[str, str]]:
        """Extract citations from snippets."""
        citations = []
        seen_titles = set()
        
        for snippet in snippets:
            if snippet.title not in seen_titles:
                citations.append({
                    "title": snippet.title,
                    "url": snippet.url
                })
                seen_titles.add(snippet.title)
        
        return citations
    
    def _extract_steps(self, text: str) -> List[str]:
        """Extract numbered steps or bullet points."""
        # Look for numbered lists
        pattern = r'(?:^|\n)\s*\d+[\.\)]\s*([^\n]+)'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        if matches:
            return [m.strip() for m in matches][:7]  # Max 7 steps
        
        # Fallback: bullet points
        pattern = r'(?:^|\n)\s*[-•*]\s*([^\n]+)'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        if matches:
            return [m.strip() for m in matches][:7]
        
        return []
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 20][:5]
    
    def _extract_prerequisites(self, text: str) -> str:
        """Extract prerequisites section."""
        pattern = r'(?:prerequisite|requirement|before you begin)[s]?:?\s*([^\n.]+(?:\.[^\n.]+)?)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_pitfalls(self, text: str) -> str:
        """Extract common pitfalls or warnings."""
        patterns = [
            r'(?:pitfall|warning|caution|note)[s]?:?\s*([^\n.]+(?:\.[^\n.]+)?)',
            r'(?:avoid|do not|don\'t)\s+([^\n.]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def _extract_problem_description(self, query: str, text: str) -> str:
        """Extract problem description."""
        # Use first sentence from content or query
        sentences = self._extract_sentences(text)
        if sentences:
            return sentences[0]
        return query
    
    def _extract_diagnostics(self, text: str) -> str:
        """Extract diagnostic information."""
        pattern = r'(?:diagnos|check|verify|inspect)[^\n.]+(?:\.[^\n.]+)?'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0).strip() if match else ""
    
    def _extract_resolution(self, text: str) -> str:
        """Extract resolution steps."""
        # Look for solution/fix/resolution keywords
        patterns = [
            r'(?:solution|resolution|fix)[:\s]+([^\n]+(?:\n[^\n]+){0,2})',
            r'(?:to resolve|to fix)[,\s]+([^\n]+(?:\n[^\n]+){0,2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fallback: first 2 sentences
        sentences = self._extract_sentences(text)[:2]
        return " ".join(sentences) if sentences else "See documentation for resolution steps."
    
    def _extract_verification(self, text: str) -> str:
        """Extract verification steps."""
        pattern = r'(?:verif|test|confirm)[^\n.]+(?:\.[^\n.]+)?'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(0).strip() if match else ""
    
    def _extract_concept(self, text: str) -> str:
        """Extract concept definition."""
        # First 2-3 sentences
        sentences = self._extract_sentences(text)[:3]
        return " ".join(sentences) if sentences else "No description available."
    
    def _extract_use_cases(self, text: str) -> str:
        """Extract use cases."""
        pattern = r'(?:use case|used for|useful when)[s]?[:\s]+([^\n]+(?:\n[^\n]+)?)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip() if match else ""
    
    def _extract_related_features(self, snippets: List[DocumentSnippet]) -> str:
        """Extract related features from metadata."""
        # Get unique document titles (excluding current)
        titles = list(set(s.title for s in snippets))
        if len(titles) > 1:
            return ", ".join(titles[:3])
        return ""
    
    def _generate_generic_clarification(self, query: str, intent: IntentType) -> str:
        """Generate generic clarification question."""
        if intent == IntentType.HOW_TO:
            return "Could you provide more details about which specific component or feature you want to work with?"
        elif intent == IntentType.BUG_RESOLUTION:
            return "Could you provide the specific error message or error code you're encountering?"
        elif intent == IntentType.TOOL_EXPLANATION:
            return "Which specific tool or feature would you like me to explain?"
        else:
            return "Could you provide more details about what you're looking for?"
