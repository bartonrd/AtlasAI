"""
Answer synthesis module.

Generates grounded answers with citations based on retrieved documents.
Provides intent-specific formatting.
"""

import re
from typing import List, Dict, Any, Optional, Protocol
from dataclasses import dataclass

from .retriever import RetrievedDoc
from .intent_classifier import (
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
)


@dataclass
class SynthesizedAnswer:
    """A synthesized answer with citations."""
    answer: str
    citations: List[Dict[str, str]]
    intent_formatting_applied: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "answer": self.answer,
            "citations": self.citations,
            "intent_formatting_applied": self.intent_formatting_applied,
        }


class LLMProvider(Protocol):
    """Protocol for LLM providers used in synthesis."""
    
    def generate(self, prompt: str, max_tokens: int = 384) -> str:
        """Generate text from prompt."""
        ...


class AnswerSynthesizer:
    """
    Synthesizes answers from retrieved documents with citations.
    
    Behavior:
    - Only uses retrieved snippets (grounding)
    - Includes 2-3 citations with title+url
    - Intent-specific formatting:
      * how_to: numbered steps, prerequisites, time estimate, pitfalls
      * bug_resolution: signature, diagnostics, resolution, verification, rollback
      * tool_explanation: concept, analogy, use cases, related features, examples
    """
    
    def __init__(self, llm_provider: Optional[LLMProvider] = None):
        """
        Initialize the answer synthesizer.
        
        Args:
            llm_provider: Optional LLM provider for answer generation
        """
        self.llm_provider = llm_provider
    
    def synthesize(
        self,
        user_query: str,
        intent: str,
        retrieved_docs: List[RetrievedDoc],
        max_tokens: int = 384,
    ) -> SynthesizedAnswer:
        """
        Synthesize an answer from retrieved documents.
        
        Args:
            user_query: The user's query
            intent: The detected intent
            retrieved_docs: List of retrieved documents
            max_tokens: Maximum tokens in answer
        
        Returns:
            SynthesizedAnswer with formatted answer and citations
        """
        if not retrieved_docs:
            return SynthesizedAnswer(
                answer="I couldn't find relevant information to answer your question.",
                citations=[],
                intent_formatting_applied=False,
            )
        
        # Build context from retrieved docs
        context = self._build_context(retrieved_docs)
        
        # Generate answer using LLM if available
        if self.llm_provider:
            answer = self._generate_with_llm(
                user_query=user_query,
                intent=intent,
                context=context,
                max_tokens=max_tokens,
            )
        else:
            # Fallback: Extract relevant snippets
            answer = self._extract_snippets(user_query, retrieved_docs, intent)
        
        # Apply intent-specific formatting
        formatted_answer = self._apply_intent_formatting(answer, intent)
        
        # Extract citations (top 2-3 sources)
        citations = self._extract_citations(retrieved_docs[:3])
        
        return SynthesizedAnswer(
            answer=formatted_answer,
            citations=citations,
            intent_formatting_applied=True,
        )
    
    def _build_context(self, retrieved_docs: List[RetrievedDoc]) -> str:
        """Build context string from retrieved documents."""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_parts.append(f"[Source {i}] {doc.title}\n{doc.content}")
        return "\n\n".join(context_parts)
    
    def _generate_with_llm(
        self,
        user_query: str,
        intent: str,
        context: str,
        max_tokens: int,
    ) -> str:
        """Generate answer using LLM provider."""
        prompt = self._build_prompt(user_query, intent, context)
        
        try:
            answer = self.llm_provider.generate(prompt, max_tokens=max_tokens)
            return answer.strip()
        except Exception as e:
            # Fallback on LLM failure
            return f"Error generating answer: {str(e)}"
    
    def _build_prompt(self, user_query: str, intent: str, context: str) -> str:
        """Build intent-specific prompt for LLM."""
        base_instructions = """You are a helpful technical assistant. Answer the user's question using ONLY the provided context.
Include specific references to sources (e.g., [Source 1], [Source 2]).
Do not make up information not present in the context."""
        
        if intent == INTENT_HOW_TO:
            specific_instructions = """
Format your answer as:
1. Prerequisites (if any)
2. Step-by-step instructions (numbered)
3. Estimated time (if mentioned)
4. Common pitfalls to avoid (if mentioned)
"""
        elif intent == INTENT_BUG_RESOLUTION:
            specific_instructions = """
Format your answer as:
1. Error signature/description
2. Diagnostic steps
3. Resolution steps
4. Verification steps
5. Rollback procedure (if applicable)
"""
        elif intent == INTENT_TOOL_EXPLANATION:
            specific_instructions = """
Format your answer as:
1. Core concept explanation
2. Simple analogy or example
3. Common use cases
4. Related features or modules
"""
        else:
            specific_instructions = "\nProvide a clear, concise answer in bullet points."
        
        prompt = f"""{base_instructions}
{specific_instructions}

Context:
{context}

Question: {user_query}

Answer:"""
        
        return prompt
    
    def _extract_snippets(
        self,
        user_query: str,
        retrieved_docs: List[RetrievedDoc],
        intent: str,
    ) -> str:
        """
        Fallback: Extract relevant snippets when no LLM is available.
        
        Uses simple heuristics to combine document content.
        """
        snippets = []
        query_terms = set(user_query.lower().split())
        
        for i, doc in enumerate(retrieved_docs[:3], start=1):
            # Take first 200 chars or find most relevant sentence
            content = doc.content
            sentences = re.split(r'[.!?]\s+', content)
            
            # Score sentences by query term overlap
            best_sentence = ""
            best_score = 0
            for sentence in sentences:
                sentence_terms = set(sentence.lower().split())
                overlap = len(query_terms & sentence_terms)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sentence
            
            if best_sentence:
                snippets.append(f"[Source {i}] {best_sentence}")
            else:
                # Fallback to first sentence
                snippets.append(f"[Source {i}] {sentences[0] if sentences else content[:200]}")
        
        return "\n".join(snippets)
    
    def _apply_intent_formatting(self, answer: str, intent: str) -> str:
        """
        Apply intent-specific formatting to the answer.
        
        Ensures the answer follows the expected structure for each intent type.
        """
        # If answer is already well-formatted with bullets or numbers, return as-is
        if self._is_well_formatted(answer):
            return answer
        
        # Otherwise, apply basic formatting based on intent
        if intent == INTENT_HOW_TO:
            return self._format_how_to(answer)
        elif intent == INTENT_BUG_RESOLUTION:
            return self._format_bug_resolution(answer)
        elif intent == INTENT_TOOL_EXPLANATION:
            return self._format_tool_explanation(answer)
        else:
            return self._format_as_bullets(answer)
    
    def _is_well_formatted(self, text: str) -> bool:
        """Check if text is already formatted with bullets or numbers."""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return False
        
        # Check for bullet points or numbered lists
        formatted_lines = sum(
            1 for line in lines
            if re.match(r'^\s*[-*•]\s+', line) or re.match(r'^\s*\d+[\.)]\s+', line)
        )
        
        return formatted_lines >= len(lines) * 0.5  # At least 50% formatted
    
    def _format_how_to(self, answer: str) -> str:
        """Format answer for how-to queries."""
        # Try to identify steps
        sentences = re.split(r'[.!?]\s+', answer)
        
        formatted = "**Steps:**\n"
        for i, sentence in enumerate(sentences, start=1):
            if sentence.strip():
                formatted += f"{i}. {sentence.strip()}\n"
        
        return formatted
    
    def _format_bug_resolution(self, answer: str) -> str:
        """Format answer for bug resolution queries."""
        # Try to structure the answer
        lines = answer.strip().split('\n')
        
        formatted = "**Error Information:**\n"
        for line in lines:
            if line.strip():
                formatted += f"- {line.strip()}\n"
        
        return formatted
    
    def _format_tool_explanation(self, answer: str) -> str:
        """Format answer for tool explanation queries."""
        # Format as concept explanation
        lines = answer.strip().split('\n')
        
        formatted = "**Explanation:**\n"
        for line in lines:
            if line.strip():
                formatted += f"- {line.strip()}\n"
        
        return formatted
    
    def _format_as_bullets(self, answer: str) -> str:
        """Format answer as bullet points."""
        sentences = re.split(r'[.!?]\s+', answer)
        
        formatted = ""
        for sentence in sentences:
            if sentence.strip():
                formatted += f"- {sentence.strip()}\n"
        
        return formatted
    
    def _extract_citations(self, retrieved_docs: List[RetrievedDoc]) -> List[Dict[str, str]]:
        """
        Extract citations from top retrieved documents.
        
        Returns list of dicts with title and url.
        """
        citations = []
        for i, doc in enumerate(retrieved_docs, start=1):
            citations.append({
                "index": str(i),
                "title": doc.title,
                "url": doc.url,
            })
        return citations
