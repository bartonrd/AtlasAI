"""
Query rewriting module.

Rewrites user queries based on intent to improve retrieval effectiveness.
Extracts entities and constraints for better search targeting.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .intent_classifier import (
    INTENT_HOW_TO,
    INTENT_BUG_RESOLUTION,
    INTENT_TOOL_EXPLANATION,
)


@dataclass
class RewrittenQuery:
    """Result of query rewriting."""
    rewritten_query: str
    entities: List[str]
    constraints: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "rewritten_query": self.rewritten_query,
            "entities": self.entities,
            "constraints": self.constraints,
        }


class QueryRewriter:
    """
    Rewrites user queries to improve retrieval effectiveness.
    
    Intent-aware behavior:
    - how_to: Expand verbs and add procedural keywords
    - bug_resolution: Extract error codes, versions, OS, log keywords
    - tool_explanation: Add synonyms and related module terms
    """
    
    def __init__(self):
        """Initialize the query rewriter."""
        # Verb expansions for how-to queries
        self.verb_expansions = {
            "install": ["setup", "configure", "deploy"],
            "configure": ["setup", "set up", "initialize"],
            "run": ["execute", "start", "launch"],
            "debug": ["troubleshoot", "diagnose", "fix"],
            "update": ["upgrade", "patch", "modify"],
            "remove": ["uninstall", "delete", "clean up"],
        }
        
        # Concept synonyms for tool explanation
        self.concept_synonyms = {
            "api": ["interface", "endpoint", "service"],
            "database": ["db", "datastore", "repository"],
            "cache": ["buffer", "storage", "memory"],
            "module": ["component", "library", "package"],
            "service": ["daemon", "process", "application"],
        }
    
    def rewrite(self, user_query: str, intent: str) -> RewrittenQuery:
        """
        Rewrite query based on intent.
        
        Args:
            user_query: Original user query
            intent: Detected intent
        
        Returns:
            RewrittenQuery with rewritten query, entities, and constraints
        """
        if not user_query or not user_query.strip():
            return RewrittenQuery(
                rewritten_query=user_query,
                entities=[],
                constraints=[]
            )
        
        if intent == INTENT_HOW_TO:
            return self._rewrite_how_to(user_query)
        elif intent == INTENT_BUG_RESOLUTION:
            return self._rewrite_bug_resolution(user_query)
        elif intent == INTENT_TOOL_EXPLANATION:
            return self._rewrite_tool_explanation(user_query)
        else:
            # For other intents, minimal rewriting
            return RewrittenQuery(
                rewritten_query=user_query,
                entities=self._extract_entities(user_query),
                constraints=[]
            )
    
    def _rewrite_how_to(self, query: str) -> RewrittenQuery:
        """
        Rewrite how-to queries by expanding verbs and adding procedural context.
        
        Example:
            "How to install the module" →
            "How to install setup configure the module step by step procedure"
        """
        rewritten = query
        query_lower = query.lower()
        
        # Expand key verbs
        for verb, expansions in self.verb_expansions.items():
            if verb in query_lower:
                # Add expansions but avoid duplication
                for expansion in expansions:
                    if expansion not in query_lower:
                        rewritten += f" {expansion}"
        
        # Add procedural keywords if not present
        procedural_keywords = ["step", "procedure", "guide", "instructions"]
        has_procedural = any(kw in query_lower for kw in procedural_keywords)
        if not has_procedural:
            rewritten += " step by step procedure"
        
        # Extract entities (nouns, module names, technical terms)
        entities = self._extract_entities(query)
        
        # Extract constraints (versions, platforms, etc.)
        constraints = self._extract_constraints(query)
        
        return RewrittenQuery(
            rewritten_query=rewritten.strip(),
            entities=entities,
            constraints=constraints
        )
    
    def _rewrite_bug_resolution(self, query: str) -> RewrittenQuery:
        """
        Rewrite bug resolution queries by extracting error details.
        
        Extracts:
        - Error codes (errno, exit codes)
        - Version numbers
        - OS/platform info
        - Stack trace keywords
        - Log keywords
        """
        rewritten = query
        
        # Extract error codes
        error_codes = self._extract_error_codes(query)
        
        # Extract version numbers
        versions = self._extract_versions(query)
        
        # Extract OS/platform
        os_platform = self._extract_os_platform(query)
        
        # Combine all extracted information
        entities = error_codes + versions + os_platform
        
        # Add diagnostic keywords if not present
        diagnostic_keywords = ["error", "exception", "diagnosis", "troubleshoot", "fix"]
        query_lower = query.lower()
        has_diagnostic = any(kw in query_lower for kw in diagnostic_keywords)
        if not has_diagnostic and error_codes:
            rewritten += " error diagnosis troubleshooting"
        
        # Constraints are error codes and versions
        constraints = error_codes + versions
        
        return RewrittenQuery(
            rewritten_query=rewritten.strip(),
            entities=entities,
            constraints=constraints
        )
    
    def _rewrite_tool_explanation(self, query: str) -> RewrittenQuery:
        """
        Rewrite tool explanation queries by adding synonyms.
        
        Example:
            "What is the API module" →
            "What is the API interface endpoint service module component"
        """
        rewritten = query
        query_lower = query.lower()
        
        # Add concept synonyms
        for concept, synonyms in self.concept_synonyms.items():
            if concept in query_lower:
                for synonym in synonyms:
                    if synonym not in query_lower:
                        rewritten += f" {synonym}"
        
        # Add explanation keywords
        explanation_keywords = ["explanation", "description", "definition", "purpose"]
        has_explanation = any(kw in query_lower for kw in explanation_keywords)
        if not has_explanation:
            rewritten += " explanation description"
        
        # Extract entities
        entities = self._extract_entities(query)
        
        return RewrittenQuery(
            rewritten_query=rewritten.strip(),
            entities=entities,
            constraints=[]
        )
    
    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract technical entities from query.
        
        Looks for:
        - CamelCase terms (e.g., ModelManager) - requires at least 2 capital letters
        - ALL_CAPS terms (e.g., ADMS)
        - Quoted terms
        - Technical abbreviations
        """
        entities = []
        
        # CamelCase terms - requires at least two capitalized segments
        camel_case = re.findall(r'\b[A-Z][a-z]+[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', query)
        entities.extend(camel_case)
        
        # ALL_CAPS terms (at least 2 chars)
        all_caps = re.findall(r'\b[A-Z]{2,}\b', query)
        entities.extend(all_caps)
        
        # Quoted terms
        quoted = re.findall(r'"([^"]+)"', query)
        entities.extend(quoted)
        quoted = re.findall(r"'([^']+)'", query)
        entities.extend(quoted)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_constraints(self, query: str) -> List[str]:
        """
        Extract constraints like versions, platforms, etc.
        """
        constraints = []
        
        # Version numbers
        versions = self._extract_versions(query)
        constraints.extend(versions)
        
        # OS/platform
        os_platform = self._extract_os_platform(query)
        constraints.extend(os_platform)
        
        return constraints
    
    def _extract_error_codes(self, query: str) -> List[str]:
        """Extract error codes from query."""
        error_codes = []
        
        # errno patterns
        errno_matches = re.findall(r'\berrno\s*[-:=]?\s*(\d+)', query, re.IGNORECASE)
        error_codes.extend([f"errno {code}" for code in errno_matches])
        
        # Exit code patterns
        exit_matches = re.findall(r'\bexit\s+code\s*[-:=]?\s*(\d+)', query, re.IGNORECASE)
        error_codes.extend([f"exit code {code}" for code in exit_matches])
        
        # Error: patterns
        error_name_matches = re.findall(r'\b(?:error|exception)\s*:\s*(\w+)', query, re.IGNORECASE)
        error_codes.extend(error_name_matches)
        
        return error_codes
    
    def _extract_versions(self, query: str) -> List[str]:
        """Extract version numbers from query with context."""
        versions = []
        
        # Version patterns with more context to avoid false positives
        version_patterns = [
            r'\bv(\d+\.\d+(?:\.\d+)?)\b',  # v1.2.3
            r'\bversion\s+(\d+(?:\.\d+)?)\b',  # version 1.2
            # Match version-like patterns followed by version-related words
            r'(\d+\.\d+(?:\.\d+)?)\s+(?:release|version|bug|error|issue)',
            # Or preceded by version-related words
            r'(?:running|using|with|version|v)\s+(\d+\.\d+(?:\.\d+)?)',
        ]
        
        for pattern in version_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                v = match if isinstance(match, str) else match
                versions.append(f"v{v}" if not v.startswith("v") else v)
        
        return versions
    
    def _extract_os_platform(self, query: str) -> List[str]:
        """Extract OS/platform information from query."""
        platforms = []
        
        # Known OS/platform keywords
        os_keywords = [
            "windows", "linux", "macos", "mac os", "unix",
            "ubuntu", "centos", "debian", "redhat", "fedora",
            "android", "ios"
        ]
        
        query_lower = query.lower()
        for os_kw in os_keywords:
            if os_kw in query_lower:
                platforms.append(os_kw)
        
        return platforms
