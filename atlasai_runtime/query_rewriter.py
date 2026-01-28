"""
Query Rewriter Module

Rewrites user queries for better retrieval based on intent.
"""

import re
from typing import Dict, Any, List
from .intent_classifier import IntentType


class QueryRewriteResult:
    """Result of query rewriting."""
    
    def __init__(self, rewritten_query: str, entities: List[str], constraints: List[str]):
        """
        Initialize rewrite result.
        
        Args:
            rewritten_query: The rewritten query optimized for retrieval
            entities: Extracted entities (product names, versions, etc.)
            constraints: Extracted constraints (OS, version, etc.)
        """
        self.rewritten_query = rewritten_query
        self.entities = entities
        self.constraints = constraints
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "rewritten_query": self.rewritten_query,
            "entities": self.entities,
            "constraints": self.constraints
        }


class QueryRewriter:
    """
    Rewrites queries based on intent for improved retrieval.
    
    Behavior by intent:
    - how_to: Expand with action verbs, product/module names
    - bug_resolution: Extract error code, version, OS/log keywords
    - tool_explanation: Add synonyms and related module terms
    """
    
    # Common action verbs for how_to expansion
    ACTION_VERBS = [
        "configure", "setup", "install", "create", "deploy", "enable",
        "disable", "update", "modify", "change", "run", "execute"
    ]
    
    # Product/module detection patterns
    PRODUCT_PATTERNS = [
        r"\b[A-Z][A-Z0-9]+\b",  # ADMS, API, etc.
        r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b"  # CamelCase names
    ]
    
    # Version patterns
    VERSION_PATTERNS = [
        r"\bv?\d+\.\d+(?:\.\d+)?(?:\.\d+)?\b",  # 1.0, v2.1.3, etc.
        r"\bversion\s+\d+\b"
    ]
    
    # OS/platform patterns
    OS_PATTERNS = [
        r"\b(?:windows|linux|unix|macos|mac os|ubuntu|debian|centos|rhel)\b",
        r"\b(?:win\d+|win10|win11)\b"
    ]
    
    # Error code patterns
    ERROR_CODE_PATTERNS = [
        r"\b[A-Z]+[-_]?\d{3,}\b",  # ERR_123, HTTP_404
        r"\b\d{3,4}\b",  # 404, 5000
        r"\bHTTP\s*[45]\d{2}\b"  # HTTP 404, HTTP 500
    ]
    
    def __init__(self):
        """Initialize query rewriter."""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._product_regex = [re.compile(p) for p in self.PRODUCT_PATTERNS]
        self._version_regex = [re.compile(p, re.IGNORECASE) for p in self.VERSION_PATTERNS]
        self._os_regex = [re.compile(p, re.IGNORECASE) for p in self.OS_PATTERNS]
        self._error_code_regex = [re.compile(p) for p in self.ERROR_CODE_PATTERNS]
    
    def rewrite(self, user_query: str, intent: IntentType) -> QueryRewriteResult:
        """
        Rewrite query based on intent.
        
        Args:
            user_query: Original user query
            intent: Classified intent
            
        Returns:
            QueryRewriteResult with rewritten query, entities, and constraints
        """
        if not user_query or not user_query.strip():
            return QueryRewriteResult(
                rewritten_query="",
                entities=[],
                constraints=[]
            )
        
        # Extract entities and constraints
        entities = self._extract_entities(user_query)
        constraints = self._extract_constraints(user_query)
        
        # Rewrite based on intent
        if intent == IntentType.HOW_TO:
            rewritten = self._rewrite_how_to(user_query, entities)
        elif intent == IntentType.BUG_RESOLUTION:
            rewritten = self._rewrite_bug_resolution(user_query, entities, constraints)
        elif intent == IntentType.TOOL_EXPLANATION:
            rewritten = self._rewrite_tool_explanation(user_query, entities)
        else:
            # Default: minimal expansion
            rewritten = self._expand_abbreviations(user_query)
        
        return QueryRewriteResult(
            rewritten_query=rewritten,
            entities=entities,
            constraints=constraints
        )
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract product/module entities from text."""
        entities = []
        
        # Find products/modules
        for pattern in self._product_regex:
            matches = pattern.findall(text)
            entities.extend(matches)
        
        # Deduplicate while preserving order
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_constraints(self, text: str) -> List[str]:
        """Extract constraints like version, OS, error codes."""
        constraints = []
        
        # Version
        for pattern in self._version_regex:
            matches = pattern.findall(text)
            constraints.extend([f"version:{m}" for m in matches])
        
        # OS
        for pattern in self._os_regex:
            matches = pattern.findall(text)
            constraints.extend([f"os:{m.lower()}" for m in matches])
        
        # Error codes
        for pattern in self._error_code_regex:
            matches = pattern.findall(text)
            constraints.extend([f"error:{m}" for m in matches])
        
        # Deduplicate
        return list(dict.fromkeys(constraints))
    
    def _rewrite_how_to(self, query: str, entities: List[str]) -> str:
        """
        Rewrite how_to query with expanded action verbs and entities.
        
        Args:
            query: Original query
            entities: Extracted entities
            
        Returns:
            Rewritten query
        """
        query_lower = query.lower()
        
        # Find action verbs in query
        found_verbs = [v for v in self.ACTION_VERBS if v in query_lower]
        
        # Build expanded query
        parts = [query]
        
        # Add related action verbs
        if found_verbs:
            # Add synonyms for found verbs
            verb_synonyms = {
                "configure": ["setup", "set up"],
                "install": ["deploy", "setup"],
                "create": ["generate", "make"],
                "enable": ["activate", "turn on"],
                "disable": ["deactivate", "turn off"]
            }
            for verb in found_verbs:
                if verb in verb_synonyms:
                    parts.extend(verb_synonyms[verb])
        
        # Add entities to emphasize them
        if entities:
            entity_str = " ".join(entities)
            parts.append(f"procedure {entity_str}")
        else:
            parts.append("procedure steps guide")
        
        return " ".join(parts)
    
    def _rewrite_bug_resolution(self, query: str, entities: List[str], constraints: List[str]) -> str:
        """
        Rewrite bug resolution query with diagnostic keywords.
        
        Args:
            query: Original query
            entities: Extracted entities
            constraints: Extracted constraints
            
        Returns:
            Rewritten query
        """
        parts = [query]
        
        # Add diagnostic terms
        parts.append("troubleshoot diagnose resolution fix")
        
        # Emphasize error codes
        error_constraints = [c for c in constraints if c.startswith("error:")]
        if error_constraints:
            error_codes = [c.split(":", 1)[1] for c in error_constraints]
            parts.append(" ".join(error_codes))
        
        # Add version/OS context
        version_constraints = [c for c in constraints if c.startswith("version:")]
        os_constraints = [c for c in constraints if c.startswith("os:")]
        
        if version_constraints:
            versions = [c.split(":", 1)[1] for c in version_constraints]
            parts.append(" ".join(versions))
        
        if os_constraints:
            oses = [c.split(":", 1)[1] for c in os_constraints]
            parts.append(" ".join(oses))
        
        # Add product context
        if entities:
            parts.append(" ".join(entities))
        
        return " ".join(parts)
    
    def _rewrite_tool_explanation(self, query: str, entities: List[str]) -> str:
        """
        Rewrite tool explanation query with concept keywords.
        
        Args:
            query: Original query
            entities: Extracted entities
            
        Returns:
            Rewritten query
        """
        parts = [query]
        
        # Add conceptual terms
        parts.append("overview concept documentation feature capability")
        
        # Add module/component terms
        if entities:
            entity_str = " ".join(entities)
            parts.append(f"{entity_str} module component architecture")
        
        # Add common synonyms for "what is"
        query_lower = query.lower()
        if "what is" in query_lower or "what does" in query_lower:
            parts.append("definition purpose use case")
        
        return " ".join(parts)
    
    def _expand_abbreviations(self, query: str) -> str:
        """
        Expand common abbreviations.
        
        Args:
            query: Original query
            
        Returns:
            Query with abbreviations expanded
        """
        # Common abbreviations
        expansions = {
            r"\bapi\b": "API application programming interface",
            r"\bui\b": "UI user interface",
            r"\bdb\b": "database DB",
            r"\bos\b": "operating system OS",
            r"\bcli\b": "CLI command line interface",
            r"\bgui\b": "GUI graphical user interface"
        }
        
        expanded = query
        for abbr, expansion in expansions.items():
            expanded = re.sub(abbr, expansion, expanded, flags=re.IGNORECASE)
        
        return expanded
