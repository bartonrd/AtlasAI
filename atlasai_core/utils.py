"""
Text formatting utilities for AtlasAI.

Handles response formatting and text cleaning.
"""

import re
import html


def strip_boilerplate(text: str) -> str:
    """
    Remove common footer/header boilerplate from text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text with boilerplate removed
    """
    if not text:
        return text
    
    patterns = [
        r"\bProprietary\s*-\s*See\s*Copyright\s*Page\b",
        r"\bContents\b",
        r"\bADMS\s*[\d\.]+\s*Modeling\s*Overview\s*and\s*Converter\s*User\s*Guide\b",
        r"\bDistribution\s*Model\s*Manager\s*User\s*Guide\b",
        r"^\s*Page\s*\d+\s*$",
    ]
    
    cleaned = text
    for pat in patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE)
    
    # Remove extra spaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned


# Bullet point regex pattern - matches various bullet characters
# • (U+2022), ▪ (U+25AA), ● (U+25CF), · (U+00B7), and ASCII -, –, *
BULLET_PATTERN = r"^[•\-–\*\u2022\u25AA\u25CF\u00B7]+\s*"


def to_bullets(text: str, min_items: int = 3, max_items: int = 10) -> str:
    """
    Convert free-form LLM answer into Markdown bullets.
    
    Args:
        text: Input text
        min_items: Minimum number of bullet points
        max_items: Maximum number of bullet points
        
    Returns:
        Formatted markdown bullet list
    """
    if not text:
        return ""
    
    # Remove boilerplate if the model echoed it
    text = strip_boilerplate(text)
    
    # Normalize whitespace for fallback splitter
    normalized = re.sub(r"\s+", " ", text).strip()
    
    # Primary: split on explicit bullet glyphs or line-start hyphens/newlines
    parts = re.split(
        r"(?:\n+|[•▪●·]\s+|(?:^|\s)-\s+)",
        text,
        flags=re.UNICODE
    )
    parts = [p.strip() for p in parts if p and p.strip()]
    
    # Fallback: sentence splitting if bullets weren't found
    if len(parts) <= 1:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
        parts = [p.strip() for p in parts if p and p.strip()]
    
    # Final cleanup: strip any leading bullets or numbering
    cleaned = []
    for p in parts:
        p = re.sub(BULLET_PATTERN, "", p)  # Remove bullet characters
        p = re.sub(r"^(?:\d+|[A-Za-z])[\.\)\:]\s*", "", p)  # numbering
        p = p.strip()
        if p:
            cleaned.append(p)
    
    # If still too few, try splitting by special chars
    if len(cleaned) < min_items:
        if "•" in text:
            cleaned = [
                re.sub(BULLET_PATTERN, "", p).strip() 
                for p in text.split("•") 
                if p.strip()
            ]
        elif ";" in text:
            cleaned = [
                re.sub(BULLET_PATTERN, "", p).strip() 
                for p in text.split(";") 
                if p.strip()
            ]
    
    # Emit Markdown bullets, capped
    bullets = [f"- {p}" for p in cleaned[:max_items]]
    return "\n".join(bullets)


def thinking_message(text: str) -> str:
    """
    Format a thinking/processing message with lighter color and italic styling.
    
    Args:
        text: The message to display (e.g., "Thinking...", "Loading documents...")
    
    Returns:
        Formatted HTML string with consistent styling
    """
    # Escape HTML to prevent XSS vulnerabilities
    escaped_text = html.escape(text)
    return f'<span style="color: #888888; font-style: italic;">{escaped_text}</span>'


def generate_chat_name(message: str, max_length: int = 30) -> str:
    """
    Generate a short chat name from a message.
    
    Args:
        message: Input message
        max_length: Maximum length of generated name
        
    Returns:
        Short chat name
    """
    if not message:
        return "New Chat"
    
    # Remove extra whitespace and newlines
    clean_msg = " ".join(message.split())
    
    # Truncate to max_length
    if len(clean_msg) <= max_length:
        return clean_msg
    
    # Truncate and add ellipsis
    ellipsis = "..."
    max_text_length = max_length - len(ellipsis)
    truncated = clean_msg[:max_text_length]
    
    # Try to break at word boundary
    space_pos = truncated.rfind(' ')
    if space_pos > 0:
        return truncated[:space_pos] + ellipsis
    else:
        return truncated + ellipsis
