"""
Formatting utilities for displaying content
"""

import html
import re


def format_answer_as_bullets(text: str, max_items: int = 10) -> str:
    """
    Convert a free-form LLM answer into Markdown bullets
    
    Args:
        text: Raw text from LLM
        max_items: Maximum number of bullet points
        
    Returns:
        Formatted markdown with bullet points
    """
    if not text:
        return ""
    
    # Normalize whitespace
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
        p = re.sub(r"^[•\-–\*\u2022\u25AA\u25CF\u00B7]+\s*", "", p)
        p = re.sub(r"^(?:\d+|[A-Za-z])[\.\)\:]\s*", "", p)
        p = p.strip()
        if p:
            cleaned.append(p)
    
    # If still too few, try splitting by '•' or semicolons
    if len(cleaned) < 3:
        if "•" in text:
            cleaned = [re.sub(r"^[•\-\*\s]+", "", p).strip() for p in text.split("•") if p.strip()]
        elif ";" in text:
            cleaned = [re.sub(r"^[•\-\*\s]+", "", p).strip() for p in text.split(";") if p.strip()]
    
    # Emit Markdown bullets, capped
    bullets = [f"- {p}" for p in cleaned[:max_items]]
    return "\n".join(bullets)


def thinking_message(text: str) -> str:
    """
    Format a thinking/processing message with styling
    
    Args:
        text: The message to display
        
    Returns:
        Formatted HTML string with consistent styling
    """
    # Escape HTML to prevent XSS vulnerabilities
    escaped_text = html.escape(text)
    return f'<span style="color: #888888; font-style: italic;">{escaped_text}</span>'
