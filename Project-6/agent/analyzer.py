"""Query analysis and intent classification module."""

import re

# Defined Action Constants
ACTION_LLM_REASONING = "LLM_REASONING"
ACTION_RAG_KNOWLEDGE = "RAG_KNOWLEDGE"
ACTION_TOOL_API = "TOOL_API"
ACTION_ERROR = "ERROR"

# Keywords & Regex Patterns for classification
TOOL_PATTERNS = [
    r"\bcalculate\b",
    r"\bmath\b",
    r"\bconvert\b",
    r"\d+\s*[\+\-\*\/×÷\^]\s*\d+",  # e.g., 125 × 8, 25 * 8, 100 / 4
    r"\bcompute\b",
    r"\bweather\s+in\b",
    r"\bcurrent\s+time\b",
    r"\bstock\s+price\b",
]

RAG_PATTERNS = [
    r"\bknowledge\s+base\b",
    r"\bproject\s+doc(s|ument|uments)?\b",
    r"\binternal\s+doc(s|ument|uments)?\b",
    r"\bcompany\s+polic(y|ies)\b",
    r"\bsearch\s+documents\b",
    r"\bsearch\s+knowledge\b",
    r"\bretrieve\s+from\b",
    r"\bproject\s+wiki\b",
]


def classify_query(query: str) -> str:
    """Inspects the incoming query and categorizes it into an action type.

    Args:
        query: The raw string query from the user.

    Returns:
        One of ACTION_LLM_REASONING, ACTION_RAG_KNOWLEDGE, ACTION_TOOL_API, or ACTION_ERROR.
    """
    if not query or not query.strip():
        return ACTION_ERROR

    cleaned = query.strip().lower()

    # 1. Check for Tool / API patterns (Calculations, external API requests)
    for pattern in TOOL_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return ACTION_TOOL_API

    # 2. Check for RAG / Knowledge Base patterns
    for pattern in RAG_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return ACTION_RAG_KNOWLEDGE

    # 3. Default to General LLM Reasoning / Explanation
    return ACTION_LLM_REASONING
