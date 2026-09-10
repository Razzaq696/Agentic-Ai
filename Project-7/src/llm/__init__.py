"""LLM integration module: factories, providers, and prompt templates."""

from src.llm.factory import get_llm, MockRequirementLLM, get_embeddings, DeterministicMockEmbeddings
from src.llm.prompts import (
    REQUIREMENT_ANALYSIS_SYSTEM_PROMPT,
    REQUIREMENT_ANALYSIS_USER_TEMPLATE,
)

__all__ = [
    "get_llm",
    "MockRequirementLLM",
    "get_embeddings",
    "DeterministicMockEmbeddings",
    "REQUIREMENT_ANALYSIS_SYSTEM_PROMPT",
    "REQUIREMENT_ANALYSIS_USER_TEMPLATE",
]
