"""Typed state definition for the Knowledge-Based Decision Agent graph."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Represents the state passing through the full LangGraph Decision Agent workflow.

    Fields:
        query: The raw user query.
        retrieval_query: The refined search keywords for the retriever.
        context: List of retrieved context chunks from Chroma.
        answer: The grounded LLM response, recommendation, or decision.
        sources: List of source filenames used for the response.
        error: Optional error message if any step encounters an issue.
    """
    query: str
    retrieval_query: str
    context: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    error: Optional[str]
