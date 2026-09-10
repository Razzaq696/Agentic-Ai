"""State definition for the LangGraph Agentic Assistant."""

from typing import TypedDict, Optional, List, Dict, Any


class AgentState(TypedDict, total=False):
    """Clean LangGraph Agent State for Phase 1 & Phase 2 workflow.

    Fields:
        user_query: The raw input query provided by the user.
        query_analysis: The categorized intent of the query (LLM_REASONING, RAG_KNOWLEDGE, TOOL_API, ERROR).
        selected_action: The selected action route matching the analysis.
        retrieved_context: List of text chunks retrieved from the RAG knowledge base.
        tool_result: Structured result dictionary returned from tool execution.
        processed_result: Standardized internal result dictionary (success, source, result, error).
        validation_status: Status of the validation step (VALID, INVALID, ERROR).
        result: Intermediate result produced by the selected route node.
        final_response: The final generated response ready for delivery.
        error: Optional error description if any stage encounters an issue.
    """

    user_query: str
    query_analysis: str
    selected_action: str
    retrieved_context: Optional[List[str]]
    tool_result: Optional[Dict[str, Any]]
    processed_result: Optional[Dict[str, Any]]
    validation_status: Optional[str]
    result: Optional[str]
    final_response: Optional[str]
    error: Optional[str]
