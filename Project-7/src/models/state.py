"""LangGraph state definition for the shopping agent workflow."""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class ShoppingState(TypedDict, total=False):
    """Core state tracked across the LangGraph shopping workflow nodes.

    Required fields:
        user_request: The raw user input query.
        product_category: Extracted product type or category.
        budget: Structured budget information dictionary.
        requirements: List of mandatory features/specs.
        preferences: List of nice-to-have user preferences.
        priorities: Key decision trade-offs or priorities.
        interpreted_requirements: Full structured requirements dictionary.
        next_action: Selected next step ('proceed_to_product_research', 'request_clarification', etc.).
        final_response: Formatted summary or clarification message presented to user.
    """

    user_request: str
    product_category: Optional[str]
    budget: Optional[Dict[str, Any]]
    requirements: List[str]
    preferences: List[str]
    priorities: List[str]
    intended_use: Optional[str]
    interpreted_requirements: Optional[Dict[str, Any]]
    missing_or_ambiguous_info: List[str]
    validation_errors: List[str]
    next_action: Optional[str]
    final_response: Optional[str]

    # Phase 2: RAG State Fields
    retrieved_products: List[Dict[str, Any]]
    rag_output: Optional[Dict[str, Any]]
    retrieval_sufficient: Optional[bool]
    research_needed: Optional[bool]
    retrieval_decision: Optional[str]
    retrieval_retry_count: int
    retrieval_query: Optional[str]

    # Phase 3: Web Search, Playwright & ReAct Fields
    search_query: Optional[str]
    search_results: List[Dict[str, Any]]
    candidate_urls: List[str]
    extracted_pages: List[Dict[str, Any]]
    research_result: Optional[Dict[str, Any]]
    react_iterations: int
    tool_history: List[Dict[str, Any]]
    research_status: Optional[str]

    # Phase 4: Multi-Agent Validation, Comparison & Decision Fields
    validated_products: List[Dict[str, Any]]
    validation_summary: Optional[Dict[str, Any]]
    comparison_result: Optional[Dict[str, Any]]
    product_scores: List[Dict[str, Any]]
    decision_result: Optional[Dict[str, Any]]
    final_decision: Optional[Dict[str, Any]]

    # Phase 5: Automation & Notification Dispatch Results
    integration_results: Optional[Dict[str, Any]]
