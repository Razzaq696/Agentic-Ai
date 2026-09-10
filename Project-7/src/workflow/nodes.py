"""LangGraph node implementations for Phase 1 Requirement Analysis workflow."""

from typing import Any, Dict, List, Optional, Tuple
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.guardrails.validators import (
    validate_input_request,
    check_requirement_completeness,
)
from src.models.schemas import NextAction, RequirementAnalysisOutput, BudgetInfo, Product
from src.models.state import ShoppingState
from src.utils.logger import logger


def analyze_shopping_request(
    state: ShoppingState,
    agent: Optional[RequirementAnalysisAgent] = None,
) -> Dict[str, Any]:
    """Node 1: Validates raw input and invokes requirement analysis agent.

    Args:
        state: The current LangGraph shopping state.
        agent: Optional RequirementAnalysisAgent instance (useful for dependency injection).

    Returns:
        State update dictionary.
    """
    user_request = state.get("user_request", "")
    logger.info(f"[Node: analyze_shopping_request] Input: {user_request!r}")

    # Step 1: Input guardrail check
    is_valid, error_msg = validate_input_request(user_request)
    if not is_valid:
        logger.warning(f"[Node: analyze_shopping_request] Input rejected: {error_msg}")
        return {
            "validation_errors": [error_msg],
            "next_action": NextAction.INVALID_INPUT.value,
            "final_response": f"Input validation failed: {error_msg}",
            "product_category": None,
            "budget": BudgetInfo().model_dump(),
            "requirements": [],
            "preferences": [],
            "priorities": [],
            "intended_use": None,
            "interpreted_requirements": None,
            "missing_or_ambiguous_info": [error_msg],
        }

    # Step 2: Invoke agent
    active_agent = agent or RequirementAnalysisAgent()
    output: RequirementAnalysisOutput = active_agent.analyze(user_request)

    return {
        "product_category": output.product_category,
        "budget": output.budget.model_dump(),
        "requirements": output.required_features,
        "preferences": output.preferences,
        "priorities": output.priorities,
        "intended_use": output.intended_use,
        "interpreted_requirements": output.model_dump(),
        "missing_or_ambiguous_info": list(output.ambiguities_or_missing_info),
        "validation_errors": [],
    }


def validate_requirements(state: ShoppingState) -> Dict[str, Any]:
    """Node 2: Sanity-checks interpreted requirements and evaluates completeness.

    Rules:
    - If already flagged as INVALID_INPUT, passes through.
    - Handles missing budget gracefully (sets note, does not block).
    - Checks for missing category or critical ambiguities.

    Args:
        state: Current LangGraph shopping state.

    Returns:
        State update dictionary.
    """
    if state.get("next_action") == NextAction.INVALID_INPUT.value:
        logger.info("[Node: validate_requirements] Skipping validation for invalid input.")
        return {}

    interpreted_dict = state.get("interpreted_requirements")
    if not interpreted_dict:
        logger.warning("[Node: validate_requirements] Missing interpreted_requirements in state.")
        return {
            "validation_errors": ["No interpreted requirements found to validate."],
            "missing_or_ambiguous_info": ["Requirements could not be extracted."],
        }

    # Reconstruct Pydantic schema for validation
    output = RequirementAnalysisOutput.model_validate(interpreted_dict)
    is_sufficient, issues = check_requirement_completeness(output)

    existing_missing = list(state.get("missing_or_ambiguous_info") or [])
    combined_issues = list(dict.fromkeys(existing_missing + issues))

    # Graceful handling of missing budget
    budget_dict = state.get("budget") or {}
    is_budget_specified = budget_dict.get("is_specified", False)
    validation_notes = list(state.get("validation_errors") or [])

    if not is_budget_specified:
        logger.info("[Node: validate_requirements] Budget is not specified. Treating as flexible.")
        # We do not add this as a blocking issue, just a note if needed

    logger.info(
        f"[Node: validate_requirements] Sufficiency: {is_sufficient}, Issues detected: {combined_issues}"
    )

    return {
        "missing_or_ambiguous_info": combined_issues,
        "validation_errors": validation_notes,
    }


def determine_next_action(state: ShoppingState) -> Dict[str, Any]:
    """Node 3: Decides next action (proceed to research vs. request clarification).

    Decision logic:
    - If input was rejected: Action is INVALID_INPUT.
    - If critical missing info or ambiguous category: Action is REQUEST_CLARIFICATION.
    - If enough information exists: Action is PROCEED_TO_PRODUCT_RESEARCH.

    Args:
        state: Current LangGraph shopping state.

    Returns:
        State update dictionary.
    """
    logger.info("[Node: determine_next_action] Evaluating state for next action...")

    if state.get("next_action") == NextAction.INVALID_INPUT.value:
        return {
            "next_action": NextAction.INVALID_INPUT.value,
            "final_response": state.get("final_response") or "Request could not be processed.",
        }

    category = state.get("product_category")
    missing_issues = state.get("missing_or_ambiguous_info") or []

    # Category missing or critical ambiguities
    if not category or any("category" in issue.lower() for issue in missing_issues):
        clarification_msg = (
            "I could not clearly identify the product you are looking to purchase. "
            "Could you please specify the exact product or category (e.g., laptop, headphones, mechanical keyboard)?"
        )
        logger.info("[Node: determine_next_action] Decided: REQUEST_CLARIFICATION")
        return {
            "next_action": NextAction.REQUEST_CLARIFICATION.value,
            "final_response": clarification_msg,
        }

    # If category is known and requirements are captured, even if some optional ambiguities exist,
    # we have enough core information to begin product research.
    # If missing_issues contains blocking criteria (like completely unclear request):
    if len(missing_issues) > 0 and not category:
        issues_list = "\n - ".join(missing_issues)
        clarification_msg = (
            f"To help you find the best options, please provide a bit more detail on:\n - {issues_list}"
        )
        logger.info("[Node: determine_next_action] Decided: REQUEST_CLARIFICATION")
        return {
            "next_action": NextAction.REQUEST_CLARIFICATION.value,
            "final_response": clarification_msg,
        }

    # Sufficient information to proceed
    budget_info = state.get("budget") or {}
    budget_str = "Flexible / Market pricing"
    if budget_info.get("is_specified"):
        min_amt = budget_info.get("min_amount")
        max_amt = budget_info.get("max_amount")
        curr = budget_info.get("currency", "$")
        curr_sym = "$" if curr.upper() in ("USD", "$") else f"{curr} "
        
        def fmt_num(val):
            return f"{int(val):,}" if val == int(val) else f"{val:,.2f}"

        if min_amt is not None and max_amt is not None:
            budget_str = f"{curr_sym}{fmt_num(min_amt)} - {curr_sym}{fmt_num(max_amt)}"
        elif max_amt is not None:
            budget_str = f"Up to {curr_sym}{fmt_num(max_amt)}"
        elif min_amt is not None:
            budget_str = f"Starting at {curr_sym}{fmt_num(min_amt)}"

    reqs_str = ", ".join(state.get("requirements") or []) or "Standard specifications"
    prefs_str = ", ".join(state.get("preferences") or []) or "None specified"
    priorities_str = ", ".join(state.get("priorities") or []) or "Balanced value"

    response_summary = (
        f"Understood! Here are your analyzed requirements:\n"
        f"• Product Category: {category}\n"
        f"• Budget: {budget_str}\n"
        f"• Key Requirements: {reqs_str}\n"
        f"• Preferences: {prefs_str}\n"
        f"• Priorities: {priorities_str}\n\n"
        f"Ready to proceed to product research."
    )

    logger.info("[Node: determine_next_action] Decided: PROCEED_TO_PRODUCT_RESEARCH")
    return {
        "next_action": NextAction.PROCEED_TO_PRODUCT_RESEARCH.value,
        "final_response": response_summary,
    }


# =====================================================================
# PHASE 2: AGENTIC RAG & RETRIEVAL NODES
# =====================================================================

from src.models.schemas import RAGDecision, StructuredRAGOutput
from src.rag.retriever import ProductRetriever
from src.rag.evaluator import AgenticRAGEvaluator
from src.guardrails.validators import validate_structured_rag_output


def retrieve_product_knowledge(
    state: ShoppingState,
    retriever: Optional[ProductRetriever] = None,
) -> Dict[str, Any]:
    """Node 4: Retrieves matching products from Chroma vector store based on requirements.

    Args:
        state: The current LangGraph shopping state.
        retriever: Optional injected ProductRetriever instance.

    Returns:
        State update with retrieved products and query.
    """
    logger.info("[Node: retrieve_product_knowledge] Executing product knowledge retrieval")

    # If already stopped by earlier node, pass through
    if state.get("next_action") in (NextAction.INVALID_INPUT.value, NextAction.REQUEST_CLARIFICATION.value):
        return {}

    active_retriever = retriever or ProductRetriever()
    try:
        retrieved_products = active_retriever.retrieve_for_state(state)
        # Store as serializable dicts
        products_data = [p.model_dump() for p in retrieved_products]
        query_used = active_retriever.construct_query(
            category=state.get("product_category"),
            requirements=state.get("requirements") or [],
            preferences=state.get("preferences") or [],
            intended_use=state.get("intended_use"),
        )
        logger.info(
            f"[Node: retrieve_product_knowledge] Retrieved {len(products_data)} candidate products"
        )
        return {
            "retrieved_products": products_data,
            "retrieval_query": query_used,
        }
    except Exception as e:
        logger.error(f"[Node: retrieve_product_knowledge] Vector store retrieval error: {e}", exc_info=True)
        return {
            "retrieved_products": [],
            "retrieval_query": "",
        }


def evaluate_retrieval(
    state: ShoppingState,
    evaluator: Optional[AgenticRAGEvaluator] = None,
) -> Dict[str, Any]:
    """Node 5: Evaluates retrieved product relevance, sufficiency, and determines next research action.

    Args:
        state: The current LangGraph shopping state.
        evaluator: Optional injected AgenticRAGEvaluator instance.

    Returns:
        State update with StructuredRAGOutput, decision, and augmented response.
    """
    logger.info("[Node: evaluate_retrieval] Evaluating product retrieval sufficiency")

    # If earlier guardrail stopped execution, pass through
    if state.get("next_action") in (NextAction.INVALID_INPUT.value, NextAction.REQUEST_CLARIFICATION.value):
        return {}

    active_evaluator = evaluator or AgenticRAGEvaluator()
    rag_output: StructuredRAGOutput = active_evaluator.evaluate_for_state(state)
    rag_output = validate_structured_rag_output(rag_output)

    current_retries = state.get("retrieval_retry_count", 0)
    new_retry_count = current_retries
    if rag_output.decision == RAGDecision.REFINE_QUERY:
        new_retry_count += 1

    # Enhance final_response with RAG knowledge facts and source attribution
    base_response = state.get("final_response") or ""
    rag_summary_lines = []

    if rag_output.retrieval_sufficient and rag_output.retrieved_products:
        rag_summary_lines.append("\n\n[Knowledge Base Matches Found]:")
        for p in rag_output.retrieved_products:
            rag_summary_lines.append(
                f"• {p.product_name} ({p.brand}) — ${p.price:,.2f} {p.currency}\n"
                f"  Specs/Features: {', '.join(p.features) if p.features else 'Standard specifications'}\n"
                f"  Match: {p.match_notes or 'Matches search criteria'}\n"
                f"  Source: {p.source}"
            )
        rag_summary_lines.append(f"\nRetrieval Assessment: {rag_output.reasoning}")
    elif rag_output.research_needed:
        rag_summary_lines.append(
            f"\n\n[Knowledge Base Status]: {rag_output.reasoning}\n"
            f"External research will be conducted in the next phase."
        )

    augmented_response = base_response + "".join(rag_summary_lines)

    logger.info(
        f"[Node: evaluate_retrieval] Decision: {rag_output.decision.value} "
        f"(Sufficient: {rag_output.retrieval_sufficient}, Confidence: {rag_output.retrieval_confidence})"
    )

    return {
        "rag_output": rag_output.model_dump(),
        "retrieval_sufficient": rag_output.retrieval_sufficient,
        "research_needed": rag_output.research_needed,
        "retrieval_decision": rag_output.decision.value,
        "retrieval_retry_count": new_retry_count,
        "final_response": augmented_response,
    }


# =====================================================================
# PHASE 3: WEB SEARCH, PLAYWRIGHT & ReAct RESEARCH NODES
# =====================================================================

from src.models.schemas import Product, ProductResearchResult, SearchResultItem
from src.tools.search_tool import WebSearchTool
from src.tools.browser_tool import ProductPageExtractorTool
from src.guardrails.validators import (
    validate_url,
    validate_search_results,
    validate_product_research_output,
    deduplicate_products,
)


def decide_research(
    state: ShoppingState,
    search_tool: Optional[WebSearchTool] = None,
) -> Dict[str, Any]:
    """Node 6: Determines whether external web research is needed or skipped."""
    logger.info("[Node: decide_research] Evaluating if external web research is required")

    # If already stopped by earlier node, pass through
    if state.get("next_action") in (NextAction.INVALID_INPUT.value, NextAction.REQUEST_CLARIFICATION.value):
        return {}

    # Rule: If Phase 2 determined retrieval is already sufficient, skip web research!
    if state.get("retrieval_sufficient") is True and not state.get("research_needed"):
        logger.info("[Node: decide_research] Local Chroma retrieval is already sufficient. Skipping external web research.")
        return {
            "research_status": "skipped",
            "react_iterations": 0,
        }

    active_search = search_tool or WebSearchTool()
    budget_dict = state.get("budget") or {}
    b_info = BudgetInfo(**budget_dict) if budget_dict else None

    query = active_search.synthesize_query(
        category=state.get("product_category"),
        requirements=state.get("requirements") or [],
        budget=b_info,
        priorities=state.get("priorities") or [],
        intended_use=state.get("intended_use"),
    )

    logger.info(f"[Node: decide_research] Initiating external research with query: {query!r}")
    return {
        "research_status": "in_progress",
        "search_query": query,
        "react_iterations": 0,
        "tool_history": [],
    }


def execute_web_search(
    state: ShoppingState,
    search_tool: Optional[WebSearchTool] = None,
) -> Dict[str, Any]:
    """Node 7: Executes web search tool and stores structured search results."""
    logger.info("[Node: execute_web_search] Executing web search")

    if state.get("research_status") == "skipped":
        return {}

    active_search = search_tool or WebSearchTool()
    query = state.get("search_query") or state.get("user_request") or ""

    search_output = active_search.search(query)
    valid_items = validate_search_results(search_output.results)
    results_dicts = [item.model_dump() for item in valid_items]

    history = list(state.get("tool_history") or [])
    history.append({
        "action": "web_search",
        "query": query,
        "results_count": len(results_dicts),
    })

    logger.info(f"[Node: execute_web_search] Retrieved {len(results_dicts)} valid search results")
    return {
        "search_results": results_dicts,
        "tool_history": history,
    }


def evaluate_search_results(state: ShoppingState) -> Dict[str, Any]:
    """Node 8: Evaluates search snippets and filters top product URLs to inspect."""
    logger.info("[Node: evaluate_search_results] Evaluating search result URLs")

    if state.get("research_status") == "skipped":
        return {}

    raw_results = state.get("search_results") or []
    target_urls: List[str] = []

    for r in raw_results:
        u = r.get("url", "")
        is_valid, _ = validate_url(u)
        if is_valid:
            target_urls.append(u)

    logger.info(f"[Node: evaluate_search_results] Identified {len(target_urls)} candidate URLs for extraction")
    return {
        "candidate_urls": target_urls[:2],  # Inspect top 2 URLs
    }


def extract_product_pages(
    state: ShoppingState,
    browser_tool: Optional[ProductPageExtractorTool] = None,
) -> Dict[str, Any]:
    """Node 9: Uses Playwright to extract product details from target URLs."""
    logger.info("[Node: extract_product_pages] Extracting product page contents")

    if state.get("research_status") == "skipped":
        return {}

    active_browser = browser_tool or ProductPageExtractorTool()
    candidate_urls = state.get("candidate_urls") or [
        r.get("url") for r in (state.get("search_results") or []) if r.get("url")
    ][:2]
    existing_extracted = list(state.get("extracted_pages") or [])
    history = list(state.get("tool_history") or [])

    new_extracted = []
    for url in candidate_urls:
        try:
            page_data = active_browser.extract(url)
            new_extracted.append(page_data.model_dump())
            history.append({
                "action": "extract_product_page",
                "url": url,
                "product_name": page_data.name,
                "price": page_data.price,
                "confidence": page_data.data_confidence,
            })
        except Exception as e:
            logger.error(f"Error extracting product page {url}: {e}")

    combined = existing_extracted + new_extracted
    logger.info(f"[Node: extract_product_pages] Total extracted pages: {len(combined)}")
    return {
        "extracted_pages": combined,
        "tool_history": history,
    }


def validate_research_results(state: ShoppingState) -> Dict[str, Any]:
    """Node 10: Validates, deduplicates, and structures external research output."""
    logger.info("[Node: validate_research_results] Validating research results")

    if state.get("research_status") == "skipped":
        return {}

    extracted_pages = state.get("extracted_pages") or []
    search_results = state.get("search_results") or []
    current_iterations = state.get("react_iterations", 0) + 1

    products: List[Product] = []
    sources: List[str] = []

    for ep in extracted_pages:
        if isinstance(ep, dict):
            p_name = ep.get("name", "")
            if p_name and "Empty Page" not in p_name:
                prod = Product(
                    name=p_name,
                    brand=ep.get("brand"),
                    category=ep.get("category") or state.get("product_category"),
                    price=ep.get("price"),
                    currency=ep.get("currency", "USD"),
                    specifications=ep.get("specifications") or {},
                    features=ep.get("features") or [],
                    availability=ep.get("availability"),
                    url=ep.get("url"),
                    source=ep.get("source", "product_page"),
                    data_confidence=ep.get("data_confidence", 0.8),
                )
                products.append(prod)
                if prod.url:
                    sources.append(prod.url)

    # If no pages extracted, extract candidates from search snippets as fallback
    if not products and search_results:
        for sr in search_results:
            if isinstance(sr, dict):
                p_title = sr.get("title", "")
                if p_title:
                    prod = Product(
                        name=p_title,
                        brand=None,
                        category=state.get("product_category"),
                        price=None,
                        currency="USD",
                        specifications={},
                        features=[],
                        availability="Unknown",
                        url=sr.get("url"),
                        source=sr.get("source", "web_search"),
                        data_confidence=0.5,
                    )
                    products.append(prod)
                    if prod.url:
                        sources.append(prod.url)

    unique_products = deduplicate_products(products)
    research_output = ProductResearchResult(
        products=unique_products,
        search_queries=[state.get("search_query", "")],
        sources=list(set(sources)),
        research_confidence=0.85 if unique_products else 0.0,
        research_complete=True,
        research_needed=len(unique_products) == 0,
        errors=[],
    )

    validated_result = validate_product_research_output(research_output)

    # Append external research findings to final_response
    base_response = state.get("final_response") or ""
    research_summary_lines = []

    if validated_result.products:
        research_summary_lines.append("\n\n[External Web Research Results]:")
        for p in validated_result.products:
            price_str = f"${p.price:,.2f} {p.currency}" if p.price is not None else "Price available on site"
            brand_str = f" ({p.brand})" if p.brand else ""
            feats_str = f", Features: {', '.join(p.features)}" if p.features else ""
            research_summary_lines.append(
                f"• {p.name}{brand_str} — {price_str}{feats_str}\n"
                f"  Source URL: {p.url or p.source}"
            )
        research_summary_lines.append(
            f"\nResearch Assessment: Successfully retrieved and verified {len(validated_result.products)} "
            f"external product option(s) via web research and Playwright extraction."
        )
    else:
        research_summary_lines.append(
            "\n\n[External Web Research Status]: Reliable external product information could not be obtained. "
            "No verifiable product listings matched the criteria."
        )

    augmented_response = base_response + "".join(research_summary_lines)
    logger.info(f"[Node: validate_research_results] Finalized {len(validated_result.products)} verified external products")

    return {
        "research_result": validated_result.model_dump(),
        "research_status": "complete",
        "react_iterations": current_iterations,
        "final_response": augmented_response,
    }


# =====================================================================
# PHASE 4: MULTI-AGENT VALIDATION, COMPARISON, SCORING & DECISION NODES
# =====================================================================

from src.agents.validation_agent import ProductValidationAgent
from src.agents.comparison_agent import ProductComparisonAgent
from src.agents.decision_agent import DecisionAgent
from src.workflow.scoring import score_products as calculate_product_scores
from src.guardrails.validators import validate_final_decision
from src.models.schemas import (
    ComparisonResult,
    DecisionResult,
    DecisionStatus,
    FinalDecision,
    ProductScoreResult,
    ScoreBreakdown,
    ValidationAgentOutput,
)


def _collect_candidate_products_from_state(state: ShoppingState) -> List[Product]:
    """Aggregate candidate products from both Phase 2 RAG and Phase 3 Web Research."""
    candidates: List[Product] = []

    # 1. From Phase 2 local Chroma retrieval
    retrieved_dicts = state.get("retrieved_products") or []
    for rp in retrieved_dicts:
        if isinstance(rp, dict):
            p = Product(
                name=rp.get("product_name") or rp.get("name", ""),
                brand=rp.get("brand"),
                category=rp.get("category"),
                price=rp.get("price"),
                currency=rp.get("currency", "USD"),
                specifications=rp.get("specifications") or {},
                features=rp.get("features") or [],
                availability="In Stock",
                url=None,
                source=rp.get("source", "Local Knowledge Base"),
                data_confidence=0.95,
            )
            candidates.append(p)

    # 2. From Phase 3 external web research
    research_dict = state.get("research_result") or {}
    research_prods = research_dict.get("products") or []
    for ep in research_prods:
        if isinstance(ep, Product):
            candidates.append(ep)
        elif isinstance(ep, dict):
            try:
                candidates.append(Product.model_validate(ep))
            except Exception:
                pass

    unique = deduplicate_products(candidates)
    target_category = state.get("product_category")
    if target_category:
        tc_clean = target_category.strip().lower()
        matching = [
            p for p in unique
            if (p.category and (tc_clean in p.category.lower() or p.category.lower() in tc_clean))
            or (tc_clean in p.name.lower())
        ]
        if matching:
            return matching

    return unique


def validate_candidate_products_node(
    state: ShoppingState,
    agent: Optional[ProductValidationAgent] = None,
) -> Dict[str, Any]:
    """Phase 4 Node 1: Product Validation Agent checks candidate products.

    Filters out invalid or corrupted items, flags missing fields, and keeps
    validated candidates for comparison and scoring.
    """
    logger.info("[Node: validate_candidate_products_node] Starting product validation...")
    candidate_products = _collect_candidate_products_from_state(state)
    target_category = state.get("product_category")

    validation_agent = agent or ProductValidationAgent()
    val_output: ValidationAgentOutput = validation_agent.validate_products(
        candidate_products,
        expected_category=target_category,
    )

    validated_dicts = [p.model_dump() for p in val_output.validated_products]
    logger.info(
        f"[Node: validate_candidate_products_node] Verified {len(val_output.validated_products)} products "
        f"(Valid: {val_output.total_valid}, Partially Valid: {val_output.total_partially_valid}, Invalid: {val_output.total_invalid})"
    )

    return {
        "validated_products": validated_dicts,
        "validation_summary": val_output.model_dump(),
    }


def compare_products_node(
    state: ShoppingState,
    agent: Optional[ProductComparisonAgent] = None,
) -> Dict[str, Any]:
    """Phase 4 Node 2: Product Comparison Agent compares validated products

    against user requirements, budget, preferences, and priorities.
    """
    logger.info("[Node: compare_products_node] Starting comparative analysis...")
    validated_dicts = state.get("validated_products") or []
    products: List[Product] = []
    for d in validated_dicts:
        try:
            products.append(Product.model_validate(d))
        except Exception:
            pass

    interpreted = state.get("interpreted_requirements") or {}
    user_reqs = RequirementAnalysisOutput.model_validate(interpreted) if interpreted else RequirementAnalysisOutput(
        product_category=state.get("product_category"),
        budget=BudgetInfo.model_validate(state.get("budget") or {}),
        required_features=state.get("requirements") or [],
        preferences=state.get("preferences") or [],
        priorities=state.get("priorities") or [],
    )

    comparison_agent = agent or ProductComparisonAgent()
    comp_result: ComparisonResult = comparison_agent.compare_products(products, user_reqs)

    logger.info(
        f"[Node: compare_products_node] Comparison completed across {len(comp_result.comparisons)} products."
    )

    return {
        "comparison_result": comp_result.model_dump(),
    }


def score_products_node(state: ShoppingState) -> Dict[str, Any]:
    """Phase 4 Node 3: Product Scoring engine evaluates validated products deterministically."""
    logger.info("[Node: score_products_node] Computing transparent scores...")
    validated_dicts = state.get("validated_products") or []
    products: List[Product] = []
    for d in validated_dicts:
        try:
            products.append(Product.model_validate(d))
        except Exception:
            pass

    interpreted = state.get("interpreted_requirements") or {}
    user_reqs = RequirementAnalysisOutput.model_validate(interpreted) if interpreted else RequirementAnalysisOutput(
        product_category=state.get("product_category"),
        budget=BudgetInfo.model_validate(state.get("budget") or {}),
        required_features=state.get("requirements") or [],
        preferences=state.get("preferences") or [],
        priorities=state.get("priorities") or [],
    )

    scored_results = calculate_product_scores(products, user_reqs)
    scores_dump = [s.model_dump() for s in scored_results]

    logger.info(f"[Node: score_products_node] Scored and ranked {len(scored_results)} products.")
    return {
        "product_scores": scores_dump,
    }


def make_decision_node(
    state: ShoppingState,
    agent: Optional[DecisionAgent] = None,
) -> Dict[str, Any]:
    """Phase 4 Node 4: Decision Agent selects best recommendation and alternative."""
    logger.info("[Node: make_decision_node] Synthesizing final shopping decision...")

    scores_dump = state.get("product_scores") or []
    scored_results = [ProductScoreResult.model_validate(s) for s in scores_dump]

    comp_dict = state.get("comparison_result") or {}
    comp_result = ComparisonResult.model_validate(comp_dict) if comp_dict else ComparisonResult()

    interpreted = state.get("interpreted_requirements") or {}
    user_reqs = RequirementAnalysisOutput.model_validate(interpreted) if interpreted else RequirementAnalysisOutput(
        product_category=state.get("product_category"),
        budget=BudgetInfo.model_validate(state.get("budget") or {}),
        required_features=state.get("requirements") or [],
        preferences=state.get("preferences") or [],
        priorities=state.get("priorities") or [],
    )

    decision_agent = agent or DecisionAgent()
    decision: DecisionResult = decision_agent.decide(scored_results, comp_result, user_reqs)

    logger.info(f"[Node: make_decision_node] Decision reached: status={decision.decision_status.value}")
    return {
        "decision_result": decision.model_dump(),
    }


def validate_final_decision_node(state: ShoppingState) -> Dict[str, Any]:
    """Phase 4 Node 5: Guardrail validation of final decision and generation

    of structured FinalDecision and user-facing presentation.
    """
    logger.info("[Node: validate_final_decision_node] Executing decision guardrails...")

    decision_dict = state.get("decision_result") or {}
    decision = DecisionResult.model_validate(decision_dict) if decision_dict else DecisionResult(
        recommended_product=None,
        alternative_product=None,
        recommendation_reason="No decision was formulated.",
        confidence=0.0,
        decision_status=DecisionStatus.INSUFFICIENT_DATA,
    )

    validated_dicts = state.get("validated_products") or []
    candidate_products = [Product.model_validate(d) for d in validated_dicts]

    interpreted = state.get("interpreted_requirements") or {}
    user_reqs = RequirementAnalysisOutput.model_validate(interpreted) if interpreted else RequirementAnalysisOutput(
        product_category=state.get("product_category"),
        budget=BudgetInfo.model_validate(state.get("budget") or {}),
        required_features=state.get("requirements") or [],
        preferences=state.get("preferences") or [],
        priorities=state.get("priorities") or [],
    )

    is_valid, issues = validate_final_decision(decision, candidate_products, user_reqs)
    if not is_valid:
        logger.warning(f"[Node: validate_final_decision_node] Decision guardrails flagged issues: {issues}")

    # Build sources list
    sources: List[str] = []
    if decision.recommended_product:
        p = decision.recommended_product
        sources.append(p.url or p.source)
    if decision.alternative_product:
        p = decision.alternative_product
        sources.append(p.url or p.source)
    sources = list(dict.fromkeys(filter(None, sources)))

    # Get recommended score breakdown
    scores_dump = state.get("product_scores") or []
    rec_breakdown = None
    if decision.recommended_product and scores_dump:
        for s in scores_dump:
            if s.get("product_name") == decision.recommended_product.name:
                rec_breakdown = s.get("breakdown")
                break

    comp_dict = state.get("comparison_result") or {}
    comp_summary = comp_dict.get("hard_requirements_summary", "")

    final_decision = FinalDecision(
        user_requirements=user_reqs,
        recommended_product=decision.recommended_product,
        alternative_product=decision.alternative_product,
        comparison_summary=comp_summary,
        score_breakdown=ScoreBreakdown.model_validate(rec_breakdown) if rec_breakdown else None,
        key_reasons=[decision.recommendation_reason] if decision.recommendation_reason else [],
        tradeoffs=decision.tradeoffs,
        unmet_requirements=decision.unmet_requirements,
        sources=sources,
        confidence=decision.confidence,
        decision_status=decision.decision_status,
    )

    # Format user-facing structured recommendation report
    report_lines: List[str] = []
    report_lines.append("\n\n=======================================================")
    report_lines.append("           AI SMART SHOPPING DECISION REPORT           ")
    report_lines.append("=======================================================")

    if final_decision.decision_status == DecisionStatus.RECOMMENDED and final_decision.recommended_product:
        rp = final_decision.recommended_product
        price_str = f"${rp.price:,.2f} {rp.currency}" if rp.price is not None else "Price available on site"
        score_val = final_decision.score_breakdown.total_score if final_decision.score_breakdown else 0.0

        report_lines.append(f"\n[TOP RECOMMENDATION]: {rp.name} ({price_str})")
        report_lines.append(f"  Overall Score: {score_val:.1f}/100 | Confidence: {final_decision.confidence * 100:.0f}%")
        report_lines.append(f"  Source/Citation: {rp.url or rp.source}")
        report_lines.append(f"\n  Why this product was chosen:")
        report_lines.append(f"  {decision.recommendation_reason}")

        if decision.key_advantages:
            report_lines.append("\n  Key Advantages:")
            for adv in decision.key_advantages:
                report_lines.append(f"  + {adv}")

        if decision.tradeoffs:
            report_lines.append("\n  Trade-offs & Considerations:")
            for t in decision.tradeoffs:
                report_lines.append(f"  * {t}")

        if final_decision.alternative_product:
            ap = final_decision.alternative_product
            ap_price = f"${ap.price:,.2f} {ap.currency}" if ap.price is not None else "N/A"
            report_lines.append(f"\n[CLOSEST ALTERNATIVE]: {ap.name} ({ap_price})")
            report_lines.append(f"  Source: {ap.url or ap.source}")

    elif final_decision.decision_status == DecisionStatus.NO_SATISFYING_PRODUCT:
        report_lines.append("\n[STATUS]: No product fully satisfies the required criteria.")
        report_lines.append("  We do not recommend forcing an unsuitable product.")
        report_lines.append(f"\n  Reason:")
        report_lines.append(f"  {decision.recommendation_reason}")

        if decision.unmet_requirements:
            report_lines.append("\n  Unmet Criteria:")
            for u in decision.unmet_requirements:
                report_lines.append(f"  - {u}")

        if final_decision.alternative_product:
            ap = final_decision.alternative_product
            ap_price = f"${ap.price:,.2f} {ap.currency}" if ap.price is not None else "N/A"
            report_lines.append(f"\n[CLOSEST AVAILABLE ALTERNATIVE]: {ap.name} ({ap_price})")
            report_lines.append(f"  Source: {ap.url or ap.source}")

    else:
        report_lines.append("\n[STATUS]: Insufficient product data available to formulate a verified decision.")
        report_lines.append(f"  {decision.recommendation_reason}")

    report_lines.append("=======================================================")

    full_response = (state.get("final_response") or "") + "\n".join(report_lines)

    return {
        "final_decision": final_decision.model_dump(),
        "final_response": full_response,
    }


# =====================================================================
# PHASE 5: AUTOMATION & NOTIFICATION INTEGRATION NODE
# =====================================================================

from datetime import datetime, timezone
from src.integrations.n8n_client import N8nClient
from src.integrations.sendgrid_client import SendGridClient
from src.integrations.pushover_client import PushoverClient
from src.models.schemas import (
    IntegrationDispatchStatus,
    IntegrationDispatchSummary,
    ServiceDispatchResult,
)


def dispatch_integrations_node(
    state: ShoppingState,
    n8n_client: Optional[N8nClient] = None,
    sendgrid_client: Optional[SendGridClient] = None,
    pushover_client: Optional[PushoverClient] = None,
) -> Dict[str, Any]:
    """Phase 5 Node: Dispatches final decision to n8n, SendGrid, and Pushover.

    Strictly decoupled: Every integration is isolated with try/except blocks so that
    network failures, missing credentials, or API timeouts in downstream services
    never block or crash the shopping decision result.
    """
    logger.info("[Node: dispatch_integrations_node] Evaluating external integrations dispatch...")

    final_decision_dict = state.get("final_decision")
    if not final_decision_dict:
        logger.warning("[Node: dispatch_integrations_node] No final_decision available in state. Skipping integrations.")
        return {}

    try:
        final_decision = FinalDecision.model_validate(final_decision_dict)
    except Exception as e:
        logger.error(f"[Node: dispatch_integrations_node] Could not parse final_decision: {e}")
        return {}

    now_iso = datetime.now(timezone.utc).isoformat()
    notice_lines: List[str] = []

    # 1. n8n Dispatch
    n8n = n8n_client or N8nClient()
    try:
        n8n_success, n8n_msg = n8n.send_decision(final_decision)
        n8n_status = IntegrationDispatchStatus.SUCCESS if n8n_success else (
            IntegrationDispatchStatus.SKIPPED if "disabled" in n8n_msg or "not configured" in n8n_msg
            else IntegrationDispatchStatus.FAILED
        )
        if n8n_success:
            notice_lines.append(f"  * n8n Workflow: {n8n_msg}")
    except Exception as e:
        logger.error(f"[Node: dispatch_integrations_node] n8n dispatch exception: {e}")
        n8n_status = IntegrationDispatchStatus.FAILED
        n8n_msg = f"Exception: {str(e)}"

    n8n_result = ServiceDispatchResult(
        service_name="n8n",
        status=n8n_status,
        message=n8n_msg,
        timestamp=now_iso,
    )

    # 2. SendGrid Email Dispatch
    sendgrid = sendgrid_client or SendGridClient()
    try:
        sg_success, sg_msg = sendgrid.send_report(final_decision)
        sg_status = IntegrationDispatchStatus.SUCCESS if sg_success else (
            IntegrationDispatchStatus.SKIPPED if "disabled" in sg_msg or "not configured" in sg_msg
            else IntegrationDispatchStatus.FAILED
        )
        if sg_success:
            notice_lines.append(f"  * Email Report (SendGrid): {sg_msg}")
    except Exception as e:
        logger.error(f"[Node: dispatch_integrations_node] SendGrid dispatch exception: {e}")
        sg_status = IntegrationDispatchStatus.FAILED
        sg_msg = f"Exception: {str(e)}"

    sg_result = ServiceDispatchResult(
        service_name="sendgrid",
        status=sg_status,
        message=sg_msg,
        timestamp=now_iso,
    )

    # 3. Pushover Notification Dispatch
    pushover = pushover_client or PushoverClient()
    try:
        po_success, po_msg = pushover.notify_decision(final_decision)
        po_status = IntegrationDispatchStatus.SUCCESS if po_success else (
            IntegrationDispatchStatus.SKIPPED if "disabled" in po_msg or "not configured" in po_msg
            else IntegrationDispatchStatus.FAILED
        )
        if po_success:
            notice_lines.append(f"  * Push Notification (Pushover): {po_msg}")
    except Exception as e:
        logger.error(f"[Node: dispatch_integrations_node] Pushover dispatch exception: {e}")
        po_status = IntegrationDispatchStatus.FAILED
        po_msg = f"Exception: {str(e)}"

    po_result = ServiceDispatchResult(
        service_name="pushover",
        status=po_status,
        message=po_msg,
        timestamp=now_iso,
    )

    any_failures = any(r.status == IntegrationDispatchStatus.FAILED for r in [n8n_result, sg_result, po_result])
    dispatch_summary = IntegrationDispatchSummary(
        n8n=n8n_result,
        sendgrid=sg_result,
        pushover=po_result,
        any_failures=any_failures,
    )

    # If any integrations were triggered successfully, append notice to final_response
    current_response = state.get("final_response") or ""
    if notice_lines:
        integrations_header = "\n\n[Active Integrations & Delivery]:\n" + "\n".join(notice_lines)
        current_response += integrations_header

    return {
        "integration_results": dispatch_summary.model_dump(),
        "final_response": current_response,
    }
