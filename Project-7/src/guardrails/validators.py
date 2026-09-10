"""Input and output guardrails to ensure robust requirement processing."""

import json
import re
from typing import Any, List, Optional, Tuple
from pydantic import ValidationError

from src.models.schemas import BudgetInfo, RequirementAnalysisOutput
from src.utils.logger import logger


class GuardrailValidationError(Exception):
    """Raised when an input or output violates guardrail constraints."""

    def __init__(self, message: str, details: Optional[List[str]] = None):
        super().__init__(message)
        self.details = details or []


def validate_input_request(user_request: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Guardrail to validate the user's initial shopping request.

    Rejects:
    - None or non-string inputs
    - Empty or whitespace-only inputs
    - Extremely short inputs (< 3 characters)
    - Inputs containing only punctuation or numbers (e.g. '???', '1234')

    Args:
        user_request: The raw user prompt text.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])
    """
    if user_request is None:
        return False, "Shopping request cannot be empty (received None)."

    if not isinstance(user_request, str):
        return False, f"Invalid request format: expected text, got {type(user_request).__name__}."

    cleaned = user_request.strip()
    if not cleaned:
        return False, "Shopping request cannot be empty or whitespace only."

    if len(cleaned) < 3:
        return False, "Shopping request is too brief to convey meaningful purchase intent."

    # Check for inputs that have no alphabetic letters (e.g., '12345', '!??$#')
    if not re.search(r"[a-zA-Z]", cleaned):
        return False, "Shopping request must contain valid descriptive text or product names."

    return True, None


def validate_structured_output(raw_output: Any) -> RequirementAnalysisOutput:
    """Guardrail to validate and safely cast LLM output into RequirementAnalysisOutput.

    Prevents malformed structured outputs from corrupting the agent workflow.

    Args:
        raw_output: Object returned by the LLM or parser.

    Returns:
        Validated RequirementAnalysisOutput instance.

    Raises:
        GuardrailValidationError: If output cannot be validated or parsed into schema.
    """
    if raw_output is None:
        raise GuardrailValidationError("LLM returned empty (None) structured output.")

    if isinstance(raw_output, RequirementAnalysisOutput):
        # Guarantee budget object integrity
        if raw_output.budget is None:
            raw_output.budget = BudgetInfo(is_specified=False, is_flexible=True)
        return raw_output

    if isinstance(raw_output, dict):
        try:
            return RequirementAnalysisOutput.model_validate(raw_output)
        except ValidationError as e:
            errors = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
            logger.error(f"Malformed dict in structured output: {errors}")
            raise GuardrailValidationError(
                "Malformed LLM structured output does not conform to RequirementAnalysisOutput schema.",
                details=errors,
            )

    if isinstance(raw_output, str):
        try:
            parsed_json = json.loads(raw_output)
            return RequirementAnalysisOutput.model_validate(parsed_json)
        except (json.JSONDecodeError, ValidationError) as e:
            msg = f"Failed to parse LLM string response into valid JSON/schema: {str(e)}"
            logger.error(msg)
            raise GuardrailValidationError(msg)

    raise GuardrailValidationError(
        f"Unsupported structured output type: {type(raw_output).__name__}."
    )


def check_requirement_completeness(
    requirements: RequirementAnalysisOutput,
) -> Tuple[bool, List[str]]:
    """Determine whether extracted requirements are clear enough to proceed to research.

    Rules:
    - Missing product category -> Not clear (must clarify).
    - Explicitly flagged as unclear (`is_clear=False`) -> Not clear (must clarify).
    - Missing budget is completely permissible (defaults to flexible / market price).
    - If critical ambiguities exist, must clarify.

    Args:
        requirements: Parsed RequirementAnalysisOutput instance.

    Returns:
        Tuple of (is_sufficient: bool, missing_or_ambiguous_points: List[str])
    """
    issues: List[str] = []

    # Category check
    category = (requirements.product_category or "").strip()
    if not category or category.lower() in ["none", "unknown", "product", "item"]:
        issues.append("Product category or item type could not be identified.")

    # Clarity flag
    if not requirements.is_clear:
        if requirements.ambiguities_or_missing_info:
            issues.extend(requirements.ambiguities_or_missing_info)
        else:
            issues.append("The shopping request is too ambiguous or underspecified.")

    # Deduplicate issues while maintaining order
    seen = set()
    unique_issues = []
    for issue in issues:
        if issue not in seen:
            seen.add(issue)
            unique_issues.append(issue)

    is_sufficient = len(unique_issues) == 0
    return is_sufficient, unique_issues


# =====================================================================
# PHASE 2: RAG & PRODUCT RETRIEVAL GUARDRAILS
# =====================================================================

from src.models.schemas import ProductRecord, RetrievedProduct, StructuredRAGOutput, RAGDecision


def validate_product_record_dict(data: Any) -> Tuple[bool, Optional[str]]:
    """Guardrail to validate a raw dictionary against ProductRecord schema.

    Args:
        data: Dict representing a product candidate.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
    """
    if not isinstance(data, dict):
        return False, f"Expected dict, got {type(data).__name__}"
    try:
        ProductRecord.model_validate(data)
        return True, None
    except ValidationError as ve:
        err = ve.errors()[0]
        return False, f"Field '{err['loc'][0]}': {err['msg']}"


def validate_retrieved_products(products: Any) -> List[RetrievedProduct]:
    """Guardrail to validate and filter a list of retrieved products.

    Drops corrupted or malformed items safely without crashing the workflow.

    Args:
        products: Iterable of product objects or dicts.

    Returns:
        List of validated RetrievedProduct objects.
    """
    if not products:
        return []

    if not isinstance(products, list):
        logger.warning(f"Expected list of products, got {type(products).__name__}")
        return []

    valid_list: List[RetrievedProduct] = []
    for item in products:
        if isinstance(item, RetrievedProduct):
            valid_list.append(item)
        elif isinstance(item, dict):
            try:
                valid_list.append(RetrievedProduct.model_validate(item))
            except ValidationError as ve:
                logger.warning(f"Skipping malformed retrieved product dict: {ve}")
        else:
            logger.warning(f"Unsupported product item type: {type(item).__name__}")

    return valid_list


def validate_structured_rag_output(raw_output: Any) -> StructuredRAGOutput:
    """Guardrail to validate that RAG evaluation conforms to StructuredRAGOutput.

    Recovers safely from malformed or incomplete evaluations without crashing.

    Args:
        raw_output: Object produced by RAG evaluator.

    Returns:
        Validated StructuredRAGOutput instance.
    """
    if isinstance(raw_output, StructuredRAGOutput):
        return raw_output

    if isinstance(raw_output, dict):
        try:
            return StructuredRAGOutput.model_validate(raw_output)
        except ValidationError as ve:
            logger.error(f"Malformed dict in RAG evaluation: {ve}")

    logger.warning(
        "Invalid RAG evaluation output encountered. Falling back to default EXTERNAL_RESEARCH_NEEDED output."
    )
    return StructuredRAGOutput(
        retrieved_products=[],
        relevant_information=[],
        retrieval_confidence=0.0,
        retrieval_sufficient=False,
        research_needed=True,
        decision=RAGDecision.EXTERNAL_RESEARCH_NEEDED,
        reasoning="Guardrail triggered: RAG output was malformed. Falling back to external research.",
    )


# =====================================================================
# PHASE 3: RESEARCH & PLAYWRIGHT VALIDATION GUARDRAILS
# =====================================================================

import urllib.parse
from src.models.schemas import Product, ProductResearchResult, SearchResultItem


def validate_url(url: Optional[str]) -> Tuple[bool, Optional[str]]:
    """Guardrail to validate web URLs before making external browser calls.

    Args:
        url: Candidate URL string.

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str]).
    """
    if not url or not isinstance(url, str):
        return False, "URL cannot be empty or non-string."

    cleaned = url.strip()
    if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
        return False, f"URL must start with http:// or https:// (received: '{cleaned}')."

    parsed = urllib.parse.urlparse(cleaned)
    if not parsed.netloc:
        return False, f"URL missing domain host: '{cleaned}'."

    return True, None


def validate_search_results(raw_results: Any) -> List[SearchResultItem]:
    """Guardrail to validate and filter web search results.

    Args:
        raw_results: Candidate list of search result items.

    Returns:
        List of valid SearchResultItem instances.
    """
    if not raw_results or not isinstance(raw_results, list):
        return []

    valid_items: List[SearchResultItem] = []
    for r in raw_results:
        if isinstance(r, SearchResultItem):
            if r.url and r.title:
                valid_items.append(r)
        elif isinstance(r, dict):
            try:
                item = SearchResultItem.model_validate(r)
                if item.url and item.title:
                    valid_items.append(item)
            except ValidationError:
                pass

    return valid_items


def deduplicate_products(products: List[Product]) -> List[Product]:
    """Remove duplicate products based on normalized name or URL."""
    seen_names = set()
    seen_urls = set()
    unique: List[Product] = []

    for p in products:
        norm_name = re.sub(r"[^a-zA-Z0-9]", "", p.name.lower())
        norm_url = (p.url or "").strip().lower()

        if norm_name and norm_name in seen_names:
            continue
        if norm_url and norm_url in seen_urls:
            continue

        if norm_name:
            seen_names.add(norm_name)
        if norm_url:
            seen_urls.add(norm_url)

        unique.append(p)

    return unique


def validate_product_research_output(raw_output: Any) -> ProductResearchResult:
    """Guardrail to validate and sanitize complete ProductResearchResult.

    Args:
        raw_output: Object or dict representing product research output.

    Returns:
        Validated, deduplicated ProductResearchResult.
    """
    if isinstance(raw_output, ProductResearchResult):
        raw_output.products = deduplicate_products(raw_output.products)
        return raw_output

    if isinstance(raw_output, dict):
        try:
            res = ProductResearchResult.model_validate(raw_output)
            res.products = deduplicate_products(res.products)
            return res
        except ValidationError as ve:
            logger.warning(f"Malformed dict in ProductResearchResult: {ve}")

    logger.warning("Falling back to empty ProductResearchResult due to validation error.")
    return ProductResearchResult(
        products=[],
        search_queries=[],
        sources=[],
        research_confidence=0.0,
        research_complete=False,
        research_needed=True,
        errors=["Research output failed schema validation"],
    )


# =====================================================================
# PHASE 4: MULTI-AGENT COMPARISON & DECISION GUARDRAILS
# =====================================================================

from src.models.schemas import (
    DecisionResult,
    DecisionStatus,
    FinalDecision,
    RequirementAnalysisOutput,
)


def validate_product_for_recommendation(product: Optional[Product]) -> Tuple[bool, List[str]]:
    """Guardrail: Ensure a recommended product is valid, well-formed, and supported.

    Args:
        product: The product proposed for recommendation.

    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
    """
    if product is None:
        return False, ["No product provided for recommendation"]

    issues: List[str] = []

    # Name check
    name = (product.name or "").strip()
    if not name or name.lower() in ("product", "item", "unknown", "untitled", "n/a"):
        issues.append("Recommended product has an invalid or placeholder name")

    # Price check
    if product.price is not None and product.price < 0:
        issues.append(f"Recommended product has invalid negative price: {product.price}")

    # Confidence check
    if product.data_confidence is not None and product.data_confidence < 0.2:
        issues.append(f"Product data confidence too low ({product.data_confidence}) for reliable recommendation")

    # Source check
    if not product.url and not product.source:
        issues.append("Product lacks source citation or URL for verification")

    return len(issues) == 0, issues


def validate_final_decision(
    decision: DecisionResult,
    candidate_products: List[Product],
    user_requirements: RequirementAnalysisOutput,
) -> Tuple[bool, List[str]]:
    """Guardrail: Comprehensive sanity check on final DecisionResult before user presentation.

    Checks:
    1. If a product is recommended, verify it exists in the candidate products pool.
    2. Recommended product must not violate hard budget constraints unless explicitly flagged.
    3. If no product satisfied requirements, ensure decision_status reflects this.
    4. Guard against empty reasoning or fabricated claims.
    5. Confidence must be non-zero when recommending a product.

    Args:
        decision: DecisionResult to validate.
        candidate_products: Pool of verified products available to the agents.
        user_requirements: User's structured requirements.

    Returns:
        Tuple of (is_valid: bool, issues: List[str]).
    """
    issues: List[str] = []

    if decision.decision_status == DecisionStatus.RECOMMENDED:
        rec = decision.recommended_product
        if rec is None:
            issues.append("Decision status is RECOMMENDED but recommended_product is None")
            return False, issues

        # 1. Existence check in candidate pool
        candidate_names = [p.name.strip().lower() for p in candidate_products]
        if rec.name.strip().lower() not in candidate_names:
            issues.append(f"Recommended product '{rec.name}' does not exist in the candidate products pool")

        # 2. Product field validity
        prod_valid, prod_issues = validate_product_for_recommendation(rec)
        if not prod_valid:
            issues.extend(prod_issues)

        # 3. Budget constraint violation check
        budget = user_requirements.budget
        if budget and budget.is_specified and budget.max_amount is not None:
            if rec.price is not None and rec.price > budget.max_amount:
                overage = rec.price - budget.max_amount
                # Hard budget constraint violation must not silently slip through as a clean recommendation
                if not any("exceeds" in u.lower() or "budget" in u.lower() for u in decision.unmet_requirements):
                    issues.append(
                        f"Recommended product '{rec.name}' (${rec.price:,.2f}) violates hard budget of "
                        f"${budget.max_amount:,.2f} without documented unmet requirement"
                    )

        # 4. Reason grounding
        if not decision.recommendation_reason or len(decision.recommendation_reason.strip()) < 10:
            issues.append("Decision recommendation_reason is missing or too terse")

        # 5. Confidence check
        if decision.confidence <= 0.0:
            issues.append("Decision confidence score must be greater than 0.0")

    elif decision.decision_status == DecisionStatus.NO_SATISFYING_PRODUCT:
        if decision.recommended_product is not None:
            issues.append("Status is NO_SATISFYING_PRODUCT but recommended_product was set")
        if not decision.unmet_requirements:
            issues.append("Status is NO_SATISFYING_PRODUCT but unmet_requirements list is empty")

    return len(issues) == 0, issues
