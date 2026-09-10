"""Centralized deterministic scoring engine for Phase 4 product comparison and decision making."""

from typing import Any, Dict, List, Optional, Tuple
from src.models.schemas import (
    BudgetInfo,
    Product,
    ProductScoreResult,
    RequirementAnalysisOutput,
    ScoreBreakdown,
)
from src.utils.logger import logger


# =====================================================================
# CENTRALIZED SCORING WEIGHTS & CONSTANTS
# =====================================================================
WEIGHT_HARD_REQUIREMENTS = 40.0
WEIGHT_BUDGET_FIT = 25.0
WEIGHT_SPEC_MATCH = 15.0
WEIGHT_FEATURE_MATCH = 10.0
WEIGHT_PREFERENCES_PRIORITIES = 5.0
WEIGHT_DATA_CONFIDENCE = 5.0

MAX_TOTAL_SCORE = 100.0


def _normalize_text(text: str) -> str:
    """Normalize text for consistent substring matching."""
    return (text or "").strip().lower()


def _get_product_text_corpus(product: Product) -> str:
    """Construct searchable text corpus from product attributes, specs, and features."""
    parts = [
        product.name,
        product.brand or "",
        product.category or "",
        " ".join(product.features),
    ]
    for k, v in product.specifications.items():
        parts.append(f"{k} {v}")
    return _normalize_text(" ".join(parts))


def evaluate_hard_requirements(
    product: Product,
    requirements: List[str],
) -> Tuple[float, List[str]]:
    """Evaluate non-negotiable requirements against product data.

    Missing data is strictly treated as unmet (never assumed).

    Args:
        product: Verified product model.
        requirements: List of mandatory requirement strings.

    Returns:
        Tuple of (score out of WEIGHT_HARD_REQUIREMENTS, list of unmet requirement strings).
    """
    if not requirements:
        return WEIGHT_HARD_REQUIREMENTS, []

    corpus = _get_product_text_corpus(product)
    unmet: List[str] = []
    matched_count = 0

    for req in requirements:
        req_norm = _normalize_text(req)
        # Check direct inclusion or token overlap
        req_words = [w for w in req_norm.split() if len(w) > 2]
        if req_norm in corpus or (req_words and all(w in corpus for w in req_words)):
            matched_count += 1
        else:
            unmet.append(req)

    fraction = matched_count / len(requirements)
    score = round(fraction * WEIGHT_HARD_REQUIREMENTS, 2)
    return score, unmet


def evaluate_budget_fit(
    product: Product,
    budget: Optional[BudgetInfo],
) -> Tuple[float, str, Optional[str]]:
    """Evaluate budget compliance.

    Rules:
    - If budget is not specified or flexible: award base high score (21.25 / 25.0, i.e., 85%).
    - If price is within max_amount: award 100% (25.0).
    - If price exceeds budget by <= 10%: partial penalty (10.0 / 25.0, 40%).
    - If price exceeds budget by > 10%: 0.0 points and hard violation flagged.
    - If product price is missing: partial penalty (10.0 / 25.0) and marked as unverified.

    Args:
        product: Verified product model.
        budget: User's budget information.

    Returns:
        Tuple of (budget_score out of 25.0, budget_fit status string, violation string or None).
    """
    if not budget or not budget.is_specified or budget.max_amount is None:
        return round(WEIGHT_BUDGET_FIT * 0.85, 2), "flexible", None

    if product.price is None:
        return round(WEIGHT_BUDGET_FIT * 0.4, 2), "price_unknown", "Product price is unlisted/unavailable"

    max_allowed = budget.max_amount
    price = product.price

    if price <= max_allowed:
        # Extra slight bonus if comfortably within budget range
        if budget.min_amount is not None and price < budget.min_amount:
            # Below requested minimum, still acceptable
            return WEIGHT_BUDGET_FIT, "within_budget", None
        return WEIGHT_BUDGET_FIT, "within_budget", None

    # Exceeds budget
    overage_pct = (price - max_allowed) / max_allowed
    if overage_pct <= 0.10:
        # Slight overage (under 10%)
        return round(WEIGHT_BUDGET_FIT * 0.4, 2), "slightly_exceeds_budget", f"Price exceeds budget by {overage_pct * 100:.1f}% (${price:,.2f} vs max ${max_allowed:,.2f})"

    # Serious overage
    violation = f"Price ${price:,.2f} exceeds budget limit of ${max_allowed:,.2f} by {overage_pct * 100:.1f}%"
    return 0.0, "exceeds_budget", violation


def evaluate_spec_and_features(
    product: Product,
    requirements: List[str],
) -> Tuple[float, float]:
    """Evaluate detailed specification and feature alignment.

    Args:
        product: Verified product model.
        requirements: Extracted requirements.

    Returns:
        Tuple of (spec_score out of 15.0, feature_score out of 10.0).
    """
    corpus = _get_product_text_corpus(product)

    # Specs completeness
    spec_keys = len(product.specifications)
    spec_score = min(WEIGHT_SPEC_MATCH, round((spec_keys / 4.0) * WEIGHT_SPEC_MATCH, 2)) if spec_keys > 0 else 5.0

    # Feature match against requirements
    matched_features = 0
    if product.features and requirements:
        for f in product.features:
            f_norm = _normalize_text(f)
            if any(_normalize_text(r) in f_norm or f_norm in _normalize_text(r) for r in requirements):
                matched_features += 1
        feature_score = min(WEIGHT_FEATURE_MATCH, round((matched_features / max(len(requirements), 1)) * WEIGHT_FEATURE_MATCH, 2))
    elif product.features:
        feature_score = round(WEIGHT_FEATURE_MATCH * 0.8, 2)
    else:
        feature_score = 0.0

    return spec_score, feature_score


def evaluate_preferences_and_priorities(
    product: Product,
    preferences: List[str],
    priorities: List[str],
) -> float:
    """Evaluate alignment with soft user preferences and priorities.

    Args:
        product: Verified product model.
        preferences: Nice-to-have preferences.
        priorities: Decision priorities.

    Returns:
        Score out of 5.0.
    """
    total_items = len(preferences) + len(priorities)
    if total_items == 0:
        return WEIGHT_PREFERENCES_PRIORITIES

    corpus = _get_product_text_corpus(product)
    matches = 0
    for p in preferences + priorities:
        p_norm = _normalize_text(p)
        words = [w for w in p_norm.split() if len(w) > 2]
        if p_norm in corpus or (words and any(w in corpus for w in words)):
            matches += 1

    return round((matches / total_items) * WEIGHT_PREFERENCES_PRIORITIES, 2)


def score_single_product(
    product: Product,
    user_requirements: RequirementAnalysisOutput,
) -> ProductScoreResult:
    """Compute deterministic, transparent score for a product candidate.

    Args:
        product: Validated product instance.
        user_requirements: User's structured requirements.

    Returns:
        ProductScoreResult containing total score and detailed breakdown.
    """
    reqs = user_requirements.required_features or []
    budget = user_requirements.budget
    prefs = user_requirements.preferences or []
    prios = user_requirements.priorities or []

    hard_score, unmet_hard = evaluate_hard_requirements(product, reqs)
    budget_score, budget_fit, budget_violation = evaluate_budget_fit(product, budget)
    spec_score, feature_score = evaluate_spec_and_features(product, reqs)
    pref_score = evaluate_preferences_and_priorities(product, prefs, prios)

    # Data confidence score (out of 5.0)
    conf_score = round((product.data_confidence or 0.8) * WEIGHT_DATA_CONFIDENCE, 2)

    total_score = round(
        hard_score + budget_score + spec_score + feature_score + pref_score + conf_score,
        2,
    )
    total_score = max(0.0, min(MAX_TOTAL_SCORE, total_score))

    violations: List[str] = []
    if budget_violation:
        violations.append(budget_violation)
    if unmet_hard:
        for u in unmet_hard:
            violations.append(f"Missing hard requirement: '{u}'")

    breakdown = ScoreBreakdown(
        hard_requirement_score=hard_score,
        budget_score=budget_score,
        spec_match_score=spec_score,
        feature_match_score=feature_score,
        preference_priority_score=pref_score,
        data_confidence_score=conf_score,
        total_score=total_score,
        violations=violations,
    )

    is_satisfied = len(violations) == 0

    return ProductScoreResult(
        product_name=product.name,
        product=product,
        breakdown=breakdown,
        is_hard_criteria_satisfied=is_satisfied,
    )


def score_products(
    products: List[Product],
    user_requirements: RequirementAnalysisOutput,
) -> List[ProductScoreResult]:
    """Score all candidate products and sort descending by total_score.

    Products satisfying hard criteria rank strictly above those with hard violations,
    preventing budget-violating or requirement-missing products from becoming false winners.

    Args:
        products: List of verified products.
        user_requirements: User's structured requirements.

    Returns:
        Ranked list of ProductScoreResult.
    """
    scored_items: List[ProductScoreResult] = []
    for p in products:
        score_res = score_single_product(p, user_requirements)
        scored_items.append(score_res)

    # Sorting priority:
    # 1. Hard criteria satisfied (True comes first)
    # 2. Total score descending
    scored_items.sort(
        key=lambda item: (
            1 if item.is_hard_criteria_satisfied else 0,
            item.breakdown.total_score,
        ),
        reverse=True,
    )

    logger.info(
        f"Product scoring completed for {len(products)} products. "
        f"Top scorer: {scored_items[0].product_name if scored_items else 'None'} "
        f"(Score: {scored_items[0].breakdown.total_score if scored_items else 0.0})"
    )
    return scored_items
