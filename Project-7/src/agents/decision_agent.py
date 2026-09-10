"""Decision Agent for Phase 4."""

from typing import Any, Dict, List, Optional
from src.models.schemas import (
    ComparisonResult,
    DecisionResult,
    DecisionStatus,
    Product,
    ProductScoreResult,
    RequirementAnalysisOutput,
)
from src.utils.logger import logger


class DecisionAgent:
    """Agent responsible for selecting the optimal product recommendation,

    identifying an alternative, grounding reasons in verified data, and
    explaining trade-offs and unmet requirements.
    """

    def __init__(self):
        pass

    def decide(
        self,
        scored_products: List[ProductScoreResult],
        comparison_result: ComparisonResult,
        user_requirements: RequirementAnalysisOutput,
    ) -> DecisionResult:
        """Formulate a grounded decision based on scored products and comparison results.

        Rules:
        - If no products are provided: returns INSUFFICIENT_DATA.
        - If products exist but NONE satisfy hard criteria:
            Does NOT force a recommendation.
            Sets decision_status=NO_SATISFYING_PRODUCT,
            recommended_product=None,
            alternative_product=closest candidate,
            lists unmet_requirements explicitly.
        - If a product satisfies all hard criteria:
            Recommends the top-ranked satisfying product.
            Selects the second best satisfying (or closest runner-up) as alternative.
            Synthesizes key advantages, trade-offs, and grounded reasoning.

        Args:
            scored_products: Ranked list of ProductScoreResult (from scoring engine).
            comparison_result: Structured ComparisonResult.
            user_requirements: User's requirements.

        Returns:
            Structured DecisionResult.
        """
        logger.info(f"DecisionAgent: Evaluating decisions for {len(scored_products)} candidates...")

        if not scored_products:
            logger.warning("DecisionAgent: No products available to decide.")
            return DecisionResult(
                recommended_product=None,
                alternative_product=None,
                recommendation_reason="No valid product candidates were found matching the search criteria.",
                key_advantages=[],
                tradeoffs=[],
                unmet_requirements=["No candidate products available"],
                confidence=0.0,
                decision_status=DecisionStatus.INSUFFICIENT_DATA,
            )

        # Separate products satisfying hard criteria from those with violations
        satisfying = [p for p in scored_products if p.is_hard_criteria_satisfied]
        non_satisfying = [p for p in scored_products if not p.is_hard_criteria_satisfied]

        # Case 1: No product fully satisfies the required criteria
        if not satisfying:
            logger.info("DecisionAgent: No product satisfies all hard criteria. Not forcing recommendation.")
            closest = non_satisfying[0] if non_satisfying else scored_products[0]
            violations = list(closest.breakdown.violations)

            runner_up = non_satisfying[1].product if len(non_satisfying) > 1 else None

            unmet_list = violations if violations else ["Failed to meet specified budget or hardware requirements"]

            reason = (
                "No product in the verified knowledge base or web research fully satisfies all mandatory "
                "requirements and budget constraints. We do not recommend forcing an unsuitable product. "
                f"The closest available alternative is {closest.product_name}, but it requires compromises."
            )

            return DecisionResult(
                recommended_product=None,
                alternative_product=closest.product,
                recommendation_reason=reason,
                key_advantages=[
                    f"Closest match by overall feature alignment ({closest.breakdown.total_score:.1f}/100)",
                ],
                tradeoffs=[
                    f"Requires compromise on: {', '.join(unmet_list)}",
                ],
                unmet_requirements=unmet_list,
                confidence=0.4,
                decision_status=DecisionStatus.NO_SATISFYING_PRODUCT,
            )

        # Case 2: One or more products satisfy hard criteria
        top_pick = satisfying[0]
        rec_product = top_pick.product

        # Runner-up alternative
        alt_product: Optional[Product] = None
        if len(satisfying) > 1:
            alt_product = satisfying[1].product
        elif non_satisfying:
            alt_product = non_satisfying[0].product

        # Derive advantages from comparison items and scoring
        matching_comp = next(
            (c for c in comparison_result.comparisons if c.product_name == rec_product.name),
            None,
        )

        advantages: List[str] = []
        if matching_comp:
            advantages.extend(matching_comp.strengths[:4])
        if not advantages:
            advantages = [
                f"Achieved highest overall alignment score of {top_pick.breakdown.total_score:.1f}/100",
                "Fully satisfies all non-negotiable requirements without budget violation",
            ]

        # Derive trade-offs
        tradeoffs: List[str] = []
        if matching_comp and matching_comp.weaknesses:
            tradeoffs.extend(matching_comp.weaknesses[:3])
        if alt_product and alt_product.price and rec_product.price:
            if rec_product.price > alt_product.price:
                diff = rec_product.price - alt_product.price
                tradeoffs.append(
                    f"Priced ${diff:,.2f} higher than alternative {alt_product.name}"
                )

        # Unmet requirements
        unmet_requirements = list(top_pick.breakdown.violations)

        # Formulate grounded reason
        price_str = f"${rec_product.price:,.2f}" if rec_product.price is not None else "market pricing"
        reason = (
            f"The {rec_product.name} ({price_str}) is selected as the best choice because it fully satisfies "
            f"all mandatory specifications and budget parameters with the highest composite score "
            f"({top_pick.breakdown.total_score:.1f}/100)."
        )

        # Confidence calculation
        margin = 0.15 if len(satisfying) > 1 and satisfying[0].breakdown.total_score > satisfying[1].breakdown.total_score + 5 else 0.05
        confidence = min(0.98, max(0.65, (rec_product.data_confidence or 0.8) + margin))

        logger.info(
            f"DecisionAgent: Recommended '{rec_product.name}' with score "
            f"{top_pick.breakdown.total_score:.1f} and confidence {confidence:.2f}"
        )

        return DecisionResult(
            recommended_product=rec_product,
            alternative_product=alt_product,
            recommendation_reason=reason,
            key_advantages=advantages,
            tradeoffs=tradeoffs,
            unmet_requirements=unmet_requirements,
            confidence=round(confidence, 2),
            decision_status=DecisionStatus.RECOMMENDED,
        )
