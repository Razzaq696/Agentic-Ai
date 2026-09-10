"""Product Comparison Agent for Phase 4."""

from typing import Any, Dict, List, Optional
from src.models.schemas import (
    ComparisonResult,
    Product,
    ProductComparisonItem,
    RequirementAnalysisOutput,
    RequirementMatchDetail,
)
from src.utils.logger import logger


class ProductComparisonAgent:
    """Agent responsible for comparing validated candidate products against user requirements,

    budget, preferences, and priorities to generate structured, traceable comparison data.
    """

    def __init__(self):
        pass

    def _normalize(self, text: str) -> str:
        return (text or "").strip().lower()

    def _evaluate_match(self, requirement: str, product: Product, is_hard: bool) -> RequirementMatchDetail:
        """Traceably match a specific requirement against product features and specifications."""
        req_norm = self._normalize(requirement)
        req_words = [w for w in req_norm.split() if len(w) > 2]

        evidence = None
        is_matched = False

        # 1. Search in features
        for f in product.features:
            f_norm = self._normalize(f)
            if req_norm in f_norm or (req_words and all(w in f_norm for w in req_words)):
                is_matched = True
                evidence = f"Feature: '{f}'"
                break

        # 2. Search in specifications
        if not is_matched:
            for k, v in product.specifications.items():
                combined = self._normalize(f"{k} {v}")
                if req_norm in combined or (req_words and all(w in combined for w in req_words)):
                    is_matched = True
                    evidence = f"Specification: {k}='{v}'"
                    break

        # 3. Search in product name or category
        if not is_matched:
            combined_header = self._normalize(f"{product.name} {product.brand or ''} {product.category or ''}")
            if req_norm in combined_header or (req_words and all(w in combined_header for w in req_words)):
                is_matched = True
                evidence = f"Product Title/Metadata: '{product.name}'"

        return RequirementMatchDetail(
            requirement=requirement,
            is_hard_requirement=is_hard,
            is_matched=is_matched,
            supporting_evidence=evidence if is_matched else "Not found in verified product specifications",
        )

    def compare_single_product(
        self,
        product: Product,
        requirements: RequirementAnalysisOutput,
    ) -> ProductComparisonItem:
        """Build structured comparison details for a single product."""
        matches: List[RequirementMatchDetail] = []

        # Hard requirements
        for req in requirements.required_features or []:
            matches.append(self._evaluate_match(req, product, is_hard=True))

        # Soft preferences
        for pref in requirements.preferences or []:
            matches.append(self._evaluate_match(pref, product, is_hard=False))

        # Budget fit
        budget = requirements.budget
        budget_fit = "flexible"
        weaknesses: List[str] = []
        strengths: List[str] = []
        missing_info: List[str] = []

        if budget and budget.is_specified and budget.max_amount is not None:
            if product.price is None:
                budget_fit = "price_unverified"
                missing_info.append("Product pricing is unavailable or unverified.")
                weaknesses.append("Price is unverified; could exceed budget limit.")
            elif product.price <= budget.max_amount:
                budget_fit = "within_budget"
                diff = budget.max_amount - product.price
                strengths.append(f"Comfortably within budget (${product.price:,.2f} vs max ${budget.max_amount:,.2f}, saving ${diff:,.2f})")
            else:
                over = product.price - budget.max_amount
                budget_fit = "exceeds_budget"
                weaknesses.append(f"Exceeds target budget by ${over:,.2f} (${product.price:,.2f} vs ${budget.max_amount:,.2f})")

        # Strengths from matched requirements
        for m in matches:
            if m.is_matched and m.is_hard_requirement:
                strengths.append(f"Satisfies mandatory requirement: '{m.requirement}' ({m.supporting_evidence})")
            elif not m.is_matched and m.is_hard_requirement:
                weaknesses.append(f"Fails mandatory requirement: '{m.requirement}'")

        # Record missing data items
        if not product.specifications:
            missing_info.append("Detailed technical specifications are absent.")
        if not product.features:
            missing_info.append("Curated feature list is absent.")

        # Comparison notes
        comp_notes = (
            f"{product.name}: "
            f"{sum(1 for m in matches if m.is_matched)}/{len(matches) if matches else 1} criteria matched. "
            f"Budget status: {budget_fit}."
        )

        return ProductComparisonItem(
            product_name=product.name,
            brand=product.brand,
            price=product.price,
            currency=product.currency,
            requirement_matches=matches,
            budget_fit=budget_fit,
            strengths=strengths,
            weaknesses=weaknesses,
            missing_information=missing_info,
            comparison_notes=comp_notes,
        )

    def compare_products(
        self,
        products: List[Product],
        requirements: RequirementAnalysisOutput,
    ) -> ComparisonResult:
        """Perform comprehensive comparative analysis across all valid products."""
        logger.info(f"ProductComparisonAgent: Comparing {len(products)} products against user requirements...")
        comparisons: List[ProductComparisonItem] = []

        for p in products:
            comp_item = self.compare_single_product(p, requirements)
            comparisons.append(comp_item)

        # Summary of hard requirement fulfillment
        total_prods = len(products)
        fully_satisfied = sum(
            1 for c in comparisons
            if all(m.is_matched for m in c.requirement_matches if m.is_hard_requirement)
            and c.budget_fit in ("within_budget", "flexible")
        )
        hard_summary = (
            f"{fully_satisfied} of {total_prods} evaluated products fully meet all hard requirements and budget limits."
        )

        # Main trade-offs
        tradeoffs: List[str] = []
        if len(comparisons) > 1:
            p1 = comparisons[0]
            p2 = comparisons[1]
            if p1.price is not None and p2.price is not None:
                diff = abs(p1.price - p2.price)
                cheaper = p1.product_name if p1.price < p2.price else p2.product_name
                pricier = p2.product_name if p1.price < p2.price else p1.product_name
                tradeoffs.append(
                    f"{cheaper} is ${diff:,.2f} more economical than {pricier}."
                )

        logger.info("ProductComparisonAgent: Comparison complete.")
        return ComparisonResult(
            comparisons=comparisons,
            hard_requirements_summary=hard_summary,
            key_tradeoffs=tradeoffs,
        )
