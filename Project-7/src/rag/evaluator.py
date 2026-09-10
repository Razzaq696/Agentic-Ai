"""Agentic RAG Evaluator for determining retrieval sufficiency and next research actions."""

from typing import Any, Dict, List, Optional
from src.config import settings
from src.models.schemas import (
    BudgetInfo,
    RAGDecision,
    RetrievedProduct,
    StructuredRAGOutput,
)
from src.models.state import ShoppingState
from src.utils.logger import logger


class AgenticRAGEvaluator:
    """Evaluates retrieved product knowledge against structured user shopping requirements."""

    def __init__(self, max_retries: Optional[int] = None):
        """Initialize evaluator.

        Args:
            max_retries: Maximum refinement attempts allowed. Defaults to settings.rag_max_retries.
        """
        self.max_retries = max_retries if max_retries is not None else settings.rag_max_retries

    def evaluate(
        self,
        retrieved_products: List[RetrievedProduct],
        category: Optional[str] = None,
        budget: Optional[BudgetInfo] = None,
        requirements: Optional[List[str]] = None,
        preferences: Optional[List[str]] = None,
        priorities: Optional[List[str]] = None,
        retry_count: int = 0,
    ) -> StructuredRAGOutput:
        """Evaluate retrieved products against user requirements and make an agentic decision.

        Args:
            retrieved_products: Products retrieved by the retriever.
            category: Product category requested.
            budget: Parsed budget constraints.
            requirements: Non-negotiable specs requested.
            preferences: Soft user preferences.
            priorities: Decision priorities.
            retry_count: Number of query refinement attempts already conducted.

        Returns:
            StructuredRAGOutput containing decision, confidence, facts, and research flag.
        """
        reqs = requirements or []
        prefs = preferences or []

        # Case 1: Empty retrieval
        if not retrieved_products:
            logger.info("Evaluation: No products were retrieved. External research required.")
            return StructuredRAGOutput(
                retrieved_products=[],
                relevant_information=[],
                retrieval_confidence=0.0,
                retrieval_sufficient=False,
                research_needed=True,
                decision=RAGDecision.EXTERNAL_RESEARCH_NEEDED,
                reasoning=(
                    f"No product knowledge exists in the local catalog for category '{category or 'unknown'}'. "
                    "External web research is required to discover matching products."
                ),
            )

        # Case 2: Category relevance analysis
        category_matches = []
        if category:
            cat_clean = category.lower()
            category_matches = [
                p for p in retrieved_products if cat_clean in p.category.lower() or p.category.lower() in cat_clean
            ]
        else:
            category_matches = retrieved_products

        if not category_matches:
            # None of the retrieved products match the category
            if retry_count < self.max_retries:
                logger.info(
                    f"Evaluation: Category mismatch. Retry count {retry_count} < max {self.max_retries}. Triggering query refinement."
                )
                return StructuredRAGOutput(
                    retrieved_products=retrieved_products,
                    relevant_information=[],
                    retrieval_confidence=0.25,
                    retrieval_sufficient=False,
                    research_needed=True,
                    decision=RAGDecision.REFINE_QUERY,
                    reasoning=(
                        f"Retrieved products did not closely match requested category '{category}'. "
                        "Refining search query for a targeted second attempt."
                    ),
                )
            else:
                logger.info("Evaluation: Category mismatch and retry limit reached. External research required.")
                return StructuredRAGOutput(
                    retrieved_products=retrieved_products,
                    relevant_information=[],
                    retrieval_confidence=0.2,
                    retrieval_sufficient=False,
                    research_needed=True,
                    decision=RAGDecision.EXTERNAL_RESEARCH_NEEDED,
                    reasoning=(
                        f"Local catalog contains no matching products for category '{category}'. "
                        "External web research will be required."
                    ),
                )

        # Case 3: Budget check
        max_budget = budget.max_amount if (budget and budget.is_specified) else None
        budget_compliant = []
        if max_budget is not None:
            # Allow 5% flexible margin if user is marked flexible
            upper_bound = max_budget * (1.05 if budget.is_flexible else 1.0)
            budget_compliant = [p for p in category_matches if p.price <= upper_bound]
        else:
            budget_compliant = category_matches

        if max_budget is not None and not budget_compliant:
            # Products match category, but all exceed budget
            cheapest_found = min(p.price for p in category_matches)
            if retry_count < self.max_retries:
                return StructuredRAGOutput(
                    retrieved_products=category_matches,
                    relevant_information=[
                        f"Found options in category '{category}', but lowest price is ${cheapest_found:,.2f} exceeding your budget of ${max_budget:,.2f}."
                    ],
                    retrieval_confidence=0.4,
                    retrieval_sufficient=False,
                    research_needed=True,
                    decision=RAGDecision.REFINE_QUERY,
                    reasoning=(
                        f"Retrieved {category} options exceed the target budget of ${max_budget:,.2f}. "
                        "Attempting query refinement targeting budget options."
                    ),
                )
            else:
                return StructuredRAGOutput(
                    retrieved_products=category_matches,
                    relevant_information=[
                        f"Local options for '{category}' start at ${cheapest_found:,.2f} (above stated ${max_budget:,.2f} limit)."
                    ],
                    retrieval_confidence=0.5,
                    retrieval_sufficient=False,
                    research_needed=True,
                    decision=RAGDecision.EXTERNAL_RESEARCH_NEEDED,
                    reasoning=(
                        f"All matching local catalog products exceed the specified budget ceiling of ${max_budget:,.2f}. "
                        "External web research required to identify budget-friendly alternatives."
                    ),
                )

        # Case 4: Feature match evaluation & fact extraction
        usable_products = budget_compliant if budget_compliant else category_matches
        relevant_info: List[str] = []
        feature_coverage_scores: List[float] = []

        for p in usable_products:
            # Extract factual bullet point
            info_bullet = (
                f"{p.product_name} ({p.brand}) — ${p.price:,.2f} {p.currency}. "
                f"Features: {', '.join(p.features) if p.features else 'Standard'}. "
                f"[Source: {p.source}]"
            )
            relevant_info.append(info_bullet)

            # Score required feature coverage
            if reqs:
                matched_cnt = len(p.features)
                coverage = min(1.0, matched_cnt / len(reqs))
                feature_coverage_scores.append(coverage)
            else:
                feature_coverage_scores.append(1.0)

        avg_feature_coverage = (
            sum(feature_coverage_scores) / len(feature_coverage_scores) if feature_coverage_scores else 0.5
        )

        # Calculate confidence
        base_confidence = 0.75
        if max_budget is not None and budget_compliant:
            base_confidence += 0.1
        if avg_feature_coverage >= 0.5:
            base_confidence += 0.1
        confidence = min(0.98, max(0.6, base_confidence))

        # Decision: Sufficient knowledge retrieved
        logger.info(
            f"Evaluation: Found {len(usable_products)} sufficient products. Confidence: {confidence:.2f}."
        )
        return StructuredRAGOutput(
            retrieved_products=usable_products,
            relevant_information=relevant_info,
            retrieval_confidence=round(confidence, 2),
            retrieval_sufficient=True,
            research_needed=False,
            decision=RAGDecision.SUFFICIENT,
            reasoning=(
                f"Successfully retrieved {len(usable_products)} matching product(s) in category '{category}' "
                f"from local knowledge base that satisfy required specifications and budget constraints."
            ),
        )

    def evaluate_for_state(self, state: ShoppingState) -> StructuredRAGOutput:
        """Convenience evaluation directly from LangGraph ShoppingState."""
        raw_retrieved = state.get("retrieved_products") or []
        retrieved_objs = []
        for item in raw_retrieved:
            if isinstance(item, RetrievedProduct):
                retrieved_objs.append(item)
            elif isinstance(item, dict):
                try:
                    retrieved_objs.append(RetrievedProduct.model_validate(item))
                except Exception:
                    pass

        budget_dict = state.get("budget") or {}
        budget_info = BudgetInfo(**budget_dict) if budget_dict else None

        return self.evaluate(
            retrieved_products=retrieved_objs,
            category=state.get("product_category"),
            budget=budget_info,
            requirements=state.get("requirements") or [],
            preferences=state.get("preferences") or [],
            priorities=state.get("priorities") or [],
            retry_count=state.get("retrieval_retry_count", 0),
        )
