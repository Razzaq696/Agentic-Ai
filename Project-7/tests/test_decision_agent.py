"""Unit tests for Phase 4 Decision Agent and Final Validation Guardrails."""

import pytest
from src.agents.comparison_agent import ProductComparisonAgent
from src.agents.decision_agent import DecisionAgent
from src.guardrails.validators import validate_final_decision
from src.models.schemas import (
    BudgetInfo,
    DecisionResult,
    DecisionStatus,
    Product,
    RequirementAnalysisOutput,
)
from src.workflow.scoring import score_products


@pytest.fixture
def decision_agent():
    return DecisionAgent()


@pytest.fixture
def comparison_agent():
    return ProductComparisonAgent()


@pytest.fixture
def user_requirements():
    return RequirementAnalysisOutput(
        product_category="Espresso Machine",
        budget=BudgetInfo(max_amount=1000.0, is_specified=True, is_flexible=False),
        required_features=["Dual boiler", "PID temperature control"],
        preferences=["Compact footprint"],
        priorities=["Temperature stability"],
    )


@pytest.fixture
def satisfying_espresso():
    return Product(
        name="Gaggia Classic Pro Dual Mod",
        brand="Gaggia",
        category="Espresso Machine",
        price=899.00,
        currency="USD",
        specifications={"boiler": "Dual boiler system", "pid": "Digital PID temperature control"},
        features=["Dual boiler", "PID temperature control", "Compact footprint"],
        url="https://gaggia.com/classic-dual",
        source="gaggia.com",
        data_confidence=0.9,
    )


@pytest.fixture
def alternative_espresso():
    return Product(
        name="Rancilio Silvia Pro X",
        brand="Rancilio",
        category="Espresso Machine",
        price=980.00,
        currency="USD",
        specifications={"boiler": "Dual boiler system", "pid": "PID temperature control"},
        features=["Dual boiler", "PID temperature control"],
        url="https://rancilio.com/silvia-pro-x",
        source="rancilio.com",
        data_confidence=0.88,
    )


@pytest.fixture
def over_budget_espresso():
    return Product(
        name="La Marzocco Micra",
        brand="La Marzocco",
        category="Espresso Machine",
        price=3900.00,  # $3900 vs $1000 budget
        currency="USD",
        specifications={"boiler": "Dual boiler", "pid": "PID control"},
        features=["Dual boiler", "PID temperature control"],
        url="https://lamarzocco.com/micra",
        source="lamarzocco.com",
        data_confidence=0.95,
    )


class TestDecisionAgentAndGuardrails:
    """Test suite for DecisionAgent selection, no-satisfaction fallback, and guardrails."""

    def test_best_product_selected_correctly(
        self, decision_agent, comparison_agent, user_requirements, satisfying_espresso, alternative_espresso
    ):
        """Test 8: Best product is selected as recommended and runner-up as alternative."""
        products = [satisfying_espresso, alternative_espresso]
        comp = comparison_agent.compare_products(products, user_requirements)
        scored = score_products(products, user_requirements)

        decision: DecisionResult = decision_agent.decide(scored, comp, user_requirements)

        assert decision.decision_status == DecisionStatus.RECOMMENDED
        assert decision.recommended_product is not None
        assert decision.recommended_product.name == satisfying_espresso.name
        assert decision.alternative_product is not None
        assert decision.alternative_product.name == alternative_espresso.name
        assert decision.confidence > 0.6
        assert len(decision.key_advantages) > 0

    def test_no_product_satisfies_requirements_does_not_force_winner(
        self, decision_agent, comparison_agent, user_requirements, over_budget_espresso
    ):
        """Test 9: When no product satisfies hard requirements or budget, winner is NOT forced."""
        products = [over_budget_espresso]
        comp = comparison_agent.compare_products(products, user_requirements)
        scored = score_products(products, user_requirements)

        decision: DecisionResult = decision_agent.decide(scored, comp, user_requirements)

        assert decision.decision_status == DecisionStatus.NO_SATISFYING_PRODUCT
        assert decision.recommended_product is None  # Must not force false recommendation
        assert decision.alternative_product is not None  # Closest candidate provided
        assert decision.alternative_product.name == over_budget_espresso.name
        assert len(decision.unmet_requirements) > 0
        assert any("budget" in u.lower() for u in decision.unmet_requirements)

    def test_decision_agent_returns_valid_pydantic_output(
        self, decision_agent, comparison_agent, user_requirements, satisfying_espresso
    ):
        """Test 10: Decision agent returns a strictly valid Pydantic DecisionResult model."""
        products = [satisfying_espresso]
        comp = comparison_agent.compare_products(products, user_requirements)
        scored = score_products(products, user_requirements)

        decision = decision_agent.decide(scored, comp, user_requirements)

        assert isinstance(decision, DecisionResult)
        dumped = decision.model_dump()
        assert "recommended_product" in dumped
        assert "decision_status" in dumped
        assert dumped["decision_status"] == "recommended"

    def test_final_validation_rejects_unsupported_or_hallucinated_recommendations(
        self, user_requirements, satisfying_espresso
    ):
        """Test 11: Final validation guardrail rejects phantom products not in candidate pool."""
        phantom_product = Product(
            name="Imaginary Magic Machine",
            price=200.0,
            currency="USD",
        )

        invalid_decision = DecisionResult(
            recommended_product=phantom_product,
            alternative_product=None,
            recommendation_reason="Made up imaginary product.",
            confidence=0.9,
            decision_status=DecisionStatus.RECOMMENDED,
        )

        is_valid, issues = validate_final_decision(
            invalid_decision,
            candidate_products=[satisfying_espresso],  # Phantom is not in candidate list
            user_requirements=user_requirements,
        )

        assert is_valid is False
        assert any("candidate products pool" in issue for issue in issues)
