"""Unit tests for Phase 4 Comparison Agent and Deterministic Scoring Engine."""

import pytest
from src.agents.comparison_agent import ProductComparisonAgent
from src.models.schemas import BudgetInfo, Product, RequirementAnalysisOutput
from src.workflow.scoring import score_products, score_single_product


@pytest.fixture
def comparison_agent():
    return ProductComparisonAgent()


@pytest.fixture
def user_requirements():
    return RequirementAnalysisOutput(
        product_category="Laptop",
        budget=BudgetInfo(max_amount=1200.0, is_specified=True, is_flexible=False),
        required_features=["16GB RAM", "512GB SSD"],
        preferences=["Lightweight", "Long battery life"],
        priorities=["Portability"],
    )


@pytest.fixture
def laptop_within_budget():
    return Product(
        name="Dell Inspiron 14 Plus",
        brand="Dell",
        category="Laptop",
        price=999.00,
        currency="USD",
        specifications={"ram": "16GB LPDDR5", "storage": "512GB NVMe SSD", "weight": "3.1 lbs"},
        features=["16GB RAM", "512GB SSD", "Lightweight design", "12-hour battery life"],
        url="https://dell.com/inspiron14",
        source="dell.com",
        data_confidence=0.9,
    )


@pytest.fixture
def laptop_exceeding_budget():
    return Product(
        name="Dell XPS 15 Pro",
        brand="Dell",
        category="Laptop",
        price=1800.00,  # Exceeds $1200 budget by 50%
        currency="USD",
        specifications={"ram": "32GB RAM", "storage": "1TB SSD"},
        features=["16GB RAM", "512GB SSD", "OLED 4K Display", "Long battery life"],
        url="https://dell.com/xps15",
        source="dell.com",
        data_confidence=0.9,
    )


@pytest.fixture
def laptop_missing_ram():
    return Product(
        name="Budget Notebook 14",
        brand="BudgetCo",
        category="Laptop",
        price=600.00,
        currency="USD",
        specifications={"storage": "512GB SSD"},  # Missing 16GB RAM
        features=["512GB SSD", "Lightweight"],
        url="https://budget.com/nb14",
        source="budget.com",
        data_confidence=0.8,
    )


class TestComparisonAndScoring:
    """Test suite for Requirement Matching, Priority Weights, Budget Penalties, and Scoring."""

    def test_comparison_correctly_matches_requirements(self, comparison_agent, user_requirements, laptop_within_budget):
        """Test 4: Comparison agent produces traceable, factual match details."""
        comp = comparison_agent.compare_single_product(laptop_within_budget, user_requirements)

        assert comp.product_name == laptop_within_budget.name
        assert comp.budget_fit == "within_budget"

        # Check matched requirements
        matched_reqs = [m for m in comp.requirement_matches if m.is_matched]
        matched_texts = [m.requirement for m in matched_reqs]
        assert "16GB RAM" in matched_texts
        assert "512GB SSD" in matched_texts

        # Traceable evidence provided
        evidence = [m.supporting_evidence for m in matched_reqs]
        assert any("16GB RAM" in ev for ev in evidence)
        assert len(comp.strengths) > 0

    def test_hard_requirements_have_higher_priority_in_scoring(self, user_requirements, laptop_within_budget, laptop_missing_ram):
        """Test 5: Fulfilling hard requirements awards significantly more points (40 pts) than soft features."""
        score_full = score_single_product(laptop_within_budget, user_requirements)
        score_missing_hard = score_single_product(laptop_missing_ram, user_requirements)

        # Full match receives maximum 40.0 hard requirement score
        assert score_full.breakdown.hard_requirement_score == 40.0
        assert score_full.is_hard_criteria_satisfied is True

        # Missing one of two hard requirements gets 20.0
        assert score_missing_hard.breakdown.hard_requirement_score == 20.0
        assert score_missing_hard.is_hard_criteria_satisfied is False
        assert any("Missing hard requirement" in v for v in score_missing_hard.breakdown.violations)

    def test_budget_constraint_affects_scoring_and_prevents_false_winner(
        self, user_requirements, laptop_within_budget, laptop_exceeding_budget
    ):
        """Test 6: A product exceeding the budget by >10% receives 0 budget points and is flagged."""
        score_under = score_single_product(laptop_within_budget, user_requirements)
        score_over = score_single_product(laptop_exceeding_budget, user_requirements)

        assert score_under.breakdown.budget_score == 25.0
        assert score_over.breakdown.budget_score == 0.0
        assert score_over.is_hard_criteria_satisfied is False
        assert any("exceeds budget" in v.lower() for v in score_over.breakdown.violations)

        # In ranking, satisfying product must rank first even if expensive laptop has higher specs
        ranked = score_products([laptop_exceeding_budget, laptop_within_budget], user_requirements)
        assert ranked[0].product_name == laptop_within_budget.name

    def test_product_scoring_is_deterministic(self, user_requirements, laptop_within_budget):
        """Test 7: Scoring produces identical, deterministic numeric results across repeated calls."""
        score1 = score_single_product(laptop_within_budget, user_requirements)
        score2 = score_single_product(laptop_within_budget, user_requirements)

        assert score1.breakdown.total_score == score2.breakdown.total_score
        assert score1.breakdown.hard_requirement_score == score2.breakdown.hard_requirement_score
        assert score1.breakdown.budget_score == score2.breakdown.budget_score
