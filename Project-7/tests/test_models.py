"""Unit tests for Pydantic models and schemas."""

from src.models.schemas import BudgetInfo, RequirementAnalysisOutput, NextAction


class TestBudgetInfoModel:
    """Tests for BudgetInfo parsing and validation."""

    def test_default_values(self):
        budget = BudgetInfo()
        assert budget.min_amount is None
        assert budget.max_amount is None
        assert budget.currency == "USD"
        assert budget.is_flexible is True
        assert budget.is_specified is False

    def test_string_amount_parsing(self):
        budget = BudgetInfo(min_amount="$500", max_amount="$1,250.50", is_specified=True)
        assert budget.min_amount == 500.0
        assert budget.max_amount == 1250.50
        assert budget.is_specified is True

    def test_numeric_float_assignment(self):
        budget = BudgetInfo(min_amount=300.0, max_amount=700.0, is_specified=True)
        assert budget.min_amount == 300.0
        assert budget.max_amount == 700.0


class TestRequirementAnalysisOutputModel:
    """Tests for RequirementAnalysisOutput schema."""

    def test_default_initialization(self):
        output = RequirementAnalysisOutput()
        assert output.product_category is None
        assert output.budget.is_specified is False
        assert output.required_features == []
        assert output.preferences == []
        assert output.priorities == []
        assert output.is_clear is True
        assert output.ambiguities_or_missing_info == []

    def test_serialization_and_deserialization(self, sample_structured_output):
        dumped = sample_structured_output.model_dump()
        assert dumped["product_category"] == "Laptop"
        assert dumped["budget"]["max_amount"] == 1200.0
        assert "16GB RAM" in dumped["required_features"]

        recreated = RequirementAnalysisOutput.model_validate(dumped)
        assert recreated.product_category == sample_structured_output.product_category
        assert recreated.budget.max_amount == sample_structured_output.budget.max_amount


class TestNextActionEnum:
    """Tests for NextAction enum values."""

    def test_enum_constants(self):
        assert NextAction.PROCEED_TO_PRODUCT_RESEARCH.value == "proceed_to_product_research"
        assert NextAction.REQUEST_CLARIFICATION.value == "request_clarification"
        assert NextAction.INVALID_INPUT.value == "invalid_input"
