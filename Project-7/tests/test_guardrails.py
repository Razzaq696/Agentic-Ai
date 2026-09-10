"""Unit tests for input and output guardrails."""

import pytest
from src.guardrails.validators import (
    validate_input_request,
    validate_structured_output,
    check_requirement_completeness,
    GuardrailValidationError,
)
from src.models.schemas import BudgetInfo, RequirementAnalysisOutput


class TestInputGuardrails:
    """Tests for validating user shopping requests."""

    def test_empty_string_rejected(self):
        is_valid, msg = validate_input_request("")
        assert not is_valid
        assert "cannot be empty" in msg

    def test_whitespace_only_rejected(self):
        is_valid, msg = validate_input_request("     ")
        assert not is_valid
        assert "whitespace" in msg

    def test_none_input_rejected(self):
        is_valid, msg = validate_input_request(None)
        assert not is_valid
        assert "received None" in msg

    def test_too_short_rejected(self):
        is_valid, msg = validate_input_request("a")
        assert not is_valid
        assert "too brief" in msg

    def test_punctuation_or_numbers_only_rejected(self):
        is_valid, msg = validate_input_request("???!!!")
        assert not is_valid
        assert "valid descriptive text" in msg

        is_valid, msg = validate_input_request("123456789")
        assert not is_valid
        assert "valid descriptive text" in msg

    def test_valid_request_accepted(self):
        is_valid, msg = validate_input_request("Looking for a gaming headset under $100")
        assert is_valid
        assert msg is None


class TestOutputGuardrails:
    """Tests for validating structured outputs from LLM."""

    def test_valid_pydantic_instance_passes(self, sample_structured_output):
        validated = validate_structured_output(sample_structured_output)
        assert validated.product_category == "Laptop"
        assert validated.budget.max_amount == 1200.0

    def test_valid_dict_passes(self):
        data = {
            "product_category": "Headphones",
            "budget": {"max_amount": 250.0, "is_specified": True},
            "required_features": ["ANC"],
            "preferences": [],
            "priorities": [],
            "is_clear": True,
            "summary": "Noise cancelling headphones.",
        }
        validated = validate_structured_output(data)
        assert validated.product_category == "Headphones"
        assert validated.budget.max_amount == 250.0

    def test_none_output_raises_guardrail_error(self):
        with pytest.raises(GuardrailValidationError) as excinfo:
            validate_structured_output(None)
        assert "empty (None)" in str(excinfo.value)

    def test_unsupported_type_raises_guardrail_error(self):
        with pytest.raises(GuardrailValidationError) as excinfo:
            validate_structured_output([1, 2, 3])
        assert "Unsupported structured output type" in str(excinfo.value)


class TestRequirementCompleteness:
    """Tests for evaluating requirement sufficiency."""

    def test_complete_requirements_pass(self, sample_structured_output):
        is_sufficient, issues = check_requirement_completeness(sample_structured_output)
        assert is_sufficient
        assert len(issues) == 0

    def test_missing_category_fails(self):
        output = RequirementAnalysisOutput(
            product_category=None,
            budget=BudgetInfo(is_specified=False),
            is_clear=True,
        )
        is_sufficient, issues = check_requirement_completeness(output)
        assert not is_sufficient
        assert any("category" in issue.lower() for issue in issues)

    def test_unclear_flag_fails(self):
        output = RequirementAnalysisOutput(
            product_category="Monitor",
            is_clear=False,
            ambiguities_or_missing_info=["Missing screen resolution and refresh rate"],
        )
        is_sufficient, issues = check_requirement_completeness(output)
        assert not is_sufficient
        assert "Missing screen resolution and refresh rate" in issues

    def test_missing_budget_handled_gracefully(self):
        """Missing budget should NOT mark requirements as incomplete."""
        output = RequirementAnalysisOutput(
            product_category="Mechanical Keyboard",
            budget=BudgetInfo(is_specified=False, is_flexible=True),
            required_features=["Tactile switches", "Wireless"],
            is_clear=True,
        )
        is_sufficient, issues = check_requirement_completeness(output)
        assert is_sufficient
        assert len(issues) == 0
