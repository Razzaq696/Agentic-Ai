"""End-to-end functional tests for the LangGraph shopping workflow."""

import pytest
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.llm.factory import MockRequirementLLM
from src.models.schemas import BudgetInfo, NextAction, RequirementAnalysisOutput
from src.workflow.graph import build_shopping_graph


class TestLangGraphShoppingWorkflow:
    """End-to-end tests verifying the 6 required scenarios in Phase 1."""

    def test_normal_shopping_request(self, default_workflow):
        """Scenario 1: Normal shopping request with product, specs, and preferences."""
        request = "Looking for a mechanical keyboard with tactile switches for daily coding"
        state = {"user_request": request}

        result = default_workflow.invoke(state)

        assert result["next_action"] == NextAction.PROCEED_TO_PRODUCT_RESEARCH.value
        assert result["product_category"] == "Mechanical Keyboard"
        assert "Mechanical switches" in result["requirements"]
        assert result["interpreted_requirements"] is not None
        assert "Ready to proceed to product research" in result["final_response"]

    def test_request_with_budget(self, default_workflow):
        """Scenario 2: Shopping request with explicit budget constraint."""
        request = "I need a gaming laptop under $1500 with 16GB RAM"
        state = {"user_request": request}

        result = default_workflow.invoke(state)

        assert result["next_action"] == NextAction.PROCEED_TO_PRODUCT_RESEARCH.value
        assert result["product_category"] == "Laptop"
        assert result["budget"]["is_specified"] is True
        assert result["budget"]["max_amount"] == 1500.0
        assert "$1,500" in result["final_response"]

    def test_request_without_budget(self, default_workflow):
        """Scenario 3: Shopping request without budget is handled gracefully (not blocked)."""
        request = "Recommend high-end noise cancelling headphones for long flights"
        state = {"user_request": request}

        result = default_workflow.invoke(state)

        # Missing budget should NOT block product research
        assert result["next_action"] == NextAction.PROCEED_TO_PRODUCT_RESEARCH.value
        assert result["product_category"] == "Headphones"
        assert result["budget"]["is_specified"] is False
        assert result["budget"]["is_flexible"] is True
        assert "Flexible / Market pricing" in result["final_response"]

    def test_incomplete_or_ambiguous_request(self, default_workflow):
        """Scenario 4: Incomplete/ambiguous request leads to REQUEST_CLARIFICATION."""
        request = "I want to buy something good"
        state = {"user_request": request}

        result = default_workflow.invoke(state)

        assert result["next_action"] == NextAction.REQUEST_CLARIFICATION.value
        assert result["product_category"] is None
        assert "specify the exact product or category" in result["final_response"]

    def test_invalid_or_empty_input(self, default_workflow):
        """Scenario 5: Empty/invalid input is rejected by guardrails at Node 1."""
        for invalid_input in ["", "     ", "???", "12345"]:
            state = {"user_request": invalid_input}
            result = default_workflow.invoke(state)

            assert result["next_action"] == NextAction.INVALID_INPUT.value
            assert len(result["validation_errors"]) > 0
            assert "Input validation failed" in result["final_response"]

    def test_malformed_llm_structured_output(self):
        """Scenario 6: Malformed LLM output or failure recovered gracefully without workflow crash."""
        failing_mock = MockRequirementLLM(should_fail=True)
        agent = RequirementAnalysisAgent(llm=failing_mock)
        workflow = build_shopping_graph(agent=agent)

        state = {"user_request": "I want a new laptop for work"}
        result = workflow.invoke(state)

        # The workflow finishes gracefully and asks for clarification/reports the issue
        assert result["next_action"] == NextAction.REQUEST_CLARIFICATION.value
        assert result["product_category"] is None
        assert "final_response" in result
