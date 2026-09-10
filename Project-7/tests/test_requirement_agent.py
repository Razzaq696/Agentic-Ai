"""Unit tests for the RequirementAnalysisAgent."""

from src.agents.requirement_agent import RequirementAnalysisAgent
from src.llm.factory import MockRequirementLLM
from src.models.schemas import BudgetInfo, RequirementAnalysisOutput


class TestRequirementAnalysisAgent:
    """Tests for agent parsing and resilience."""

    def test_successful_analysis(self, requirement_agent):
        request = "I want a mechanical keyboard with tactile switches for typing under $150"
        output = requirement_agent.analyze(request)

        assert isinstance(output, RequirementAnalysisOutput)
        assert output.product_category == "Mechanical Keyboard"
        assert output.budget.is_specified is True
        assert output.budget.max_amount == 150.0
        assert "Mechanical switches" in output.required_features

    def test_llm_failure_resilience(self, failing_llm):
        agent = RequirementAnalysisAgent(llm=failing_llm)
        output = agent.analyze("Find me running shoes")

        # Agent should not crash; it should return a safe failure schema
        assert isinstance(output, RequirementAnalysisOutput)
        assert output.product_category is None
        assert output.is_clear is False
        assert any("Error" in issue for issue in output.ambiguities_or_missing_info)

    def test_forced_structured_output(self):
        custom_output = RequirementAnalysisOutput(
            product_category="Espresso Machine",
            budget=BudgetInfo(max_amount=600.0, is_specified=True),
            required_features=["15 bar pump", "Steam wand"],
            is_clear=True,
        )
        mock_llm = MockRequirementLLM(forced_output=custom_output)
        agent = RequirementAnalysisAgent(llm=mock_llm)
        output = agent.analyze("Any coffee maker query")

        assert output.product_category == "Espresso Machine"
        assert output.budget.max_amount == 600.0
        assert "Steam wand" in output.required_features
