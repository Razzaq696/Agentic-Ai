"""End-to-end integration tests for Phase 4: Multi-Agent Comparison, Scoring, and Decision in LangGraph."""

import pytest
from src.agents.comparison_agent import ProductComparisonAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.agents.validation_agent import ProductValidationAgent
from src.models.schemas import DecisionStatus, NextAction
from src.tools.browser_tool import ProductPageExtractorTool
from src.tools.search_tool import WebSearchTool
from src.workflow.graph import build_shopping_graph


@pytest.fixture
def phase4_workflow(requirement_agent, product_retriever, rag_evaluator):
    """Full workflow configured with all Phase 1-4 agents, tools, and mock search."""
    search_tool = WebSearchTool(provider="mock")
    browser_tool = ProductPageExtractorTool()
    validation_agent = ProductValidationAgent()
    comparison_agent = ProductComparisonAgent()
    decision_agent = DecisionAgent()

    return build_shopping_graph(
        agent=requirement_agent,
        retriever=product_retriever,
        evaluator=rag_evaluator,
        search_tool=search_tool,
        browser_tool=browser_tool,
        validation_agent=validation_agent,
        comparison_agent=comparison_agent,
        decision_agent=decision_agent,
    )


class TestPhase4Workflow:
    """End-to-end LangGraph workflow tests verifying Phase 4 multi-agent decision pipeline."""

    def test_end_to_end_local_rag_to_decision(self, phase4_workflow):
        """Scenario: User requests mechanical keyboard found in local Chroma KB -> validates -> compares -> decides."""
        request = "Looking for a mechanical keyboard with tactile switches for daily coding under $250"
        state = {"user_request": request}

        result = phase4_workflow.invoke(state)

        # Requirements & RAG
        assert result["product_category"] == "Mechanical Keyboard"
        assert result["retrieval_sufficient"] is True
        assert len(result["retrieved_products"]) > 0

        # Phase 4 Multi-Agent State
        assert "validated_products" in result
        assert len(result["validated_products"]) > 0
        assert "comparison_result" in result
        assert "product_scores" in result
        assert len(result["product_scores"]) > 0
        assert "decision_result" in result
        assert "final_decision" in result

        final_dec = result["final_decision"]
        assert final_dec["decision_status"] == DecisionStatus.RECOMMENDED.value
        assert final_dec["recommended_product"] is not None
        assert "Keychron" in final_dec["recommended_product"]["name"] or "Logitech" in final_dec["recommended_product"]["name"]

        # Final response formatting
        assert "AI SMART SHOPPING DECISION REPORT" in result["final_response"]
        assert "[TOP RECOMMENDATION]:" in result["final_response"]
        assert "Overall Score:" in result["final_response"]

    def test_end_to_end_web_research_to_decision(self, phase4_workflow):
        """Scenario: Product absent from local KB (espresso machine) -> triggers web research -> multi-agent decision."""
        request = "I need an espresso machine with dual boiler and PID control under $2000"
        state = {"user_request": request}

        result = phase4_workflow.invoke(state)

        # Web research executed
        assert result["retrieval_sufficient"] is False
        assert result["research_status"] == "complete"
        assert len(result["research_result"]["products"]) > 0

        # Multi-agent decision executed on external products
        final_dec = result["final_decision"]
        assert final_dec["decision_status"] == DecisionStatus.RECOMMENDED.value
        assert final_dec["recommended_product"] is not None
        assert "Espresso" in final_dec["recommended_product"]["name"] or "Breville" in final_dec["recommended_product"]["name"] or "Gaggia" in final_dec["recommended_product"]["name"]

        # Report contains source citation
        assert "Source/Citation:" in result["final_response"]
        assert "Key Advantages:" in result["final_response"]

    def test_end_to_end_impossible_budget_returns_no_satisfying_product(self, phase4_workflow):
        """Scenario: User asks for a MacBook Pro under $50 (impossible budget) -> safely outputs NO_SATISFYING_PRODUCT."""
        request = "Looking for a laptop with 16GB RAM under $50"
        state = {"user_request": request}

        result = phase4_workflow.invoke(state)

        final_dec = result["final_decision"]
        assert final_dec["decision_status"] == DecisionStatus.NO_SATISFYING_PRODUCT.value
        assert final_dec["recommended_product"] is None
        assert final_dec["alternative_product"] is not None  # Closest alternative provided
        assert len(final_dec["unmet_requirements"]) > 0

        # Report clearly explains no product met criteria
        assert "[STATUS]: No product fully satisfies the required criteria" in result["final_response"]
        assert "[CLOSEST AVAILABLE ALTERNATIVE]:" in result["final_response"]
        assert "We do not recommend forcing an unsuitable product" in result["final_response"]

    def test_clarification_preservation_short_circuits_without_decision(self, phase4_workflow):
        """Scenario: Ambiguous input asking for clarification short-circuits at determine_next_action."""
        state = {"user_request": "I want to buy something nice"}
        result = phase4_workflow.invoke(state)

        assert result["next_action"] == NextAction.REQUEST_CLARIFICATION.value
        assert "final_decision" not in result or result.get("final_decision") is None
        assert "AI SMART SHOPPING DECISION REPORT" not in (result.get("final_response") or "")
