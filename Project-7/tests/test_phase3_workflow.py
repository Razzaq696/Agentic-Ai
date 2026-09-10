"""End-to-end tests for Phase 3: Web Search + Playwright + Tool Calling + ReAct in LangGraph."""

import pytest
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.models.schemas import NextAction, ProductResearchResult
from src.tools.browser_tool import ProductPageExtractorTool
from src.tools.search_tool import WebSearchTool
from src.workflow.graph import build_shopping_graph


@pytest.fixture
def phase3_workflow(requirement_agent, product_retriever, rag_evaluator):
    """Workflow configured with mock search and page extractor tools."""
    search_tool = WebSearchTool(provider="mock")
    browser_tool = ProductPageExtractorTool()
    return build_shopping_graph(
        agent=requirement_agent,
        retriever=product_retriever,
        evaluator=rag_evaluator,
        search_tool=search_tool,
        browser_tool=browser_tool,
    )


class TestPhase3Workflow:
    """End-to-end LangGraph tests verifying Phase 3 capabilities and integration."""

    def test_rag_sufficient_skips_web_research(self, phase3_workflow):
        """Scenario 10: When local Chroma retrieval is sufficient, web research is skipped."""
        # This product is well covered in data/products.json
        request = "Looking for a mechanical keyboard with tactile switches for daily coding under $250"
        state = {"user_request": request}

        result = phase3_workflow.invoke(state)

        # Requirements and RAG succeeded
        assert result["product_category"] == "Mechanical Keyboard"
        assert result["retrieval_sufficient"] is True
        assert result["research_needed"] is False

        # Web research was skipped
        assert result.get("research_status") in ("skipped", None)
        assert result.get("search_results") is None or result.get("search_results") == []
        assert "Knowledge Base Matches Found" in result["final_response"]
        assert "External Web Research Results" not in result["final_response"]

    def test_rag_insufficient_triggers_web_research_and_playwright(self, phase3_workflow):
        """Scenario 11: When Chroma retrieval is insufficient, web search & extraction execute."""
        # Espresso machine is absent from the local catalog
        request = "I need an espresso machine with dual boiler and PID control under $2000"
        state = {"user_request": request}

        result = phase3_workflow.invoke(state)

        # Requirements analyzed
        assert result["product_category"] is not None

        # RAG was insufficient, triggering research
        assert result["retrieval_sufficient"] is False
        assert result["research_status"] == "complete"

        # Web search executed
        assert "search_results" in result
        assert len(result["search_results"]) > 0

        # Research result is structured and contains verified products
        assert "research_result" in result
        res_dict = result["research_result"]
        assert len(res_dict["products"]) > 0
        assert res_dict["research_complete"] is True

        # Final response contains external research results with source URLs
        final_resp = result["final_response"]
        assert "External Web Research Results" in final_resp
        assert "Breville" in final_resp or "Gaggia" in final_resp or "Rancilio" in final_resp
        assert "Source URL:" in final_resp

    def test_react_loop_respects_max_iterations(self, phase3_workflow):
        """Scenario 9: ReAct loop terminates within maximum iteration bounds."""
        request = "Looking for an espresso machine"
        state = {"user_request": request}

        result = phase3_workflow.invoke(state)
        assert result.get("react_iterations", 0) <= 2

    def test_clarification_preservation_without_web_research(self, phase3_workflow):
        """Verify ambiguous input routes to clarification without triggering research."""
        request = "I want to buy something good"
        state = {"user_request": request}

        result = phase3_workflow.invoke(state)
        assert result["next_action"] == NextAction.REQUEST_CLARIFICATION.value
        assert result.get("search_results") is None
        assert result.get("research_result") is None

    def test_invalid_input_preservation(self, phase3_workflow):
        """Verify empty input is rejected by guardrails without invoking tools."""
        for invalid in ["", "   ", "???", "12345"]:
            state = {"user_request": invalid}
            result = phase3_workflow.invoke(state)
            assert result["next_action"] == NextAction.INVALID_INPUT.value
            assert result.get("search_results") is None

    def test_structured_research_output_validation(self, phase3_workflow):
        """Scenario 13: Verify output adheres strictly to ProductResearchResult Pydantic schema."""
        request = "I need an espresso machine"
        state = {"user_request": request}

        result = phase3_workflow.invoke(state)
        if result.get("research_result"):
            model = ProductResearchResult.model_validate(result["research_result"])
            assert isinstance(model, ProductResearchResult)
            assert model.research_complete is True
            assert isinstance(model.products, list)
