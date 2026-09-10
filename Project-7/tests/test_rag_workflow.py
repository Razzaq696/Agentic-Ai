"""End-to-end tests verifying LangGraph execution with Agentic RAG in Phase 2."""

import pytest
from src.models.schemas import NextAction, RAGDecision
from src.workflow.graph import build_shopping_graph


class TestLangGraphAgenticRAGWorkflow:
    """End-to-end LangGraph tests verifying Phase 2 requirements."""

    def test_successful_product_retrieval_flow(self, rag_workflow):
        """Verify full flow for a request that has matching products in local knowledge base."""
        request = "Looking for a mechanical keyboard with tactile switches for coding under $250"
        state = {"user_request": request}

        result = rag_workflow.invoke(state)

        # Requirements analysis passed
        assert result["product_category"] == "Mechanical Keyboard"
        assert result["next_action"] == NextAction.PROCEED_TO_PRODUCT_RESEARCH.value

        # RAG evaluation ran
        assert "retrieved_products" in result
        assert len(result["retrieved_products"]) > 0
        assert result["retrieval_sufficient"] is True
        assert result["research_needed"] is False
        assert result["retrieval_decision"] == RAGDecision.SUFFICIENT.value

        # Final response contains knowledge matches and source reference
        final_resp = result["final_response"]
        assert "Knowledge Base Matches Found" in final_resp
        assert "Source:" in final_resp
        assert "Keychron" in final_resp or "Logitech" in final_resp

    def test_product_absent_from_knowledge_base_triggers_external_research(self, rag_workflow):
        """Verify request for product absent from catalog triggers external research flag."""
        request = "I want to buy a high-end espresso machine with dual boiler under $2000"
        state = {"user_request": request}

        result = rag_workflow.invoke(state)

        # Requirements analysis identifies category
        assert result["product_category"] is not None

        # RAG evaluation detects no matching items in local catalog
        assert result["retrieval_sufficient"] is False
        assert result["research_needed"] is True
        assert result["retrieval_decision"] == RAGDecision.EXTERNAL_RESEARCH_NEEDED.value
        assert "External research will be conducted" in result["final_response"]

    def test_clarification_path_preserves_phase1_behavior_without_rag(self, rag_workflow):
        """Verify ambiguous input stops at clarify node without executing RAG retrieval."""
        request = "I want to buy something good"
        state = {"user_request": request}

        result = rag_workflow.invoke(state)

        # Should preserve clarification behavior
        assert result["next_action"] == NextAction.REQUEST_CLARIFICATION.value
        assert result["product_category"] is None
        assert "specify the exact product or category" in result["final_response"]

        # RAG nodes were not executed
        assert result.get("retrieval_sufficient") is None
        assert result.get("retrieved_products") is None

    def test_invalid_input_preserves_guardrail_rejection(self, rag_workflow):
        """Verify invalid input is rejected at Node 1 without running RAG."""
        for bad_query in ["", "   ", "???", "99999"]:
            state = {"user_request": bad_query}
            result = rag_workflow.invoke(state)

            assert result["next_action"] == NextAction.INVALID_INPUT.value
            assert "Input validation failed" in result["final_response"]
            assert result.get("retrieved_products") is None

    def test_retrieval_with_flexible_budget(self, rag_workflow):
        """Verify search without explicit budget still retrieves successfully."""
        request = "I need noise-cancelling headphones for flights"
        state = {"user_request": request}

        result = rag_workflow.invoke(state)

        assert result["product_category"] == "Headphones"
        assert result["retrieval_sufficient"] is True
        assert result["research_needed"] is False
        assert any(brand in result["final_response"] for brand in ["Sennheiser", "Sony", "Bose"])
