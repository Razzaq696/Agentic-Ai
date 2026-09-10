"""Comprehensive test suite for Project 6 LangGraph Agent (Phase 1 & Phase 2)."""

import pytest
from agent.runner import run_agent
from agent.graph import create_agent_graph, agent_app
from agent.analyzer import (
    classify_query,
    ACTION_LLM_REASONING,
    ACTION_RAG_KNOWLEDGE,
    ACTION_TOOL_API,
    ACTION_ERROR,
)
from agent.tools import calculate_expression, get_current_datetime
from agent.rag import retrieve_knowledge


class TestLangGraphStructure:
    """Verifies that the agent uses LangGraph with all required Phase 2 nodes and edges."""

    def test_graph_nodes_exist(self):
        """Verify all required Phase 2 nodes exist in the LangGraph workflow."""
        app = create_agent_graph()
        node_keys = set(app.get_graph().nodes.keys())
        expected_nodes = {
            "__start__",
            "analyze_query",
            "decide_action",
            "llm_reasoning",
            "rag_knowledge",
            "tools_api",
            "process_result",
            "validate_result",
            "generate_final_reply",
            "__end__",
        }
        for node in expected_nodes:
            assert node in node_keys, f"Expected node '{node}' not found in graph nodes: {node_keys}"

    def test_graph_compiles_successfully(self):
        """Ensure graph compiles and is runnable."""
        assert agent_app is not None


class TestPhase2CoreRoutes:
    """Verifies the core functional execution paths for LLM, RAG, and Tools."""

    def test_llm_reasoning_route(self):
        """TEST 1 — LLM REASONING
        Input: Explain why Python is commonly used in AI development.
        Expected:
            Analyze -> LLM_REASONING
            Decide Action -> LLM_REASONING
            LLM Reasoning executed
            Process Result -> normalized
            Validate Result -> VALID
            Generate Final Reply delivers response
        """
        query = "Explain why Python is commonly used in AI development."
        result = run_agent(query)

        assert result["query_analysis"] == ACTION_LLM_REASONING
        assert result["selected_action"] == ACTION_LLM_REASONING
        assert result["processed_result"]["success"] is True
        assert result["processed_result"]["source"] == ACTION_LLM_REASONING
        assert result["validation_status"] == "VALID"
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 0
        assert "python" in result["final_response"].lower()

    def test_rag_knowledge_route(self):
        """TEST 2 — RAG KNOWLEDGE
        Input: What are the core features of the Telegram Agentic AI Assistant?
        Expected:
            Analyze -> RAG_KNOWLEDGE
            Decide Action -> RAG_KNOWLEDGE
            Retriever retrieves relevant chunks
            LLM generates grounded response
            Process Result -> normalized
            Validate Result -> VALID
            Generate Final Reply delivers grounded response
        """
        query = "What are the core features and knowledge base of the Telegram Agentic AI Assistant?"
        result = run_agent(query)

        assert result["query_analysis"] == ACTION_RAG_KNOWLEDGE
        assert result["selected_action"] == ACTION_RAG_KNOWLEDGE
        assert result["retrieved_context"] is not None
        assert len(result["retrieved_context"]) > 0
        assert result["processed_result"]["success"] is True
        assert result["processed_result"]["source"] == ACTION_RAG_KNOWLEDGE
        assert result["validation_status"] == "VALID"
        assert result["final_response"] is not None
        assert "knowledge" in result["final_response"].lower() or "assistant" in result["final_response"].lower()

    def test_tool_calculator_route(self):
        """TEST 3 — TOOL CALCULATION ROUTE
        Input: Calculate 125 × 8.
        Expected:
            Analyze -> TOOL_API
            Decide Action -> TOOL_API
            Calculator tool executed safely
            Process Result -> normalized
            Validate Result -> VALID
            Generate Final Reply delivers formatted calculation
        """
        query = "Calculate 125 × 8."
        result = run_agent(query)

        assert result["query_analysis"] == ACTION_TOOL_API
        assert result["selected_action"] == ACTION_TOOL_API
        assert result["tool_result"]["status"] == "SUCCESS"
        assert result["tool_result"]["result"] == "1000" or result["tool_result"]["result"] == "1,000"
        assert result["processed_result"]["success"] is True
        assert result["validation_status"] == "VALID"
        assert "1000" in result["final_response"] or "1,000" in result["final_response"]


class TestFailureAndEdgeCases:
    """Verifies graceful failure handling for Tool errors, RAG misses, and invalid inputs."""

    def test_tool_division_by_zero_failure(self):
        """Simulate tool failure (division by zero). Workflow must not crash."""
        query = "Calculate 10 / 0"
        result = run_agent(query)

        assert result["query_analysis"] == ACTION_TOOL_API
        assert result["selected_action"] == ACTION_TOOL_API
        assert result["tool_result"]["status"] == "FAILURE"
        assert "zero" in result["tool_result"]["error"].lower()
        assert result["processed_result"]["success"] is False
        assert result["validation_status"] == "INVALID"
        assert "division by zero" in result["final_response"].lower() or "error" in result["final_response"].lower()

    def test_tool_invalid_math_expression(self):
        """Simulate calculation error with non-arithmetic syntax."""
        query = "Calculate ??? +++"
        result = run_agent(query)

        assert result["query_analysis"] == ACTION_TOOL_API
        assert result["tool_result"]["status"] == "FAILURE"
        assert result["validation_status"] == "INVALID"
        assert "couldn't complete" in result["final_response"].lower() or "error" in result["final_response"].lower()

    def test_empty_query_error(self):
        """Empty string input should be normalized and return clear error message."""
        result = run_agent("")

        assert result["query_analysis"] == ACTION_ERROR
        assert result["selected_action"] == ACTION_ERROR
        assert result["processed_result"]["success"] is False
        assert result["validation_status"] == "INVALID"
        assert "empty" in result["final_response"].lower() or "error" in result["final_response"].lower()

    def test_whitespace_query_error(self):
        """Whitespace query should be normalized and return clear error message."""
        result = run_agent("    ")

        assert result["query_analysis"] == ACTION_ERROR
        assert result["selected_action"] == ACTION_ERROR
        assert result["validation_status"] == "INVALID"


class TestQueryRoutingDecision:
    """Verifies that the agent dynamically routes queries based on intent before branch execution."""

    def test_routing_diversity(self):
        """Confirm normal reasoning, knowledge queries, and tools route to distinct branches."""
        reasoning_res = run_agent("Explain why functional programming is useful.")
        assert reasoning_res["selected_action"] == ACTION_LLM_REASONING

        knowledge_res = run_agent("According to the project knowledge base, what is the architecture?")
        assert knowledge_res["selected_action"] == ACTION_RAG_KNOWLEDGE

        tool_res = run_agent("Calculate (50 + 25) * 4")
        assert tool_res["selected_action"] == ACTION_TOOL_API
