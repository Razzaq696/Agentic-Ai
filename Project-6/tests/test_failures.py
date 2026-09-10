"""Tests focusing on failure handling, RAG edge cases, and validation."""

import pytest
from agent.runner import run_agent
from agent.validation import validate_execution_result, ProcessedResultSchema
from agent.nodes import rag_knowledge_node


class TestFailureSimulations:
    """Test suite ensuring that component failures do not crash the LangGraph workflow."""

    def test_direct_validation_failure_on_empty_result(self):
        """Validation fails when success is expected but result is empty."""
        processed = {
            "success": True,
            "source": "LLM_REASONING",
            "result": "",
            "error": None,
            "raw_data": None,
        }
        status, err = validate_execution_result(processed, "LLM_REASONING")
        assert status == "INVALID"
        assert "empty content" in err

    def test_direct_validation_failure_on_rag_empty_context(self):
        """Validation fails when RAG branch retrieves 0 documents."""
        processed = {
            "success": True,
            "source": "RAG_KNOWLEDGE",
            "result": "Some response",
            "error": None,
            "raw_data": None,
        }
        status, err = validate_execution_result(processed, "RAG_KNOWLEDGE", retrieved_context=[])
        assert status == "INVALID"
        assert "no relevant knowledge" in err

    def test_direct_validation_failure_on_tool_status_failure(self):
        """Validation catches tool failure in raw_data payload."""
        processed = {
            "success": True,
            "source": "TOOL_API",
            "result": "None",
            "error": None,
            "raw_data": {"status": "FAILURE", "error": "Hardware fault"},
        }
        status, err = validate_execution_result(processed, "TOOL_API")
        assert status == "INVALID"
        assert "Hardware fault" in err

    def test_rag_node_handles_exception_safely(self, monkeypatch):
        """Simulate unexpected database error in ChromaDB retriever."""
        import agent.nodes

        def mock_retrieve_err(query, top_k=2):
            raise RuntimeError("Database connection timed out")

        monkeypatch.setattr("agent.nodes.retrieve_knowledge", mock_retrieve_err)

        state = {"user_query": "What is the knowledge base?"}
        res = agent.nodes.rag_knowledge_node(state)
        assert res["result"] is None
        assert "RAG knowledge retrieval error" in res["error"]
