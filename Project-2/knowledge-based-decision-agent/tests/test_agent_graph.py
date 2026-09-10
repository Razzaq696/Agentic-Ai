"""Phase 2 Agent + LangGraph Test Suite."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from agent.state import AgentState
from agent.graph import build_retrieval_graph, run_retrieval_graph, analyze_query, retrieve_context_node


class TestAgentGraph(unittest.TestCase):
    """Test suite for Phase 2 LangGraph agent workflow."""

    def test_01_graph_construction(self):
        """Test 1: Verify LangGraph compiles and contains required nodes and edges."""
        graph = build_retrieval_graph()
        self.assertIsNotNone(graph, "Compiled graph should not be None.")
        # Verify node names in the graph
        nodes = graph.nodes
        self.assertIn("analyze_query", nodes)
        self.assertIn("retrieve_context", nodes)
        print("\n[PASS] LangGraph constructed and compiled successfully with nodes: analyze_query, retrieve_context.")

    def test_02_attendance_query_flow(self):
        """Test 2: Verify attendance query execution through LangGraph."""
        user_query = "What are the attendance requirements for students?"
        result = run_retrieval_graph(user_query)

        self.assertEqual(result["query"], user_query)
        self.assertTrue(bool(result["retrieval_query"]), "retrieval_query must not be empty.")
        self.assertGreater(len(result["context"]), 0, "Retrieved context must not be empty.")

        top_sources = [c["source"] for c in result["context"]]
        self.assertIn("attendance_policy.txt", top_sources)
        self.assertEqual(result["context"][0]["source"], "attendance_policy.txt")
        print(f"[PASS] Attendance query flow: query='{user_query}' -> retrieval_query='{result['retrieval_query']}' -> top source='{top_sources[0]}'")

    def test_03_scholarship_query_flow(self):
        """Test 3: Verify scholarship query execution through LangGraph."""
        user_query = "Who is eligible for the university merit scholarship?"
        result = run_retrieval_graph(user_query)

        self.assertEqual(result["query"], user_query)
        self.assertTrue(bool(result["retrieval_query"]))
        self.assertGreater(len(result["context"]), 0)

        top_sources = [c["source"] for c in result["context"]]
        self.assertIn("scholarship_policy.txt", top_sources)
        self.assertEqual(result["context"][0]["source"], "scholarship_policy.txt")
        print(f"[PASS] Scholarship query flow: query='{user_query}' -> retrieval_query='{result['retrieval_query']}' -> top source='{top_sources[0]}'")

    def test_04_examination_query_flow(self):
        """Test 4: Verify examination query execution through LangGraph."""
        user_query = "What are the rules and regulations for final examinations?"
        result = run_retrieval_graph(user_query)

        self.assertEqual(result["query"], user_query)
        self.assertTrue(bool(result["retrieval_query"]))
        self.assertGreater(len(result["context"]), 0)

        top_sources = [c["source"] for c in result["context"]]
        self.assertIn("examination_policy.txt", top_sources)
        self.assertEqual(result["context"][0]["source"], "examination_policy.txt")
        print(f"[PASS] Examination query flow: query='{user_query}' -> retrieval_query='{result['retrieval_query']}' -> top source='{top_sources[0]}'")

    def test_05_empty_query_handling(self):
        """Test 5: Verify graceful error handling on empty queries."""
        result = run_retrieval_graph("")
        self.assertEqual(result["context"], [])
        self.assertTrue(bool(result["error"]))
        print(f"[PASS] Empty query handled gracefully: error='{result['error']}'")


if __name__ == "__main__":
    unittest.main(verbosity=2)
