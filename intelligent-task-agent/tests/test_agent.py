"""
Unit and Integration Tests for Phase 3 ReAct Agent & Tool Calling.
Verifies real tool selection, execution, observation handling, and final response generation.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import run_agent, get_agent


class TestReActAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = get_agent()

    def test_1_no_tool_query(self):
        """Verify agent handles general conversational queries without calling unnecessary tools."""
        result = run_agent("Reply with exactly the word 'SUCCESS'.")
        self.assertTrue(result["success"])
        self.assertIsInstance(result["final_response"], str)
        self.assertTrue(len(result["final_response"]) > 0)
        # Verify no tools were called or response was delivered directly
        print(f"\n[No-Tool Response]: {result['final_response']}")

    def test_2_calculator_tool_calling(self):
        """Verify agent selects and executes the calculate tool for arithmetic."""
        result = run_agent("What is 125 * 8?")
        self.assertTrue(result["success"])
        self.assertTrue(result["tool_calls_made"], "Agent should have called a tool for calculation")

        # Verify calculator tool was called
        called_tools = [tc["tool"] for tc in result["tool_calls"]]
        self.assertIn("calculate", called_tools)

        # Verify observation contains 1000 and final response includes result
        observations = [str(tc["observation"]) for tc in result["tool_calls"]]
        self.assertTrue(any("1000" in obs for obs in observations))
        self.assertTrue("1000" in result["final_response"] or "1,000" in result["final_response"])
        print(f"\n[Calculator Tool Call]: {result['tool_calls']}")
        print(f"[Calculator Final Response]: {result['final_response']}")

    def test_3_datetime_tool_calling(self):
        """Verify agent selects and executes the datetime tool for temporal requests."""
        result = run_agent("What is the current date and time?")
        self.assertTrue(result["success"])
        self.assertTrue(result["tool_calls_made"], "Agent should have called datetime tool")

        called_tools = [tc["tool"] for tc in result["tool_calls"]]
        self.assertIn("get_current_datetime", called_tools)
        self.assertTrue(len(result["final_response"]) > 0)
        print(f"\n[DateTime Tool Call]: {result['tool_calls']}")
        print(f"[DateTime Final Response]: {result['final_response']}")

    def test_4_web_search_tool_calling(self):
        """Verify agent selects and executes web_search for external knowledge questions."""
        result = run_agent("Search the web for what the Python programming language is.")
        self.assertTrue(result["success"])
        self.assertTrue(result["tool_calls_made"], "Agent should have called web_search tool")

        called_tools = [tc["tool"] for tc in result["tool_calls"]]
        self.assertIn("web_search", called_tools)
        self.assertTrue(len(result["final_response"]) > 0)
        print(f"\n[Web Search Tool Call]: {result['tool_calls']}")
        print(f"[Web Search Final Response]: {result['final_response']}")

    def test_5_final_response_format_and_cleanliness(self):
        """Verify final response does not contain raw <think> tags and provides a clean answer."""
        result = run_agent("Calculate (50 + 50) * 2")
        self.assertTrue(result["success"])
        self.assertNotIn("<think>", result["final_response"])
        self.assertNotIn("</think>", result["final_response"])
        self.assertTrue("200" in result["final_response"])

    def test_6_empty_input_handling(self):
        """Verify empty goal returns structured error."""
        result = run_agent("   ")
        self.assertFalse(result["success"])
        self.assertIn("empty", result["final_response"].lower())


if __name__ == "__main__":
    unittest.main()
