"""
Unit and Integration Tests for Phase 4: Agent Decision + Continue / Finish Flow.
Verifies autonomous multi-step execution, single-step termination, direct answers, and error recovery.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent import run_agent


class TestAgentDecisions(unittest.TestCase):
    def test_1_no_tool_decision(self):
        """Verify agent answers non-tool questions directly with an immediate FINISH decision."""
        result = run_agent("Reply with exactly the word 'READY'.")
        self.assertTrue(result["success"])
        self.assertFalse(result["tool_calls_made"])
        self.assertEqual(result["total_steps"], 0)
        self.assertIn("FINISH", result["decision_flow"][0])
        self.assertIn("READY", result["final_response"])
        print(f"\n[Test 1 No-Tool Decision Flow]: {result['decision_flow']}")
        print(f"[Test 1 Final Response]: {result['final_response']}")

    def test_2_single_tool_decision_finish(self):
        """Verify simple calculations execute one tool, observe the result, and immediately FINISH."""
        result = run_agent("What is 25 * 20?")
        self.assertTrue(result["success"])
        self.assertTrue(result["tool_calls_made"])
        self.assertEqual(result["total_steps"], 1)

        # Step 1 must have decided to FINISH
        step1 = result["steps"][0]
        self.assertEqual(step1["tool"], "calculate")
        self.assertEqual(step1["observation"], "500")
        self.assertEqual(step1["decision"], "FINISH")
        self.assertEqual(result["decision_flow"], ["FINISH"])
        self.assertIn("500", result["final_response"])
        print(f"\n[Test 2 Single-Step Decision Flow]: {result['decision_flow']}")
        print(f"[Test 2 Final Response]: {result['final_response']}")

    def test_3_multi_step_continue_finish(self):
        """
        Verify multi-step goal requiring date lookup first, then arithmetic:
        Tool 1 (datetime) -> Observation -> CONTINUE -> Tool 2 (calculate) -> Observation -> FINISH.
        """
        prompt = (
            "First look up the current date and time to find what the current year is. "
            "Then calculate what that year plus 50 will be."
        )
        result = run_agent(prompt)
        self.assertTrue(result["success"])
        self.assertTrue(result["tool_calls_made"])
        self.assertGreaterEqual(result["total_steps"], 2, "Multi-step task should execute at least 2 steps")

        # Verify CONTINUE followed by FINISH
        decisions = [s["decision"] for s in result["steps"]]
        self.assertIn("CONTINUE", decisions[:-1], "Initial step(s) must have decided to CONTINUE")
        self.assertEqual(decisions[-1], "FINISH", "Last step must decide to FINISH")

        tools_called = [s["tool"] for s in result["steps"]]
        self.assertIn("get_current_datetime", tools_called)
        self.assertIn("calculate", tools_called)

        print(f"\n[Test 3 Multi-Step Decision Flow]: {result['decision_flow']}")
        print(f"[Test 3 Tools Sequence]: {tools_called}")
        print(f"[Test 3 Final Response]: {result['final_response']}")

    def test_4_error_handling_graceful_finish(self):
        """Verify tool errors (e.g. division by zero) are observed and agent decides FINISH with explanation."""
        result = run_agent("Calculate 1000 / 0")
        self.assertTrue(result["success"])
        self.assertTrue(result["tool_calls_made"])

        # Tool was called and returned an error observation
        step1 = result["steps"][0]
        self.assertEqual(step1["tool"], "calculate")
        self.assertIn("Division by zero", step1["observation"])
        self.assertEqual(step1["decision"], "FINISH")

        # Agent formulated helpful response explaining the division by zero error
        self.assertTrue(
            "zero" in result["final_response"].lower() or "division" in result["final_response"].lower()
        )
        print(f"\n[Test 4 Error Recovery Decision]: {step1['decision']}")
        print(f"[Test 4 Final Response]: {result['final_response']}")


if __name__ == "__main__":
    unittest.main()
