"""
Automated Streamlit UI Test Suite for Phase 5.
Verifies complete ReAct workflow, input validation, single-step execution,
multi-step execution trace, and final response rendering.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest


class TestStreamlitUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_file = str(PROJECT_ROOT / "app.py")

    def test_1_app_initial_rendering(self):
        """Verify the app loads with all required Phase 5 elements."""
        at = AppTest.from_file(self.app_file, default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App threw an exception on render: {at.exception}")

        # Check text area exists
        self.assertEqual(len(at.text_area), 1)
        self.assertEqual(at.text_area[0].label, "Enter your task")

        # Check Run Agent button exists
        run_buttons = [b for b in at.button if b.label == "Run Agent"]
        self.assertEqual(len(run_buttons), 1)

    def test_2_empty_input_validation(self):
        """Verify submitting empty task displays 'Please enter a task.' warning."""
        at = AppTest.from_file(self.app_file, default_timeout=30)
        at.run()

        # Click Run Agent without entering text
        run_button = [b for b in at.button if b.label == "Run Agent"][0]
        run_button.click().run()

        warnings = [w.value for w in at.warning]
        self.assertTrue(
            any("Please enter a task." in w for w in warnings),
            f"Expected warning not found. Actual warnings: {warnings}",
        )

    def test_3_calculator_execution_flow(self):
        """Verify single-step calculator task displays execution trace and final response."""
        at = AppTest.from_file(self.app_file, default_timeout=120)
        at.run()

        at.text_area[0].input("What is 125 * 8?").run()
        run_button = [b for b in at.button if b.label == "Run Agent"][0]
        run_button.click().run()

        self.assertFalse(at.exception, f"Exception during execution: {at.exception}")

        # Check for execution trace components in markdown
        markdowns = [m.value for m in at.markdown]
        has_tool_selected = any("calculate" in m for m in markdowns)
        self.assertTrue(has_tool_selected, f"Tool selection trace not found. Markdowns: {markdowns}")

        # Check for final response
        has_1000 = any("1000" in m or "1,000" in m for m in markdowns)
        self.assertTrue(has_1000, f"Final response with 1000 not found: {markdowns}")

    def test_4_no_tool_execution_flow(self):
        """Verify no-tool queries display direct reasoning flow and final response."""
        at = AppTest.from_file(self.app_file, default_timeout=60)
        at.run()

        at.text_area[0].input("Reply with exactly the word 'SUCCESS'.").run()
        run_button = [b for b in at.button if b.label == "Run Agent"][0]
        run_button.click().run()

        self.assertFalse(at.exception, f"Exception during execution: {at.exception}")

        # Check info or success badge
        infos = [i.value for i in at.info]
        self.assertTrue(
            any("No external tool was required" in i for i in infos),
            f"Direct reasoning info message not found. Infos: {infos}",
        )

        markdowns = [m.value for m in at.markdown]
        self.assertTrue(any("SUCCESS" in m for m in markdowns))


if __name__ == "__main__":
    unittest.main()
