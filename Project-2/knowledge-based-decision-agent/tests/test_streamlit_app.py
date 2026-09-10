"""Phase 4 Streamlit App Integration Test Suite."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import unittest
from streamlit.testing.v1 import AppTest


class TestStreamlitApp(unittest.TestCase):
    """Test suite for Phase 4 Streamlit app interface and interactions."""

    def test_01_app_initial_render(self):
        """Test 1: App initializes with title, description, textarea, and action buttons."""
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        at.run(timeout=30)

        self.assertFalse(at.exception)
        self.assertTrue(any("Knowledge-Based Decision Agent" in m.value for m in at.markdown))
        self.assertEqual(len(at.text_area), 1)
        self.assertTrue(any("Ask Decision Agent" in b.label for b in at.button))
        print("[PASS] Streamlit initial UI components rendered successfully.")

    def test_02_empty_input_validation(self):
        """Test 2: Submitting empty query triggers warning validation message."""
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        at.run(timeout=30)

        ask_btn = next(b for b in at.button if "Ask Decision Agent" in b.label)
        at.text_area[0].input("").run(timeout=30)
        ask_btn.click().run(timeout=30)

        self.assertFalse(at.exception)
        self.assertGreater(len(at.warning), 0)
        self.assertIn("Please enter a question", at.warning[0].value)
        print("[PASS] Empty input validation verified.")

    def test_03_attendance_question_execution(self):
        """Test 3: Querying attendance policy displays answer and sources in UI."""
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        at.run(timeout=30)

        ask_btn = next(b for b in at.button if "Ask Decision Agent" in b.label)
        at.text_area[0].input("What are the attendance requirements?").run(timeout=30)
        ask_btn.click().run(timeout=60)

        self.assertFalse(at.exception)
        markdown_texts = [m.value for m in at.markdown]
        self.assertTrue(any("Final Answer / Decision" in t for t in markdown_texts))
        self.assertTrue(any("Grounded Source References" in t for t in markdown_texts))
        print("[PASS] Attendance question executed in Streamlit UI.")

    def test_04_scholarship_decision_execution(self):
        """Test 4: Querying scholarship decision displays decision and sources in UI."""
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        at.run(timeout=30)

        ask_btn = next(b for b in at.button if "Ask Decision Agent" in b.label)
        at.text_area[0].input("Based on the scholarship policy, am I eligible if my GPA is 3.4?").run(timeout=30)
        ask_btn.click().run(timeout=60)

        self.assertFalse(at.exception)
        markdown_texts = [m.value for m in at.markdown]
        self.assertTrue(any("Final Answer / Decision" in t for t in markdown_texts))
        print("[PASS] Scholarship decision query executed in Streamlit UI.")

    def test_05_unsupported_query_execution(self):
        """Test 5: Querying unsupported out-of-domain question displays unavailable message."""
        at = AppTest.from_file(str(PROJECT_ROOT / "app.py"))
        at.run(timeout=30)

        ask_btn = next(b for b in at.button if "Ask Decision Agent" in b.label)
        at.text_area[0].input("Who won the FIFA World Cup in 2022?").run(timeout=30)
        ask_btn.click().run(timeout=60)

        self.assertFalse(at.exception)
        markdown_texts = [m.value for m in at.markdown]
        self.assertTrue(any("Final Answer / Decision" in t for t in markdown_texts))
        print("[PASS] Unsupported query handled in Streamlit UI.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
