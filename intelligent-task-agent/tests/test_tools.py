"""
Unit and Integration Tests for Phase 2 Tools & Registry.
Tests calculator, web search, datetime tool, and LangChain tool-binding compatibility.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import calculate, web_search, get_current_datetime, TOOLS, TOOL_MAP, get_tool_by_name
from agent.llm import get_llm


class TestCalculatorTool(unittest.TestCase):
    def test_addition(self):
        res = calculate.invoke({"expression": "25 + 75"})
        self.assertEqual(res, "100")

    def test_multiplication(self):
        res = calculate.invoke({"expression": "25 * 8"})
        self.assertEqual(res, "200")

    def test_division(self):
        res = calculate.invoke({"expression": "1500 / 3"})
        self.assertEqual(res, "500")

    def test_parentheses_and_precedence(self):
        res = calculate.invoke({"expression": "(20 + 10) * 5"})
        self.assertEqual(res, "150")

    def test_powers_and_functions(self):
        res_pow = calculate.invoke({"expression": "2 ** 8"})
        self.assertEqual(res_pow, "256")
        res_sqrt = calculate.invoke({"expression": "sqrt(144)"})
        self.assertEqual(res_sqrt, "12")

    def test_division_by_zero(self):
        res = calculate.invoke({"expression": "100 / 0"})
        self.assertTrue(res.startswith("Error"), f"Expected error but got: {res}")
        self.assertIn("Division by zero", res)

    def test_invalid_syntax(self):
        res = calculate.invoke({"expression": "25 +* 8"})
        self.assertTrue(res.startswith("Error"), f"Expected error but got: {res}")

    def test_empty_expression(self):
        res = calculate.invoke({"expression": "   "})
        self.assertTrue(res.startswith("Error"), f"Expected error but got: {res}")

    def test_security_rejection_of_arbitrary_code(self):
        res = calculate.invoke({"expression": "__import__('os').system('dir')"})
        self.assertTrue(res.startswith("Error"), "Calculator must not execute arbitrary code")


class TestWebSearchTool(unittest.TestCase):
    def test_normal_query(self):
        res = web_search.invoke({"query": "Python programming language"})
        self.assertIsInstance(res, str)
        self.assertTrue(len(res) > 20)
        self.assertIn("Search Results", res)

    def test_empty_query(self):
        res = web_search.invoke({"query": ""})
        self.assertTrue(res.startswith("Error"), f"Expected error but got: {res}")

    def test_structured_content(self):
        res = web_search.invoke({"query": "Artificial Intelligence"})
        self.assertIsInstance(res, str)
        self.assertTrue("Snippet:" in res or "Search Results" in res)


class TestDateTimeTool(unittest.TestCase):
    def test_default_all_execution(self):
        res = get_current_datetime.invoke({})
        self.assertIsInstance(res, str)
        self.assertIn("Current Date:", res)
        self.assertIn("Current Local Time:", res)
        self.assertIn("UTC Time:", res)

    def test_date_query(self):
        res = get_current_datetime.invoke({"query": "date"})
        self.assertIn("Current Date:", res)

    def test_time_query(self):
        res = get_current_datetime.invoke({"query": "time"})
        self.assertIn("Current Local Time:", res)

    def test_weekday_query(self):
        res = get_current_datetime.invoke({"query": "weekday"})
        self.assertIn("Current Day of the Week:", res)


class TestToolRegistry(unittest.TestCase):
    def test_registry_contains_three_tools(self):
        self.assertEqual(len(TOOLS), 3)

    def test_tool_names(self):
        tool_names = [t.name for t in TOOLS]
        self.assertIn("calculate", tool_names)
        self.assertIn("web_search", tool_names)
        self.assertIn("get_current_datetime", tool_names)

    def test_tool_descriptions_present(self):
        for tool_obj in TOOLS:
            self.assertIsNotNone(tool_obj.description)
            self.assertTrue(len(tool_obj.description.strip()) > 10)

    def test_tool_map_and_lookup(self):
        calc_tool = get_tool_by_name("calculate")
        self.assertEqual(calc_tool.name, "calculate")
        with self.assertRaises(KeyError):
            get_tool_by_name("non_existent_tool")


class TestLangChainToolBindingCompatibility(unittest.TestCase):
    def test_llm_bind_tools(self):
        """Verify tools can be bound to ChatOllama without schema errors."""
        llm = get_llm()
        llm_with_tools = llm.bind_tools(TOOLS)
        self.assertIsNotNone(llm_with_tools)


if __name__ == "__main__":
    unittest.main()
