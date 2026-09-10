import pytest
from tools.tools import calculate, research_lookup, safe_eval_math

def test_safe_eval_math_basic_arithmetic():
    assert safe_eval_math("15 * 24") == "360"
    assert safe_eval_math("3 * 24.00") == "72"
    assert safe_eval_math("100 + 50 - 25") == "125"
    assert safe_eval_math("10 / 2") == "5"
    assert safe_eval_math("2 ** 3") == "8"

def test_safe_eval_math_division_by_zero():
    res = safe_eval_math("10 / 0")
    assert "Error" in res
    assert "zero" in res.lower()

def test_safe_eval_math_rejects_unsafe_code():
    res = safe_eval_math("__import__('os').system('dir')")
    assert "Error" in res

def test_calculate_tool_invocation():
    result = calculate.invoke({"expression": "15 * 24"})
    assert result == "360"

def test_research_lookup_python_vs_javascript():
    result = research_lookup.invoke({"query": "Compare Python and JavaScript for beginner web development."})
    assert "Python" in result
    assert "JavaScript" in result

def test_research_lookup_pricing():
    result = research_lookup.invoke({"query": "average price information"})
    assert "$24.00" in result

def test_research_lookup_fallback():
    result = research_lookup.invoke({"query": "nonexistent_topic_xyz"})
    assert "No direct data found" in result
