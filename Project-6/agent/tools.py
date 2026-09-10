"""Tools and API execution module with structured returns."""

import ast
import operator
import re
from datetime import datetime
from typing import Dict, Any, Optional

# Supported operators for safe AST arithmetic evaluation
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_ast_node(node):
    """Recursively evaluates an AST arithmetic expression node."""
    if isinstance(node, ast.Constant):  # Numbers
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):  # Binary operations (+, -, *, /, etc.)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            left = _eval_ast_node(node.left)
            right = _eval_ast_node(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero is not allowed.")
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):  # Unary operations (-5, +3)
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            operand = _eval_ast_node(node.operand)
            return _ALLOWED_OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type}")
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node)}")


def calculate_expression(expression_str: str) -> Dict[str, Any]:
    """Safely evaluates an arithmetic expression using AST parsing.

    Args:
        expression_str: Math expression string (e.g. '125 × 8', '100 / 4').

    Returns:
        Structured tool result dictionary.
    """
    cleaned = expression_str.replace("×", "*").replace("÷", "/").replace("^", "**")
    # Extract only the mathematical portion
    math_match = re.search(r"[-+]?\s*\(?\s*\d+(?:\.\d+)?(?:[\s\+\-\*\/\%\(\)]+\d+(?:\.\d+)?)*\)?", cleaned)
    if not math_match:
        return {
            "tool_name": "calculator",
            "status": "FAILURE",
            "result": None,
            "error": f"Could not extract a valid mathematical expression from '{expression_str}'.",
        }

    target_expr = math_match.group(0).strip()

    try:
        parsed_tree = ast.parse(target_expr, mode="eval")
        calc_result = _eval_ast_node(parsed_tree.body)

        # Format integer results neatly
        if isinstance(calc_result, float) and calc_result.is_integer():
            formatted_result = str(int(calc_result))
        else:
            formatted_result = f"{calc_result:,}" if isinstance(calc_result, (int, float)) else str(calc_result)

        return {
            "tool_name": "calculator",
            "status": "SUCCESS",
            "result": formatted_result,
            "error": None,
        }
    except ZeroDivisionError as zde:
        return {
            "tool_name": "calculator",
            "status": "FAILURE",
            "result": None,
            "error": str(zde),
        }
    except Exception as e:
        return {
            "tool_name": "calculator",
            "status": "FAILURE",
            "result": None,
            "error": f"Calculation error: {str(e)}",
        }


def get_current_datetime() -> Dict[str, Any]:
    """Returns the current date and time."""
    now = datetime.now()
    return {
        "tool_name": "datetime",
        "status": "SUCCESS",
        "result": now.strftime("%Y-%m-%d %H:%M:%S (%A)"),
        "error": None,
    }


def execute_tool_by_query(query: str) -> Dict[str, Any]:
    """Selects and executes the appropriate tool based on user query intent."""
    cleaned = query.lower()

    if any(k in cleaned for k in ["time", "date", "clock", "today", "now"]):
        return get_current_datetime()

    # Default to arithmetic calculator
    return calculate_expression(query)
