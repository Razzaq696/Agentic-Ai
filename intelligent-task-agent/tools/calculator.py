"""
Calculator Tool for Intelligent Task Execution Agent.
Phase 2: Tool Development

Safely evaluates mathematical expressions using Python's Abstract Syntax Tree (AST).
Prevents code injection by rejecting arbitrary code execution and unrestricted eval.
"""

import ast
import math
import operator
from typing import Any, Union
from langchain_core.tools import tool

# Supported operators mapping
OPERATORS = {
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

# Safe math functions
SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "ceil": math.ceil,
    "floor": math.floor,
}

# Safe math constants
SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_ast(node: ast.AST) -> Union[int, float]:
    """Recursively evaluates an AST node containing mathematical operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)

    # Constant numbers
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    # Unary operations (e.g., -5, +3)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in OPERATORS:
            operand = _safe_eval_ast(node.operand)
            return OPERATORS[op_type](operand)
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")

    # Binary operations (e.g., 2 + 3, 10 * 5)
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in OPERATORS:
            left = _safe_eval_ast(node.left)
            right = _safe_eval_ast(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")

    # Function calls (e.g., sqrt(16), round(3.1415, 2))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            func = SAFE_FUNCTIONS[node.func.id]
            args = [_safe_eval_ast(arg) for arg in node.args]
            return func(*args)
        func_name = getattr(node.func, "id", "unknown")
        raise ValueError(f"Unsupported function call: '{func_name}'")

    # Named constants (e.g., pi, e)
    if isinstance(node, ast.Name):
        if node.id in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[node.id]
        raise ValueError(f"Undefined variable or constant: '{node.id}'")

    raise ValueError(f"Unsupported expression construct: {type(node).__name__}")


@tool
def calculate(expression: str) -> str:
    """
    Perform mathematical calculations and evaluate arithmetic expressions.
    Use this tool whenever you need to compute numbers, perform basic arithmetic
    (addition, subtraction, multiplication, division, powers), or solve math problems.

    Args:
        expression: The mathematical expression to evaluate (e.g., '25 * 8', '(100 + 50) / 2', '2 ** 10').

    Returns:
        The calculated result as a string, or an error message if the expression is invalid.
    """
    if not expression or not expression.strip():
        return "Error: Expression cannot be empty."

    cleaned_expr = expression.strip()

    try:
        parsed_tree = ast.parse(cleaned_expr, mode="eval")
        result = _safe_eval_ast(parsed_tree)

        # Format integer results cleanly if whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)

    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except (SyntaxError, ValueError) as e:
        return f"Error: Invalid mathematical expression '{cleaned_expr}'. ({e})"
    except Exception as e:
        return f"Error: Failed to evaluate expression. ({e})"
