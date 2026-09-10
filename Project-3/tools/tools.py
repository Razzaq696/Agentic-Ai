import ast
import operator
from typing import Union, Dict
from langchain_core.tools import tool

# --- Safe Arithmetic Calculator Tool ---

_SAFE_OPERATORS = {
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

def _eval_node(node: ast.AST) -> Union[int, float]:
    """Recursively evaluates an AST node safely without calling eval()."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            return _SAFE_OPERATORS[op_type](_eval_node(node.operand))
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _SAFE_OPERATORS:
            left_val = _eval_node(node.left)
            right_val = _eval_node(node.right)
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right_val == 0:
                raise ZeroDivisionError("Division or modulo by zero is not allowed.")
            return _SAFE_OPERATORS[op_type](left_val, right_val)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    else:
        raise ValueError(f"Unsafe or unsupported syntax element: {type(node).__name__}")

def safe_eval_math(expression: str) -> str:
    """Safely evaluates a basic mathematical string expression."""
    clean_expr = expression.strip()
    if not clean_expr:
        return "Error: Empty expression provided."
    try:
        parsed = ast.parse(clean_expr, mode="eval")
        result = _eval_node(parsed)
        # Format clean integer representation if whole number
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError as zde:
        return f"Error: {str(zde)}"
    except Exception as e:
        return f"Error: Invalid arithmetic expression '{clean_expr}'. Details: {str(e)}"

@tool
def calculate(expression: str) -> str:
    """
    Safely calculates the result of a mathematical expression.
    Supports basic arithmetic operations: addition (+), subtraction (-),
    multiplication (*), division (/), floor division (//), modulo (%), and exponentiation (**).
    Does NOT use eval() and strictly rejects non-arithmetic syntax.

    Args:
        expression: The mathematical expression string to calculate (e.g. '15 * 24' or '3 * 24.00').
    """
    return safe_eval_math(expression)


# --- Deterministic Research & Data Lookup Tool ---

RESEARCH_DATA: Dict[str, str] = {
    "python vs javascript": (
        "Comparison: Python vs JavaScript for Beginner Web Development:\n"
        "- Python: Renowned for clean, readable syntax and a very gentle learning curve. "
        "Excellent for backend web development (frameworks: Django, FastAPI, Flask), AI, and data processing. "
        "Cannot run natively in web browsers without specialized runtimes.\n"
        "- JavaScript: The standard language of web browsers, essential for interactive frontend web development. "
        "Supports full-stack development via Node.js backend. Faster visual feedback for web apps, but syntax "
        "and asynchronous paradigms can have a steeper initial learning curve for absolute beginners.\n"
        "- Recommendation: JavaScript is essential for interactive front-end browsers, while Python is often preferred for beginners starting with backend logic and data services."
    ),
    "pricing catalog": (
        "Product Pricing Catalog:\n"
        "- Standard Widget: Average price is $24.00 per unit.\n"
        "- Premium Widget: Average price is $48.00 per unit.\n"
        "- Basic Component: Unit cost is $12.00 per unit.\n"
        "Summary: The overall average price across standard catalog items is $24.00 per unit."
    ),
    "average price": (
        "Price Information:\n"
        "- Standard Widget unit price: $24.00\n"
        "- General catalog benchmark average price: $24.00 per unit.\n"
        "- Quantity calculations should use the standard unit price of $24.00 unless specified otherwise."
    ),
    "python": (
        "Python Overview: High-level language known for clear syntax, readability, and versatile libraries. "
        "Popular in backend web development (Django, FastAPI, Flask), data science, machine learning, and automation."
    ),
    "javascript": (
        "JavaScript Overview: Dynamic scripting language native to all web browsers. "
        "Enables interactive frontend interfaces, single-page applications, and server-side runtimes via Node.js."
    ),
    "web development": (
        "Web Development Overview:\n"
        "- Frontend: Built using HTML, CSS, and JavaScript for UI interactivity.\n"
        "- Backend: Handles database management, business logic, and APIs using languages like Python (Django/FastAPI) or JavaScript (Node.js)."
    ),
}

@tool
def research_lookup(query: str) -> str:
    """
    Deterministic research and data lookup tool.
    Retrieves accurate factual data on technical comparisons (e.g. Python vs JavaScript)
    and product pricing information (e.g. average prices, widget costs).

    Args:
        query: The research topic or search query string (e.g. 'Python vs JavaScript', 'average price', 'pricing catalog').
    """
    q = query.strip().lower()
    if not q:
        return "Error: Empty query provided to research lookup."

    # Multi-term priority check
    if ("python" in q and "javascript" in q) or "compare python" in q:
        return RESEARCH_DATA["python vs javascript"]
    if "average price" in q or "price information" in q:
        return RESEARCH_DATA["average price"]
    if "catalog" in q or "pricing" in q or "widget" in q:
        return RESEARCH_DATA["pricing catalog"]

    # Exact key match (longest key first)
    sorted_keys = sorted(RESEARCH_DATA.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in q:
            return RESEARCH_DATA[key]

    # Token overlap match
    query_tokens = set(q.replace("-", " ").replace("_", " ").replace(".", " ").split())
    best_match = None
    max_overlap = 0

    for key, data in RESEARCH_DATA.items():
        key_tokens = set(key.split())
        overlap = len(query_tokens.intersection(key_tokens))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = data

    if best_match and max_overlap > 0:
        return best_match

    available_topics = ", ".join([f"'{k}'" for k in RESEARCH_DATA.keys()])
    return (
        f"No direct data found for '{query}'. "
        f"Available research topics in catalog: {available_topics}."
    )
