"""
Tools package for Intelligent Task Execution Agent.
Phase 2: Tool Development.

Exposes calculator, web_search, and datetime tools along with the central TOOLS registry.
"""

from tools.calculator import calculate
from tools.web_search import web_search
from tools.datetime_tool import get_current_datetime
from tools.registry import TOOLS, TOOL_MAP, get_all_tools, get_tool_by_name

__all__ = [
    "calculate",
    "web_search",
    "get_current_datetime",
    "TOOLS",
    "TOOL_MAP",
    "get_all_tools",
    "get_tool_by_name",
]
