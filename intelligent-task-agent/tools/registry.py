"""
Tool Registry for Intelligent Task Execution Agent.
Phase 2: Tool Development

Centralizes all available LangChain tools for discovery, registration,
and binding to the LLM agent.
"""

from typing import List, Dict
from langchain_core.tools import BaseTool

from tools.calculator import calculate
from tools.web_search import web_search
from tools.datetime_tool import get_current_datetime

# Centralized collection of registered LangChain tools
TOOLS: List[BaseTool] = [
    calculate,
    web_search,
    get_current_datetime,
]

# Mapping by tool name for easy lookup
TOOL_MAP: Dict[str, BaseTool] = {tool.name: tool for tool in TOOLS}


def get_all_tools() -> List[BaseTool]:
    """Returns the list of all registered LangChain tools."""
    return list(TOOLS)


def get_tool_by_name(name: str) -> BaseTool:
    """Retrieves a registered tool by its name."""
    if name not in TOOL_MAP:
        raise KeyError(f"Tool '{name}' not found in registry. Available tools: {list(TOOL_MAP.keys())}")
    return TOOL_MAP[name]
