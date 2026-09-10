"""Tool registry managing tool definitions, function schemas, and dispatching."""

from typing import Any, Callable, Dict, List, Optional
from src.models.schemas import ExtractedProductPageData, WebSearchOutput
from src.models.state import ShoppingState
from src.tools.search_tool import WebSearchTool
from src.tools.browser_tool import ProductPageExtractorTool
from src.utils.logger import logger


class ToolRegistry:
    """Registry coordinating available agent research tools."""

    def __init__(
        self,
        search_tool: Optional[WebSearchTool] = None,
        browser_tool: Optional[ProductPageExtractorTool] = None,
    ):
        """Initialize registry with search and browser tools."""
        self.search_tool = search_tool or WebSearchTool()
        self.browser_tool = browser_tool or ProductPageExtractorTool()

        self._tools: Dict[str, Callable[..., Any]] = {
            "web_search": self.search_tool.search,
            "product_page_extractor": self.browser_tool.extract,
        }

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return function calling schemas for LLM tool binding."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for products matching user specifications and budget.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query containing product category, features, and price.",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "product_page_extractor",
                    "description": "Open a product page URL using Playwright and extract specifications and price.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "Web URL of the product page to inspect.",
                            }
                        },
                        "required": ["url"],
                    },
                },
            },
        ]

    def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a registered tool by name with arguments.

        Args:
            tool_name: 'web_search' or 'product_page_extractor'.
            **kwargs: Arguments to pass to the tool.

        Returns:
            Structured tool output.
        """
        if tool_name not in self._tools:
            err = f"Unknown tool '{tool_name}'. Available tools: {list(self._tools.keys())}"
            logger.error(err)
            raise ValueError(err)

        logger.info(f"Executing tool '{tool_name}' with kwargs: {kwargs}")
        tool_fn = self._tools[tool_name]
        return tool_fn(**kwargs)

    def decide_next_action(self, state: ShoppingState) -> Dict[str, Any]:
        """Determine whether to invoke web_search, product_page_extractor, or finish research.

        Args:
            state: Current LangGraph ShoppingState.

        Returns:
            Dictionary specifying {'action': 'web_search' | 'extract_pages' | 'complete', 'args': dict}.
        """
        search_results = state.get("search_results") or []
        extracted_pages = state.get("extracted_pages") or []
        iterations = state.get("react_iterations", 0)

        # Rule 1: If no search has been executed yet, search first
        if not search_results:
            budget_dict = state.get("budget") or {}
            from src.models.schemas import BudgetInfo
            b_info = BudgetInfo(**budget_dict) if budget_dict else None
            query = self.search_tool.synthesize_query(
                category=state.get("product_category"),
                requirements=state.get("requirements") or [],
                budget=b_info,
                priorities=state.get("priorities") or [],
                intended_use=state.get("intended_use"),
            )
            return {"action": "web_search", "args": {"query": query}}

        # Rule 2: If search hits exist and extracted pages are fewer than search results
        extracted_urls = {p.get("url") for p in extracted_pages if isinstance(p, dict)}
        unvisited = [r for r in search_results if r.get("url") not in extracted_urls and r.get("url")]

        if unvisited and len(extracted_pages) < 2:
            target_url = unvisited[0].get("url")
            return {"action": "extract_pages", "args": {"url": target_url}}

        # Rule 3: We have search and extracted pages -> complete
        return {"action": "complete", "args": {}}
