"""Tests for ToolRegistry and ReAct reasoning loop."""

import pytest
from src.models.schemas import Product, SearchResultItem
from src.models.state import ShoppingState
from src.tools.registry import ToolRegistry
from src.guardrails.validators import (
    deduplicate_products,
    validate_search_results,
    validate_url,
)


class TestReActToolLoop:
    """Unit tests for tool calling decisions and loop control."""

    def test_tool_calling_selects_web_search_first(self):
        """Scenario 7a: Tool registry decides web_search when no search results exist."""
        registry = ToolRegistry()
        state: ShoppingState = {
            "product_category": "Espresso Machine",
            "requirements": ["Dual Boiler"],
            "search_results": [],
            "extracted_pages": [],
            "react_iterations": 0,
        }
        decision = registry.decide_next_action(state)
        assert decision["action"] == "web_search"
        assert "args" in decision
        assert "query" in decision["args"]

    def test_tool_calling_selects_product_extractor_when_search_done(self):
        """Scenario 7b: Tool registry selects product_page_extractor when URLs are unextracted."""
        registry = ToolRegistry()
        state: ShoppingState = {
            "product_category": "Espresso Machine",
            "search_results": [
                {"title": "Machine 1", "url": "https://www.example.com/p1"},
                {"title": "Machine 2", "url": "https://www.example.com/p2"},
            ],
            "extracted_pages": [],
            "react_iterations": 1,
        }
        decision = registry.decide_next_action(state)
        assert decision["action"] == "extract_pages"
        assert decision["args"]["url"] == "https://www.example.com/p1"

    def test_react_loop_stops_after_sufficient_information(self):
        """Scenario 8: Tool registry decides 'complete' when pages are extracted."""
        registry = ToolRegistry()
        state: ShoppingState = {
            "product_category": "Espresso Machine",
            "search_results": [
                {"title": "Machine 1", "url": "https://www.example.com/p1"},
            ],
            "extracted_pages": [
                {"name": "Machine 1", "url": "https://www.example.com/p1", "price": 1200.0},
                {"name": "Machine 2", "url": "https://www.example.com/p2", "price": 1500.0},
            ],
            "react_iterations": 1,
        }
        decision = registry.decide_next_action(state)
        assert decision["action"] == "complete"

    def test_tool_registry_execution(self):
        """Verify tool execution by name through registry."""
        registry = ToolRegistry()
        search_res = registry.execute_tool("web_search", query="espresso machine")
        assert len(search_res.results) > 0

        page_res = registry.execute_tool(
            "product_page_extractor",
            url="https://www.coffeereviewhub.com/breville-dual-boiler-review",
        )
        assert page_res.name != ""

    def test_validate_url_guardrail(self):
        """Verify URL validation rejects non-HTTP schemes or invalid formats."""
        is_valid, _ = validate_url("https://www.amazon.com/dp/B001")
        assert is_valid is True

        is_valid_http, _ = validate_url("http://example.com/item")
        assert is_valid_http is True

        is_invalid_scheme, err = validate_url("ftp://files.example.com/item")
        assert is_invalid_scheme is False
        assert "http" in err

        is_empty, _ = validate_url("")
        assert is_empty is False

    def test_deduplicate_products(self):
        """Scenario 12: Deduplicate identical product listings by name or URL."""
        prods = [
            Product(name="Breville Dual Boiler", price=1599.95, url="https://example.com/p1"),
            Product(name="Breville Dual Boiler", price=1599.95, url="https://example.com/p1_diff"),
            Product(name="Gaggia Classic Pro", price=449.00, url="https://example.com/p2"),
        ]
        unique = deduplicate_products(prods)
        assert len(unique) == 2
        assert unique[0].name == "Breville Dual Boiler"
        assert unique[1].name == "Gaggia Classic Pro"
