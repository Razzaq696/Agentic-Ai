"""Tests for WebSearchTool and query synthesis."""

import pytest
from src.models.schemas import BudgetInfo, SearchResultItem, WebSearchOutput
from src.tools.search_tool import WebSearchTool


class TestWebSearchTool:
    """Unit tests for WebSearchTool."""

    def test_search_returns_valid_structured_results(self):
        """Scenario 1: Web search returns valid structured results."""
        tool = WebSearchTool(provider="mock", max_results=3)
        output = tool.search("best espresso machine with dual boiler under $2000")

        assert isinstance(output, WebSearchOutput)
        assert output.total_results > 0
        assert len(output.results) <= 3

        first_hit = output.results[0]
        assert isinstance(first_hit, SearchResultItem)
        assert len(first_hit.title) > 0
        assert first_hit.url.startswith("http")
        assert len(first_hit.snippet) > 0
        assert first_hit.source != ""

    def test_query_synthesis_from_requirements(self):
        """Verify search query includes category, requirements, and budget."""
        tool = WebSearchTool()
        budget = BudgetInfo(max_amount=1500.0, is_specified=True, is_flexible=False)
        query = tool.synthesize_query(
            category="Espresso Machine",
            requirements=["Dual Boiler", "PID Temperature Control"],
            budget=budget,
            intended_use="Home Barista",
        )

        assert "best Espresso Machine" in query
        assert "Dual Boiler" in query
        assert "under $1500" in query
        assert "for Home Barista" in query

    def test_empty_query_returns_safe_output(self):
        """Scenario 2: Web search handles empty/whitespace query safely."""
        tool = WebSearchTool()
        out_empty = tool.search("")
        assert out_empty.total_results == 0
        assert out_empty.results == []
        assert out_empty.error is not None

        out_whitespace = tool.search("    ")
        assert out_whitespace.total_results == 0

    def test_missing_api_key_graceful_fallback(self):
        """Scenario 3: Web search provider falls back safely when API key is missing."""
        # Provider configured as tavily without key
        tool = WebSearchTool(provider="tavily", tavily_api_key=None)
        output = tool.search("mechanical keyboard")

        # Must not crash; falls back gracefully to mock results
        assert isinstance(output, WebSearchOutput)
        assert len(output.results) > 0

        # Provider configured as serper without key
        tool_serper = WebSearchTool(provider="serper", serper_api_key=None)
        output_serper = tool_serper.search("laptop")
        assert isinstance(output_serper, WebSearchOutput)
        assert len(output_serper.results) > 0

    def test_max_results_limit_enforced(self):
        """Verify max_results parameter restricts hit count."""
        tool = WebSearchTool(provider="mock", max_results=2)
        output = tool.search("espresso machine", max_results=1)
        assert len(output.results) == 1
