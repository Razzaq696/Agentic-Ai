"""Web search tool supporting multiple providers, query synthesis, and deterministic mocking."""

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional
from src.config import settings
from src.models.schemas import BudgetInfo, SearchResultItem, WebSearchOutput
from src.utils.logger import logger


class WebSearchTool:
    """Reusable web search tool for e-commerce product discovery."""

    def __init__(
        self,
        provider: Optional[str] = None,
        max_results: Optional[int] = None,
        tavily_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
    ):
        """Initialize web search tool.

        Args:
            provider: Search provider ('mock', 'tavily', 'serper', 'duckduckgo'). Defaults to settings.search_provider.
            max_results: Maximum number of results to fetch. Defaults to settings.search_max_results.
            tavily_api_key: Optional Tavily API key override.
            serper_api_key: Optional Serper API key override.
        """
        self.provider = (provider or settings.search_provider).lower()
        self.max_results = max_results or settings.search_max_results
        self.tavily_api_key = tavily_api_key or settings.tavily_api_key
        self.serper_api_key = serper_api_key or settings.serper_api_key

    def synthesize_query(
        self,
        category: Optional[str],
        requirements: Optional[List[str]] = None,
        budget: Optional[BudgetInfo] = None,
        priorities: Optional[List[str]] = None,
        intended_use: Optional[str] = None,
    ) -> str:
        """Synthesize an expressive, high-precision web search query from shopping requirements."""
        parts: List[str] = []

        if category:
            parts.append(f"best {category}")
        else:
            parts.append("best products")

        if requirements:
            # Add top 2 mandatory specifications
            parts.extend(requirements[:2])

        if budget and budget.is_specified and budget.max_amount:
            parts.append(f"under ${int(budget.max_amount)}")

        if intended_use and intended_use != "General / specified use case":
            parts.append(f"for {intended_use}")

        query = " ".join(parts).strip()
        logger.debug(f"Synthesized web search query: {query!r}")
        return query

    def search(self, query: str, max_results: Optional[int] = None) -> WebSearchOutput:
        """Execute web search using the configured provider.

        Args:
            query: The search query string.
            max_results: Optional result limit override.

        Returns:
            Structured WebSearchOutput.
        """
        limit = max_results or self.max_results
        cleaned_query = (query or "").strip()

        if not cleaned_query:
            logger.warning("Empty search query passed to WebSearchTool")
            return WebSearchOutput(query="", results=[], total_results=0, error="Empty query")

        logger.info(f"Executing web search with provider '{self.provider}' for query: {cleaned_query!r}")

        if self.provider == "tavily":
            return self._search_tavily(cleaned_query, limit)
        elif self.provider == "serper":
            return self._search_serper(cleaned_query, limit)
        elif self.provider == "duckduckgo":
            return self._search_duckduckgo(cleaned_query, limit)
        else:
            # Default to mock
            return self._search_mock(cleaned_query, limit)

    def _search_mock(self, query: str, limit: int) -> WebSearchOutput:
        """Deterministic mock search providing realistic e-commerce product links."""
        q_lower = query.lower()
        items: List[SearchResultItem] = []

        if "espresso" in q_lower or "coffee" in q_lower:
            items = [
                SearchResultItem(
                    title="Breville Dual Boiler BES920XL Espresso Machine Review",
                    url="https://www.coffeereviewhub.com/breville-dual-boiler-review",
                    snippet="The Breville Dual Boiler delivers commercial quality espresso extraction with dedicated steam and brew boilers, PID control, and 58mm portafilter under $1600.",
                    source="coffeereviewhub.com",
                ),
                SearchResultItem(
                    title="Gaggia Classic Pro Espresso Machine — Specs & Details",
                    url="https://www.espressogearguide.com/gaggia-classic-pro",
                    snippet="Solid stainless steel body, 58mm commercial portafilter, 15 bar pump, and traditional steam wand. Excellent budget entry machine at $450.",
                    source="espressogearguide.com",
                ),
                SearchResultItem(
                    title="Rancilio Silvia Pro X Dual Boiler Espresso Machine",
                    url="https://www.coffeegeekspecs.org/rancilio-silvia-pro-x",
                    snippet="Features dual PID boilers, soft infusion technology, heavy brass group head, and professional steam pressure priced around $1940.",
                    source="coffeegeekspecs.org",
                ),
            ]
        elif "laptop" in q_lower:
            items = [
                SearchResultItem(
                    title="Framework Laptop 13 (AMD Ryzen 7040 Series) Specs",
                    url="https://www.techlaptopguide.com/framework-13-amd-review",
                    snippet="Modular and easily repairable 13.5-inch laptop with up to 64GB DDR5, USB-4 expansion cards, 2.8K screen, starting at $1049.",
                    source="techlaptopguide.com",
                ),
                SearchResultItem(
                    title="Acer Swift Go 14 OLED Laptop Review & Pricing",
                    url="https://www.ultrabookspecs.com/acer-swift-go-14-oled",
                    snippet="Intel Core Ultra 7 with 16GB LPDDR5X, 14-inch 2.8K 90Hz OLED panel, lightweight 2.9 lbs chassis, priced under $850.",
                    source="ultrabookspecs.com",
                ),
            ]
        elif "keyboard" in q_lower:
            items = [
                SearchResultItem(
                    title="Epomaker RT100 Retro Mechanical Keyboard 97 Keys",
                    url="https://www.mechanicalkeyboardclub.com/epomaker-rt100",
                    snippet="Gasket-mounted mechanical keyboard with smart mini TV display, Kailh Sea Salt Silent switches, Bluetooth 5.0 and 2.4GHz wireless for $105.",
                    source="mechanicalkeyboardclub.com",
                ),
                SearchResultItem(
                    title="NuPhy Air75 V2 Low-Profile Wireless Mechanical Keyboard",
                    url="https://www.desksetupreviews.com/nuphy-air75-v2",
                    snippet="Ultra-thin 75% layout, QMK/VIA support, 1000Hz polling rate, hot-swappable Cowberry linear switches, priced at $119.",
                    source="desksetupreviews.com",
                ),
            ]
        else:
            # Generic fallback mock search hit
            cat_keyword = query.split()[0].capitalize()
            items = [
                SearchResultItem(
                    title=f"Top Rated {cat_keyword} Buying Guide & Reviews 2024",
                    url=f"https://www.consumershopperhub.com/best-{query.replace(' ', '-')[:30]}",
                    snippet=f"Comprehensive buying guide covering the best options, verified hardware specs, and user reviews for {query}.",
                    source="consumershopperhub.com",
                )
            ]

        results = items[:limit]
        return WebSearchOutput(query=query, results=results, total_results=len(results))

    def _search_tavily(self, query: str, limit: int) -> WebSearchOutput:
        """Tavily search API integration with fallback on missing key or network error."""
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY is not set. Falling back to mock search.")
            return self._search_mock(query, limit)

        try:
            url = "https://api.tavily.com/search"
            headers = {"Content-Type": "application/json"}
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": limit,
            }
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            for r in data.get("results", []):
                results.append(
                    SearchResultItem(
                        title=r.get("title", "Untitled"),
                        url=r.get("url", ""),
                        snippet=r.get("content", ""),
                        source="tavily",
                    )
                )
            return WebSearchOutput(query=query, results=results, total_results=len(results))
        except Exception as e:
            logger.error(f"Tavily search error: {e}. Falling back to mock search.")
            return self._search_mock(query, limit)

    def _search_serper(self, query: str, limit: int) -> WebSearchOutput:
        """Serper.dev search API integration."""
        if not self.serper_api_key:
            logger.warning("SERPER_API_KEY is not set. Falling back to mock search.")
            return self._search_mock(query, limit)

        try:
            url = "https://google.serper.dev/search"
            headers = {
                "X-API-KEY": self.serper_api_key,
                "Content-Type": "application/json",
            }
            payload = {"q": query, "num": limit}
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                data = json.loads(response.read().decode("utf-8"))

            results = []
            for item in data.get("organic", []):
                results.append(
                    SearchResultItem(
                        title=item.get("title", ""),
                        url=item.get("link", ""),
                        snippet=item.get("snippet", ""),
                        source="serper",
                    )
                )
            return WebSearchOutput(query=query, results=results, total_results=len(results))
        except Exception as e:
            logger.error(f"Serper search error: {e}. Falling back to mock search.")
            return self._search_mock(query, limit)

    def _search_duckduckgo(self, query: str, limit: int) -> WebSearchOutput:
        """DuckDuckGo instant search with fallback."""
        try:
            params = urllib.parse.urlencode({"q": query, "format": "json", "no_html": 1})
            url = f"https://api.duckduckgo.com/?{params}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for topic in data.get("RelatedTopics", []):
                if "Text" in topic and "FirstURL" in topic:
                    results.append(
                        SearchResultItem(
                            title=topic.get("Text", "")[:60],
                            url=topic.get("FirstURL", ""),
                            snippet=topic.get("Text", ""),
                            source="duckduckgo",
                        )
                    )
                    if len(results) >= limit:
                        break

            if not results:
                return self._search_mock(query, limit)

            return WebSearchOutput(query=query, results=results, total_results=len(results))
        except Exception as e:
            logger.info(f"DuckDuckGo search error: {e}. Falling back to mock search.")
            return self._search_mock(query, limit)
