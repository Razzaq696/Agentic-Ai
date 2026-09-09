"""
Web Search Tool for Intelligent Task Execution Agent.
Phase 2: Tool Development

Retrieves relevant, up-to-date web information without requiring external API keys.
Uses DuckDuckGo search integration with resilient multi-tier fallback.
"""

import json
import warnings
import urllib.parse
import urllib.request
from typing import List, Dict, Any
from langchain_core.tools import tool

# Suppress harmless package rename notice
warnings.filterwarnings("ignore", category=RuntimeWarning)


def _search_ddg_instant(query: str) -> List[Dict[str, str]]:
    """Fetches instant answer knowledge from DuckDuckGo Instant Answer API."""
    results = []
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))

            abstract = data.get("AbstractText", "").strip()
            heading = data.get("Heading", "").strip()
            source_url = data.get("AbstractURL", "").strip()

            if abstract:
                results.append({
                    "title": heading or f"Overview: {query}",
                    "snippet": abstract,
                    "url": source_url or "https://duckduckgo.com",
                })

            # Check related topics for additional relevant snippets
            related = data.get("RelatedTopics", [])
            for item in related[:3]:
                if isinstance(item, dict) and "Text" in item and "FirstURL" in item:
                    text = item.get("Text", "").strip()
                    first_url = item.get("FirstURL", "").strip()
                    if text:
                        results.append({
                            "title": text.split(" - ")[0] if " - " in text else heading or query,
                            "snippet": text,
                            "url": first_url,
                        })
    except Exception:
        pass
    return results


def _search_wikipedia_fallback(query: str) -> List[Dict[str, str]]:
    """Fallback encyclopedia search using Wikipedia API."""
    results = []
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_query}&format=json&utf8=1"
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "IntelligentTaskAgent/1.0 (academic university project)"},
        )
        with urllib.request.urlopen(req, timeout=6) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            search_items = data.get("query", {}).get("search", [])
            for item in search_items[:3]:
                title = item.get("title", "")
                snippet = item.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": page_url,
                })
    except Exception:
        pass
    return results


def _search_ddg_library(query: str, max_results: int) -> List[Dict[str, str]]:
    """Attempts search using the installed duckduckgo_search / ddgs library."""
    results = []
    try:
        from duckduckgo_search import DDGS
        ddgs = DDGS()
        raw_results = ddgs.text(query, max_results=max_results)
        if raw_results:
            for r in raw_results:
                results.append({
                    "title": r.get("title", "Web Result"),
                    "snippet": r.get("body", "").strip(),
                    "url": r.get("href", ""),
                })
    except Exception:
        pass
    return results


@tool
def web_search(query: str) -> str:
    """
    Search the web for real-world information, current events, documentation, and external knowledge.
    Use this tool whenever the user's request requires up-to-date facts, definitions,
    explanations, or information not computable via direct arithmetic.

    Args:
        query: The search query string describing what information to look up.

    Returns:
        A formatted list of relevant search results with titles, snippets, and source URLs.
    """
    if not query or not query.strip():
        return "Error: Search query cannot be empty."

    cleaned_query = query.strip()

    # Tier 1: Try duckduckgo_search library
    results = _search_ddg_library(cleaned_query, max_results=4)

    # Tier 2: Try DuckDuckGo Instant Answer API
    if not results:
        results = _search_ddg_instant(cleaned_query)

    # Tier 3: Try Wikipedia Search API fallback
    if not results:
        results = _search_wikipedia_fallback(cleaned_query)

    if not results:
        return f"Search Result: No relevant information found for query '{cleaned_query}'."

    formatted_entries = []
    for i, res in enumerate(results[:4], start=1):
        title = res.get("title", "Result")
        snippet = res.get("snippet", "No description available.")
        url = res.get("url", "N/A")
        formatted_entries.append(
            f"[{i}] {title}\n"
            f"    Snippet: {snippet}\n"
            f"    Source: {url}"
        )

    return f"Search Results for '{cleaned_query}':\n\n" + "\n\n".join(formatted_entries)
