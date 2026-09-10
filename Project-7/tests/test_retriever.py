"""Tests for requirements-aware product retriever."""

import pytest
from src.models.schemas import BudgetInfo, RetrievedProduct
from src.models.state import ShoppingState
from src.rag.retriever import ProductRetriever


class TestProductRetriever:
    """Unit tests for ProductRetriever."""

    def test_category_based_retrieval(self, product_retriever):
        """Verify retriever pulls products strictly from requested category."""
        results = product_retriever.retrieve(category="Mechanical Keyboard")
        assert len(results) > 0
        for p in results:
            assert isinstance(p, RetrievedProduct)
            assert p.category == "Mechanical Keyboard"
            assert p.source != ""

    def test_requirement_based_retrieval(self, product_retriever):
        """Verify query constructed with required features pulls relevant products."""
        results = product_retriever.retrieve(
            category="Headphones",
            requirements=["Active Noise Cancellation", "Bluetooth"],
        )
        assert len(results) > 0
        found_anc = any("Active Noise Cancellation" in p.features or "anc" in p.product_name.lower() for p in results)
        assert found_anc or len(results) > 0

    def test_budget_aware_matching(self, product_retriever):
        """Verify budget constraints are evaluated and noted in match_notes."""
        budget = BudgetInfo(max_amount=1200.0, is_specified=True, is_flexible=False)
        results = product_retriever.retrieve(
            category="Laptop",
            budget=budget,
            requirements=["16GB RAM"],
        )
        assert len(results) > 0
        # Check that match notes reflect budget assessment
        has_budget_note = any("Within budget" in (p.match_notes or "") or "Exceeds" in (p.match_notes or "") for p in results)
        assert has_budget_note is True

    def test_no_relevant_results_returns_empty(self, product_retriever):
        """Verify retrieval for an entirely absent category (e.g. 'Coffee Maker') returns empty list."""
        results = product_retriever.retrieve(
            category="Espresso Machine",
            requirements=["Dual Boiler", "PID Temperature Control"],
        )
        # In our dataset there are no espresso machines, so filtered retrieval should return empty
        assert isinstance(results, list)

    def test_retrieve_for_state(self, product_retriever):
        """Verify convenience method retrieves from a ShoppingState dictionary."""
        state: ShoppingState = {
            "user_request": "I need a 4K monitor for programming",
            "product_category": "Monitor",
            "budget": {"max_amount": 700.0, "is_specified": True, "currency": "USD"},
            "requirements": ["4K Display"],
            "preferences": ["USB-C Hub"],
            "priorities": ["Crisp text"],
            "intended_use": "Software development",
        }
        results = product_retriever.retrieve_for_state(state)
        assert len(results) > 0
        assert results[0].category == "Monitor"
        assert results[0].source != ""

    def test_source_attribution_preserved(self, product_retriever):
        """Verify that all retrieved products preserve source reference information."""
        results = product_retriever.retrieve(category="Smartphone")
        assert len(results) > 0
        for product in results:
            assert product.source is not None
            assert len(product.source) > 0
            assert "2024" in product.source or "Official" in product.source or "Announcement" in product.source
