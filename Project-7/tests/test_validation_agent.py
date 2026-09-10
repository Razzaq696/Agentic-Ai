"""Unit tests for Phase 4 Product Validation Agent."""

import pytest
from src.agents.validation_agent import ProductValidationAgent
from src.models.schemas import Product, ValidationStatus


@pytest.fixture
def validation_agent():
    return ProductValidationAgent()


@pytest.fixture
def sample_valid_product():
    return Product(
        name="Keychron Q1 Pro Mechanical Keyboard",
        brand="Keychron",
        category="Mechanical Keyboard",
        price=199.00,
        currency="USD",
        specifications={
            "switch_type": "Keychron K Pro Red (Linear)",
            "layout": "75% ANSI layout",
            "connectivity": "Bluetooth 5.1 & USB-C wired",
        },
        features=["Hot-swappable switches", "Full aluminum CNC body", "QMK/VIA remappable"],
        availability="In Stock",
        url="https://keychron.com/q1-pro",
        source="keychron.com",
        data_confidence=0.9,
    )


class TestProductValidationAgent:
    """Test suite for ProductValidationAgent checks, detections, and statuses."""

    def test_product_validation_succeeds_for_complete_product(self, validation_agent, sample_valid_product):
        """Test 1: Complete and well-formed product receives VALID status."""
        rec = validation_agent.validate_product(sample_valid_product, expected_category="Mechanical Keyboard")
        assert rec.status == ValidationStatus.VALID
        assert rec.is_eligible_for_recommendation is True
        assert rec.data_completeness == 1.0
        assert len(rec.missing_fields) == 0
        assert len(rec.conflicting_fields) == 0

    def test_invalid_product_missing_or_placeholder_name_rejected(self, validation_agent):
        """Test 2: Product with empty or placeholder name is marked INVALID."""
        invalid_p = Product(
            name="   ",
            price=150.0,
            currency="USD",
            specifications={},
            features=[],
            source="unknown",
        )
        rec = validation_agent.validate_product(invalid_p)
        assert rec.status == ValidationStatus.INVALID
        assert rec.is_eligible_for_recommendation is False
        assert any("name" in f.lower() for f in rec.conflicting_fields + rec.missing_fields)

        placeholder_p = Product(name="product", price=100.0)
        rec2 = validation_agent.validate_product(placeholder_p)
        assert rec2.status == ValidationStatus.INVALID

    def test_missing_product_data_handled_correctly(self, validation_agent):
        """Test 3: Missing price or specifications explicitly flagged without hallucinating values."""
        partial_p = Product(
            name="Custom Artisan Keyboard",
            brand="Artisan",
            category="Mechanical Keyboard",
            price=None,  # Price unavailable
            currency="USD",
            specifications={},  # No specs
            features=["Handmade wood case"],
            url="https://artisan.com/kb",
            source="artisan.com",
            data_confidence=0.75,
        )
        rec = validation_agent.validate_product(partial_p, expected_category="Mechanical Keyboard")
        assert rec.status == ValidationStatus.PARTIALLY_VALID
        assert "price" in rec.missing_fields
        assert "specifications" in rec.missing_fields
        # Never invent missing values
        assert partial_p.price is None
        assert rec.data_completeness < 1.0

    def test_negative_price_or_unreliable_confidence_rejected(self, validation_agent):
        """Test: Products with conflicting/unreliable attributes are rejected as INVALID."""
        neg_price_p = Product(
            name="Faulty Listing Keyboard",
            price=-50.0,
            currency="USD",
            specifications={"layout": "Full"},
            features=["RGB"],
            url="https://example.com/bad",
        )
        rec = validation_agent.validate_product(neg_price_p)
        assert rec.status == ValidationStatus.INVALID
        assert rec.is_eligible_for_recommendation is False

        low_conf_p = Product(
            name="Scraped Junk Listing",
            price=99.0,
            currency="USD",
            data_confidence=0.1,  # Too low
        )
        rec_conf = validation_agent.validate_product(low_conf_p)
        assert rec_conf.status == ValidationStatus.INVALID

    def test_validate_products_batch_filters_and_tallies(self, validation_agent, sample_valid_product):
        """Test: Batch validation accurately separates valid and invalid candidates."""
        bad_p = Product(name="", price=10.0)
        out = validation_agent.validate_products([sample_valid_product, bad_p])

        assert out.total_valid == 1
        assert out.total_invalid == 1
        assert len(out.validated_products) == 1
        assert out.validated_products[0].name == sample_valid_product.name
