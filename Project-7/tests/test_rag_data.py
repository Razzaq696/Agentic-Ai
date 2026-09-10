"""Tests for product dataset loading, normalization, and document preparation."""

import json
import pytest
from src.models.schemas import ProductRecord
from src.rag.loader import load_product_dataset, prepare_documents, product_to_text
from src.guardrails.validators import validate_product_record_dict


class TestProductDataLoader:
    """Unit tests for product data ingestion and schema validation."""

    def test_load_valid_products(self):
        """Verify the local product dataset loads valid ProductRecord instances."""
        products, errors = load_product_dataset("data/products.json")
        assert len(products) >= 10
        assert len(errors) == 0

        first_product = products[0]
        assert isinstance(first_product, ProductRecord)
        assert first_product.product_id != ""
        assert first_product.price > 0
        assert first_product.category in [
            "Laptop",
            "Mechanical Keyboard",
            "Headphones",
            "Monitor",
            "Smartphone",
        ]
        assert first_product.source != ""

    def test_prepare_documents(self):
        """Verify conversion of ProductRecord objects to LangChain Documents with metadata."""
        products, _ = load_product_dataset("data/products.json")
        docs = prepare_documents(products)

        assert len(docs) == len(products)
        sample_doc = docs[0]
        assert "Product:" in sample_doc.page_content
        assert "Source Reference:" in sample_doc.page_content
        assert "product_id" in sample_doc.metadata
        assert "category" in sample_doc.metadata
        assert "price" in sample_doc.metadata
        assert "source" in sample_doc.metadata

    def test_product_to_text_contains_all_critical_fields(self):
        """Verify textual representation includes specs, features, pros, cons, and sources."""
        product = ProductRecord(
            product_id="test-001",
            product_name="Test Mechanical Keyboard",
            category="Mechanical Keyboard",
            brand="TestBrand",
            price=150.0,
            specifications={"switches": "Tactile Brown", "layout": "75%"},
            features=["Hot-swappable", "RGB"],
            use_cases=["Coding", "Typing"],
            pros=["Great acoustics"],
            cons=["Heavy"],
            source="Test Manufacturer Specs 2024",
        )
        text = product_to_text(product)
        assert "Test Mechanical Keyboard" in text
        assert "TestBrand" in text
        assert "$150.00 USD" in text
        assert "Tactile Brown" in text
        assert "Hot-swappable" in text
        assert "Great acoustics" in text
        assert "Test Manufacturer Specs 2024" in text

    def test_missing_file_handling(self):
        """Verify non-existent file path returns empty list and error message without crashing."""
        products, errors = load_product_dataset("non_existent_file.json")
        assert products == []
        assert len(errors) == 1
        assert "not found" in errors[0].lower()

    def test_malformed_json_handling(self, tmp_path):
        """Verify malformed JSON syntax is caught gracefully."""
        bad_json_file = tmp_path / "bad.json"
        bad_json_file.write_text("{ incomplete json", encoding="utf-8")

        products, errors = load_product_dataset(str(bad_json_file))
        assert products == []
        assert len(errors) == 1
        assert "malformed json" in errors[0].lower()

    def test_malformed_product_record_handling(self, tmp_path):
        """Verify invalid records (e.g. negative price or missing category) are skipped."""
        mixed_data = [
            {
                "product_id": "valid-1",
                "product_name": "Valid Mouse",
                "category": "Peripherals",
                "brand": "Logitech",
                "price": 50.0,
                "source": "Catalog",
            },
            {
                "product_id": "bad-1",
                "product_name": "Bad Mouse with Negative Price",
                "category": "Peripherals",
                "brand": "Logitech",
                "price": -99.0,  # Negative price violates ge=0
                "source": "Catalog",
            },
            {
                "product_id": "bad-2",
                # missing product_name and category
                "price": 100.0,
            },
        ]
        test_file = tmp_path / "mixed.json"
        test_file.write_text(json.dumps(mixed_data), encoding="utf-8")

        products, errors = load_product_dataset(str(test_file))
        assert len(products) == 1
        assert products[0].product_id == "valid-1"
        assert len(errors) == 2

    def test_validate_product_record_dict(self):
        """Verify guardrail validator accepts valid dict and catches errors."""
        valid_dict = {
            "product_id": "p-1",
            "product_name": "Sample",
            "category": "Cat",
            "brand": "Brand",
            "price": 99.0,
        }
        is_valid, err = validate_product_record_dict(valid_dict)
        assert is_valid is True
        assert err is None

        invalid_dict = {"product_id": "p-1", "price": -10.0}
        is_valid, err = validate_product_record_dict(invalid_dict)
        assert is_valid is False
        assert err is not None
