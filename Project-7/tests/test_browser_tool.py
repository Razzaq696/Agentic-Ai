"""Tests for Playwright product page extractor tool."""

import pytest
from src.models.schemas import ExtractedProductPageData
from src.tools.browser_tool import ProductPageExtractorTool


class TestProductPageExtractorTool:
    """Unit tests for ProductPageExtractorTool."""

    def test_extract_mock_product_page(self):
        """Scenario 4: Extractor successfully parses product details from target URL."""
        extractor = ProductPageExtractorTool()
        url = "https://www.coffeereviewhub.com/breville-dual-boiler-review"
        result = extractor.extract(url)

        assert isinstance(result, ExtractedProductPageData)
        assert "Breville Dual Boiler" in result.name
        assert result.brand == "Breville"
        assert result.price == 1599.95
        assert result.currency == "USD"
        assert len(result.features) > 0
        assert result.url == url
        assert result.source != ""
        assert result.data_confidence >= 0.85

    def test_extract_from_html_with_json_ld(self):
        """Verify extraction from HTML content containing Schema.org JSON-LD."""
        extractor = ProductPageExtractorTool()
        sample_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@context": "https://schema.org/",
                "@type": "Product",
                "name": "Profitec Pro 700 Espresso Machine",
                "brand": { "@type": "Brand", "name": "Profitec" },
                "offers": {
                    "@type": "Offer",
                    "price": "2999.00",
                    "priceCurrency": "USD",
                    "availability": "https://schema.org/InStock"
                }
            }
            </script>
        </head>
        <body>
            <h1>Profitec Pro 700</h1>
            <ul>
                <li>Dual stainless steel boilers with rotary pump</li>
                <li>PID display with shot timer</li>
                <li>Direct water line connection capability</li>
            </ul>
        </body>
        </html>
        """
        result = extractor.extract_from_html(sample_html, "https://www.example.com/profitec-700")

        assert result.name == "Profitec Pro 700 Espresso Machine"
        assert result.brand == "Profitec"
        assert result.price == 2999.0
        assert result.currency == "USD"
        assert result.availability == "InStock"
        assert len(result.features) >= 1
        assert "Dual stainless steel boilers" in result.features[0]

    def test_inaccessible_page_handling(self):
        """Scenario 5: Extractor handles inaccessible / non-responsive page gracefully."""
        extractor = ProductPageExtractorTool()
        # Empty or invalid domain
        result = extractor.extract("")
        assert result.data_confidence == 0.0
        assert result.extraction_error is not None

    def test_product_page_with_missing_fields_never_fabricates(self):
        """Scenario 6: Product page with missing price or brand leaves fields None without inventing values."""
        extractor = ProductPageExtractorTool()
        sparse_html = """
        <html>
        <head><title>Unbranded Item Overview</title></head>
        <body>
            <h1>Unbranded Replacement Part</h1>
            <p>Description without price or brand tags.</p>
        </body>
        </html>
        """
        result = extractor.extract_from_html(sparse_html, "https://www.example.com/item-no-price")

        assert result.name == "Unbranded Replacement Part"
        assert result.brand is None
        assert result.price is None  # Must NOT fabricate price
        assert result.features == []
        assert result.data_confidence < 0.8  # Lower confidence due to missing price

    def test_empty_html_content(self):
        """Verify empty HTML string returns structured error."""
        extractor = ProductPageExtractorTool()
        result = extractor.extract_from_html("", "https://www.example.com/blank")
        assert result.name == "Empty Page"
        assert result.data_confidence == 0.0
        assert result.extraction_error == "HTML content was empty"
