"""Unit and integration tests for Phase 6: Streamlit UI and Presentation Layer."""

import pytest
from src.agents.comparison_agent import ProductComparisonAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.agents.validation_agent import ProductValidationAgent
from src.models.schemas import DecisionStatus, NextAction
from src.tools.browser_tool import ProductPageExtractorTool
from src.tools.search_tool import WebSearchTool
from src.ui.helpers import (
    extract_closest_options,
    extract_comparison_display,
    extract_decision_display,
    extract_integrations_display,
    extract_products_display,
    extract_requirement_matching,
    extract_requirements_display,
    extract_score_breakdown_bars,
    extract_scores_display,
    extract_sources_trust,
    format_currency,
    format_percentage,
    format_score,
    sanitize_error_message,
)
from src.workflow.graph import build_shopping_graph



@pytest.fixture
def phase6_workflow(requirement_agent, product_retriever, rag_evaluator):
    """Full workflow configured for Phase 6 UI verification."""
    search_tool = WebSearchTool(provider="mock")
    browser_tool = ProductPageExtractorTool()
    validation_agent = ProductValidationAgent()
    comparison_agent = ProductComparisonAgent()
    decision_agent = DecisionAgent()

    return build_shopping_graph(
        agent=requirement_agent,
        retriever=product_retriever,
        evaluator=rag_evaluator,
        search_tool=search_tool,
        browser_tool=browser_tool,
        validation_agent=validation_agent,
        comparison_agent=comparison_agent,
        decision_agent=decision_agent,
    )


class TestUIHelpers:
    """Unit tests for formatting and presentation helper functions."""

    def test_format_currency(self):
        assert format_currency(199.99, "USD") == "$199.99 USD"
        assert format_currency(1500.0, "$") == "$1,500.00 USD"
        assert format_currency(250.0, "EUR") == "250.00 EUR"
        assert format_currency(None) == "Not specified"
        assert format_currency(-10.0) == "Not specified"

    def test_format_score(self):
        assert format_score(97.75) == "97.8/100"
        assert format_score(85.0, 100.0) == "85.0/100"
        assert format_score(None) == "N/A"

    def test_format_percentage(self):
        assert format_percentage(0.95) == "95%"
        assert format_percentage(0.8) == "80%"
        assert format_percentage(92.0) == "92%"
        assert format_percentage(None) == "N/A"

    def test_sanitize_error_message_redacts_secrets_and_paths(self):
        raw_err = (
            "API failure with key sk-abcdef1234567890 and sendgrid SG.xyz9876543210 "
            "located at C:\\Users\\secret_user\\Desktop\\PROJECT 7\\src\\api.py"
        )
        sanitized = sanitize_error_message(raw_err)

        assert "sk-" not in sanitized
        assert "SG.xyz" not in sanitized
        assert "secret_user" not in sanitized
        assert "[REDACTED" in sanitized
        assert "[LOCAL_FILE_PATH]" in sanitized

    def test_sanitize_error_message_handles_none_or_empty(self):
        assert "unexpected error" in sanitize_error_message(None).lower()
        assert "unexpected error" in sanitize_error_message("").lower()


class TestDataExtractors:
    """Unit tests for UI data extraction functions from workflow state."""

    def test_extract_requirements_display(self):
        state = {
            "product_category": "Mechanical Keyboard",
            "budget_info": {"max_amount": 250.0, "currency": "USD", "is_flexible": True},
            "required_features": ["Tactile switches", "Wireless"],
            "preferences": ["Compact footprint"],
            "priorities": ["Typing feel"],
        }
        res = extract_requirements_display(state)

        assert res["category"] == "Mechanical Keyboard"
        assert "$250.00 USD (Flexible)" in res["budget"]
        assert "Tactile switches" in res["features"]
        assert "Compact footprint" in res["preferences"]
        assert "Typing feel" in res["priorities"]

    def test_extract_products_display(self):
        state = {
            "validated_products": [
                {
                    "name": "Keychron Q1 Pro",
                    "brand": "Keychron",
                    "price": 199.99,
                    "currency": "USD",
                    "specifications": {"switches": "Banana tactile"},
                    "features": ["Wireless", "Hot-swap"],
                    "url": "https://keychron.com/q1-pro",
                    "source": "keychron.com",
                }
            ]
        }
        items = extract_products_display(state)

        assert len(items) == 1
        assert items[0]["name"] == "Keychron Q1 Pro"
        assert items[0]["price"] == "$199.99 USD"
        assert items[0]["url"] == "https://keychron.com/q1-pro"

    def test_extract_comparison_display(self):
        state = {
            "comparison_result": {
                "comparison_summary": "Compared 2 mechanical keyboards.",
                "key_tradeoffs": ["Keychron is heavier aluminum."],
                "products": [
                    {
                        "product_name": "Keychron Q1 Pro",
                        "requirement_matches": ["Meets budget ($199 vs $250)"],
                        "strengths": ["Premium aluminum build"],
                        "weaknesses": ["Heavier for travel"],
                        "missing_info": [],
                    }
                ],
            }
        }
        comp = extract_comparison_display(state)

        assert "Compared 2 mechanical keyboards" in comp["summary"]
        assert len(comp["items"]) == 1
        assert comp["items"][0]["product_name"] == "Keychron Q1 Pro"
        assert "Keychron is heavier aluminum." in comp["key_tradeoffs"]

    def test_extract_scores_display_sorts_descending(self):
        state = {
            "product_scores": [
                {
                    "product_name": "Product Lower",
                    "breakdown": {"total_score": 82.5, "hard_requirement_score": 35.0},
                    "is_hard_criteria_satisfied": True,
                },
                {
                    "product_name": "Product Higher",
                    "breakdown": {"total_score": 96.0, "hard_requirement_score": 40.0},
                    "is_hard_criteria_satisfied": True,
                },
            ]
        }
        scores = extract_scores_display(state)

        assert len(scores) == 2
        assert scores[0]["Product"] == "Product Higher"
        assert scores[0]["Total Score"] == 96.0
        assert scores[1]["Product"] == "Product Lower"

    def test_extract_decision_display_recommended(self):
        state = {
            "final_decision": {
                "decision_status": "recommended",
                "recommended_product": {
                    "name": "Keychron Q1 Pro",
                    "brand": "Keychron",
                    "price": 199.99,
                    "currency": "USD",
                    "url": "https://keychron.com",
                    "features": ["Wireless"],
                },
                "alternative_product": {
                    "name": "Logitech MX Mechanical",
                    "brand": "Logitech",
                    "price": 149.99,
                    "currency": "USD",
                },
                "confidence": 0.95,
                "key_reasons": ["Highest build quality"],
                "tradeoffs": ["Heavy aluminum body"],
                "unmet_requirements": [],
                "sources": ["https://keychron.com"],
            }
        }
        dec = extract_decision_display(state)

        assert dec["is_recommended"] is True
        assert dec["recommended"]["name"] == "Keychron Q1 Pro"
        assert dec["alternative"]["name"] == "Logitech MX Mechanical"
        assert dec["confidence_pct"] == "95%"
        assert "Highest build quality" in dec["key_reasons"]

    def test_extract_decision_display_no_satisfying_product(self):
        state = {
            "final_decision": {
                "decision_status": "no_satisfying_product",
                "recommended_product": None,
                "alternative_product": None,
                "confidence": 0.0,
                "key_reasons": [],
                "tradeoffs": [],
                "unmet_requirements": ["Budget under $20 not viable for espresso machines"],
                "sources": [],
            }
        }
        dec = extract_decision_display(state)

        assert dec["is_recommended"] is False
        assert dec["status"] == "no_satisfying_product"
        assert len(dec["unmet_requirements"]) == 1

    def test_extract_integrations_display(self):
        state = {
            "integration_results": {
                "any_failures": False,
                "n8n": {"status": "skipped", "message": "Disabled via settings"},
                "sendgrid": {"status": "success", "message": "Email sent (HTTP 202)"},
                "pushover": {"status": "failed", "message": "Network timeout"},
            }
        }
        integ = extract_integrations_display(state)

        assert integ["n8n"]["status"] == "skipped"
        assert integ["sendgrid"]["status"] == "success"
        assert integ["pushover"]["status"] == "failed"


class TestStreamlitWorkflowIntegration:
    """End-to-end integration tests checking the full pipeline via Streamlit data flow."""

    def test_workflow_with_budget_and_ui_extraction(self, phase6_workflow):
        req = "Looking for a mechanical keyboard with tactile switches for daily coding under $250"
        state = {"user_request": req}

        result = phase6_workflow.invoke(state)

        # Requirements extraction
        req_disp = extract_requirements_display(result)
        assert req_disp["category"] == "Mechanical Keyboard"
        assert "$250" in req_disp["budget"]

        # Products extraction
        prods = extract_products_display(result)
        assert len(prods) > 0

        # Comparison extraction
        comp = extract_comparison_display(result)
        assert len(comp["items"]) > 0

        # Scores extraction
        scores = extract_scores_display(result)
        assert len(scores) > 0
        assert scores[0]["Total Score"] > 0

        # Decision extraction
        decision = extract_decision_display(result)
        assert decision["is_recommended"] is True
        assert decision["recommended"] is not None

        # Integrations extraction
        integ = extract_integrations_display(result)
        assert "n8n" in integ
        assert "sendgrid" in integ
        assert "pushover" in integ

    def test_workflow_ambiguous_short_circuits_gracefully(self, phase6_workflow):
        state = {"user_request": "I want to buy something good"}
        result = phase6_workflow.invoke(state)

        assert result["next_action"] == NextAction.REQUEST_CLARIFICATION.value
        assert "specify the exact product" in result["final_response"].lower()

    def test_workflow_invalid_input_handled_gracefully(self, phase6_workflow):
        for invalid_input in ["", "     ", "???", "12345"]:
            state = {"user_request": invalid_input}
            result = phase6_workflow.invoke(state)

            assert result["next_action"] == NextAction.INVALID_INPUT.value
            assert len(result["validation_errors"]) > 0
            assert "Input validation failed" in result["final_response"]

    def test_new_ui_helpers_with_workflow_state(self, phase6_workflow):
        req = "Looking for a mechanical keyboard with tactile switches for daily coding under $250"
        state = {"user_request": req}
        result = phase6_workflow.invoke(state)

        # Requirement matching
        matches = extract_requirement_matching(result)
        assert len(matches) > 0
        assert any("Budget" in m["requirement"] for m in matches)

        # Score breakdown bars
        bars = extract_score_breakdown_bars(result)
        assert len(bars) == 6
        assert bars[0]["dimension"] == "Hard Requirements"
        assert 0.0 <= bars[0]["percentage"] <= 1.0

        # Closest options
        closest = extract_closest_options(result)
        assert len(closest) > 0
        assert closest[0]["score"] > 0

        # Sources trust
        sources = extract_sources_trust(result)
        assert len(sources) > 0

