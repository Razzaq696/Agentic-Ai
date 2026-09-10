"""Tests for Phase 5: n8n, SendGrid, Pushover, and LangSmith Integrations."""

import io
import json
import os
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from src.agents.comparison_agent import ProductComparisonAgent
from src.agents.decision_agent import DecisionAgent
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.agents.validation_agent import ProductValidationAgent
from src.config import Settings
from src.integrations.langsmith import configure_langsmith
from src.integrations.n8n_client import N8nClient
from src.integrations.pushover_client import PushoverClient
from src.integrations.sendgrid_client import SendGridClient
from src.models.schemas import (
    BudgetInfo,
    DecisionStatus,
    FinalDecision,
    IntegrationDispatchStatus,
    Product,
    RequirementAnalysisOutput,
    ScoreBreakdown,
)
from src.tools.browser_tool import ProductPageExtractorTool
from src.tools.search_tool import WebSearchTool
from src.workflow.graph import build_shopping_graph
from src.workflow.nodes import dispatch_integrations_node


@pytest.fixture
def sample_final_decision():
    """Create a realistic FinalDecision fixture for testing payloads."""
    requirements = RequirementAnalysisOutput(
        product_category="Mechanical Keyboard",
        budget=BudgetInfo(max_amount=250.0, is_specified=True, is_flexible=False),
        required_features=["Tactile switches", "Wireless"],
        preferences=["Compact footprint"],
        priorities=["Typing feel"],
    )

    top_pick = Product(
        name="Keychron Q1 Pro",
        brand="Keychron",
        category="Mechanical Keyboard",
        price=199.99,
        currency="USD",
        specifications={"switch_type": "Keychron K Pro Banana (Tactile)", "connectivity": "Bluetooth 5.1 / Type-C"},
        features=["Wireless", "Hot-swappable", "Tactile switches"],
        url="https://keychron.com/q1-pro",
        source="keychron.com",
        data_confidence=0.95,
    )

    scores = ScoreBreakdown(
        hard_requirement_score=40.0,
        budget_score=25.0,
        spec_match_score=15.0,
        feature_match_score=10.0,
        preference_priority_score=5.0,
        data_confidence_score=4.8,
        total_score=99.8,
        violations=[],
    )

    return FinalDecision(
        user_requirements=requirements,
        recommended_product=top_pick,
        alternative_product=None,
        comparison_summary="Keychron Q1 Pro is the top recommendation within the $250 budget.",
        score_breakdown=scores,
        key_reasons=["Excellent tactile switches", "Solid wireless connectivity", "Under $250 budget"],
        tradeoffs=["Heavier aluminum body not ideal for travel"],
        unmet_requirements=[],
        sources=["https://keychron.com/q1-pro"],
        confidence=0.95,
        decision_status=DecisionStatus.RECOMMENDED,
    )


@pytest.fixture
def phase5_workflow(requirement_agent, product_retriever, rag_evaluator):
    """Full workflow configured with all Phase 1-5 agents, tools, and mock search."""
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


class TestN8nClient:
    """Unit tests for n8n webhook integration."""

    def test_n8n_payload_generation(self, sample_final_decision):
        client = N8nClient(enabled=True, webhook_url="https://n8n.example.com/webhook/shopping")
        payload = client.build_payload(sample_final_decision)

        assert payload["event"] == "shopping_recommendation_completed"
        assert payload["decision_status"] == DecisionStatus.RECOMMENDED.value
        assert payload["recommended_product"]["name"] == "Keychron Q1 Pro"
        assert payload["recommended_product"]["price"] == 199.99
        assert payload["confidence"] == 0.95
        assert "timestamp" in payload
        assert payload["category"] == "Mechanical Keyboard"

    def test_n8n_skipped_when_disabled(self, sample_final_decision):
        client = N8nClient(enabled=False, webhook_url="https://n8n.example.com/webhook")
        success, msg = client.send_decision(sample_final_decision)

        assert success is False
        assert "disabled" in msg.lower()

    def test_n8n_skipped_when_url_missing(self, sample_final_decision):
        client = N8nClient(enabled=True, webhook_url="")
        success, msg = client.send_decision(sample_final_decision)

        assert success is False
        assert "not configured" in msg.lower()

    @patch("urllib.request.urlopen")
    def test_n8n_webhook_success(self, mock_urlopen, sample_final_decision):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = b'{"success": true}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = N8nClient(enabled=True, webhook_url="https://n8n.example.com/webhook/shopping")
        success, msg = client.send_decision(sample_final_decision)

        assert success is True
        assert "200" in msg
        assert mock_urlopen.called

    @patch("urllib.request.urlopen")
    def test_n8n_webhook_failure_handled_gracefully(self, mock_urlopen, sample_final_decision):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://n8n.example.com/webhook",
            code=500,
            msg="Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"server crashed"),
        )

        client = N8nClient(enabled=True, webhook_url="https://n8n.example.com/webhook")
        success, msg = client.send_decision(sample_final_decision)

        assert success is False
        assert "500" in msg


class TestSendGridClient:
    """Unit tests for SendGrid email reporting integration."""

    def test_sendgrid_payload_and_content_generation(self, sample_final_decision):
        client = SendGridClient(
            enabled=True,
            api_key="SG.fake_key_123",
            from_email="noreply@shopping-agent.ai",
            to_email="user@example.com",
        )
        subject, plain_text, html_text = client.format_email_content(sample_final_decision)

        assert "Keychron Q1 Pro" in subject
        assert "Keychron Q1 Pro" in plain_text
        assert "AI SMART SHOPPING DECISION REPORT" in plain_text
        assert "Keychron Q1 Pro" in html_text
        assert "<html" in html_text

        payload = client.build_payload(sample_final_decision)
        assert payload["personalizations"][0]["to"][0]["email"] == "user@example.com"
        assert payload["from"]["email"] == "noreply@shopping-agent.ai"
        assert len(payload["content"]) == 2

    def test_sendgrid_skipped_when_disabled_or_missing_keys(self, sample_final_decision):
        client_disabled = SendGridClient(enabled=False, api_key="SG.xxx")
        success1, msg1 = client_disabled.send_report(sample_final_decision)
        assert success1 is False
        assert "disabled" in msg1.lower()

        client_no_key = SendGridClient(enabled=True, api_key="")
        success2, msg2 = client_no_key.send_report(sample_final_decision)
        assert success2 is False
        assert "not configured" in msg2.lower()

    @patch("urllib.request.urlopen")
    def test_sendgrid_success(self, mock_urlopen, sample_final_decision):
        mock_resp = MagicMock()
        mock_resp.status = 202
        mock_resp.getcode.return_value = 202
        mock_resp.read.return_value = b""
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = SendGridClient(
            enabled=True,
            api_key="SG.valid_mock_key",
            from_email="agent@shop.ai",
            to_email="buyer@test.com",
        )
        success, msg = client.send_report(sample_final_decision)

        assert success is True
        assert "202" in msg

    @patch("urllib.request.urlopen")
    def test_sendgrid_failure_handled_gracefully(self, mock_urlopen, sample_final_decision):
        mock_urlopen.side_effect = urllib.error.HTTPError(
            url="https://api.sendgrid.com/v3/mail/send",
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b'{"errors":[{"message":"The provided authorization grant is invalid"}]}'),
        )

        client = SendGridClient(
            enabled=True,
            api_key="SG.bad_key",
            from_email="agent@shop.ai",
            to_email="buyer@test.com",
        )
        success, msg = client.send_report(sample_final_decision)

        assert success is False
        assert "401" in msg


class TestPushoverClient:
    """Unit tests for Pushover push notifications integration."""

    def test_pushover_notification_content_recommended(self, sample_final_decision):
        client = PushoverClient(
            enabled=True,
            user_key="user_key_mock",
            api_token="token_mock",
        )
        title, msg, priority = client.format_notification(sample_final_decision)

        assert "Keychron Q1 Pro" in title
        assert "Keychron Q1 Pro" in msg
        assert "$199.99" in msg
        assert priority == 0

    def test_pushover_notification_content_no_product(self):
        requirements = RequirementAnalysisOutput(
            product_category="Mechanical Keyboard",
            budget=BudgetInfo(max_amount=20.0, is_specified=True, is_flexible=False),
        )
        decision = FinalDecision(
            user_requirements=requirements,
            recommended_product=None,
            alternative_product=None,
            comparison_summary="No product satisfied requirements under $20.",
            score_breakdown=None,
            key_reasons=[],
            tradeoffs=["Price floor exceeded"],
            unmet_requirements=["No mechanical keyboard with Hall Effect switches under $20"],
            sources=[],
            confidence=0.0,
            decision_status=DecisionStatus.NO_SATISFYING_PRODUCT,
        )
        client = PushoverClient(enabled=True, user_key="u", api_token="t")
        title, msg, priority = client.format_notification(decision)

        assert "No Mechanical Keyboard" in title
        assert "Search ended without recommendation" in msg
        assert priority == 1

    def test_pushover_skipped_when_disabled_or_missing_keys(self, sample_final_decision):
        client = PushoverClient(enabled=False, user_key="u", api_token="t")
        success, msg = client.notify_decision(sample_final_decision)
        assert success is False
        assert "disabled" in msg.lower()

        client_missing = PushoverClient(enabled=True, user_key="", api_token="t")
        success2, msg2 = client_missing.notify_decision(sample_final_decision)
        assert success2 is False
        assert "not configured" in msg2.lower()

    @patch("urllib.request.urlopen")
    def test_pushover_success(self, mock_urlopen, sample_final_decision):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.getcode.return_value = 200
        mock_resp.read.return_value = b'{"status": 1, "request": "req-123"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        client = PushoverClient(enabled=True, user_key="u", api_token="t")
        success, msg = client.notify_decision(sample_final_decision)

        assert success is True
        assert "200" in msg

    @patch("urllib.request.urlopen")
    def test_pushover_failure_handled_gracefully(self, mock_urlopen, sample_final_decision):
        mock_urlopen.side_effect = urllib.error.URLError("Network unreachable")

        client = PushoverClient(enabled=True, user_key="u", api_token="t")
        success, msg = client.notify_decision(sample_final_decision)

        assert success is False
        assert "Network unreachable" in msg


class TestLangSmithConfiguration:
    """Unit tests for LangSmith tracing configuration."""

    def test_langsmith_enabled_sets_env(self, monkeypatch):
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
        monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)

        settings = Settings(
            langchain_tracing_v2=True,
            langchain_api_key="lsv2_pt_testkey123",
            langchain_project="test-project",
            langchain_endpoint="https://api.smith.langchain.com",
        )
        enabled = configure_langsmith(config=settings)

        assert enabled is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "lsv2_pt_testkey123"
        assert os.environ.get("LANGCHAIN_PROJECT") == "test-project"

    def test_langsmith_disabled_does_not_enable(self, monkeypatch):
        monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
        settings = Settings(langchain_tracing_v2=False)
        enabled = configure_langsmith(config=settings)

        assert enabled is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "false"


class TestDispatchIntegrationsNode:
    """Unit tests for dispatch_integrations_node and graceful degradation."""

    def test_dispatch_with_all_disabled(self, sample_final_decision):
        state = {
            "final_decision": sample_final_decision.model_dump(),
            "final_response": "Here is your recommendation",
            "user_request": "Buy a mechanical keyboard",
        }
        res = dispatch_integrations_node(state)

        assert "integration_results" in res
        summary = res["integration_results"]
        assert summary["any_failures"] is False
        assert summary["n8n"]["status"] == IntegrationDispatchStatus.SKIPPED.value
        assert summary["sendgrid"]["status"] == IntegrationDispatchStatus.SKIPPED.value
        assert summary["pushover"]["status"] == IntegrationDispatchStatus.SKIPPED.value
        # Final response intact
        assert "Here is your recommendation" in res["final_response"]

    @patch("src.integrations.n8n_client.N8nClient.send_decision")
    @patch("src.integrations.sendgrid_client.SendGridClient.send_report")
    @patch("src.integrations.pushover_client.PushoverClient.notify_decision")
    def test_final_decision_remains_available_when_all_fail(
        self, mock_push, mock_sg, mock_n8n, sample_final_decision
    ):
        mock_n8n.return_value = (False, "Connection timed out")
        mock_sg.return_value = (False, "Invalid API key")
        mock_push.return_value = (False, "Rate limit exceeded")

        state = {
            "final_decision": sample_final_decision.model_dump(),
            "final_response": "Here is your recommendation",
            "user_request": "Buy a mechanical keyboard",
        }
        res = dispatch_integrations_node(state)

        assert "integration_results" in res
        summary = res["integration_results"]
        assert summary["any_failures"] is True
        assert summary["n8n"]["status"] == IntegrationDispatchStatus.FAILED.value
        assert summary["sendgrid"]["status"] == IntegrationDispatchStatus.FAILED.value
        assert summary["pushover"]["status"] == IntegrationDispatchStatus.FAILED.value

        # Crucial check: Core state preserved without corruption
        assert state["final_decision"] is not None
        assert state["final_response"] == "Here is your recommendation"


class TestFullPhase1To5WorkflowEndToEnd:
    """Full workflow test ensuring Phase 1 to Phase 5 runs cleanly end-to-end."""

    def test_phase5_workflow_includes_integration_results(self, phase5_workflow):
        request = "Looking for a mechanical keyboard with tactile switches for daily coding under $250"
        state = {"user_request": request}

        result = phase5_workflow.invoke(state)

        # Core Phase 1-4 asserts
        assert result["retrieval_sufficient"] is True
        assert result["final_decision"] is not None
        assert result["final_decision"]["decision_status"] == DecisionStatus.RECOMMENDED.value

        # Phase 5 assert: integration_results present in output state
        assert "integration_results" in result
        integ = result["integration_results"]
        assert "any_failures" in integ
        assert "n8n" in integ
        assert "sendgrid" in integ
        assert "pushover" in integ
        assert integ["n8n"]["status"] == IntegrationDispatchStatus.SKIPPED.value
        assert integ["sendgrid"]["status"] == IntegrationDispatchStatus.SKIPPED.value
        assert integ["pushover"]["status"] == IntegrationDispatchStatus.SKIPPED.value
