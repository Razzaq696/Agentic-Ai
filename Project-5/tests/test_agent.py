"""
Comprehensive Test Suite for Project 5 (Phase 1).
Tests Agent Decision Making, Tool Calling (SendGrid, Pushover, Audit Logger),
No Action behavior, Tool Failure handling, and Exclusive Tool Selection.
"""

import pytest
from src.models import EventRequest, ActionType
from src.agent import CommunicationAgent
from src.logger import ConfirmationLogger
from src.tools.sendgrid_tool import send_email_via_sendgrid
from src.tools.pushover_tool import send_pushover_notification
from src.tools.audit_logger_tool import record_audit_log


@pytest.fixture
def agent():
    """Provides a fresh CommunicationAgent with an isolated ConfirmationLogger."""
    logger = ConfirmationLogger()
    return CommunicationAgent(logger=logger)


# ==============================================================================
# TEST 1 — EMAIL DECISION (SendGrid)
# ==============================================================================
def test_email_decision(agent):
    """
    Test 1: A high-priority event requires notifying the user by email.
    Expected: Agent -> SEND_EMAIL -> SendGrid Tool -> Confirmation / Log.
    """
    event = EventRequest(
        event_type="security_alert",
        message="Unauthorized login attempt detected on production database.",
        priority="HIGH",
        recipient="DevSecOps Lead",
        email="devops@company.com"
    )

    confirmation = agent.process_event(event, force_mock=True)

    assert confirmation.selected_action == ActionType.SEND_EMAIL
    assert confirmation.tool_used == "sendgrid"
    assert confirmation.status == "success"
    assert confirmation.tool_result is not None
    assert confirmation.tool_result.tool == "sendgrid"
    assert confirmation.tool_result.mode == "mock"
    assert confirmation.tool_result.recipient == "devops@company.com"
    assert confirmation.tool_result.message_id is not None
    assert "security_alert" in confirmation.summary or "sendgrid" in confirmation.summary.lower()


# ==============================================================================
# TEST 2 — NOTIFICATION DECISION (Pushover)
# ==============================================================================
def test_notification_decision(agent):
    """
    Test 2: A high-priority event requires an urgent push notification.
    Expected: Agent -> SEND_NOTIFICATION -> Pushover Tool -> Confirmation / Log.
    """
    event = EventRequest(
        event_type="service_outage",
        message="Primary payment gateway API latency exceeded 5000ms threshold.",
        priority="CRITICAL",
        recipient="OnCall Engineer",
        notification_message="CRITICAL: Payment Gateway Outage! Immediate response needed.",
        conditions={"notification_enabled": True}
    )

    confirmation = agent.process_event(event, force_mock=True)

    assert confirmation.selected_action == ActionType.SEND_NOTIFICATION
    assert confirmation.tool_used == "pushover"
    assert confirmation.status == "success"
    assert confirmation.tool_result is not None
    assert confirmation.tool_result.tool == "pushover"
    assert confirmation.tool_result.mode == "mock"
    assert confirmation.tool_result.message_id is not None
    assert "Pushover" in confirmation.summary or "pushover" in confirmation.summary.lower()


# ==============================================================================
# TEST 3 — NO ACTION
# ==============================================================================
def test_no_action(agent):
    """
    Test 3: A low-priority informational event that does not require communication.
    Expected: Agent -> NO_ACTION -> Confirmation / Log (No external tool called).
    """
    event = EventRequest(
        event_type="routine_healthcheck",
        message="Disk utilization is normal at 32%. Memory utilization at 41%.",
        priority="LOW",
        recipient="System Monitor"
    )

    confirmation = agent.process_event(event)

    assert confirmation.selected_action == ActionType.NO_ACTION
    assert confirmation.tool_used is None
    assert confirmation.status == "no_action"
    assert confirmation.tool_result is None
    assert "No communication required" in confirmation.summary


# ==============================================================================
# TEST 4 — TOOL FAILURE HANDLING
# ==============================================================================
def test_tool_failure_handling(agent):
    """
    Test 4: Simulate a tool failure.
    Verify: Tool -> Failure -> Error Status -> Confirmation / Log without crashing.
    """
    event = EventRequest(
        event_type="server_unreachable",
        message="Secondary load balancer failed healthcheck.",
        priority="HIGH",
        recipient="Infrastructure Admin",
        email="infra@company.com"
    )

    # Force simulated failure
    confirmation = agent.process_event(event, force_mock=True, simulate_tool_failure=True)

    assert confirmation.selected_action == ActionType.SEND_EMAIL
    assert confirmation.tool_used == "sendgrid"
    assert confirmation.status == "failed"
    assert confirmation.tool_result is not None
    assert confirmation.tool_result.status == "failed"
    assert confirmation.error is not None
    assert "Simulated SendGrid API connection failure" in confirmation.error
    assert "failed" in confirmation.summary.lower()


# ==============================================================================
# TEST 5 — OTHER TOOL DECISION (Audit Logger)
# ==============================================================================
def test_other_tool_decision(agent):
    """
    Test 5: An internal audit / record-keeping event.
    Expected: Agent -> OTHER_TOOL -> Audit Logger Tool -> Confirmation / Log.
    """
    event = EventRequest(
        event_type="audit_compliance_scan",
        message="Monthly SOC2 permission review completed automatically.",
        priority="MEDIUM",
        recipient="Compliance Officer",
        conditions={"log_only": True}
    )

    confirmation = agent.process_event(event)

    assert confirmation.selected_action == ActionType.OTHER_TOOL
    assert confirmation.tool_used == "audit_logger"
    assert confirmation.status == "success"
    assert confirmation.tool_result is not None
    assert confirmation.tool_result.tool == "audit_logger"
    assert confirmation.tool_result.message_id.startswith("audit-")


# ==============================================================================
# TEST 6 — EXCLUSIVE TOOL SELECTION VERIFICATION
# ==============================================================================
def test_exclusive_tool_selection(agent, monkeypatch):
    """
    Test 6: Verify that the Agent ONLY calls the selected tool and does NOT
    call other tools sequentially.
    """
    calls = {"sendgrid": 0, "pushover": 0, "audit_logger": 0}

    def mock_sg(*args, **kwargs):
        calls["sendgrid"] += 1
        return send_email_via_sendgrid(*args, force_mock=True, **kwargs)

    def mock_po(*args, **kwargs):
        calls["pushover"] += 1
        return send_pushover_notification(*args, force_mock=True, **kwargs)

    def mock_audit(*args, **kwargs):
        calls["audit_logger"] += 1
        return record_audit_log(*args, **kwargs)

    monkeypatch.setattr("src.agent.send_email_via_sendgrid", mock_sg)
    monkeypatch.setattr("src.agent.send_pushover_notification", mock_po)
    monkeypatch.setattr("src.agent.record_audit_log", mock_audit)

    # 1. Email request -> only sendgrid
    email_event = EventRequest(
        event_type="db_alert",
        message="Database replica lag exceeds threshold.",
        priority="HIGH",
        email="dba@company.com"
    )
    agent.process_event(email_event)
    assert calls["sendgrid"] == 1
    assert calls["pushover"] == 0
    assert calls["audit_logger"] == 0

    # 2. Push Notification request -> only pushover
    push_event = EventRequest(
        event_type="incident_page",
        message="P1 Outage in region us-east-1",
        priority="CRITICAL",
        conditions={"notification_enabled": True}
    )
    agent.process_event(push_event)
    assert calls["sendgrid"] == 1
    assert calls["pushover"] == 1
    assert calls["audit_logger"] == 0

    # 3. No action request -> 0 calls
    no_act_event = EventRequest(
        event_type="info_ping",
        message="Worker ping acknowledged.",
        priority="LOW"
    )
    agent.process_event(no_act_event)
    assert calls["sendgrid"] == 1
    assert calls["pushover"] == 1
    assert calls["audit_logger"] == 0

    # 4. Other tool request -> only audit_logger
    audit_event = EventRequest(
        event_type="audit_log",
        message="System certificate renewed.",
        priority="MEDIUM",
        conditions={"log_only": True}
    )
    agent.process_event(audit_event)
    assert calls["sendgrid"] == 1
    assert calls["pushover"] == 1
    assert calls["audit_logger"] == 1


# ==============================================================================
# TEST 7 — INPUT VALIDATION AND ERROR HANDLING
# ==============================================================================
def test_sendgrid_validation_error():
    """Verify SendGrid handles invalid email gracefully."""
    res_empty = send_email_via_sendgrid(recipient="", subject="Test", message="Body", force_mock=True)
    assert res_empty.status == "failed"
    assert "Missing recipient" in res_empty.error

    res_invalid = send_email_via_sendgrid(recipient="invalid-email-no-domain", subject="Test", message="Body", force_mock=True)
    assert res_invalid.status == "failed"
    assert "Invalid recipient email" in res_invalid.error


def test_pushover_validation_error():
    """Verify Pushover handles missing message gracefully."""
    res_empty = send_pushover_notification(message="", force_mock=True)
    assert res_empty.status == "failed"
    assert "Missing notification message" in res_empty.error
