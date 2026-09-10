"""
Automated Integration Tests for Streamlit UI (app.py) using Streamlit's AppTest framework.
Tests:
1. Initial app render
2. Scenario 1: Email Decision (SendGrid)
3. Scenario 2: Notification Decision (Pushover)
4. Scenario 3: No Action Decision
5. Scenario 4: Tool Failure Simulation
6. Scenario 5: Empty Input Validation Error
"""

import pytest
from streamlit.testing.v1 import AppTest


def test_streamlit_initial_load():
    """Verify app renders initial form and defaults correctly."""
    at = AppTest.from_file("app.py", default_timeout=10).run()
    assert not at.exception
    assert at.text_input[0].value == "security_alert"
    selectbox_values = [sb.value for sb in at.selectbox]
    assert "HIGH" in selectbox_values
    assert at.button[0].label == "🚀 Analyze & Process"


def test_streamlit_email_flow():
    """Verify default High Priority Security Alert triggers SEND_EMAIL -> SendGrid."""
    at = AppTest.from_file("app.py", default_timeout=10).run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.success) >= 1
    # Check session state confirmation
    conf = at.session_state["last_confirmation"]
    assert conf is not None
    assert conf.selected_action.value == "SEND_EMAIL"
    assert conf.tool_used == "sendgrid"
    assert conf.status == "success"
    assert conf.tool_result.mode == "mock"


def test_streamlit_notification_flow():
    """Verify Critical Outage with notification triggers SEND_NOTIFICATION -> Pushover."""
    at = AppTest.from_file("app.py", default_timeout=10).run()

    # Change to preset 2 (Critical System Outage)
    at.sidebar.selectbox[0].select("2. Critical Priority - System Outage (Pushover Push)").run()
    at.button[0].click().run()

    assert not at.exception
    conf = at.session_state["last_confirmation"]
    assert conf is not None
    assert conf.selected_action.value == "SEND_NOTIFICATION"
    assert conf.tool_used == "pushover"
    assert conf.status == "success"
    assert conf.tool_result.mode == "mock"


def test_streamlit_no_action_flow():
    """Verify Low Priority Routine Healthcheck triggers NO_ACTION."""
    at = AppTest.from_file("app.py", default_timeout=10).run()

    # Change to preset 3 (Low Priority Healthcheck)
    at.sidebar.selectbox[0].select("3. Low Priority - Routine Healthcheck (No Action)").run()
    at.button[0].click().run()

    assert not at.exception
    conf = at.session_state["last_confirmation"]
    assert conf is not None
    assert conf.selected_action.value == "NO_ACTION"
    assert conf.tool_used is None
    assert conf.status == "no_action"
    assert len(at.info) >= 1


def test_streamlit_tool_failure_flow():
    """Verify simulated tool failure is captured gracefully in the UI."""
    at = AppTest.from_file("app.py", default_timeout=10).run()

    # Change to preset 4 (Tool Failure Simulation)
    at.sidebar.selectbox[0].select("4. Tool Failure Simulation (Error Handling)").run()
    at.button[0].click().run()

    assert not at.exception
    conf = at.session_state["last_confirmation"]
    assert conf is not None
    assert conf.selected_action.value == "SEND_EMAIL"
    assert conf.status == "failed"
    assert "Simulated SendGrid API connection failure" in conf.error
    assert len(at.error) >= 1


def test_streamlit_empty_input_validation():
    """Verify empty event message triggers validation error."""
    at = AppTest.from_file("app.py", default_timeout=10).run()

    # Clear message input
    at.text_area[0].input("").run()
    at.button[0].click().run()

    assert not at.exception
    assert len(at.error) >= 1
    assert "Please enter an event or request." in at.error[0].value
