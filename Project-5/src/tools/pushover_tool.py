"""
Pushover Notification Tool.
Supports push notifications via Pushover REST API and safe mock testing mode.
"""

import uuid
import requests
from typing import Optional, Dict, Any
from src.config import Config
from src.models import ToolResult


def send_pushover_notification(
    message: str,
    title: Optional[str] = "Alert Notification",
    priority: int = 0,
    user_key: Optional[str] = None,
    api_token: Optional[str] = None,
    force_mock: Optional[bool] = None,
    simulate_failure: bool = False
) -> ToolResult:
    """
    Send a push notification using Pushover API or safe mock mode.

    Parameters:
    - message: The notification message body
    - title: Optional title for the notification
    - priority: Pushover priority integer (-2 to 2)
    - user_key: Target user key (defaults to Config.PUSHOVER_USER_KEY)
    - api_token: Application API token (defaults to Config.PUSHOVER_API_TOKEN)
    - force_mock: Override to force mock mode
    - simulate_failure: If True, simulates a tool failure for test scenarios

    Returns:
    - ToolResult: Structured execution output
    """
    # 1. Validation
    if not message or not message.strip():
        return ToolResult(
            status="failed",
            tool="pushover",
            mode="mock" if force_mock else "real",
            error="Validation Error: Missing notification message body."
        )

    if simulate_failure:
        return ToolResult(
            status="failed",
            tool="pushover",
            mode="mock",
            error="Simulated Pushover API network timeout / service unavailable."
        )

    target_user = user_key or Config.PUSHOVER_USER_KEY
    token = api_token or Config.PUSHOVER_API_TOKEN
    is_mock = force_mock if force_mock is not None else not Config.is_pushover_configured()

    # 2. Mock Mode Execution
    if is_mock:
        mock_request_id = f"mock-push-{uuid.uuid4().hex[:12]}"
        return ToolResult(
            status="success",
            tool="pushover",
            mode="mock",
            recipient=target_user if target_user else "mock_device_user",
            message_id=mock_request_id,
            data={
                "title": title,
                "priority": priority,
                "message_preview": message[:100] + ("..." if len(message) > 100 else ""),
                "note": "Mock mode: push notification was simulated safely without external API delivery."
            }
        )

    # 3. Real Mode Execution via Pushover API
    endpoint = "https://api.pushover.net/1/messages.json"
    payload = {
        "token": token,
        "user": target_user,
        "message": message,
        "title": title,
        "priority": priority
    }

    try:
        response = requests.post(endpoint, data=payload, timeout=10)
        res_json = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}

        if response.status_code == 200 and res_json.get("status") == 1:
            return ToolResult(
                status="success",
                tool="pushover",
                mode="real",
                recipient=target_user,
                message_id=res_json.get("request", f"push-{uuid.uuid4().hex[:10]}"),
                data={
                    "status_code": response.status_code,
                    "title": title,
                    "priority": priority
                }
            )
        else:
            errors = res_json.get("errors", [response.text])
            return ToolResult(
                status="failed",
                tool="pushover",
                mode="real",
                recipient=target_user,
                error=f"Pushover API Error (HTTP {response.status_code}): {'; '.join(errors)}"
            )
    except requests.RequestException as e:
        return ToolResult(
            status="failed",
            tool="pushover",
            mode="real",
            error=f"Network error communicating with Pushover: {str(e)}"
        )
