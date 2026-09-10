"""
SendGrid Email Tool.
Supports real email delivery via SendGrid v3 REST API and safe mock testing mode.
"""

import uuid
import requests
from typing import Optional, Dict, Any
from src.config import Config
from src.models import ToolResult


def send_email_via_sendgrid(
    recipient: str,
    subject: str,
    message: str,
    from_email: Optional[str] = None,
    force_mock: Optional[bool] = None,
    simulate_failure: bool = False
) -> ToolResult:
    """
    Send an email using SendGrid API or safe mock mode.

    Parameters:
    - recipient: Target email address
    - subject: Email subject line
    - message: Plain text or HTML content
    - from_email: Sender email address (defaults to Config.SENDGRID_FROM_EMAIL)
    - force_mock: Override to force mock mode
    - simulate_failure: If True, simulates a tool failure for test scenarios

    Returns:
    - ToolResult: Structured execution output
    """
    # 1. Validation
    if not recipient or not recipient.strip():
        return ToolResult(
            status="failed",
            tool="sendgrid",
            mode="mock" if force_mock else "real",
            recipient=recipient,
            error="Validation Error: Missing recipient email address."
        )

    if "@" not in recipient or "." not in recipient.split("@")[-1]:
        return ToolResult(
            status="failed",
            tool="sendgrid",
            mode="mock" if force_mock else "real",
            recipient=recipient,
            error=f"Validation Error: Invalid recipient email format '{recipient}'."
        )

    if simulate_failure:
        return ToolResult(
            status="failed",
            tool="sendgrid",
            mode="mock",
            recipient=recipient,
            error="Simulated SendGrid API connection failure (HTTP 500: Internal Server Error)."
        )

    sender = from_email or Config.SENDGRID_FROM_EMAIL
    is_mock = force_mock if force_mock is not None else not Config.is_sendgrid_configured()

    # 2. Mock Mode Execution
    if is_mock:
        mock_msg_id = f"mock-sg-{uuid.uuid4().hex[:12]}"
        return ToolResult(
            status="success",
            tool="sendgrid",
            mode="mock",
            recipient=recipient,
            message_id=mock_msg_id,
            data={
                "sender": sender,
                "subject": subject,
                "content_preview": message[:100] + ("..." if len(message) > 100 else ""),
                "note": "Mock mode: email was simulated safely without external API delivery."
            }
        )

    # 3. Real Mode Execution via SendGrid v3 API
    api_key = Config.SENDGRID_API_KEY
    endpoint = "https://api.sendgrid.com/v3/mail/send"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "personalizations": [
            {
                "to": [{"email": recipient.strip()}],
                "subject": subject
            }
        ],
        "from": {"email": sender.strip()},
        "content": [
            {
                "type": "text/plain",
                "value": message
            }
        ]
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
        if response.status_code in (200, 202):
            sg_message_id = response.headers.get("X-Message-Id", f"sg-{uuid.uuid4().hex[:10]}")
            return ToolResult(
                status="success",
                tool="sendgrid",
                mode="real",
                recipient=recipient,
                message_id=sg_message_id,
                data={
                    "status_code": response.status_code,
                    "sender": sender,
                    "subject": subject
                }
            )
        else:
            return ToolResult(
                status="failed",
                tool="sendgrid",
                mode="real",
                recipient=recipient,
                error=f"SendGrid API Error (HTTP {response.status_code}): {response.text}"
            )
    except requests.RequestException as e:
        return ToolResult(
            status="failed",
            tool="sendgrid",
            mode="real",
            recipient=recipient,
            error=f"Network error communicating with SendGrid: {str(e)}"
        )
