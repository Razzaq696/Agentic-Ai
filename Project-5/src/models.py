"""
Data models for Intelligent Communication Assistant.
Includes EventRequest, ActionDecision, ToolResult, and ConfirmationLog.
"""

from enum import Enum
from typing import Any, Dict, Optional
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Supported actions determined by the Agent."""
    SEND_EMAIL = "SEND_EMAIL"
    SEND_NOTIFICATION = "SEND_NOTIFICATION"
    OTHER_TOOL = "OTHER_TOOL"
    NO_ACTION = "NO_ACTION"


class EventRequest(BaseModel):
    """
    Input structure representing an incoming event or communication request.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    event_type: str = Field(..., description="Type of event, e.g. 'server_alert', 'system_update', 'audit_event'")
    message: str = Field(..., description="Main event details / description")
    priority: str = Field(default="MEDIUM", description="Priority level: HIGH, MEDIUM, LOW, CRITICAL, INFO")
    recipient: Optional[str] = Field(default=None, description="Name or identifier of the recipient")
    email: Optional[str] = Field(default=None, description="Email address if email communication is required")
    notification_message: Optional[str] = Field(default=None, description="Short notification text if different from message")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Flags/conditions such as notification_enabled, log_only, etc.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual metadata")


class ActionDecision(BaseModel):
    """
    Decision produced by the Agent after situation analysis.
    """
    action: ActionType = Field(..., description="Selected action type")
    reasoning: str = Field(..., description="Explanation of why this action and tool were chosen")
    tool_name: Optional[str] = Field(default=None, description="Target tool: 'sendgrid', 'pushover', 'audit_logger', or None")
    tool_args: Dict[str, Any] = Field(default_factory=dict, description="Arguments prepared for the selected tool")


class ToolResult(BaseModel):
    """
    Standardized structured response returned by every tool.
    """
    status: str = Field(..., description="'success' or 'failed'")
    tool: str = Field(..., description="Name of the tool executed, e.g. 'sendgrid', 'pushover', 'audit_logger'")
    mode: str = Field(..., description="'real' or 'mock'")
    recipient: Optional[str] = Field(default=None, description="Target recipient / identifier")
    message_id: Optional[str] = Field(default=None, description="Message/Delivery identifier")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional result details")
    error: Optional[str] = Field(default=None, description="Error message if status is 'failed'")


class ConfirmationLog(BaseModel):
    """
    Concise structured confirmation and log record generated after agent workflow execution.
    """
    log_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    event_id: str
    event_type: str
    priority: str
    selected_action: ActionType
    tool_used: Optional[str] = None
    status: str = Field(..., description="'success', 'no_action', or 'failed'")
    summary: str = Field(..., description="Human-readable summary of the action taken")
    tool_result: Optional[ToolResult] = None
    error: Optional[str] = None
