"""
Confirmation and Structured Logging Module.
Maintains in-memory records of communication decisions and tool execution confirmations.
"""

import json
from typing import List, Optional
from src.models import ConfirmationLog, EventRequest, ActionDecision, ToolResult, ActionType


class ConfirmationLogger:
    """Manages structured confirmation logs and history in-memory."""

    def __init__(self):
        self._history: List[ConfirmationLog] = []

    def create_and_record(
        self,
        event: EventRequest,
        decision: ActionDecision,
        tool_result: Optional[ToolResult] = None,
        override_error: Optional[str] = None
    ) -> ConfirmationLog:
        """
        Create a ConfirmationLog instance, store it in history, and return it.
        """
        # Determine overall execution status
        if decision.action == ActionType.NO_ACTION:
            status = "no_action"
            summary = f"No communication required. Reason: {decision.reasoning}"
        elif tool_result:
            status = tool_result.status
            if tool_result.status == "success":
                summary = (
                    f"Successfully processed '{event.event_type}' via tool '{tool_result.tool}' "
                    f"[{tool_result.mode.upper()} mode] for action '{decision.action.value}'. "
                    f"ID: {tool_result.message_id or 'N/A'}"
                )
            else:
                summary = f"Execution failed for event '{event.event_type}' using tool '{tool_result.tool}'. Error: {tool_result.error}"
        else:
            status = "failed"
            summary = f"Action '{decision.action.value}' failed before tool execution."

        error_msg = override_error or (tool_result.error if tool_result else None)

        log_entry = ConfirmationLog(
            event_id=event.event_id,
            event_type=event.event_type,
            priority=event.priority,
            selected_action=decision.action,
            tool_used=decision.tool_name if decision.action != ActionType.NO_ACTION else None,
            status=status,
            summary=summary,
            tool_result=tool_result,
            error=error_msg
        )

        self._history.append(log_entry)
        return log_entry

    def get_history(self) -> List[ConfirmationLog]:
        """Return all recorded confirmation logs."""
        return list(self._history)

    def clear(self) -> None:
        """Clear log history."""
        self._history.clear()

    @staticmethod
    def format_log_display(log: ConfirmationLog) -> str:
        """Format a confirmation log as a clean readable block."""
        lines = [
            "----------------------------------------------------------------",
            f"CONFIRMATION LOG #{log.log_id} | Timestamp: {log.timestamp}",
            f"  Event ID        : {log.event_id} ({log.event_type}, Priority: {log.priority})",
            f"  Selected Action : {log.selected_action.value}",
            f"  Tool Used       : {log.tool_used or 'NONE'}",
            f"  Status          : {log.status.upper()}",
            f"  Summary         : {log.summary}"
        ]
        if log.tool_result:
            lines.append(f"  Tool Mode       : {log.tool_result.mode.upper()}")
            if log.tool_result.recipient:
                lines.append(f"  Recipient       : {log.tool_result.recipient}")
            if log.tool_result.message_id:
                lines.append(f"  Message ID      : {log.tool_result.message_id}")
        if log.error:
            lines.append(f"  Error Details   : {log.error}")
        lines.append("----------------------------------------------------------------")
        return "\n".join(lines)


# Global singleton instance
global_logger = ConfirmationLogger()
