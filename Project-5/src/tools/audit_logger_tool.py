"""
Other Tool: Communication Audit / Activity Logger Tool.
Demonstrates the third tool branch for logging/recording internal actions without external communication.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from src.models import ToolResult

# In-memory storage for audit records
AUDIT_RECORDS = []


def record_audit_log(
    record_type: str,
    description: str,
    details: Optional[Dict[str, Any]] = None,
    simulate_failure: bool = False
) -> ToolResult:
    """
    Record an internal communication audit / activity entry.

    Parameters:
    - record_type: Type/Category of audit entry (e.g., 'SYSTEM_LOG', 'ARCHIVE_EVENT')
    - description: Description of the logged action
    - details: Contextual key-value metadata
    - simulate_failure: If True, simulates tool failure

    Returns:
    - ToolResult: Structured execution output
    """
    if not description or not description.strip():
        return ToolResult(
            status="failed",
            tool="audit_logger",
            mode="mock",
            error="Validation Error: Missing log description."
        )

    if simulate_failure:
        return ToolResult(
            status="failed",
            tool="audit_logger",
            mode="mock",
            error="Simulated Audit Logger disk write / database lock error."
        )

    record_id = f"audit-{uuid.uuid4().hex[:8]}"
    entry = {
        "record_id": record_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "record_type": record_type,
        "description": description,
        "details": details or {}
    }
    AUDIT_RECORDS.append(entry)

    return ToolResult(
        status="success",
        tool="audit_logger",
        mode="mock",
        recipient="internal_audit_system",
        message_id=record_id,
        data={
            "record_id": record_id,
            "record_type": record_type,
            "total_records": len(AUDIT_RECORDS),
            "note": "Audit entry recorded successfully in internal audit ledger."
        }
    )
