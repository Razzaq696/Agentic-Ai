"""
Date/Time Utility Tool for Intelligent Task Execution Agent.
Phase 2: Tool Development

Provides the current real-world date, time, weekday, and local timezone.
"""

from datetime import datetime, timezone
from langchain_core.tools import tool


@tool
def get_current_datetime(query: str = "all") -> str:
    """
    Retrieve the current real-world date, time, day of the week, and timezone.
    Use this tool whenever the user asks for today's date, the current time,
    what day it is, or any time-sensitive temporal information.

    Args:
        query: Optional specifier for the requested temporal component (e.g., 'date', 'time', 'weekday', 'all').

    Returns:
        A human-readable string containing the requested date and time information.
    """
    try:
        now_local = datetime.now().astimezone()
        now_utc = datetime.now(timezone.utc)

        q = (query or "").lower().strip()

        if q in ("date", "today", "day"):
            return f"Current Date: {now_local.strftime('%A, %B %d, %Y')}"

        if q in ("time", "now"):
            return f"Current Local Time: {now_local.strftime('%I:%M:%S %p %Z')}"

        if q in ("weekday", "day of week", "day_of_week"):
            return f"Current Day of the Week: {now_local.strftime('%A')}"

        # Default comprehensive output
        return (
            f"Current Date: {now_local.strftime('%A, %B %d, %Y')}\n"
            f"Current Local Time: {now_local.strftime('%I:%M:%S %p %Z')}\n"
            f"ISO Format: {now_local.isoformat()}\n"
            f"UTC Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    except Exception as e:
        return f"Error retrieving current date and time: {e}"
