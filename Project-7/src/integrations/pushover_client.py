"""Pushover notification client for delivering high-priority shopping event alerts."""

from typing import Any, Dict, Optional, Tuple
import urllib.parse
import urllib.request
import urllib.error

from src.config import settings
from src.models.schemas import DecisionStatus, FinalDecision
from src.utils.logger import logger


class PushoverClient:
    """Client responsible for delivering real-time mobile push notifications via Pushover."""

    PUSHOVER_API_URL = "https://api.pushover.net/1/messages.json"

    def __init__(
        self,
        user_key: Optional[str] = None,
        api_token: Optional[str] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self.user_key = user_key or settings.pushover_user_key
        self.api_token = api_token or settings.pushover_api_token
        self.timeout = timeout if timeout is not None else settings.pushover_timeout_sec
        self.enabled = enabled if enabled is not None else settings.pushover_enabled

    def format_notification(self, decision: FinalDecision) -> Tuple[str, str, int]:
        """Format notification title, concise message, and priority based on decision status.

        Only triggered for meaningful events:
        - Recommendation found (priority 0)
        - No satisfying product (priority 1)
        - Insufficient data (priority 0)

        Returns:
            Tuple of (title, message, priority).
        """
        category = decision.user_requirements.product_category or "Product"
        status = decision.decision_status

        if status == DecisionStatus.RECOMMENDED and decision.recommended_product:
            rp = decision.recommended_product
            p_price = f"${rp.price:,.2f}" if rp.price is not None else "Market price"
            title = f"Top Pick: {rp.name}"
            msg_parts = [
                f"Recommended: {rp.name} ({p_price})",
                f"Score: {decision.score_breakdown.total_score:.1f}/100" if decision.score_breakdown else "",
                f"Confidence: {decision.confidence * 100:.0f}%",
            ]
            message = " | ".join(filter(None, msg_parts))
            priority = 0

        elif status == DecisionStatus.NO_SATISFYING_PRODUCT:
            title = f"No {category} Met All Criteria"
            unmet_summary = ", ".join(decision.unmet_requirements[:2]) if decision.unmet_requirements else "Hard criteria unmet"
            message = f"Search ended without recommendation. Unmet: {unmet_summary}."
            priority = 1

        else:
            title = f"Shopping Alert: {category}"
            message = f"Status: {status.value}. Insufficient product data found."
            priority = 0

        return title, message, priority

    def send_notification(
        self,
        title: str,
        message: str,
        priority: int = 0,
    ) -> Tuple[bool, str]:
        """Dispatch a push notification to Pushover API.

        Safe execution: handles missing credentials, disabled state, timeouts, and network errors gracefully.

        Args:
            title: Notification title.
            message: Notification body text.
            priority: Pushover priority level (-2 to 2).

        Returns:
            Tuple of (success: bool, message: str).
        """
        if not self.enabled:
            logger.debug("Pushover client: Integration is disabled via settings.")
            return False, "Pushover integration disabled"

        if not self.user_key or not self.api_token:
            logger.warning("Pushover client: PUSHOVER_USER_KEY or PUSHOVER_API_TOKEN is not configured.")
            return False, "Pushover credentials not configured"

        payload = {
            "token": self.api_token,
            "user": self.user_key,
            "title": title,
            "message": message,
            "priority": priority,
        }
        encoded_data = urllib.parse.urlencode(payload).encode("utf-8")

        req = urllib.request.Request(
            self.PUSHOVER_API_URL,
            data=encoded_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            logger.info(f"Pushover client: Sending notification '{title}'...")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_code = getattr(resp, "status", None)
                if raw_code is None and hasattr(resp, "getcode"):
                    raw_code = resp.getcode()
                try:
                    status_code = int(raw_code)
                except Exception:
                    status_code = 200
                if 200 <= status_code < 300:
                    logger.info(f"Pushover client: Notification sent successfully (HTTP {status_code})")
                    return True, f"Notification sent via Pushover (HTTP {status_code})"
                else:
                    logger.warning(f"Pushover client: Unexpected status HTTP {status_code}")
                    return False, f"Pushover responded with HTTP {status_code}"

        except urllib.error.HTTPError as he:
            logger.error(f"Pushover client: HTTP error {he.code}: {he.reason}")
            return False, f"HTTP {he.code}: {he.reason}"
        except urllib.error.URLError as ue:
            logger.error(f"Pushover client: Network error: {ue.reason}")
            return False, f"Network error: {ue.reason}"
        except Exception as e:
            logger.error(f"Pushover client: Unexpected error: {e}")
            return False, f"Notification failed: {str(e)}"

    def notify_decision(self, decision: FinalDecision) -> Tuple[bool, str]:
        """Helper to format and dispatch a notification for a FinalDecision."""
        title, message, priority = self.format_notification(decision)
        return self.send_notification(title=title, message=message, priority=priority)
