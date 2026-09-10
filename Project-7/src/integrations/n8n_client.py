"""n8n webhook integration client for workflow automation."""

from datetime import datetime, timezone
import json
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error

from src.config import settings
from src.models.schemas import FinalDecision
from src.utils.logger import logger


class N8nClient:
    """Client responsible for dispatching validated shopping decisions to an n8n webhook."""

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self.webhook_url = webhook_url or settings.n8n_webhook_url
        self.timeout = timeout if timeout is not None else settings.n8n_timeout_sec
        self.enabled = enabled if enabled is not None else settings.n8n_enabled

    def build_payload(self, decision: FinalDecision) -> Dict[str, Any]:
        """Construct structured JSON payload from FinalDecision for n8n consumption.

        Args:
            decision: Validated FinalDecision instance.

        Returns:
            Structured dictionary payload.
        """
        rec = decision.recommended_product
        alt = decision.alternative_product
        score = decision.score_breakdown

        return {
            "event": "shopping_recommendation_completed",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision_status": decision.decision_status.value,
            "confidence": decision.confidence,
            "category": decision.user_requirements.product_category,
            "recommended_product": {
                "name": rec.name if rec else None,
                "brand": rec.brand if rec else None,
                "price": rec.price if rec else None,
                "currency": rec.currency if rec else "USD",
                "source": rec.url or rec.source if rec else None,
                "specifications": rec.specifications if rec else {},
                "features": rec.features if rec else [],
            } if rec else None,
            "alternative_product": {
                "name": alt.name if alt else None,
                "brand": alt.brand if alt else None,
                "price": alt.price if alt else None,
                "currency": alt.currency if alt else "USD",
                "source": alt.url or alt.source if alt else None,
            } if alt else None,
            "score_breakdown": score.model_dump() if score else None,
            "key_reasons": decision.key_reasons,
            "tradeoffs": decision.tradeoffs,
            "unmet_requirements": decision.unmet_requirements,
            "sources": decision.sources,
        }

    def send_decision(self, decision: FinalDecision) -> Tuple[bool, str]:
        """Send validated shopping decision to n8n webhook.

        Safe execution: handles missing URL, disabled status, timeouts, and HTTP errors gracefully.

        Args:
            decision: Validated FinalDecision model.

        Returns:
            Tuple of (success: bool, message: str).
        """
        if not self.enabled:
            logger.debug("n8n client: Integration is disabled via settings.")
            return False, "n8n integration disabled"

        if not self.webhook_url or not self.webhook_url.strip():
            logger.warning("n8n client: N8N_WEBHOOK_URL is not configured.")
            return False, "N8N_WEBHOOK_URL not configured"

        payload = self.build_payload(decision)
        payload_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.webhook_url,
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            logger.info("n8n client: Forwarding final decision to n8n webhook...")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_code = getattr(resp, "status", None)
                if raw_code is None and hasattr(resp, "getcode"):
                    raw_code = resp.getcode()
                try:
                    status_code = int(raw_code)
                except Exception:
                    status_code = 200
                if 200 <= status_code < 300:
                    logger.info(f"n8n client: Successfully delivered decision (HTTP {status_code})")
                    return True, f"Delivered to n8n (HTTP {status_code})"
                else:
                    logger.warning(f"n8n client: Received unexpected status HTTP {status_code}")
                    return False, f"n8n responded with HTTP {status_code}"

        except urllib.error.HTTPError as he:
            logger.error(f"n8n client: HTTP error during webhook delivery: HTTP {he.code} {he.reason}")
            return False, f"HTTP {he.code}: {he.reason}"
        except urllib.error.URLError as ue:
            logger.error(f"n8n client: Network/URL error during webhook delivery: {ue.reason}")
            return False, f"Network error: {ue.reason}"
        except Exception as e:
            logger.error(f"n8n client: Unexpected error during webhook delivery: {e}")
            return False, f"Delivery failed: {str(e)}"
