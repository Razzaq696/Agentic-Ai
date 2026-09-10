"""SendGrid email delivery client for sending shopping decision reports."""

import json
from typing import Any, Dict, Optional, Tuple
import urllib.request
import urllib.error

from src.config import settings
from src.models.schemas import DecisionStatus, FinalDecision
from src.utils.logger import logger


class SendGridClient:
    """Client responsible for delivering the final shopping recommendation report via SendGrid."""

    SENDGRID_API_URL = "https://api.sendgrid.com/v3/mail/send"

    def __init__(
        self,
        api_key: Optional[str] = None,
        from_email: Optional[str] = None,
        to_email: Optional[str] = None,
        timeout: Optional[float] = None,
        enabled: Optional[bool] = None,
    ):
        self.api_key = api_key or settings.sendgrid_api_key
        self.from_email = from_email or settings.sendgrid_from_email
        self.to_email = to_email or settings.sendgrid_to_email
        self.timeout = timeout if timeout is not None else settings.sendgrid_timeout_sec
        self.enabled = enabled if enabled is not None else settings.sendgrid_enabled

    def format_email_content(self, decision: FinalDecision) -> Tuple[str, str, str]:
        """Generate subject line, plain-text body, and HTML body from FinalDecision.

        Returns:
            Tuple of (subject, plain_text, html_text).
        """
        category = decision.user_requirements.product_category or "Product"
        status = decision.decision_status

        if status == DecisionStatus.RECOMMENDED and decision.recommended_product:
            p_name = decision.recommended_product.name
            price_str = f"${decision.recommended_product.price:,.2f}" if decision.recommended_product.price else ""
            subject = f"Your Shopping Recommendation: {p_name} {price_str}"
        elif status == DecisionStatus.NO_SATISFYING_PRODUCT:
            subject = f"Shopping Search Update: No {category} fully met required criteria"
        else:
            subject = f"Shopping Decision Report: {category}"

        # Plain text
        lines = [
            "=======================================================",
            "           AI SMART SHOPPING DECISION REPORT           ",
            "=======================================================",
            "",
            f"Product Category: {category}",
            f"Status: {status.value.upper()}",
            "",
        ]

        if decision.recommended_product:
            rp = decision.recommended_product
            p_str = f"${rp.price:,.2f} {rp.currency}" if rp.price is not None else "Market pricing"
            lines.append(f"RECOMMENDED PRODUCT: {rp.name} ({p_str})")
            lines.append(f"Confidence: {decision.confidence * 100:.0f}%")
            lines.append(f"Source: {rp.url or rp.source}")
            lines.append(f"\nReason: {decision.key_reasons[0] if decision.key_reasons else 'Selected based on requirements.'}")
            if decision.tradeoffs:
                lines.append("\nTrade-offs:")
                for t in decision.tradeoffs:
                    lines.append(f" - {t}")
            if decision.alternative_product:
                ap = decision.alternative_product
                ap_price = f"${ap.price:,.2f} {ap.currency}" if ap.price is not None else "N/A"
                lines.append(f"\nClosest Alternative: {ap.name} ({ap_price})")

        elif status == DecisionStatus.NO_SATISFYING_PRODUCT:
            lines.append("No product fully satisfied your hard requirements and budget limits.")
            if decision.unmet_requirements:
                lines.append("\nUnmet Criteria:")
                for u in decision.unmet_requirements:
                    lines.append(f" - {u}")
            if decision.alternative_product:
                ap = decision.alternative_product
                lines.append(f"\nClosest Alternative: {ap.name}")

        lines.append("\n=======================================================")
        plain_text = "\n".join(lines)

        # HTML
        html_text = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background: #1e293b; color: #fff; padding: 15px; border-radius: 6px; text-align: center;">
                <h2 style="margin: 0;">AI Smart Shopping Recommendation</h2>
            </div>
            <div style="padding: 20px 0;">
                <p><strong>Category:</strong> {category}</p>
                <p><strong>Decision Status:</strong> {status.value.upper()}</p>
                <hr style="border: none; border-top: 1px solid #e2e8f0; margin: 15px 0;">
        """

        if decision.recommended_product:
            rp = decision.recommended_product
            p_str = f"${rp.price:,.2f} {rp.currency}" if rp.price is not None else "Market pricing"
            html_text += f"""
                <div style="background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 6px; padding: 15px; margin-bottom: 15px;">
                    <h3 style="color: #0f172a; margin-top: 0;">{rp.name} ({p_str})</h3>
                    <p><strong>Confidence:</strong> {decision.confidence * 100:.0f}%</p>
                    <p><strong>Reason:</strong> {decision.key_reasons[0] if decision.key_reasons else ''}</p>
                    <p><strong>Source/Citation:</strong> <a href="{rp.url or '#'}">{rp.url or rp.source}</a></p>
                </div>
            """
        elif status == DecisionStatus.NO_SATISFYING_PRODUCT:
            html_text += f"""
                <div style="background: #fffbeb; border: 1px solid #fde68a; border-radius: 6px; padding: 15px;">
                    <h3 style="color: #92400e; margin-top: 0;">No Product Satisfied All Requirements</h3>
                    <p>We do not recommend forcing an unsuitable purchase. Please review the criteria.</p>
                </div>
            """

        html_text += """
            </div>
            <p style="font-size: 12px; color: #94a3b8; text-align: center;">Generated by AI Smart Shopping Decision Agent.</p>
        </body>
        </html>
        """

        return subject, plain_text, html_text

    def build_payload(self, decision: FinalDecision) -> Dict[str, Any]:
        """Build SendGrid v3 mail send payload dictionary."""
        subject, plain_text, html_text = self.format_email_content(decision)

        return {
            "personalizations": [
                {
                    "to": [{"email": self.to_email}],
                    "subject": subject,
                }
            ],
            "from": {"email": self.from_email},
            "content": [
                {"type": "text/plain", "value": plain_text},
                {"type": "text/html", "value": html_text},
            ],
        }

    def send_report(self, decision: FinalDecision) -> Tuple[bool, str]:
        """Send the shopping recommendation report via SendGrid.

        Safe execution: handles missing API key, disabled state, timeouts, and network errors gracefully.

        Args:
            decision: Validated FinalDecision instance.

        Returns:
            Tuple of (success: bool, message: str).
        """
        if not self.enabled:
            logger.debug("SendGrid client: Integration is disabled via settings.")
            return False, "SendGrid integration disabled"

        if not self.api_key or not self.api_key.strip():
            logger.warning("SendGrid client: SENDGRID_API_KEY is not configured.")
            return False, "SENDGRID_API_KEY not configured"

        if not self.from_email or not self.to_email:
            logger.warning("SendGrid client: Sender or recipient email not configured.")
            return False, "SendGrid sender/recipient email not configured"

        payload = self.build_payload(decision)
        payload_bytes = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            self.SENDGRID_API_URL,
            data=payload_bytes,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            logger.info(f"SendGrid client: Sending email report to {self.to_email}...")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw_code = getattr(resp, "status", None)
                if raw_code is None and hasattr(resp, "getcode"):
                    raw_code = resp.getcode()
                try:
                    status_code = int(raw_code)
                except Exception:
                    status_code = 202
                # SendGrid returns 202 Accepted on success
                if status_code in (200, 202):
                    logger.info(f"SendGrid client: Email successfully accepted (HTTP {status_code})")
                    return True, f"Email sent via SendGrid (HTTP {status_code})"
                else:
                    logger.warning(f"SendGrid client: Unexpected status HTTP {status_code}")
                    return False, f"SendGrid responded with HTTP {status_code}"

        except urllib.error.HTTPError as he:
            logger.error(f"SendGrid client: HTTP error {he.code}: {he.reason}")
            return False, f"HTTP {he.code}: {he.reason}"
        except urllib.error.URLError as ue:
            logger.error(f"SendGrid client: Network error: {ue.reason}")
            return False, f"Network error: {ue.reason}"
        except Exception as e:
            logger.error(f"SendGrid client: Unexpected error: {e}")
            return False, f"Email sending failed: {str(e)}"
