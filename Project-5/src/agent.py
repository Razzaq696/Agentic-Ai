"""
Intelligent Communication Assistant - Agent & Decision Making Engine.
Analyzes situation, determines communication necessity, selects single appropriate tool,
and coordinates execution flow with structured confirmation logging.
"""

import json
import re
from typing import Optional, Dict, Any

from src.config import Config
from src.models import (
    EventRequest,
    ActionDecision,
    ActionType,
    ToolResult,
    ConfirmationLog
)
from src.tools.sendgrid_tool import send_email_via_sendgrid
from src.tools.pushover_tool import send_pushover_notification
from src.tools.audit_logger_tool import record_audit_log
from src.logger import ConfirmationLogger, global_logger


class CommunicationAgent:
    """
    Core Intelligent Agent that analyzes events, makes tool-calling decisions,
    and coordinates execution.
    """

    def __init__(self, logger: Optional[ConfirmationLogger] = None):
        self.logger = logger or global_logger

    def analyze_and_decide(self, event: EventRequest, use_llm_if_available: bool = True) -> ActionDecision:
        """
        Analyze incoming situation and decide appropriate action and tool.

        Flow:
        1. Attempt LLM analysis if API key is configured and enabled.
        2. Fallback to deterministic expert decision engine for guaranteed zero-failure execution.
        """
        if use_llm_if_available and Config.is_llm_configured():
            llm_decision = self._llm_decision_making(event)
            if llm_decision:
                return llm_decision

        return self._deterministic_decision_making(event)

    def _deterministic_decision_making(self, event: EventRequest) -> ActionDecision:
        """
        Rule-based situation analysis and decision engine.
        Implements assignment requirements:
        - HIGH priority + email recipient -> SENDGRID EMAIL
        - HIGH priority + notification enabled / urgent -> PUSHOVER NOTIFICATION
        - Internal audit / log request -> OTHER_TOOL (audit_logger)
        - Low priority / informational / no communication required -> NO_ACTION
        """
        priority = (event.priority or "MEDIUM").upper()
        event_type = (event.event_type or "").lower()
        conditions = event.conditions or {}

        # 1. Check for explicit NO_ACTION conditions or Low priority informational events
        if conditions.get("no_communication") is True or priority in ("LOW", "INFO"):
            return ActionDecision(
                action=ActionType.NO_ACTION,
                reasoning=f"Event is '{priority}' priority informational update without urgent communication requirements.",
                tool_name=None,
                tool_args={}
            )

        # 2. Check for Other Tool (Internal Audit / Action Logger)
        if (
            conditions.get("log_only") is True or
            "audit" in event_type or
            "archive" in event_type or
            conditions.get("target_tool") == "audit_logger"
        ):
            return ActionDecision(
                action=ActionType.OTHER_TOOL,
                reasoning=f"Event type '{event.event_type}' requires recording in internal audit ledger without external alerting.",
                tool_name="audit_logger",
                tool_args={
                    "record_type": event.event_type.upper(),
                    "description": event.message,
                    "details": {
                        "priority": event.priority,
                        "recipient": event.recipient,
                        "conditions": event.conditions,
                        "metadata": event.metadata
                    }
                }
            )

        # 3. High Priority Decision Making
        if priority in ("HIGH", "CRITICAL"):
            # Check for email route
            if event.email or conditions.get("prefer_email") or conditions.get("target_tool") == "sendgrid":
                subject = f"[{priority} ALERT] {event.event_type.replace('_', ' ').title()}"
                body = (
                    f"Priority: {priority}\n"
                    f"Event Type: {event.event_type}\n"
                    f"Recipient: {event.recipient or 'User'}\n\n"
                    f"Message:\n{event.message}"
                )
                return ActionDecision(
                    action=ActionType.SEND_EMAIL,
                    reasoning=f"High priority ({priority}) event with email recipient specified. Routing to SendGrid Email Tool.",
                    tool_name="sendgrid",
                    tool_args={
                        "recipient": event.email or f"{event.recipient or 'user'}@example.com",
                        "subject": subject,
                        "message": body
                    }
                )

            # Check for Push Notification route
            if (
                conditions.get("notification_enabled") is True or
                event.notification_message or
                conditions.get("prefer_push") or
                conditions.get("target_tool") == "pushover" or
                not event.email
            ):
                notif_text = event.notification_message or f"[{priority}] {event.event_type}: {event.message}"
                notif_title = f"{priority} Alert: {event.event_type.replace('_', ' ').title()}"
                push_priority = 1 if priority == "CRITICAL" else 0

                return ActionDecision(
                    action=ActionType.SEND_NOTIFICATION,
                    reasoning=f"High priority ({priority}) event requiring urgent push notification. Routing to Pushover Tool.",
                    tool_name="pushover",
                    tool_args={
                        "message": notif_text,
                        "title": notif_title,
                        "priority": push_priority
                    }
                )

        # 4. Default for Medium priority with email
        if event.email:
            return ActionDecision(
                action=ActionType.SEND_EMAIL,
                reasoning="Standard communication event with email recipient provided. Routing to SendGrid.",
                tool_name="sendgrid",
                tool_args={
                    "recipient": event.email,
                    "subject": f"Notice: {event.event_type.replace('_', ' ').title()}",
                    "message": event.message
                }
            )

        # Default fallback: No communication required
        return ActionDecision(
            action=ActionType.NO_ACTION,
            reasoning="Situation analysis determined no active external communication channel was matched or required.",
            tool_name=None,
            tool_args={}
        )

    def _llm_decision_making(self, event: EventRequest) -> Optional[ActionDecision]:
        """
        LLM-powered situation analysis using Gemini or OpenAI.
        """
        prompt = f"""
You are the Intelligent Communication Assistant Decision Agent.
Analyze the following event and decide the appropriate action:

Event Data:
- Event Type: {event.event_type}
- Priority: {event.priority}
- Message: {event.message}
- Recipient: {event.recipient}
- Email: {event.email}
- Notification Message: {event.notification_message}
- Conditions: {json.dumps(event.conditions)}

Rules:
1. If HIGH/CRITICAL priority and email is provided or requested -> action: "SEND_EMAIL", tool_name: "sendgrid".
2. If HIGH/CRITICAL priority and urgent push/notification is needed -> action: "SEND_NOTIFICATION", tool_name: "pushover".
3. If internal logging/audit/record only -> action: "OTHER_TOOL", tool_name: "audit_logger".
4. If LOW priority or informational with no communication need -> action: "NO_ACTION", tool_name: null.

Respond ONLY with valid JSON matching this schema:
{{
    "action": "SEND_EMAIL" | "SEND_NOTIFICATION" | "OTHER_TOOL" | "NO_ACTION",
    "reasoning": "brief explanation",
    "tool_name": "sendgrid" | "pushover" | "audit_logger" | null,
    "tool_args": {{}}
}}
"""
        # Try Gemini
        if Config.GEMINI_API_KEY:
            try:
                from google import genai
                client = genai.Client(api_key=Config.GEMINI_API_KEY)
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                if response and response.text:
                    parsed = self._extract_json(response.text)
                    if parsed:
                        return ActionDecision(**parsed)
            except Exception:
                pass

        # Try OpenAI
        if Config.OPENAI_API_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=Config.OPENAI_API_KEY)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0
                )
                text = response.choices[0].message.content
                if text:
                    parsed = self._extract_json(text)
                    if parsed:
                        return ActionDecision(**parsed)
            except Exception:
                pass

        return None

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from LLM response text."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return None

    def execute_decision(
        self,
        decision: ActionDecision,
        event: EventRequest,
        force_mock: Optional[bool] = None,
        simulate_tool_failure: bool = False
    ) -> Optional[ToolResult]:
        """
        Real Tool Calling: Execute ONLY the single tool chosen by the Agent's decision.
        Does NOT execute all tools.
        """
        if decision.action == ActionType.NO_ACTION or not decision.tool_name:
            return None

        tool_name = decision.tool_name.lower()
        args = decision.tool_args.copy()

        # Route to SendGrid Tool
        if tool_name == "sendgrid" or decision.action == ActionType.SEND_EMAIL:
            return send_email_via_sendgrid(
                recipient=args.get("recipient") or event.email or "",
                subject=args.get("subject", f"Alert: {event.event_type}"),
                message=args.get("message", event.message),
                force_mock=force_mock,
                simulate_failure=simulate_tool_failure
            )

        # Route to Pushover Tool
        elif tool_name == "pushover" or decision.action == ActionType.SEND_NOTIFICATION:
            return send_pushover_notification(
                message=args.get("message") or event.notification_message or event.message,
                title=args.get("title", f"Alert: {event.event_type}"),
                priority=args.get("priority", 0),
                force_mock=force_mock,
                simulate_failure=simulate_tool_failure
            )

        # Route to Other Tool (Audit Logger)
        elif tool_name == "audit_logger" or decision.action == ActionType.OTHER_TOOL:
            return record_audit_log(
                record_type=args.get("record_type", event.event_type.upper()),
                description=args.get("description", event.message),
                details=args.get("details", {"event_id": event.event_id}),
                simulate_failure=simulate_tool_failure
            )

        # Unknown tool handling
        else:
            return ToolResult(
                status="failed",
                tool=tool_name,
                mode="mock",
                error=f"Unrecognized tool '{tool_name}' for action '{decision.action.value}'."
            )

    def process_event(
        self,
        event: EventRequest,
        force_mock: Optional[bool] = None,
        simulate_tool_failure: bool = False,
        use_llm_if_available: bool = True
    ) -> ConfirmationLog:
        """
        Full End-to-End Flow:
        1. EVENT / REQUEST
        2. AGENT / LLM (Analyze Situation + Decision Making)
        3. TOOL CALL (SendGrid OR Pushover OR Other Tool OR None)
        4. CONFIRMATION / LOG DATA
        """
        try:
            # 1 & 2: Analyze & Decide
            decision = self.analyze_and_decide(event, use_llm_if_available=use_llm_if_available)

            # 3: Tool Call (Selected Tool Only)
            tool_result = self.execute_decision(
                decision=decision,
                event=event,
                force_mock=force_mock,
                simulate_tool_failure=simulate_tool_failure
            )

            # 4: Generate Structured Confirmation Log
            confirmation = self.logger.create_and_record(
                event=event,
                decision=decision,
                tool_result=tool_result
            )
            return confirmation

        except Exception as ex:
            # Basic error handling: capture exception gracefully without crashing
            fallback_decision = ActionDecision(
                action=ActionType.NO_ACTION,
                reasoning=f"System error during agent processing: {str(ex)}"
            )
            return self.logger.create_and_record(
                event=event,
                decision=fallback_decision,
                tool_result=None,
                override_error=f"Agent Processing Exception: {str(ex)}"
            )
