"""
Intelligent Communication Assistant - Interactive Demo Script.
Demonstrates the full workflow:
EVENT / REQUEST -> AGENT / LLM (Analyze Situation + Decision Making) -> TOOL CALL -> CONFIRMATION / LOG
"""

from src.models import EventRequest
from src.agent import CommunicationAgent
from src.logger import ConfirmationLogger


def main():
    print("=" * 70)
    print("PROJECT 5 - INTELLIGENT COMMUNICATION ASSISTANT (PHASE 1 CORE BACKEND)")
    print("=" * 70)
    print("Architecture Flow:")
    print("  EVENT -> AGENT/LLM (Analyze & Decide) -> [SENDGRID | PUSHOVER | AUDIT LOGGER] -> CONFIRMATION LOG\n")

    logger = ConfirmationLogger()
    agent = CommunicationAgent(logger=logger)

    scenarios = [
        (
            "SCENARIO 1: High-Priority Security Alert (SendGrid Email Decision)",
            EventRequest(
                event_type="security_alert",
                message="Suspicious root access attempt detected on Production Server cluster EU-WEST.",
                priority="HIGH",
                recipient="Security Operations Team",
                email="soc@enterprise.com"
            ),
            {"force_mock": True, "simulate_tool_failure": False}
        ),
        (
            "SCENARIO 2: Critical Outage Push Alert (Pushover Notification Decision)",
            EventRequest(
                event_type="system_outage",
                message="Core API Gateway response code 503 rate spiked to 85%.",
                priority="CRITICAL",
                recipient="Incident Commander",
                notification_message="CRITICAL: API Gateway 503 Spike (85%). Immediate triaging required!",
                conditions={"notification_enabled": True}
            ),
            {"force_mock": True, "simulate_tool_failure": False}
        ),
        (
            "SCENARIO 3: Low-Priority Informational Update (No Action Decision)",
            EventRequest(
                event_type="routine_backup_success",
                message="Nightly database backup completed in 4 minutes. SHA-256 verified.",
                priority="LOW",
                recipient="Backup Admin"
            ),
            {"force_mock": True, "simulate_tool_failure": False}
        ),
        (
            "SCENARIO 4: Simulated Tool Failure (Graceful Error Capture)",
            EventRequest(
                event_type="database_failover",
                message="Database failover initiated to replica-2.",
                priority="HIGH",
                recipient="Lead DBA",
                email="dba-lead@enterprise.com"
            ),
            {"force_mock": True, "simulate_tool_failure": True}
        ),
        (
            "SCENARIO 5: Internal Audit Record (Other Tool - Audit Logger Decision)",
            EventRequest(
                event_type="audit_permission_change",
                message="User role 'developer' elevated to 'deployer' on staging pipeline.",
                priority="MEDIUM",
                recipient="Compliance Officer",
                conditions={"log_only": True}
            ),
            {"force_mock": True, "simulate_tool_failure": False}
        ),
    ]

    for title, event, flags in scenarios:
        print(f"\n>>> Running {title}...")
        confirmation = agent.process_event(
            event=event,
            force_mock=flags.get("force_mock", True),
            simulate_tool_failure=flags.get("simulate_tool_failure", False)
        )
        print(ConfirmationLogger.format_log_display(confirmation))

    print("\n" + "=" * 70)
    print("ALL SCENARIOS PROCESSED SUCCESSFULLY. TOTAL LOGS GENERATED:", len(logger.get_history()))
    print("=" * 70)


if __name__ == "__main__":
    main()
