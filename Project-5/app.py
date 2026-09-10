"""
Project 5 — Intelligent Communication Assistant
Premium Streamlit Web Application (Upgraded UI/UX Edition)
Reuses Phase 1 Agent, Decision Engine, Tools, and Confirmation Logging.
"""

import streamlit as st
import json
from datetime import datetime

# Import existing Phase 1 backend modules without duplication
from src.models import EventRequest, ActionType
from src.agent import CommunicationAgent
from src.logger import ConfirmationLogger
from src.config import Config


# -----------------------------------------------------------------------------
# Page Configuration & Advanced Theme Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Intelligent Communication Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, premium Glassmorphic UI/UX
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    /* Global Typography */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Hero Header Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E1B4B 50%, #312E81 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 26px 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.4), 0 8px 10px -6px rgba(15, 23, 42, 0.3);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.25) 0%, rgba(99, 102, 241, 0) 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #FFFFFF 0%, #E0E7FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 6px 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #C7D2FE;
        margin: 0;
        line-height: 1.5;
        max-width: 850px;
    }

    /* Metric Summary Cards */
    .metric-container {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 24px;
    }
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 18px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02), 0 1px 2px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748B;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0F172A;
    }

    /* Pipeline Flow Stepper */
    .pipeline-wrapper {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 20px;
        margin-bottom: 22px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 8px;
        overflow-x: auto;
    }
    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #475569;
        white-space: nowrap;
    }
    .step-num {
        width: 22px;
        height: 22px;
        border-radius: 50%;
        background: #E2E8F0;
        color: #475569;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 700;
    }
    .pipeline-step.active .step-num {
        background: #4F46E5;
        color: white;
    }
    .pipeline-step.active {
        color: #4F46E5;
    }
    .pipeline-arrow {
        color: #CBD5E1;
        font-size: 0.9rem;
    }

    /* Decision & Confirmation Cards */
    .glass-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 22px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        margin-bottom: 16px;
        height: 100%;
    }
    .card-heading {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Badges */
    .action-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        letter-spacing: 0.02em;
    }
    .badge-email {
        background: #EEF2FF;
        color: #4338CA;
        border: 1px solid #C7D2FE;
    }
    .badge-push {
        background: #FAF5FF;
        color: #7E22CE;
        border: 1px solid #E9D5FF;
    }
    .badge-audit {
        background: #ECFDF5;
        color: #047857;
        border: 1px solid #A7F3D0;
    }
    .badge-none {
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
    }

    .mode-chip {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .mode-mock {
        background: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    .mode-real {
        background: #DCFCE7;
        color: #166534;
        border: 1px solid #BBF7D0;
    }

    .detail-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #F1F5F9;
        font-size: 0.9rem;
    }
    .detail-row:last-child {
        border-bottom: none;
    }
    .detail-label {
        color: #64748B;
        font-weight: 500;
    }
    .detail-value {
        color: #0F172A;
        font-weight: 600;
    }

    .reason-box {
        background: #F8FAFC;
        border-left: 4px solid #6366F1;
        padding: 12px 14px;
        border-radius: 0 8px 8px 0;
        font-size: 0.88rem;
        color: #334155;
        line-height: 1.5;
        margin-top: 12px;
    }

    /* Timeline items */
    .timeline-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 14px;
        background: #F8FAFC;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Initialize Agent & Logger in Session State
# -----------------------------------------------------------------------------
if "logger" not in st.session_state:
    st.session_state.logger = ConfirmationLogger()

if "agent" not in st.session_state:
    st.session_state.agent = CommunicationAgent(logger=st.session_state.logger)

if "last_confirmation" not in st.session_state:
    st.session_state.last_confirmation = None


# -----------------------------------------------------------------------------
# Sidebar: Quick Presets & Control Center
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ Control Center")
    st.caption("Select a scenario preset or customize parameters.")

    preset_choice = st.selectbox(
        "Load Example Preset:",
        [
            "Custom Input",
            "1. High Priority - Security Alert (SendGrid Email)",
            "2. Critical Priority - System Outage (Pushover Push)",
            "3. Low Priority - Routine Healthcheck (No Action)",
            "4. Tool Failure Simulation (Error Handling)",
            "5. Medium Priority - SOC2 Audit (Other Tool)"
        ]
    )

    st.markdown("---")
    st.markdown("#### ⚙️ Runtime Settings")
    force_mock_mode = st.checkbox(
        "Force Safe Mock Mode",
        value=True,
        help="Simulates external APIs without transmitting live messages over the internet."
    )
    simulate_failure = st.checkbox(
        "Simulate Tool Failure",
        value=(preset_choice == "4. Tool Failure Simulation (Error Handling)"),
        help="Simulates a 500 error / timeout in the selected communication tool."
    )

    st.markdown("---")
    st.markdown("#### 📡 System Telemetry")
    sendgrid_status = "🟢 Ready (Live API)" if Config.is_sendgrid_configured() else "🟡 Mock Mode (Safe)"
    pushover_status = "🟢 Ready (Live API)" if Config.is_pushover_configured() else "🟡 Mock Mode (Safe)"
    llm_status = "🟢 Connected (Gemini/OpenAI)" if Config.is_llm_configured() else "🔵 Rule Decision Engine"

    st.markdown(f"**SendGrid:** `{sendgrid_status}`")
    st.markdown(f"**Pushover:** `{pushover_status}`")
    st.markdown(f"**Decision Engine:** `{llm_status}`")

    if st.session_state.logger.get_history():
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.logger.clear()
            st.session_state.last_confirmation = None
            st.rerun()


# Preset default data population
default_event_type = "security_alert"
default_priority = "HIGH"
default_message = "Unauthorized root access attempt detected on production database cluster."
default_recipient = "DevSecOps Lead"
default_email = "devops@enterprise.com"
default_notification = ""
default_log_only = False

if preset_choice == "1. High Priority - Security Alert (SendGrid Email)":
    default_event_type = "security_alert"
    default_priority = "HIGH"
    default_message = "Unauthorized root access attempt detected on production database cluster."
    default_recipient = "DevSecOps Lead"
    default_email = "devops@enterprise.com"
    default_notification = ""
    default_log_only = False

elif preset_choice == "2. Critical Priority - System Outage (Pushover Push)":
    default_event_type = "system_outage"
    default_priority = "CRITICAL"
    default_message = "Core API Gateway response code 503 rate spiked to 85%."
    default_recipient = "OnCall Engineer"
    default_email = ""
    default_notification = "CRITICAL: API Gateway 503 Spike (85%). Immediate triaging required!"
    default_log_only = False

elif preset_choice == "3. Low Priority - Routine Healthcheck (No Action)":
    default_event_type = "routine_healthcheck"
    default_priority = "LOW"
    default_message = "Disk utilization normal at 32%. Memory utilization at 41%."
    default_recipient = "System Monitor"
    default_email = ""
    default_notification = ""
    default_log_only = False

elif preset_choice == "4. Tool Failure Simulation (Error Handling)":
    default_event_type = "database_failover"
    default_priority = "HIGH"
    default_message = "Database primary node failover initiated."
    default_recipient = "Lead DBA"
    default_email = "dba-lead@enterprise.com"
    default_notification = ""
    default_log_only = False

elif preset_choice == "5. Medium Priority - SOC2 Audit (Other Tool)":
    default_event_type = "audit_permission_scan"
    default_priority = "MEDIUM"
    default_message = "Monthly SOC2 permission review completed automatically."
    default_recipient = "Compliance Officer"
    default_email = ""
    default_notification = ""
    default_log_only = True


# -----------------------------------------------------------------------------
# Main Application Header & Metrics
# -----------------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1 class="hero-title">⚡ Intelligent Communication Assistant</h1>
    <p class="hero-subtitle">
        Autonomous situation analysis, smart decision-making, and targeted tool dispatching via SendGrid Email, Pushover Push, and Internal Audit Ledger.
    </p>
</div>
""", unsafe_allow_html=True)

# Calculate live metrics
history_list = st.session_state.logger.get_history()
total_events = len(history_list)
emails_sent = sum(1 for h in history_list if h.selected_action == ActionType.SEND_EMAIL and h.status == "success")
pushes_sent = sum(1 for h in history_list if h.selected_action == ActionType.SEND_NOTIFICATION and h.status == "success")
audits_or_none = sum(1 for h in history_list if h.selected_action in (ActionType.OTHER_TOOL, ActionType.NO_ACTION))

st.markdown(f"""
<div class="metric-container">
    <div class="metric-card">
        <div class="metric-label">Total Events Processed</div>
        <div class="metric-value">{total_events}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Emails Dispatched</div>
        <div class="metric-value" style="color: #4F46E5;">{emails_sent}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Push Alerts Sent</div>
        <div class="metric-value" style="color: #9333EA;">{pushes_sent}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Audit / No Action</div>
        <div class="metric-value" style="color: #059669;">{audits_or_none}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Visual Architecture Stepper
step_action = st.session_state.last_confirmation.selected_action.value if st.session_state.last_confirmation else "Pending"
step_tool = st.session_state.last_confirmation.tool_used or "None" if st.session_state.last_confirmation else "Pending"

st.markdown(f"""
<div class="pipeline-wrapper">
    <div class="pipeline-step active">
        <span class="step-num">1</span>
        <span>Event Ingested</span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step active">
        <span class="step-num">2</span>
        <span>Situation Analyzed</span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step active">
        <span class="step-num">3</span>
        <span>Decision: <strong>{step_action}</strong></span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step active">
        <span class="step-num">4</span>
        <span>Tool: <strong>{step_tool}</strong></span>
    </div>
    <span class="pipeline-arrow">➔</span>
    <div class="pipeline-step active">
        <span class="step-num">5</span>
        <span>Confirmation Logged</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Input Form Section
# -----------------------------------------------------------------------------
with st.container():
    st.subheader("📥 Event / Request Input")

    col1, col2 = st.columns([2, 1])
    with col1:
        event_type_input = st.text_input(
            "Event Type / Name:",
            value=default_event_type,
            placeholder="e.g. security_alert, system_outage, routine_healthcheck"
        )
    with col2:
        priority_options = ["HIGH", "CRITICAL", "MEDIUM", "LOW", "INFO"]
        priority_idx = priority_options.index(default_priority) if default_priority in priority_options else 0
        priority_input = st.selectbox("Priority Level:", priority_options, index=priority_idx)

    message_input = st.text_area(
        "Event Message / Description:",
        value=default_message,
        placeholder="Enter detailed description of the event or request...",
        height=90
    )

    col3, col4 = st.columns(2)
    with col3:
        recipient_input = st.text_input("Recipient Name / Target:", value=default_recipient)
        email_input = st.text_input("Email Address (when required):", value=default_email)
    with col4:
        notification_input = st.text_input(
            "Notification Message (when required):",
            value=default_notification,
            placeholder="Short push text (defaults to event message if blank)"
        )
        log_only_checkbox = st.checkbox("Internal Audit / Log Only Condition", value=default_log_only)

    st.write("")
    analyze_button = st.button("🚀 Analyze & Process", type="primary", use_container_width=True)


# -----------------------------------------------------------------------------
# Processing Logic (Invokes Existing Phase 1 Agent)
# -----------------------------------------------------------------------------
if analyze_button:
    # 1. Validation for empty event input
    if not message_input or not message_input.strip() or not event_type_input or not event_type_input.strip():
        st.error("⚠️ Please enter an event or request.")
    else:
        # Construct EventRequest model
        conditions_dict = {}
        if log_only_checkbox:
            conditions_dict["log_only"] = True
        if notification_input or priority_input in ("HIGH", "CRITICAL"):
            if not email_input:
                conditions_dict["notification_enabled"] = True

        event = EventRequest(
            event_type=event_type_input.strip(),
            message=message_input.strip(),
            priority=priority_input,
            recipient=recipient_input.strip() if recipient_input else None,
            email=email_input.strip() if email_input else None,
            notification_message=notification_input.strip() if notification_input else None,
            conditions=conditions_dict
        )

        with st.spinner("Agent analyzing situation and evaluating communication requirements..."):
            confirmation = st.session_state.agent.process_event(
                event=event,
                force_mock=force_mock_mode,
                simulate_tool_failure=simulate_failure
            )
            st.session_state.last_confirmation = confirmation


# -----------------------------------------------------------------------------
# Display Agent Decision & Confirmation Cards
# -----------------------------------------------------------------------------
if st.session_state.last_confirmation:
    conf = st.session_state.last_confirmation

    st.markdown("<br>", unsafe_allow_html=True)
    res_col1, res_col2 = st.columns(2)

    # 1. Agent Decision Card
    with res_col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">🧠 Agent Decision & Analysis</div>', unsafe_allow_html=True)

        action_name = conf.selected_action.value
        tool_name = conf.tool_used or "None (No external tool)"

        badge_class, badge_icon = {
            ActionType.SEND_EMAIL: ("badge-email", "✉️"),
            ActionType.SEND_NOTIFICATION: ("badge-push", "🔔"),
            ActionType.OTHER_TOOL: ("badge-audit", "📝"),
            ActionType.NO_ACTION: ("badge-none", "💤")
        }.get(conf.selected_action, ("badge-none", "⚡"))

        st.markdown(f"""
        <div style="margin-bottom: 16px;">
            <span class="action-badge {badge_class}">{badge_icon} {action_name}</span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Target Tool Selected:</span>
            <span class="detail-value"><code>{tool_name}</code></span>
        </div>
        <div class="detail-row">
            <span class="detail-label">Event Classification:</span>
            <span class="detail-value">{conf.event_type} ({conf.priority} Priority)</span>
        </div>
        <div class="reason-box">
            <strong>Agent Reasoning:</strong><br>{conf.summary}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. Confirmation Card
    with res_col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="card-heading">📋 Execution Confirmation</div>', unsafe_allow_html=True)

        if conf.status == "success":
            mode_label = "🧪 SAFE MOCK SIMULATION" if (conf.tool_result and conf.tool_result.mode == "mock") else "⚡ LIVE API DELIVERY"
            mode_class = "mode-mock" if (conf.tool_result and conf.tool_result.mode == "mock") else "mode-real"

            st.success("✅ **Action Executed Successfully**")
            st.markdown(f"""
            <div style="margin-bottom: 12px; display: flex; align-items: center; justify-content: space-between;">
                <span style="font-weight: 700; color: #16A34A; font-size: 0.95rem;">Delivery Status: CONFIRMED</span>
                <span class="mode-chip {mode_class}">{mode_label}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Status:</span>
                <span class="detail-value" style="color: #16A34A;">SUCCESS</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Tool Dispatched:</span>
                <span class="detail-value"><code>{conf.tool_used}</code></span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Delivery / Reference ID:</span>
                <span class="detail-value"><code>{conf.tool_result.message_id if conf.tool_result else 'N/A'}</code></span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Recipient:</span>
                <span class="detail-value">{conf.tool_result.recipient if (conf.tool_result and conf.tool_result.recipient) else 'N/A'}</span>
            </div>
            """, unsafe_allow_html=True)

            if conf.tool_result and conf.tool_result.data.get("note"):
                st.caption(f"ℹ️ {conf.tool_result.data['note']}")

        elif conf.status == "no_action":
            st.info("ℹ️ **Action: No Action Required**")
            st.markdown(f"""
            <div class="detail-row">
                <span class="detail-label">Status:</span>
                <span class="detail-value" style="color: #0284C7;">COMPLETED</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Tool:</span>
                <span class="detail-value">None (Suppressed)</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Resolution:</span>
                <span class="detail-value">Informational event; external alerting skipped.</span>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error("❌ **Action Execution Failed**")
            st.markdown(f"""
            <div class="detail-row">
                <span class="detail-label">Status:</span>
                <span class="detail-value" style="color: #DC2626;">FAILED</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Attempted Tool:</span>
                <span class="detail-value"><code>{conf.tool_used or 'Unknown'}</code></span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Error Details:</span>
                <span class="detail-value" style="color: #DC2626;">{conf.error or 'Unknown tool error.'}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 3. Interactive Status & Audit Logs Explorer
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📊 Status / Logs (Detailed Audit Record)", expanded=True):
        tab_timeline, tab_json, tab_arch = st.tabs(["🕒 Session Audit Timeline", "🔍 Structured JSON Log", "📐 Architecture Reference"])

        with tab_timeline:
            history = st.session_state.logger.get_history()
            if not history:
                st.write("No session records found.")
            else:
                for item in reversed(history):
                    icon = "✅" if item.status == "success" else ("ℹ️" if item.status == "no_action" else "❌")
                    status_color = "#16A34A" if item.status == "success" else ("#0284C7" if item.status == "no_action" else "#DC2626")
                    st.markdown(f"""
                    <div class="timeline-item">
                        <div>
                            <span>{icon} <strong>{item.selected_action.value}</strong></span>
                            <span style="color: #64748B; margin: 0 8px;">•</span>
                            <span style="color: #334155;">{item.event_type}</span>
                            <span style="color: #64748B; margin: 0 8px;">•</span>
                            <code>{item.tool_used or 'NO_TOOL'}</code>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span style="font-size: 0.8rem; color: {status_color}; font-weight: 700;">{item.status.upper()}</span>
                            <span style="font-size: 0.78rem; color: #94A3B8;">{item.timestamp[11:19]} UTC</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_json:
            st.json({
                "log_id": conf.log_id,
                "timestamp": conf.timestamp,
                "event_id": conf.event_id,
                "event_type": conf.event_type,
                "priority": conf.priority,
                "selected_action": conf.selected_action.value,
                "tool_used": conf.tool_used,
                "status": conf.status,
                "summary": conf.summary,
                "error": conf.error,
                "tool_result": conf.tool_result.model_dump() if conf.tool_result else None
            })

        with tab_arch:
            st.markdown("""
            ```text
            EVENT / REQUEST
                  ↓
            AGENT / LLM (Analyze Situation + Decision Making)
                  ↓
             ┌──────────────┬──────────────────┬─────────────┐
             ↓              ↓                  ↓
            SENDGRID      PUSHOVER          OTHER TOOL (Audit Logger)
            EMAIL         NOTIFICATION
             ↓              ↓                  ↓
             └──────────────┴──────────────────┘
                            ↓
                    CONFIRMATION / LOG DATA
            ```
            """)
