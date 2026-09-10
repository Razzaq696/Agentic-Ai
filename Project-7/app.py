"""AI Shopping Decision Workspace — Project 7.

"Stop searching. Start deciding."
Premium AI Shopping Decision Workspace powered by autonomous multi-agent intelligence,
semantic vector RAG, verified web research, and transparent 100-point scoring.
"""

from typing import Any, Dict, List, Optional
import urllib.parse
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="AI Shopping Decision Workspace",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from src.config import settings
from src.integrations.n8n_client import N8nClient
from src.integrations.pushover_client import PushoverClient
from src.integrations.sendgrid_client import SendGridClient
from src.models.schemas import DecisionStatus, FinalDecision
from src.rag.vectorstore import ChromaVectorStoreManager

from src.ui.helpers import (
    extract_closest_options,
    extract_comparison_display,
    extract_decision_display,
    extract_integrations_display,
    extract_products_display,
    extract_requirement_matching,
    extract_requirements_display,
    extract_score_breakdown_bars,
    extract_scores_display,
    extract_sources_trust,
    format_currency,
    format_percentage,
    format_score,
    sanitize_error_message,
)
from src.workflow.graph import build_shopping_graph, ensure_knowledge_base_indexed


# =====================================================================
# CSS DESIGN SYSTEM
# =====================================================================
def inject_workspace_styles():
    """Inject ultra-modern, clean enterprise CSS for the shopping decision workspace."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Global Reset & Typography */
        html, body, [class*="css"], [class*="st-"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
        }

        /* Streamlit Default Header Reduction */
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 0.5rem !important;
        }

        /* Container Max Width for Focused Decision Workspace */
        .block-container {
            max-width: 1180px !important;
            padding-top: 1.2rem !important;
            padding-bottom: 3.5rem !important;
        }

        /* Top Navigation Header Bar Wrapper */
        div[data-testid="stHorizontalBlock"]:has(.nav-brand) {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 50%, rgba(236, 72, 153, 0.03) 100%);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 16px;
            padding: 10px 18px;
            margin-bottom: 22px;
            backdrop-filter: blur(16px);
            align-items: center !important;
            box-shadow: 0 4px 20px -4px rgba(0, 0, 0, 0.05);
        }

        /* Top Nav Popover & Button Styling */
        div[data-testid="stHorizontalBlock"]:has(.nav-brand) .stButton > button,
        div[data-testid="stHorizontalBlock"]:has(.nav-brand) div[data-testid="stPopover"] > button {
            border-radius: 10px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            border: 1px solid rgba(148, 163, 184, 0.28) !important;
            background: rgba(148, 163, 184, 0.08) !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stHorizontalBlock"]:has(.nav-brand) .stButton > button:hover,
        div[data-testid="stHorizontalBlock"]:has(.nav-brand) div[data-testid="stPopover"] > button:hover {
            border-color: #6366f1 !important;
            background: rgba(99, 102, 241, 0.12) !important;
            transform: translateY(-1px);
        }

        .nav-brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .nav-logo-icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            border-radius: 12px;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            font-size: 1.35rem;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3);
        }

        .nav-title-group {
            display: flex;
            flex-direction: column;
        }

        .nav-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin: 0;
            line-height: 1.2;
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .nav-tagline {
            font-size: 0.78rem;
            opacity: 0.75;
            font-weight: 500;
            margin: 0;
        }

        .nav-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            background: rgba(99, 102, 241, 0.12);
            color: #6366f1;
            border: 1px solid rgba(99, 102, 241, 0.25);
        }

        /* Hero Header */
        .hero-section {
            text-align: center;
            padding: 24px 12px 16px 12px;
            margin-bottom: 8px;
        }

        .hero-badge-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 14px;
            border-radius: 9999px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            background: rgba(99, 102, 241, 0.1);
            color: #6366f1;
            border: 1px solid rgba(99, 102, 241, 0.25);
            margin-bottom: 12px;
        }

        .hero-title {
            font-size: 2.5rem;
            font-weight: 850;
            letter-spacing: -0.035em;
            line-height: 1.15;
            margin-bottom: 10px;
        }

        .hero-title span.highlight {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .hero-subtitle {
            font-size: 1.05rem;
            max-width: 720px;
            margin: 0 auto;
            opacity: 0.8;
            line-height: 1.6;
            font-weight: 400;
        }

        /* Search Bar & Pill Presets */
        .preset-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            margin-top: 14px;
            margin-bottom: 18px;
        }

        .preset-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(148, 163, 184, 0.1);
            border: 1px solid rgba(148, 163, 184, 0.22);
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .preset-pill:hover {
            border-color: #6366f1;
            background: rgba(99, 102, 241, 0.08);
            transform: translateY(-1px);
        }

        /* Requirements Summary Chips */
        .chips-panel {
            background: rgba(99, 102, 241, 0.04);
            border: 1px solid rgba(99, 102, 241, 0.16);
            border-radius: 14px;
            padding: 16px 20px;
            margin: 20px 0;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }

        .chips-list {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 10px;
        }

        .req-chip {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 10px;
            font-size: 0.82rem;
            font-weight: 600;
            background: rgba(148, 163, 184, 0.12);
            border: 1px solid rgba(148, 163, 184, 0.25);
        }

        .req-chip strong {
            color: #6366f1;
            font-weight: 700;
        }

        /* Hero Recommendation Card */
        .best-match-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.08) 0%, rgba(139, 92, 246, 0.05) 50%, rgba(236, 72, 153, 0.03) 100%);
            border: 2px solid rgba(99, 102, 241, 0.35);
            border-radius: 20px;
            padding: 26px 30px;
            margin: 24px 0;
            position: relative;
            box-shadow: 0 10px 30px -10px rgba(99, 102, 241, 0.25);
        }

        .best-match-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 12px;
            margin-bottom: 16px;
            padding-bottom: 14px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }

        .best-match-tag {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            color: #4f46e5;
            background: rgba(79, 70, 229, 0.12);
            padding: 6px 14px;
            border-radius: 9999px;
            border: 1px solid rgba(79, 70, 229, 0.25);
        }

        .match-score-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 0.88rem;
            font-weight: 800;
            color: #059669;
            background: rgba(16, 185, 129, 0.14);
            padding: 6px 14px;
            border-radius: 9999px;
            border: 1px solid rgba(16, 185, 129, 0.35);
        }

        .product-title-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 14px;
            margin-bottom: 12px;
        }

        .product-title {
            font-size: 1.85rem;
            font-weight: 850;
            letter-spacing: -0.02em;
            line-height: 1.2;
            margin: 0;
        }

        .product-brand {
            font-size: 0.95rem;
            font-weight: 600;
            opacity: 0.75;
            margin-top: 4px;
        }

        .product-price-badge {
            font-size: 1.75rem;
            font-weight: 850;
            letter-spacing: -0.02em;
            color: #4f46e5;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 14px 0 18px 0;
        }

        .feature-badge {
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 12px;
            border-radius: 8px;
            font-size: 0.78rem;
            font-weight: 600;
            background: rgba(99, 102, 241, 0.08);
            border: 1px solid rgba(99, 102, 241, 0.2);
        }

        .reasons-box {
            background: rgba(148, 163, 184, 0.06);
            border-radius: 12px;
            padding: 14px 18px;
            margin: 14px 0;
            border-left: 3px solid #6366f1;
        }

        .reasons-title {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            color: #6366f1;
            margin-bottom: 6px;
        }

        /* Section Container & Titles */
        .section-header-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 28px 0 14px 0;
            padding-bottom: 8px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.015em;
            margin: 0;
        }

        .section-subtitle {
            font-size: 0.85rem;
            opacity: 0.75;
            margin-top: 2px;
        }

        /* Requirement Mapping Card */
        .mapping-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0 8px;
        }

        .mapping-row {
            background: rgba(148, 163, 184, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 10px;
            padding: 12px 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .mapping-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .mapping-icon {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: rgba(16, 185, 129, 0.15);
            color: #059669;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.85rem;
            font-weight: 800;
        }

        .mapping-req-text {
            font-weight: 700;
            font-size: 0.92rem;
        }

        .mapping-detail-text {
            font-size: 0.86rem;
            opacity: 0.85;
        }

        /* Comparison Cards */
        .comp-card {
            background: rgba(148, 163, 184, 0.05);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 14px;
            padding: 18px 20px;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .comp-card.winner {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.07) 0%, rgba(139, 92, 246, 0.04) 100%);
            border: 2px solid rgba(99, 102, 241, 0.4);
        }

        .comp-card:hover {
            transform: translateY(-2px);
            border-color: rgba(99, 102, 241, 0.4);
        }

        .comp-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 12px;
        }

        .comp-score-tag {
            font-size: 0.82rem;
            font-weight: 800;
            padding: 4px 10px;
            border-radius: 9999px;
            background: rgba(99, 102, 241, 0.12);
            color: #6366f1;
        }

        .comp-strengths-box {
            margin-top: 10px;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        .comp-weakness-box {
            margin-top: 10px;
            font-size: 0.84rem;
            line-height: 1.5;
        }

        /* Scoring Progress Bar Rows */
        .score-bar-row {
            margin-bottom: 12px;
        }

        .score-bar-labels {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.84rem;
            margin-bottom: 4px;
        }

        .score-bar-name {
            font-weight: 700;
        }

        .score-bar-val {
            font-weight: 800;
            color: #4f46e5;
        }

        .score-progress-track {
            width: 100%;
            height: 9px;
            border-radius: 9999px;
            background: rgba(148, 163, 184, 0.18);
            overflow: hidden;
        }

        .score-progress-fill {
            height: 100%;
            border-radius: 9999px;
            background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
            transition: width 0.4s ease;
        }

        /* Trade-offs Card */
        .tradeoff-card {
            background: rgba(245, 158, 11, 0.05);
            border: 1px solid rgba(245, 158, 11, 0.25);
            border-radius: 14px;
            padding: 18px 22px;
            margin: 18px 0;
        }

        /* No Perfect Match Card */
        .no-match-banner {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.06) 0%, rgba(245, 158, 11, 0.06) 100%);
            border: 1px solid rgba(239, 68, 68, 0.28);
            border-radius: 16px;
            padding: 22px 26px;
            margin: 20px 0;
        }

        /* Trust & Sources Cards */
        .source-pill-card {
            background: rgba(148, 163, 184, 0.06);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 10px;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85rem;
        }

        /* Modern Primary Action Button */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
            border: none !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            padding: 10px 24px !important;
            border-radius: 12px !important;
            box-shadow: 0 4px 14px rgba(79, 70, 229, 0.35) !important;
            transition: all 0.2s ease !important;
        }

        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.45) !important;
        }

        /* Streamlit Text Input Polish */
        .stTextArea textarea {
            border-radius: 14px !important;
            font-size: 1rem !important;
            line-height: 1.5 !important;
            padding: 16px 18px !important;
            border: 1.5px solid rgba(148, 163, 184, 0.25) !important;
        }

        .stTextArea textarea:focus {
            border-color: #6366f1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# CACHED WORKFLOW
# =====================================================================
def get_cached_workflow():
    """Build and cache the default LangGraph workflow application."""
    if "shopping_app" not in st.session_state:
        st.session_state["shopping_app"] = build_shopping_graph()
    return st.session_state["shopping_app"]


# =====================================================================
# TOP NAVIGATION
# =====================================================================
def render_top_navigation() -> Dict[str, Any]:
    """Render sleek modern top bar with branding, status pill, New Search, and Settings popover."""
    col_brand, col_status, col_actions = st.columns([3.8, 1.4, 2.8], vertical_alignment="center")

    with col_brand:
        st.markdown(
            """
            <div class="nav-brand">
                <div class="nav-logo-icon">⚖️</div>
                <div class="nav-title-group">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span class="nav-title">AI SHOPPER</span>
                        <span class="nav-badge">Decision Workspace</span>
                    </div>
                    <span class="nav-tagline">Stop searching. Start deciding.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_status:
        st.markdown(
            """
            <div style="display: flex; justify-content: center; align-items: center;">
                <span style="display: inline-flex; align-items: center; gap: 6px; font-size: 0.74rem; font-weight: 700; color: #059669; background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.28); padding: 5px 12px; border-radius: 9999px; letter-spacing: 0.03em;">
                    <span style="width: 7px; height: 7px; border-radius: 50%; background: #10b981;"></span>
                    ENGINE READY
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    budget_cap = None
    category_hint = None
    preference_hint = None
    priority_hint = None
    opt_n8n = settings.n8n_enabled
    n8n_url = settings.n8n_webhook_url or None
    opt_sendgrid = settings.sendgrid_enabled
    to_email = settings.sendgrid_to_email or None
    opt_pushover = settings.pushover_enabled
    pushover_user = settings.pushover_user_key or None

    with col_actions:
        b_new, b_settings = st.columns([1.1, 1.1], vertical_alignment="center")
        with b_new:
            if st.button("✨ New Search", use_container_width=True, help="Clear search and start over"):
                st.session_state.pop("last_result", None)
                st.session_state.pop("query_input", None)
                st.session_state.pop("last_query", None)
                st.rerun()

        with b_settings:
            with st.popover("⚙️ Settings", use_container_width=True):
                st.markdown("#### ⚙️ Decision Settings")
                st.caption("Structured constraints, telemetry, and webhooks:")

                try:
                    vsm = ChromaVectorStoreManager()
                    count = vsm.get_collection_count()
                except Exception:
                    count = 0

                st.markdown(
                    f"""
                    <div style="background: rgba(99, 102, 241, 0.08); border: 1px solid rgba(99, 102, 241, 0.2); border-radius: 10px; padding: 10px 14px; margin-bottom: 12px;">
                        <div style="font-size: 0.72rem; font-weight: 700; color: #6366f1; text-transform: uppercase;">Local Vector Catalog</div>
                        <div style="font-size: 1.05rem; font-weight: 800; margin-top: 2px;">📚 {count} Products Indexed</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("##### 🎯 Structured Constraints")
                enable_custom_budget = st.checkbox("Enforce Custom Hard Budget Limit", value=False)
                if enable_custom_budget:
                    budget_cap = st.number_input(
                        "Max Budget Ceiling ($ USD)", min_value=10.0, max_value=100000.0, value=250.0, step=25.0
                    )
                category_hint = st.text_input("Category Hint (optional)", placeholder="e.g. Mechanical Keyboard")
                preference_hint = st.text_input("Preferences (nice-to-have)", placeholder="e.g. Compact, Wireless")
                priority_hint = st.text_input("Top Priority", placeholder="e.g. Typing feel, Battery life")

                st.markdown("---")
                st.markdown("##### 📬 Optional Integrations")
                opt_n8n = st.checkbox("n8n Webhook", value=settings.n8n_enabled)
                n8n_url = st.text_input("n8n URL", value=settings.n8n_webhook_url or "") if opt_n8n else None

                opt_sendgrid = st.checkbox("SendGrid Email", value=settings.sendgrid_enabled)
                to_email = st.text_input("Recipient Email", value=settings.sendgrid_to_email or "") if opt_sendgrid else None

                opt_pushover = st.checkbox("Pushover Mobile", value=settings.pushover_enabled)
                pushover_user = st.text_input("Pushover User Key", value=settings.pushover_user_key or "", type="password") if opt_pushover else None

    return {
        "budget_cap": budget_cap,
        "category_hint": category_hint.strip() if category_hint else None,
        "preference_hint": preference_hint.strip() if preference_hint else None,
        "priority_hint": priority_hint.strip() if priority_hint else None,
        "n8n": {"enabled": opt_n8n, "url": n8n_url},
        "sendgrid": {"enabled": opt_sendgrid, "to_email": to_email},
        "pushover": {"enabled": opt_pushover, "user_key": pushover_user},
    }



# =====================================================================
# STEP 1: HERO SEARCH
# =====================================================================
def render_hero_search(settings_config: Dict[str, Any]):
    """Render the hero search bar, one-click preset pills, and execution trigger."""
    st.markdown(
        """
        <div class="hero-section">
            <div class="hero-badge-pill">Autonomous Multi-Agent Decision Engine</div>
            <h1 class="hero-title">Stop searching. <span class="highlight">Start deciding.</span></h1>
            <p class="hero-subtitle">
                An intelligent shopping workspace that analyzes technical specifications, verifies prices,
                weighs trade-offs, and scores options on a transparent 100-point rubric.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Preset Pills
    st.markdown("<div style='text-align:center; font-size: 0.82rem; opacity: 0.7; font-weight: 600; margin-bottom: 6px;'>TRY POPULAR DECISION SCENARIOS:</div>", unsafe_allow_html=True)
    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
    selected_preset = None

    with p_col1:
        if st.button("⌨️ Mechanical Keyboard ($250)", use_container_width=True, key="p_kb"):
            selected_preset = "Looking for a durable mechanical keyboard with tactile switches for daily coding under $250"
    with p_col2:
        if st.button("☕ Espresso Machine ($2000)", use_container_width=True, key="p_esp"):
            selected_preset = "I need an espresso machine with dual boiler and digital PID temperature control under $2000"
    with p_col3:
        if st.button("🎧 ANC Headphones ($350)", use_container_width=True, key="p_hp"):
            selected_preset = "Looking for over-ear wireless headphones with active noise cancellation under $350"
    with p_col4:
        if st.button("💻 Developer Laptop ($1500)", use_container_width=True, key="p_laptop"):
            selected_preset = "Need a lightweight laptop for software engineering with 32GB RAM and all-day battery under $1500"

    # Search Text Area
    if selected_preset:
        st.session_state["query_input"] = selected_preset

    current_val = st.session_state.get("query_input", st.session_state.get("last_query", ""))

    query = st.text_area(
        label="What are you trying to decide on?",
        value=current_val,
        placeholder="e.g. I need a mechanical keyboard with tactile switches for daily programming, wireless connectivity, under $250.",
        height=100,
        key="main_query_box",
    )

    btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
    with btn_col2:
        execute_clicked = st.button("✨ Find My Best Options", type="primary", use_container_width=True)

    if execute_clicked:
        if not query.strip():
            st.warning("⚠️ Please enter a product description or select one of the popular presets above.")
            return

        st.session_state["last_query"] = query
        st.session_state["query_input"] = query

        # Ensure vector store is ready
        try:
            ensure_knowledge_base_indexed()
        except Exception as e:
            st.error(f"Catalog initialization notice: {sanitize_error_message(e)}")

        # Configure custom clients if settings provided
        custom_n8n = None
        if settings_config["n8n"]["enabled"]:
            custom_n8n = N8nClient(
                enabled=True,
                webhook_url=settings_config["n8n"]["url"] or settings.n8n_webhook_url,
            )

        custom_sendgrid = None
        if settings_config["sendgrid"]["enabled"]:
            custom_sendgrid = SendGridClient(
                enabled=True,
                to_email=settings_config["sendgrid"]["to_email"] or settings.sendgrid_to_email,
            )

        custom_pushover = None
        if settings_config["pushover"]["enabled"]:
            custom_pushover = PushoverClient(
                enabled=True,
                user_key=settings_config["pushover"]["user_key"] or settings.pushover_user_key,
            )

        if any([custom_n8n, custom_sendgrid, custom_pushover]):
            app = build_shopping_graph(
                n8n_client=custom_n8n,
                sendgrid_client=custom_sendgrid,
                pushover_client=custom_pushover,
            )
        else:
            app = get_cached_workflow()

        # Build state payload
        initial_state: Dict[str, Any] = {"user_request": query}
        hints = []
        if settings_config["budget_cap"]:
            hints.append(f"Budget cap: ${settings_config['budget_cap']:,.2f}")
        if settings_config["category_hint"]:
            hints.append(f"Category: {settings_config['category_hint']}")
        if settings_config["preference_hint"]:
            hints.append(f"Preferences: {settings_config['preference_hint']}")
        if settings_config["priority_hint"]:
            hints.append(f"Priorities: {settings_config['priority_hint']}")

        if hints:
            initial_state["user_request"] += " [Constraints: " + "; ".join(hints) + "]"

        # STEP 3: AI Research Experience
        with st.status("🔍 Autonomous AI Research in Progress...", expanded=True) as status:
            try:
                st.write("1. 🧠 **Requirement Analysis**: Extracting product category, budget bounds, and non-negotiables...")
                st.write("2. 📚 **Catalog & Web Research**: Querying Chroma Vector DB and live web product pages...")
                st.write("3. 🛡️ **Product Validation**: Auditing pricing, verifying authentic specs, and filtering corrupt data...")
                st.write("4. ⚖️ **Comparison & Trade-offs**: Building side-by-side feature matrices and identifying trade-offs...")
                st.write("5. 🏆 **Objective Decision**: Computing deterministic 100-point rubric and selecting best match...")

                result = app.invoke(initial_state)
                status.update(label="✅ Decision Analysis Complete!", state="complete", expanded=False)
                st.session_state["last_result"] = result
                st.rerun()
            except Exception as ex:
                status.update(label="❌ Pipeline Execution Error", state="error", expanded=True)
                st.error(f"Execution Error: {sanitize_error_message(ex)}")
                return


# =====================================================================
# WELCOME / EMPTY STATE
# =====================================================================
def render_welcome_state():
    """Render informative value proposition cards when no search has been executed."""
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px; padding: 22px 24px; height: 100%;">
                <div style="font-size: 1.8rem; margin-bottom: 10px;">🧠</div>
                <h4 style="margin: 0 0 8px 0; font-weight: 800;">Multi-Agent Pipeline</h4>
                <p style="font-size: 0.88rem; opacity: 0.8; line-height: 1.5; margin: 0;">
                    Specialized agents for <b>Requirements</b>, <b>Validation</b>, <b>Comparison</b>, and <b>Decision</b> collaborate to filter bias and build unbiased product evaluations.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px; padding: 22px 24px; height: 100%;">
                <div style="font-size: 1.8rem; margin-bottom: 10px;">📚</div>
                <h4 style="margin: 0 0 8px 0; font-weight: 800;">Semantic RAG + Live Web</h4>
                <p style="font-size: 0.88rem; opacity: 0.8; line-height: 1.5; margin: 0;">
                    Indexes curated candidate catalogs via <b>Chroma Vector DB</b>. If local items fall short, autonomous <b>Web Search & Playwright</b> scrape verified live specifications.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 14px; padding: 22px 24px; height: 100%;">
                <div style="font-size: 1.8rem; margin-bottom: 10px;">🛡️</div>
                <h4 style="margin: 0 0 8px 0; font-weight: 800;">Strict No-Forcing Guardrail</h4>
                <p style="font-size: 0.88rem; opacity: 0.8; line-height: 1.5; margin: 0;">
                    Governed by a transparent <b>100-point scoring algorithm</b>. If no product satisfies your hard requirements, the agent will never declare a false winner.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================================
# STEP 2: REQUIREMENTS BREAKDOWN CHIPS
# =====================================================================
def render_requirements_chips(result: Dict[str, Any]):
    """Render compact, interactive chips showing interpreted requirements with edit button."""
    req_info = extract_requirements_display(result)

    c_chips, c_edit = st.columns([5.5, 1.2], vertical_alignment="center")
    with c_chips:
        st.markdown(
            f"""
            <div class="chips-panel" style="margin: 10px 0;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 0.78rem; font-weight: 800; letter-spacing: 0.05em; text-transform: uppercase; color: #6366f1;">
                        🎯 UNDERSTOOD CRITERIA:
                    </span>
                </div>
                <div class="chips-list">
                    <span class="req-chip">📦 <strong>Category:</strong> {req_info['category']}</span>
                    <span class="req-chip">💰 <strong>Budget:</strong> {req_info['budget']}</span>
                    <span class="req-chip">⚡ <strong>Priority:</strong> {req_info['priorities'][0] if req_info['priorities'] else 'Balanced'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c_edit:
        if st.button("✏️ Edit Criteria", help="Modify query and search constraints", use_container_width=True):
            st.session_state.pop("last_result", None)
            st.rerun()



# =====================================================================
# STEP 4: HERO RECOMMENDATION CARD
# =====================================================================
def render_best_match_card(result: Dict[str, Any]):
    """Render the hero recommendation card: name, brand, price, match %, badges, and why."""
    decision_info = extract_decision_display(result)
    rec = decision_info["recommended"]
    if not rec:
        return

    conf_pct = decision_info["confidence_pct"]
    name = rec["name"]
    brand = rec["brand"]
    price = rec["price"]
    reasons = decision_info["key_reasons"]
    url = rec.get("url")

    # Feature Badges
    badges_html = ""
    for f in rec.get("features", [])[:4]:
        badges_html += f'<span class="feature-badge">✓ {f}</span>'

    reasons_html = "".join(f"<li style='margin-bottom: 4px;'>{r}</li>" for r in reasons[:3])

    card_html = (
        f'<div class="best-match-card" id="best-match-section">'
        f'<div class="best-match-header">'
        f'<span class="best-match-tag">🏆 BEST MATCH FOR YOU</span>'
        f'<span class="match-score-pill">🎯 {conf_pct} Objective Confidence</span>'
        f'</div>'
        f'<div class="product-title-row">'
        f'<div>'
        f'<h2 class="product-title">{name}</h2>'
        f'<div class="product-brand">{brand} • Verified Candidate</div>'
        f'</div>'
        f'<div class="product-price-badge">{price}</div>'
        f'</div>'
        f'<div class="badge-row">{badges_html}</div>'
        f'<div class="reasons-box">'
        f'<div class="reasons-title">💡 Why We Picked This Product</div>'
        f'<ul style="margin: 0; padding-left: 18px; font-size: 0.9rem; line-height: 1.55;">{reasons_html}</ul>'
        f'</div>'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

    # Action Buttons under hero card
    act_col1, act_col2 = st.columns([1, 1])
    with act_col1:
        if url:
            st.link_button("🔗 View Official Product Page", url=url, use_container_width=True)
        else:
            search_query = urllib.parse.quote(f"{brand} {name} official specifications")
            st.link_button("🔍 Search Verified Specs Online", url=f"https://www.google.com/search?q={search_query}", use_container_width=True)
    with act_col2:
        st.markdown(
            """
            <a href="#why-this-product-section" style="text-decoration:none;">
                <div style="background: rgba(99, 102, 241, 0.1); color: #6366f1; border: 1px solid rgba(99, 102, 241, 0.25); text-align: center; padding: 9px 14px; border-radius: 12px; font-weight: 700; font-size: 0.88rem;">
                    📋 View Criteria Match Details ➔
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )



# =====================================================================
# STEP 5: WHY WE PICKED THIS (REQUIREMENT -> MATCH MAPPING)
# =====================================================================
def render_why_this_product(result: Dict[str, Any]):
    """Render requirement-to-product mapping matrix."""
    mappings = extract_requirement_matching(result)
    if not mappings:
        return

    st.markdown("<div id='why-this-product-section'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="section-header-row">
            <div>
                <h3 class="section-title">📋 Why We Picked This</h3>
                <div class="section-subtitle">Direct mapping of your requirements against verified product specifications</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for m in mappings:
        badge_style = "background: rgba(16, 185, 129, 0.12); color: #059669; border: 1px solid rgba(16, 185, 129, 0.25);"
        if m.get("status") == "partial":
            badge_style = "background: rgba(245, 158, 11, 0.12); color: #d97706; border: 1px solid rgba(245, 158, 11, 0.25);"

        row_html = (
            f'<div class="mapping-row">'
            f'<div class="mapping-left">'
            f'<div class="mapping-icon">✓</div>'
            f'<div>'
            f'<div class="mapping-req-text">{m["requirement"]}</div>'
            f'<div class="mapping-detail-text">{m["detail"]}</div>'
            f'</div>'
            f'</div>'
            f'<span style="font-size: 0.76rem; font-weight: 700; padding: 4px 10px; border-radius: 9999px; {badge_style}">'
            f'{m.get("badge", "Verified Match")}'
            f'</span>'
            f'</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)


# =====================================================================
# STEP 6: TOP PRODUCT COMPARISON
# =====================================================================
def render_top_product_comparison(result: Dict[str, Any]):
    """Render multi-product comparison cards and difference matrix."""
    comp = extract_comparison_display(result)
    items = comp.get("items") or []
    scores = extract_scores_display(result)
    score_lookup = {s["Product"]: s["Total Score"] for s in scores}

    if not items:
        return

    st.markdown(
        """
        <div class="section-header-row">
            <div>
                <h3 class="section-title">⚖️ Compare Your Top Options</h3>
                <div class="section-subtitle">Side-by-side evaluation of verified strengths, limitations, and scores</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Columns for top 3 candidates
    top_items = items[:3]
    cols = st.columns(len(top_items))

    for idx, (col, item) in enumerate(zip(cols, top_items)):
        p_name = item["product_name"]
        is_winner = (idx == 0)
        tot_score = score_lookup.get(p_name, 0.0)

        strengths_html = "".join(f"<li>{s}</li>" for s in item.get("strengths", [])[:3])
        weaknesses_html = "".join(f"<li>{w}</li>" for w in item.get("weaknesses", [])[:2])

        with col:
            winner_class = "winner" if is_winner else ""
            winner_badge = (
                '<span style="font-size: 0.72rem; font-weight: 800; background: rgba(79, 70, 229, 0.15); color: #4f46e5; padding: 3px 8px; border-radius: 9999px;">TOP CHOICE</span>'
                if is_winner
                else ""
            )
            badge_html = f"<div>{winner_badge}</div>" if winner_badge else ""

            card_html = (
                f'<div class="comp-card {winner_class}">'
                f'<div>'
                f'<div class="comp-header">'
                f'<div>'
                f'{badge_html}'
                f'<h4 style="margin: 6px 0 2px 0; font-size: 1.05rem; font-weight: 800;">{p_name}</h4>'
                f'</div>'
                f'<span class="comp-score-tag">{tot_score:.1f} / 100</span>'
                f'</div>'
                f'<div class="comp-strengths-box">'
                f'<strong style="color: #059669; font-size: 0.78rem; text-transform: uppercase;">💪 Key Strengths:</strong>'
                f'<ul style="margin: 4px 0 10px 0; padding-left: 18px;">{strengths_html}</ul>'
                f'</div>'
                f'<div class="comp-weakness-box">'
                f'<strong style="color: #d97706; font-size: 0.78rem; text-transform: uppercase;">⚠️ Considerations:</strong>'
                f'<ul style="margin: 4px 0 0 0; padding-left: 18px;">{weaknesses_html}</ul>'
                f'</div>'
                f'</div>'
                f'</div>'
            )

            st.markdown(card_html, unsafe_allow_html=True)


# =====================================================================
# STEP 7: SCORING UI ("Why did this product win?")
# =====================================================================
def render_scoring_ui(result: Dict[str, Any]):
    """Render transparent 100-point multi-agent scoring breakdown with progress bars and table."""
    decision_info = extract_decision_display(result)
    rec = decision_info.get("recommended") or {}
    p_name = rec.get("name")

    score_bars = extract_score_breakdown_bars(result, p_name)
    scores_data = extract_scores_display(result)

    st.markdown(
        """
        <div class="section-header-row">
            <div>
                <h3 class="section-title">🔢 Why Did This Product Win?</h3>
                <div class="section-subtitle">Deterministic 100-point scoring breakdown across 6 rigorous evaluation dimensions</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    s_col1, s_col2 = st.columns([3, 2])

    with s_col1:
        st.markdown(f"##### Scoring Breakdown for **{p_name or 'Top Recommendation'}**")
        for bar in score_bars:
            pct_int = int(bar["percentage"] * 100)
            st.markdown(
                f"""
                <div class="score-bar-row">
                    <div class="score-bar-labels">
                        <span class="score-bar-name">{bar['dimension']} <span style="opacity: 0.6; font-weight: 500;">({bar['weight_label']})</span></span>
                        <span class="score-bar-val">{bar['score']:.1f} / {bar['max_score']:.0f} pts ({pct_int}%)</span>
                    </div>
                    <div class="score-progress-track">
                        <div class="score-progress-fill" style="width: {pct_int}%;"></div>
                    </div>
                    <div style="font-size: 0.74rem; opacity: 0.7; margin-top: 2px;">{bar['description']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with s_col2:
        st.markdown("##### Scoring Weight System")
        st.markdown(
            """
            <div style="background: rgba(148, 163, 184, 0.05); border: 1px solid rgba(148, 163, 184, 0.18); border-radius: 12px; padding: 16px 18px; font-size: 0.84rem; line-height: 1.6;">
                <div>🎯 <strong>Hard Requirements (40 pts):</strong> Non-negotiable specs and criteria.</div>
                <div>💰 <strong>Budget Compliance (25 pts):</strong> Savings and distance to ceiling.</div>
                <div>⚙️ <strong>Specifications (15 pts):</strong> Core hardware and performance specs.</div>
                <div>✨ <strong>Feature Match (10 pts):</strong> Secondary accessories and capabilities.</div>
                <div>⚡ <strong>Priorities (5 pts):</strong> Custom user-emphasized factors.</div>
                <div>🛡️ <strong>Data Reliability (5 pts):</strong> Provenance & verification density.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Complete interactive score table
    if scores_data:
        with st.expander("📊 View Complete Comparative Scoring Table (All Candidates)", expanded=False):
            st.dataframe(
                scores_data,
                use_container_width=True,
                column_config={
                    "Total Score": st.column_config.ProgressColumn(
                        "Total Score (/100)",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    "Hard Req (40)": st.column_config.NumberColumn("Hard Req", format="%.1f"),
                    "Budget (25)": st.column_config.NumberColumn("Budget", format="%.1f"),
                    "Specs (15)": st.column_config.NumberColumn("Specs", format="%.1f"),
                    "Features (10)": st.column_config.NumberColumn("Features", format="%.1f"),
                },
            )


# =====================================================================
# STEP 8: TRADE-OFFS SECTION
# =====================================================================
def render_tradeoffs_section(result: Dict[str, Any]):
    """Render honest, Wirecutter/RTINGS style trade-offs section."""
    dec = extract_decision_display(result)
    comp = extract_comparison_display(result)

    tradeoffs = dec.get("tradeoffs") or comp.get("key_tradeoffs") or []
    if not tradeoffs:
        return

    st.markdown(
        """
        <div class="section-header-row">
            <div>
                <h3 class="section-title">⚖️ Honest Trade-offs</h3>
                <div class="section-subtitle">What you gain and what you give up with this decision</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    t_col1, t_col2 = st.columns(2)
    with t_col1:
        st.markdown(
            """
            <div class="tradeoff-card">
                <div style="font-size: 0.82rem; font-weight: 800; color: #059669; text-transform: uppercase; margin-bottom: 8px;">
                    ✅ What You Gain
                </div>
                <div style="font-size: 0.88rem; line-height: 1.6;">
                    Highest overall composite score, full compliance with your core specifications, and verified pricing within your budget target.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with t_col2:
        tradeoffs_html = "".join(f"<li style='margin-bottom: 4px;'>{t}</li>" for t in tradeoffs)
        card_html = (
            f'<div class="tradeoff-card">'
            f'<div style="font-size: 0.82rem; font-weight: 800; color: #d97706; text-transform: uppercase; margin-bottom: 8px;">'
            f'⚠️ What You Give Up / Consider'
            f'</div>'
            f'<ul style="margin: 0; padding-left: 18px; font-size: 0.88rem; line-height: 1.55;">'
            f'{tradeoffs_html}'
            f'</ul>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)


# =====================================================================
# STEP 9: NO PERFECT MATCH STATE
# =====================================================================
def render_no_perfect_match_state(result: Dict[str, Any]):
    """Render friendly guidance when no candidate satisfied 100% of hard constraints."""
    closest = extract_closest_options(result)
    category = result.get("product_category") or "product"

    st.markdown(
        f"""
        <div class="no-match-banner">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                <span style="font-size: 1.4rem;">⚠️</span>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #dc2626;">
                    No 100% Exact Match Found for {category.title()}
                </h3>
            </div>
            <p style="font-size: 0.92rem; opacity: 0.88; line-height: 1.5; margin: 0 0 10px 0;">
                Our multi-agent system verified all available candidate models against your strict requirements.
                To protect you from an unsuitable purchase, our <b>No-Forcing Guardrail</b> refused to declare a false winner because no model met 100% of your constraints.
            </p>
            <div style="font-size: 0.84rem; font-weight: 600; color: #b45309;">
                👇 Below are the closest matching alternatives found, along with what criteria they missed:
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if closest:
        st.markdown("##### 🔍 Closest Matching Alternatives:")
        for idx, c in enumerate(closest[:3], 1):
            with st.expander(f"{idx}. {c['name']} — {c['price']} (Composite Match: {c['score']:.1f}/100)", expanded=(idx == 1)):
                c_col1, c_col2 = st.columns([3, 2])
                with c_col1:
                    st.markdown("**Criteria Gap / Unmet Specifications:**")
                    for r in c["unmet_reasons"]:
                        st.markdown(f"- ⚠️ {r}")
                    if c.get("features"):
                        st.markdown("**Verified Features:** " + ", ".join(c["features"][:4]))
                with c_col2:
                    if c.get("url"):
                        st.link_button("🔗 View Model Page", url=c["url"], use_container_width=True)
                    else:
                        search_url = f"https://www.google.com/search?q={urllib.parse.quote(str(c['name']) + ' specs')}"
                        st.link_button("🔍 Search Model Specs Online", url=search_url, use_container_width=True)

                    # Dynamic intelligent suggestions based on reasons
                    reasons_text = " ".join(str(r).lower() for r in c.get("unmet_reasons", []))
                    if "budget" in reasons_text or "price" in reasons_text:
                        st.info("💡 **Tip:** Increasing your budget target by a small margin would make this model a complete match.")
                    elif "ram" in reasons_text or "memory" in reasons_text:
                        st.info(f"💡 **Tip:** Check whether this {category} has configurable RAM variants or external expansion.")
                    else:
                        st.info(f"💡 **Tip:** Relaxing non-mandatory secondary features will make this a viable option.")
    else:
        st.info("No candidates were found in this exact category. Try broadening your search query or relaxing strict filters.")



# =====================================================================
# STEP 10: RESEARCH SOURCES & TRUST
# =====================================================================
def render_sources_trust(result: Dict[str, Any]):
    """Render verifiable sources, vector catalog attribution, and confidence markers."""
    sources = extract_sources_trust(result)

    st.markdown(
        """
        <div class="section-header-row">
            <div>
                <h3 class="section-title">🔎 Research Sources & Verification</h3>
                <div class="section-subtitle">Transparent provenance of data analyzed across vector databases and web research</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for s in sources:
        url_link = f"<a href='{s['url']}' target='_blank' style='color: #6366f1; text-decoration: none; font-weight: 600;'>{s['url']}</a>" if s.get('url') else "Internal Semantic Vector Database"

        st.markdown(
            f"""
            <div class="source-pill-card">
                <div>
                    <div style="font-weight: 700;">{s['title']}</div>
                    <div style="font-size: 0.78rem; opacity: 0.75; margin-top: 2px;">{url_link}</div>
                </div>
                <span style="font-size: 0.74rem; font-weight: 700; background: rgba(16, 185, 129, 0.12); color: #059669; padding: 4px 10px; border-radius: 9999px;">
                    {s['confidence']}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =====================================================================
# STEP 11: OPTIONAL ACTIONS (SEND REPORT / WEBHOOKS)
# =====================================================================
def render_optional_actions(result: Dict[str, Any]):
    """Render on-demand triggers for Email Report, Push Notification, and Webhooks."""
    st.markdown(
        """
        <div class="section-header-row">
            <div>
                <h3 class="section-title">📬 Actions & Integrations</h3>
                <div class="section-subtitle">Dispatch decision report to your email, mobile device, or automation workflows</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    final_dec_raw = result.get("final_decision")
    final_decision = None
    if final_dec_raw:
        try:
            final_decision = (
                FinalDecision.model_validate(final_dec_raw)
                if isinstance(final_dec_raw, dict)
                else final_dec_raw
            )
        except Exception:
            final_decision = None

    a_col1, a_col2, a_col3 = st.columns(3)

    with a_col1:
        if st.button("✉️ Send Email Report", use_container_width=True):
            if not final_decision:
                st.info("ℹ️ Decision report data is not available yet to email.")
            else:
                with st.spinner("Dispatching via SendGrid..."):
                    try:
                        client = SendGridClient(enabled=True)
                        success, msg = client.send_report(final_decision)
                        if success:
                            st.success(f"Delivered: {msg}")
                        else:
                            st.info(f"SendGrid Notice: {msg} (Set SENDGRID_API_KEY in .env to dispatch live emails)")
                    except Exception as ex:
                        st.info(f"SendGrid: {sanitize_error_message(ex)}")

    with a_col2:
        if st.button("📱 Send Mobile Push", use_container_width=True):
            if not final_decision:
                st.info("ℹ️ Decision report data is not available yet for mobile push.")
            else:
                with st.spinner("Dispatching via Pushover..."):
                    try:
                        client = PushoverClient(enabled=True)
                        success, msg = client.notify_decision(final_decision)
                        if success:
                            st.success(f"Delivered: {msg}")
                        else:
                            st.info(f"Pushover Notice: {msg} (Set PUSHOVER_USER_KEY in .env for mobile alerts)")
                    except Exception as ex:
                        st.info(f"Pushover: {sanitize_error_message(ex)}")

    with a_col3:
        if st.button("⚡ Trigger n8n Webhook", use_container_width=True):
            if not final_decision:
                st.info("ℹ️ Decision report data is not available yet to dispatch to n8n.")
            else:
                with st.spinner("Triggering n8n webhook..."):
                    try:
                        client = N8nClient(enabled=True)
                        success, msg = client.send_decision(final_decision)
                        if success:
                            st.success(f"Dispatched: {msg}")
                        else:
                            st.info(f"n8n Notice: {msg} (Configure N8N_WEBHOOK_URL in Settings or .env)")
                    except Exception as ex:
                        st.info(f"n8n: {sanitize_error_message(ex)}")



# =====================================================================
# MAIN WORKSPACE APPLICATION
# =====================================================================
def main():
    """Main application loop for the AI Shopping Decision Workspace."""
    inject_workspace_styles()
    settings_config = render_top_navigation()
    render_hero_search(settings_config)


    result = st.session_state.get("last_result")
    if not result:
        render_welcome_state()
        return

    # Check for short-circuit responses
    next_action = result.get("next_action")
    if next_action == "request_clarification":
        st.info("ℹ️ **Clarification Needed to Make an Objective Decision**")
        st.markdown(
            f"""
            <div style="background: rgba(99, 102, 241, 0.08); border-left: 4px solid #6366f1; padding: 18px 22px; border-radius: 8px; margin: 16px 0;">
                {result.get('final_response') or 'Please provide a specific product type or category to proceed with product research.'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    elif next_action == "invalid_input":
        st.warning("⚠️ **Invalid Shopping Request**")
        st.markdown(
            f"""
            <div style="background: rgba(239, 68, 68, 0.08); border-left: 4px solid #ef4444; padding: 18px 22px; border-radius: 8px; margin: 16px 0;">
                {result.get('final_response') or 'The query was flagged as non-actionable or empty.'}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Section 2: Requirements Breakdown
    render_requirements_chips(result)

    # Check Decision Status
    dec_info = extract_decision_display(result)
    is_recommended = dec_info.get("is_recommended", False)

    if is_recommended:
        # Step 4: Hero Best Match Card
        render_best_match_card(result)

        # Step 5: Why We Picked This
        render_why_this_product(result)

        # Step 6: Top Product Comparison
        render_top_product_comparison(result)

        # Step 7: Scoring Engine UI
        render_scoring_ui(result)

        # Step 8: Trade-offs
        render_tradeoffs_section(result)
    else:
        # Step 9: No Perfect Match State
        render_no_perfect_match_state(result)
        render_scoring_ui(result)

    # Step 10: Research Sources & Trust
    render_sources_trust(result)

    # Step 11: Optional Actions
    render_optional_actions(result)


if __name__ == "__main__":
    main()
