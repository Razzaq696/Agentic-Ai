"""
Intelligent Task Execution Agent - Premium Streamlit Web Application
Phase 5: Real-Time Streamlit Interface with Live ReAct Execution Flow

Demonstrates the complete ReAct agent workflow:
User Goal -> Task Analysis -> Tool Selection -> Tool Execution -> Observation -> Agent Decision (Continue/Finish) -> Final Response
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from agent import run_agent, get_llm
from config import APP_NAME, MODEL_NAME, OLLAMA_BASE_URL, TEMPERATURE

# Tool metadata for icons and badges
TOOL_META = {
    "calculate": {"icon": "🧮", "label": "Calculator", "color": "#6366f1"},
    "web_search": {"icon": "🌐", "label": "Web Search", "color": "#0ea5e9"},
    "get_current_datetime": {"icon": "🕒", "label": "Date & Time", "color": "#10b981"},
}


def inject_custom_css() -> None:
    """Injects high-end glassmorphism, modern typography, and gradient styling."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Base Typography */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Outfit', sans-serif;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        /* Hero Title Gradient */
        .hero-title {
            background: linear-gradient(135deg, #6366f1 0%, #a855f7 40%, #ec4899 80%, #f43f5e 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.6rem !important;
            font-weight: 800;
            margin-bottom: 0.2rem;
            display: inline-block;
        }

        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }

        /* Glassmorphic Container Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.45);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
            transition: all 0.2s ease-in-out;
        }
        .glass-card:hover {
            border-color: rgba(99, 102, 241, 0.3);
            box-shadow: 0 14px 30px -5px rgba(99, 102, 241, 0.15);
        }

        /* Step Badge */
        .step-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(168, 85, 247, 0.2));
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.4);
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 0.85rem;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            margin-bottom: 0.6rem;
        }

        /* Tool Name Pill */
        .tool-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(99, 102, 241, 0.35);
            color: #818cf8;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.82rem;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 8px;
        }

        /* Decision Badges */
        .decision-continue {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(245, 158, 11, 0.15);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.35rem 0.9rem;
            border-radius: 8px;
        }

        .decision-finish {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.4);
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.35rem 0.9rem;
            border-radius: 8px;
        }

        /* Final Response Card */
        .response-hero {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.85) 0%, rgba(15, 23, 42, 0.95) 100%);
            border: 1px solid rgba(99, 102, 241, 0.4);
            box-shadow: 0 0 30px -5px rgba(99, 102, 241, 0.25);
            border-radius: 18px;
            padding: 1.5rem 1.75rem;
            margin-top: 1rem;
            margin-bottom: 2rem;
        }

        /* Metric Tile */
        .metric-tile {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            text-align: center;
        }
        .metric-value {
            font-family: 'Outfit', sans-serif;
            font-size: 1.3rem;
            font-weight: 700;
            color: #f8fafc;
        }
        .metric-label {
            font-size: 0.75rem;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.2rem;
        }

        /* Sidebar Styling */
        .sidebar-card {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 0.9rem;
            margin-bottom: 0.8rem;
        }

        /* Status Dot */
        .status-dot {
            width: 9px;
            height: 9px;
            background: #10b981;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #10b981;
            margin-right: 6px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Intelligent Task Execution Agent",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()

    if "user_task_input" not in st.session_state:
        st.session_state["user_task_input"] = ""

    # =========================================================================
    # SIDEBAR: Model Selector, System Status & Tools
    # =========================================================================
    with st.sidebar:
        st.markdown("### ⚙️ Engine Settings")
        
        # Model Selection (Fast LLaMA 3.2 vs Deep Reasoning Qwen 3)
        available_models = ["llama3.2:3b", "qwen3:4b"]
        selected_model = st.selectbox(
            "Select Local LLM Model:",
            options=available_models,
            index=0,  # Default to fast llama3.2:3b for optimal responsiveness
            help="llama3.2:3b is 4x faster on CPU. qwen3:4b provides deep multi-step reasoning.",
        )

        st.markdown(
            f"""
            <div class="sidebar-card">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.4rem;">
                    <span style="font-weight:600; color:#f8fafc; font-size:0.9rem;">Ollama Core</span>
                    <span style="color:#10b981; font-size:0.8rem; font-weight:600;"><span class="status-dot"></span>Active</span>
                </div>
                <div style="font-size:0.8rem; color:#94a3b8;">
                    <b>Model:</b> <code style="color:#c084fc;">{selected_model}</code><br>
                    <b>Endpoint:</b> <code style="color:#93c5fd;">{OLLAMA_BASE_URL}</code><br>
                    <b>Temp:</b> <code>{TEMPERATURE}</code> | <b>Runtime:</b> Python {sys.version.split()[0]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🛠️ Registered Tools")
        st.markdown(
            """
            <div class="sidebar-card" style="font-size:0.82rem; line-height:1.6;">
                <div style="margin-bottom:0.45rem;">
                    <span class="tool-pill">🧮 calculate</span><br>
                    <span style="color:#94a3b8;">Safe AST arithmetic & formulas</span>
                </div>
                <div style="margin-bottom:0.45rem;">
                    <span class="tool-pill">🌐 web_search</span><br>
                    <span style="color:#94a3b8;">Real-time web & facts lookup</span>
                </div>
                <div>
                    <span class="tool-pill">🕒 get_current_datetime</span><br>
                    <span style="color:#94a3b8;">Real-world temporal anchor</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### 🔄 ReAct Decision Cycle")
        st.markdown(
            """
            <div class="sidebar-card" style="font-size:0.8rem; color:#94a3b8; line-height:1.7;">
                1️⃣ <b>User Goal</b><br>
                2️⃣ <b>Task Analysis</b><br>
                3️⃣ <b>Tool Selection</b><br>
                4️⃣ <b>Tool Execution</b><br>
                5️⃣ <b>Observation</b><br>
                6️⃣ <b>Decision</b> (<code>CONTINUE</code> / <code>FINISH</code>)<br>
                7️⃣ <b>Final Response</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # MAIN AREA: Hero Header, Example Tasks, Form, Live Execution Trace
    # =========================================================================
    st.markdown('<div class="hero-title">🤖 Intelligent Task Execution Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">An autonomous ReAct reasoning agent that receives your objective, decomposes '
        "tasks, selects & executes tools, evaluates observations, and dynamically determines whether to "
        "<b>continue</b> or <b>finish</b> to deliver a grounded final answer.</div>",
        unsafe_allow_html=True,
    )

    # Quick Example Prompts
    with st.expander("💡 Example Tasks & Quick Prompts", expanded=False):
        st.markdown(
            "- **Math Calculation:** `What is 125 * 8?`\n"
            "- **Date & Time:** `What is the current date and time?`\n"
            "- **Web Search:** `Search the web for information about artificial intelligence.`\n"
            "- **Multi-Step Goal:** `First look up the current date and time to find what the current year is. Then calculate what that year plus 50 will be.`\n"
            "- **Direct Answer:** `Explain what an algorithm is in simple terms.`"
        )

    # Task Input Form
    st.subheader("📝 Task Input")
    with st.form("agent_task_form", clear_on_submit=False):
        user_input = st.text_area(
            "Enter your task",
            placeholder="What is 125 * 8?",
            height=95,
            key="user_task_input",
            help="Type any mathematical calculation, real-time question, date query, or multi-step goal.",
        )
        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            submitted = st.form_submit_button("Run Agent", type="primary", use_container_width=True)
        with col_info:
            st.caption(f"Executing with model: `{selected_model}` (Multi-threaded CPU accelerated)")

    # Execution Handling with Live Status Stream
    if submitted:
        if not user_input or not user_input.strip():
            st.warning("Please enter a task.")
            return

        cleaned_input = user_input.strip()
        t_start = time.time()

        # Real-time progress container
        with st.status("🤖 ReAct Agent Execution in Progress...", expanded=True) as status_container:
            
            def on_progress(event_msg: str):
                st.write(event_msg)

            custom_llm = get_llm(model_name=selected_model)
            result = run_agent(
                cleaned_input,
                llm=custom_llm,
                on_progress_callback=on_progress,
            )
            
            elapsed = time.time() - t_start

            if result.get("success", False):
                status_container.update(
                    label=f"✅ Task Completed in {elapsed:.1f}s!",
                    state="complete",
                    expanded=False,
                )
            else:
                status_container.update(
                    label=f"❌ Execution failed ({elapsed:.1f}s)",
                    state="error",
                    expanded=True,
                )

        if not result.get("success", False):
            st.error(f"❌ {result.get('final_response', 'The requested task could not be completed.')}")
            return

        # Metrics Bar
        steps = result.get("steps", [])
        total_steps = result.get("total_steps", len(steps))
        decision_flow = result.get("decision_flow", ["FINISH"])
        flow_str = " ➔ ".join([f"`{d}`" for d in decision_flow])

        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-value">{total_steps}</div>'
                f'<div class="metric-label">Execution Steps</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            tools_used_str = ", ".join(list({s.get("tool") for s in steps})) if steps else "None (Direct)"
            st.markdown(
                f'<div class="metric-tile"><div class="metric-value" style="font-size:1rem; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{tools_used_str}</div>'
                f'<div class="metric-label">Tools Invoked</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-value" style="font-size:0.95rem;">{flow_str}</div>'
                f'<div class="metric-label">Decision Flow</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            st.markdown(
                f'<div class="metric-tile"><div class="metric-value" style="color:#10b981;">{elapsed:.1f}s ⚡</div>'
                '<div class="metric-label">Execution Time</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Execution Trace
        st.subheader("🔍 Execution Flow & Trace")

        if steps:
            for step in steps:
                step_num = step.get("step_number", 1)
                tool_name = step.get("tool", "tool")
                tool_input = step.get("tool_input", {})
                observation = step.get("observation", "")
                decision = step.get("decision", "FINISH")

                meta = TOOL_META.get(tool_name, {"icon": "🔧", "label": tool_name})

                with st.container():
                    st.markdown(f"#### 📍 Step {step_num}")
                    st.markdown(
                        f"""
                        <div class="glass-card" style="margin-bottom:0.75rem;">
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <div>
                                    <span style="font-size:0.9rem; color:#94a3b8;">Tool Selected:</span> 
                                    <span class="tool-pill">{meta['icon']} {tool_name}</span>
                                </div>
                                <div>
                                    {f'<span class="decision-continue">🔄 Decision: CONTINUE</span>' if decision == 'CONTINUE' else '<span class="decision-finish">🏁 Decision: FINISH</span>'}
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    c_left, c_right = st.columns([1, 1.2])
                    with c_left:
                        st.markdown("**📥 Tool Input Parameters:**")
                        st.json(tool_input)

                    with c_right:
                        st.markdown("**📤 Tool Observation / Result:**")
                        st.code(observation, language="text")

                    st.markdown("---")
        else:
            with st.container():
                st.info("ℹ️ **Task Analysis:** No external tool was required for this request.")
                st.markdown(
                    """
                    <div class="glass-card" style="margin-bottom:0.75rem;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span class="step-badge">📍 Direct Reasoning</span>
                            <span class="decision-finish">🏁 Decision: FINISH</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Final Response Display
        st.subheader("🎯 Final Response")
        st.markdown(
            f"""
            <div class="response-hero">
                <div style="font-size:1.08rem; line-height:1.7; color:#f1f5f9;">
                    {result.get("final_response", "Task completed.")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
