"""Streamlit Web Application for Project 6: Telegram & LangGraph Agentic AI Assistant.

Provides a modern, interactive web GUI to interact with the LangGraph agent workflow,
inspect execution traces, route classification, RAG retrieval, and AST tool computations.
"""

import time
import streamlit as st
from agent.runner import run_agent
from agent.analyzer import ACTION_LLM_REASONING, ACTION_RAG_KNOWLEDGE, ACTION_TOOL_API

# Configure page settings
st.set_page_config(
    page_title="Agentic AI Assistant | LangGraph & Telegram Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern, premium visual aesthetics
st.markdown(
    """
    <style>
    /* Main container styling */
    .main {
        background: radial-gradient(circle at top right, #1a1d2e 0%, #0d0f17 100%);
    }

    /* Custom Header Styles */
    .header-box {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(12px);
    }
    
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .header-subtitle {
        color: #94a3b8;
        font-size: 0.98rem;
        line-height: 1.5;
        margin-bottom: 14px;
    }

    .badge-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.025em;
    }

    .badge-langgraph {
        background-color: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.4);
    }

    .badge-rag {
        background-color: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    .badge-tool {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }

    .badge-pydantic {
        background-color: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        border: 1px solid rgba(236, 72, 153, 0.4);
    }

    /* Message routing pills */
    .route-pill-llm {
        display: inline-block;
        background: rgba(99, 102, 241, 0.2);
        color: #c7d2fe;
        border: 1px solid #6366f1;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .route-pill-rag {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #a7f3d0;
        border: 1px solid #10b981;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .route-pill-tool {
        display: inline-block;
        background: rgba(245, 158, 11, 0.2);
        color: #fef08a;
        border: 1px solid #f59e0b;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .valid-pill {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.4);
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 6px;
    }

    /* Trace Details Card */
    .trace-card {
        background-color: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 10px;
        font-size: 0.85rem;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.4) 0%, rgba(15, 23, 42, 0.5) 100%);
        border: 1px dashed rgba(148, 163, 184, 0.3);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }

    /* Stat Box */
    .stat-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    .stat-card {
        flex: 1;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 10px;
        text-align: center;
    }
    .stat-number {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f8fafc;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "route_counts" not in st.session_state:
    st.session_state.route_counts = {
        ACTION_LLM_REASONING: 0,
        ACTION_RAG_KNOWLEDGE: 0,
        ACTION_TOOL_API: 0,
    }

if "preset_prompt" not in st.session_state:
    st.session_state.preset_prompt = None

# Header Section
st.markdown(
    """
    <div class="header-box">
        <div class="header-title">🤖 Agentic AI Assistant Dashboard</div>
        <div class="header-subtitle">
            Autonomous multi-stage agentic workflow powered by <b>LangGraph DAG</b>, 
            <b>ChromaDB Semantic Vector Search</b>, <b>Safe AST Calculator Tools</b>, 
            and <b>Pydantic Schema Validation</b>.
        </div>
        <div class="badge-bar">
            <span class="badge badge-langgraph">⚡ LangGraph StateGraph</span>
            <span class="badge badge-rag">📚 ChromaDB Vector RAG</span>
            <span class="badge badge-tool">🧮 Safe AST Calculator</span>
            <span class="badge badge-pydantic">🛡️ Pydantic V2 Validation</span>
            <span class="badge badge-langgraph">📱 Telegram Bot Compatible</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar Configuration and Controls
with st.sidebar:
    st.markdown("### ⚙️ System Control & Architecture")
    
    # Workflow Architecture Visualization
    with st.expander("🗺️ LangGraph DAG Flow", expanded=True):
        st.markdown(
            """
            ```mermaid
            flowchart TD
                A([User Query]) --> B[analyze_query]
                B --> C{branch_router}
                C -->|LLM_REASONING| D[llm_reasoning]
                C -->|RAG_KNOWLEDGE| E[rag_knowledge]
                C -->|TOOL_API| F[tools_apis]
                D --> G[process_result]
                E --> G
                F --> G
                G --> H[validate_result]
                H --> I[generate_final_reply]
                I --> J([Response])
            ```
            """
        )

    # Activity Metrics
    st.markdown("### 📊 Workflow Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-number">{st.session_state.route_counts[ACTION_LLM_REASONING]}</div>
                <div class="stat-label">LLM</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-number">{st.session_state.route_counts[ACTION_RAG_KNOWLEDGE]}</div>
                <div class="stat-label">RAG</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="stat-card">
                <div class="stat-number">{st.session_state.route_counts[ACTION_TOOL_API]}</div>
                <div class="stat-label">Tools</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick Presets
    st.markdown("### 💡 Quick Prompt Presets")
    preset_prompts = [
        ("🧮 Math: Complex Calculation", "Calculate (150 * 4) + (240 / 6) - 35"),
        ("📚 RAG: Project Documentation", "Search knowledge base for project documentation and architecture details"),
        ("🧠 LLM: Conceptual Reasoning", "Explain how LangGraph enables stateful and cyclic agent workflows."),
        ("🧮 Math: Multi-operator", "Calculate 125 * 8 / 2"),
        ("📚 RAG: Company Policy", "Retrieve company policy and security guidelines from internal documents"),
    ]

    for label, prompt_text in preset_prompts:
        if st.button(label, use_container_width=True):
            st.session_state.preset_prompt = prompt_text
            st.rerun()

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.route_counts = {
            ACTION_LLM_REASONING: 0,
            ACTION_RAG_KNOWLEDGE: 0,
            ACTION_TOOL_API: 0,
        }
        st.session_state.preset_prompt = None
        st.rerun()

    st.markdown("---")
    st.caption("🚀 Project 6 — Agentic AI Assistant • Streamlit Edition")

# Helper function to render route pill
def render_route_pill(route: str) -> str:
    if route == ACTION_TOOL_API:
        return '<span class="route-pill-tool">🧮 TOOLS / CALCULATOR</span>'
    elif route == ACTION_RAG_KNOWLEDGE:
        return '<span class="route-pill-rag">📚 RAG / KNOWLEDGE</span>'
    elif route == ACTION_LLM_REASONING:
        return '<span class="route-pill-llm">🧠 LLM / REASONING</span>'
    return f'<span class="route-pill-llm">⚙️ {route}</span>'

# Render Chat History
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>👋 Welcome to the Agentic Assistant!</h3>
            <p style="color: #94a3b8; max-width: 600px; margin: 0 auto 15px auto;">
                Ask any question or pick a prompt from the sidebar. The agent will analyze your query,
                select the optimal execution path (<b>LLM Reasoning</b>, <b>ChromaDB RAG</b>, or <b>Safe Calculator</b>),
                and validate the output with Pydantic.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                route = msg.get("route", "UNKNOWN")
                valid = msg.get("validation_status", "VALID")
                pill_html = render_route_pill(route)
                valid_html = f'<span class="valid-pill">🛡️ {valid}</span>'
                
                st.markdown(f"{pill_html} {valid_html}", unsafe_allow_html=True)
                st.write(msg["content"])

                # Expandable workflow inspection trace
                if "trace" in msg and msg["trace"]:
                    with st.expander("🔍 Inspect Workflow Execution Trace", expanded=False):
                        trace = msg["trace"]
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**Selected Action Route:** `{trace.get('selected_action', 'N/A')}`")
                            st.markdown(f"**Query Analysis:** `{trace.get('query_analysis', 'N/A')}`")
                            st.markdown(f"**Validation Status:** `{trace.get('validation_status', 'N/A')}`")
                        with col_b:
                            st.markdown(f"**Error Trapped:** `{trace.get('error') or 'None (Clean Run)'}`")
                        
                        # Route-specific trace details
                        if trace.get("tool_result"):
                            st.markdown("##### 🧮 Tool Execution Output")
                            st.json(trace["tool_result"])
                        if trace.get("retrieved_context"):
                            st.markdown("##### 📚 Retrieved Knowledge Context")
                            for idx, chunk in enumerate(trace["retrieved_context"], start=1):
                                st.info(f"**Document Chunk {idx}:**\n\n{chunk}")
                        if trace.get("processed_result"):
                            st.markdown("##### ⚙️ Normalized Processed Result")
                            st.json(trace["processed_result"])

# Handle prompt input (from preset or user chat input)
prompt_to_process = None
if st.session_state.preset_prompt:
    prompt_to_process = st.session_state.preset_prompt
    st.session_state.preset_prompt = None
else:
    chat_input = st.chat_input("Ask a question, request a calculation, or query knowledge...")
    if chat_input:
        prompt_to_process = chat_input

if prompt_to_process:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_process})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(prompt_to_process)

    # Process with LangGraph Agent
    with st.chat_message("assistant", avatar="🤖"):
        status_container = st.empty()
        with status_container:
            with st.spinner("🤖 Agent analyzing query and navigating LangGraph DAG..."):
                start_time = time.time()
                state = run_agent(prompt_to_process)
                elapsed = time.time() - start_time

        status_container.empty()

        route = state.get("selected_action", "UNKNOWN")
        reply = state.get("final_response", "No response generated.")
        validation = state.get("validation_status", "VALID")

        # Update route counts
        if route in st.session_state.route_counts:
            st.session_state.route_counts[route] += 1

        # Display route badges and response
        pill_html = render_route_pill(route)
        valid_html = f'<span class="valid-pill">🛡️ {validation}</span>'
        st.markdown(f"{pill_html} {valid_html}", unsafe_allow_html=True)
        st.write(reply)

        # Trace expander
        with st.expander("🔍 Inspect Workflow Execution Trace", expanded=False):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Selected Action Route:** `{state.get('selected_action', 'N/A')}`")
                st.markdown(f"**Query Analysis:** `{state.get('query_analysis', 'N/A')}`")
                st.markdown(f"**Validation Status:** `{state.get('validation_status', 'N/A')}`")
            with col_b:
                st.markdown(f"**Execution Latency:** `{elapsed:.2f}s`")
                st.markdown(f"**Error Trapped:** `{state.get('error') or 'None (Clean Run)'}`")

            if state.get("tool_result"):
                st.markdown("##### 🧮 Tool Execution Output")
                st.json(state["tool_result"])
            if state.get("retrieved_context"):
                st.markdown("##### 📚 Retrieved Knowledge Context")
                for idx, chunk in enumerate(state["retrieved_context"], start=1):
                    st.info(f"**Document Chunk {idx}:**\n\n{chunk}")
            if state.get("processed_result"):
                st.markdown("##### ⚙️ Normalized Processed Result")
                st.json(state["processed_result"])

        # Save assistant message in session
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "route": route,
            "validation_status": validation,
            "trace": {
                "selected_action": state.get("selected_action"),
                "query_analysis": state.get("query_analysis"),
                "validation_status": state.get("validation_status"),
                "error": state.get("error"),
                "tool_result": state.get("tool_result"),
                "retrieved_context": state.get("retrieved_context"),
                "processed_result": state.get("processed_result"),
            },
        })
