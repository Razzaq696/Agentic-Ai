"""Premium Streamlit Web Interface for the Knowledge-Based Decision Agent."""

import time
import streamlit as st
from agent.graph import run_decision_agent

# Page Configuration
st.set_page_config(
    page_title="Knowledge-Based Decision Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for state-of-the-art SaaS aesthetics, typography, and card designs
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Badge & Title */
    .hero-container {
        padding: 0.5rem 0 1.5rem 0;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.25);
        color: #4f46e5;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin-bottom: 0.8rem;
        letter-spacing: 0.02em;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        line-height: 1.2;
        background: linear-gradient(135deg, #0f172a 0%, #334155 50%, #4338ca 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .hero-desc {
        font-size: 1.05rem;
        color: #64748b;
        line-height: 1.5;
        max-width: 780px;
    }

    /* Preset prompt buttons */
    .preset-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* Result Card */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 5px solid #4f46e5;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
        margin: 1rem 0;
        font-size: 1.02rem;
        line-height: 1.7;
        color: #1e293b;
    }

    /* Meta stats bar */
    .meta-bar {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.85rem;
        color: #475569;
    }
    .meta-item {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        font-weight: 500;
    }

    /* Source Tags */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border: 1px solid #cbd5e1;
        color: #334155;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        font-weight: 500;
        padding: 0.35rem 0.75rem;
        border-radius: 6px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s ease;
    }
    .source-chip:hover {
        border-color: #6366f1;
        color: #4338ca;
        transform: translateY(-1px);
    }

    /* Sidebar cards */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }
    .policy-item {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .policy-item-title {
        font-weight: 600;
        color: #1e293b;
    }
    .policy-item-meta {
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'JetBrains Mono', monospace;
    }

    /* Context Chunk Card */
    .context-chunk-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem;
        margin-bottom: 0.8rem;
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }
    .chunk-score {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background: #e0e7ff;
        color: #3730a3;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
    }
    .chunk-content {
        font-size: 0.88rem;
        color: #475569;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("### 🏛️ Knowledge Base Hub")
    st.caption("Private University Policy Knowledge Repository")

    st.markdown("#### ⚡ System Architecture")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**LLM Agent**")
        st.markdown('<span class="status-badge">● Active</span>', unsafe_allow_html=True)
        st.caption("Ollama (Local)")
    with col_s2:
        st.markdown("**Vector Store**")
        st.markdown('<span class="status-badge">● 28 Chunks</span>', unsafe_allow_html=True)
        st.caption("ChromaDB")

    st.markdown("---")
    st.markdown("#### 📚 Indexed Policy Documents")

    policies = [
        ("📋 Student Attendance Policy", "attendance_policy.txt", "80% threshold, exam debarment"),
        ("🎓 Scholarship & Financial Aid", "scholarship_policy.txt", "CGPA 3.75 waiver, grants"),
        ("📝 Examination & Assessment", "examination_policy.txt", "Hall rules, make-up exams"),
        ("🎯 Undergraduate Admission", "admission_policy.txt", "Requirements, credit transfer"),
        ("📊 Grading & Academic Standing", "grading_policy.txt", "GPA scale, probation rules"),
        ("🗓️ Student Leave of Absence", "leave_policy.txt", "Medical leave, semester pause"),
        ("💳 Tuition & Fee Payment", "fee_policy.txt", "Installments, late fee policies"),
    ]

    for title, fname, desc in policies:
        st.markdown(
            f"""
            <div class="policy-item">
                <div class="policy-item-title">{title}</div>
                <div class="policy-item-meta">{fname}</div>
                <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.caption("Google DeepMind Project 2 • Knowledge-Based Decision Agent")


# ----------------- MAIN VIEW -----------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">✨ Agentic RAG • Chroma • LangGraph Workflow</div>
        <div class="hero-title">Knowledge-Based Decision Agent</div>
        <div class="hero-desc">
            Ask questions, request policy recommendations, or evaluate academic decisions based on the private university knowledge base. The system analyzes queries, retrieves verified policy context, and produces grounded responses.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Preset Prompt Quick-Pill Selection
st.markdown('<div class="preset-title">💡 Sample Queries & Quick Prompts</div>', unsafe_allow_html=True)
p_col1, p_col2, p_col3, p_col4 = st.columns(4)

selected_preset = None
with p_col1:
    if st.button("📋 Attendance Rules", use_container_width=True):
        selected_preset = "What are the attendance requirements and penalties for students?"
with p_col2:
    if st.button("🎓 Scholarship (GPA 3.4)", use_container_width=True):
        selected_preset = "Based on the scholarship policy, am I eligible if my GPA is 3.4?"
with p_col3:
    if st.button("🚨 70% Attendance Advice", use_container_width=True):
        selected_preset = "I currently have 70% attendance. Based on the university policy, what should I do before the examination?"
with p_col4:
    if st.button("⚽ Unsupported Question", use_container_width=True):
        selected_preset = "Who won the FIFA World Cup in 2022?"

# Input State Management
if selected_preset:
    st.session_state["user_query_text"] = selected_preset

current_input_value = st.session_state.get("user_query_text", "")

# Text Area Query Input
user_question = st.text_area(
    "Enter your question",
    value=current_input_value,
    placeholder="Type your question, scenario, or decision query here...",
    height=100,
    key="user_query_textarea"
)

# Action Buttons Row
btn_col1, btn_col2 = st.columns([4, 1])
with btn_col1:
    ask_submitted = st.button("🚀 Ask Decision Agent", type="primary", use_container_width=True)
with btn_col2:
    clear_submitted = st.button("🔄 Clear", use_container_width=True)

if clear_submitted:
    st.session_state["user_query_text"] = ""
    st.rerun()

# Execution Flow
if ask_submitted:
    clean_query = user_question.strip() if user_question else ""

    if not clean_query:
        st.warning("⚠️ Please enter a question before asking the agent.")
    else:
        try:
            start_time = time.time()
            with st.spinner("🔍 Analyzing query, retrieving policy documents, and reasoning..."):
                result = run_decision_agent(clean_query)
            elapsed_time = time.time() - start_time

            # Error banner if present
            if result.get("error"):
                st.error(f"Notice: {result['error']}")

            # Execution Metadata Bar
            sources = result.get("sources", [])
            context_chunks = result.get("context", [])
            retrieval_query = result.get("retrieval_query", clean_query)

            st.markdown(
                f"""
                <div class="meta-bar">
                    <div class="meta-item">⏱️ <b>Latency:</b> {elapsed_time:.2f}s</div>
                    <div class="meta-item">🔍 <b>Analyzed Intent:</b> <code>{retrieval_query}</code></div>
                    <div class="meta-item">📄 <b>Retrieved:</b> {len(context_chunks)} Chunks</div>
                    <div class="meta-item">🏷️ <b>Sources:</b> {len(sources)} Verified Document(s)</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Final Answer Section
            st.markdown("### 🎯 Final Answer / Decision")
            answer_text = result.get("answer", "No answer generated.")
            st.markdown(f'<div class="result-card">{answer_text}</div>', unsafe_allow_html=True)

            # Sources Section
            st.markdown("### 📚 Grounded Source References")
            if sources:
                chips_html = "".join([f'<span class="source-chip">📄 {s}</span>' for s in sources])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.markdown("*(No sources referenced — query was out of domain or lacked relevant context)*")

            # Retrieved Context Expander
            with st.expander("🔎 View Retrieved Policy Context Chunks", expanded=False):
                if context_chunks:
                    for i, chunk in enumerate(context_chunks, 1):
                        source_name = chunk.get("source", "Unknown")
                        title = chunk.get("title", "")
                        score = chunk.get("score", 0.0)
                        content = chunk.get("content", "").strip()

                        st.markdown(
                            f"""
                            <div class="context-chunk-card">
                                <div class="chunk-header">
                                    <span>#{i} • <b>{title}</b> ({source_name})</span>
                                    <span class="chunk-score">Distance: {score:.3f}</span>
                                </div>
                                <div class="chunk-content">{content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info("No context chunks retrieved for this query.")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred while processing your request: {str(e)}")
