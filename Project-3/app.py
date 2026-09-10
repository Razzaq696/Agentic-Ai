import streamlit as st
from workflow.graph import run_workflow

# ---------------------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Problem Solving System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern UI / UX Styles
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header styling */
.hero-container {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.badge-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
    color: #ffffff;
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 12px;
    border-radius: 9999px;
    margin-bottom: 12px;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #f8fafc 0%, #cbd5e1 50%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
}

/* Agent Architecture Flow Card */
.flow-card {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(15, 23, 42, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 14px 20px;
    margin-bottom: 24px;
    flex-wrap: wrap;
    gap: 10px;
}

.flow-step {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
    font-weight: 600;
    color: #e2e8f0;
}

.flow-step-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}

.icon-sup { background: rgba(99, 102, 241, 0.2); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.4); }
.icon-res { background: rgba(14, 165, 233, 0.2); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.4); }
.icon-ana { background: rgba(168, 85, 247, 0.2); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }
.icon-exe { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }

.flow-arrow {
    color: #64748b;
    font-size: 0.9rem;
}

/* Agent Card Styles */
.agent-card {
    background: rgba(30, 41, 59, 0.5);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    transition: all 0.2s ease-in-out;
}

.agent-card:hover {
    border-color: rgba(99, 102, 241, 0.3);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.agent-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
}

.agent-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.pill-research { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); }
.pill-analysis { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
.pill-execution { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
.pill-supervisor { background: rgba(99, 102, 241, 0.15); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.3); }

.tool-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    background: rgba(15, 23, 42, 0.8);
    color: #e2e8f0;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.code-box {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
    padding: 10px 14px;
    color: #f1f5f9;
    margin: 8px 0;
}

/* Final Result Hero Card */
.final-result-card {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.8) 100%);
    border: 2px solid rgba(99, 102, 241, 0.4);
    border-radius: 16px;
    padding: 24px 28px;
    margin: 20px 0;
    box-shadow: 0 10px 30px -5px rgba(99, 102, 241, 0.15);
}

.final-result-title {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 14px;
}

/* Streamlit Button Overrides */
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #4f46e5 0%, #7c3aed 100%);
    color: #ffffff;
    font-weight: 600;
    font-size: 1rem;
    padding: 10px 28px;
    border-radius: 10px;
    border: none;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4);
    transition: all 0.2s ease;
    width: 100%;
}

div.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
    transform: translateY(-1px);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: System Overview & Preset Samples
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 Multi-Agent Architecture")
    st.markdown(
        """
        This system implements an intelligent coordinator workflow using **LangGraph** and **local Ollama LLMs**.
        
        **Specialized Agents:**
        - 👔 **Supervisor Agent**: Orchestrates task routing & final synthesis
        - 🔬 **Research Agent**: Deterministic information & price catalog lookup (`research_lookup`)
        - 📊 **Analysis Agent**: Structured reasoning & technical comparisons
        - ⚙️ **Execution Agent**: Safe AST arithmetic computation (`calculate`)
        """
    )
    
    st.divider()
    st.markdown("### 💡 Quick-Fill Example Tasks")
    
    preset_choice = st.selectbox(
        "Select a template to test:",
        [
            "Select an example...",
            "Test 1: Research + Analysis (Python vs JavaScript)",
            "Test 2: Execution (15 items @ $24 calculation)",
            "Test 3: Multi-Agent Collaboration (Catalog lookup + 3 units)",
        ],
    )
    
    selected_preset_text = ""
    if preset_choice == "Test 1: Research + Analysis (Python vs JavaScript)":
        selected_preset_text = "Compare Python and JavaScript for beginner web development."
    elif preset_choice == "Test 2: Execution (15 items @ $24 calculation)":
        selected_preset_text = "Calculate the total cost of 15 items at $24 each and explain the result."
    elif preset_choice == "Test 3: Multi-Agent Collaboration (Catalog lookup + 3 units)":
        selected_preset_text = "Research the average price information provided by the available research data, analyze it, and calculate the total for 3 units."

    st.divider()
    st.caption("⚡ Powered by LangGraph • ChatOllama • Zero Paid APIs")

# ---------------------------------------------------------
# Main UI Layout
# ---------------------------------------------------------

# Header
st.title("Multi-Agent Problem Solving System")
st.write("Enter a problem and let the Supervisor Agent coordinate the specialized agents.")

# Visual Interactive Pipeline Overview
st.markdown(
    """
    <div class="flow-card">
        <div class="flow-step">
            <div class="flow-step-icon icon-sup">👤</div>
            <span>User Problem</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <div class="flow-step-icon icon-sup">👔</div>
            <span>Supervisor</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <div class="flow-step-icon icon-res">🔬</div>
            <span>Research</span>
        </div>
        <span class="flow-arrow">/</span>
        <div class="flow-step">
            <div class="flow-step-icon icon-ana">📊</div>
            <span>Analysis</span>
        </div>
        <span class="flow-arrow">/</span>
        <div class="flow-step">
            <div class="flow-step-icon icon-exe">⚙️</div>
            <span>Execution</span>
        </div>
        <span class="flow-arrow">➔</span>
        <div class="flow-step">
            <div class="flow-step-icon icon-sup">🎯</div>
            <span>Final Result</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Text Input Area
default_input_val = selected_preset_text if selected_preset_text else ""
problem_input = st.text_area(
    "Enter your problem:",
    value=default_input_val,
    placeholder="e.g. Compare Python and JavaScript for beginner web development.",
    height=120,
    help="Type your problem or choose a preset from the sidebar.",
)

# Run Agents Action Button
run_clicked = st.button("Run Agents", type="primary")

# ---------------------------------------------------------
# Workflow Execution & Result Handling
# ---------------------------------------------------------
if run_clicked:
    clean_problem = problem_input.strip()
    
    if not clean_problem:
        st.warning("Please enter a problem first.")
    else:
        with st.status("🤖 Supervisor coordinating specialized agents...", expanded=True) as status:
            st.write("🔄 Initializing LangGraph state & routing supervisor...")
            result_state = run_workflow(clean_problem)
            status.update(label="✅ Multi-agent problem solving complete!", state="complete", expanded=False)
            
        # Error handling
        if result_state.get("error") and not result_state.get("final_result"):
            st.error(f"Workflow error: {result_state.get('error')}")
        else:
            # 1. Display Final Result
            st.subheader("Final Result")
            final_result_text = result_state.get("final_result", "No final result generated.")
            st.markdown(final_result_text)

            # 2. Display Agent Activity in Expandable Section
            with st.expander("Agent Activity", expanded=True):
                observations = result_state.get("observations", [])
                if not observations:
                    st.write("No agent observations recorded.")
                else:
                    for idx, obs in enumerate(observations, 1):
                        agent_name = obs.get("agent", "Agent").lower()
                        tool_name = obs.get("tool")
                        obs_content = obs.get("observation", "")

                        # Agent icon and pill class
                        if agent_name == "research":
                            icon = "🔬"
                            pill_class = "pill-research"
                            display_title = "Research Agent"
                        elif agent_name == "analysis":
                            icon = "📊"
                            pill_class = "pill-analysis"
                            display_title = "Analysis Agent"
                        elif agent_name == "execution":
                            icon = "⚙️"
                            pill_class = "pill-execution"
                            display_title = "Execution Agent"
                        else:
                            icon = "👔"
                            pill_class = "pill-supervisor"
                            display_title = "Supervisor Agent"

                        st.markdown(
                            f"""
                            <div class="agent-card">
                                <div class="agent-header">
                                    <span class="agent-pill {pill_class}">{icon} {idx}. {display_title}</span>
                                    {f'<span class="tool-badge">🔧 Tool: {tool_name}</span>' if tool_name else ''}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if "query" in obs:
                            st.caption(f"🔎 **Search Query:** `{obs['query']}`")
                        if "expression" in obs:
                            st.caption(f"🧮 **Evaluation Expression:** `{obs['expression']}`")

                        st.info(obs_content)
