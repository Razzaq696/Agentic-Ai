from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm
from workflow.state import MultiAgentState

ANALYSIS_SYSTEM_PROMPT = """You are Agent B (Analysis Agent) in a multi-agent system.
Your job is to analyze the user's problem along with any research observations collected by the Research Agent.
You must provide a structured and concise analysis containing:
1. Key Findings
2. Technical Comparison / Interpretations
3. Concise Summary & Recommendation for the Supervisor

Do NOT expose hidden chain-of-thought or raw internal reasoning. Keep the output factual, structured, and concise."""

def analysis_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Analysis Agent Node:
    Analyzes gathered facts and problem context to provide structured insights.
    """
    problem = state.get("problem", "")
    research_result = state.get("research_result", "No prior research data available.")
    
    llm = get_llm()

    user_content = (
        f"User Problem: {problem}\n\n"
        f"Research Observation Provided: {research_result}\n\n"
        "Please provide your concise structured analysis."
    )

    messages = [
        SystemMessage(content=ANALYSIS_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        analysis_text = response.content.strip()
    except Exception as e:
        analysis_text = f"Analysis Agent Error: {str(e)}"

    new_observation_entry = {
        "agent": "analysis",
        "observation": analysis_text,
    }

    current_observations = list(state.get("observations", []))
    current_observations.append(new_observation_entry)

    return {
        "analysis_result": analysis_text,
        "observations": current_observations,
        "current_agent": "analysis",
        "step_count": state.get("step_count", 0) + 1,
    }
