import json
from typing import Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm
from workflow.state import MultiAgentState

SUPERVISOR_SYNTHESIS_PROMPT = """You are the Supervisor Agent in a multi-agent system.
Your specialized agents (Research, Analysis, Execution) have gathered observations for the user's problem.
Provide a clear, well-structured, professional final answer to the user's problem based strictly on the observations provided.
Do not invent unverified facts."""

def decide_next_agent(
    problem: str,
    has_research: bool,
    has_analysis: bool,
    has_execution: bool,
) -> str:
    """
    Evaluates the problem and current observation state to decide the next agent.
    Routes dynamically to Research, Analysis, Execution, or FINISH.
    """
    p_lower = problem.lower()

    # Detect task requirements
    needs_calc = any(term in p_lower for term in ["calculate", "cost", "total", "multiply", "$", "price for", "units", "items at", "*"])
    needs_research = any(term in p_lower for term in ["research", "what is", "difference", "price information", "pricing data", "catalog", "compare", "lookup"])
    needs_analysis = any(term in p_lower for term in ["compare", "analyze", "analysis", "vs", "which is better", "recommend", "explain"])

    # Multi-Agent Collaboration Workflow: Research -> Analysis -> Execution -> FINISH
    if ("research" in p_lower or "price information" in p_lower or "catalog" in p_lower) and ("analyze" in p_lower or "analysis" in p_lower) and needs_calc:
        if not has_research:
            return "research"
        if not has_analysis:
            return "analysis"
        if not has_execution:
            return "execution"
        return "FINISH"

    # Research + Analysis Workflow: Research -> Analysis -> FINISH
    if ("compare" in p_lower or "vs" in p_lower or needs_analysis) and not needs_calc:
        if not has_research:
            return "research"
        if not has_analysis:
            return "analysis"
        return "FINISH"

    # Pure Execution Workflow: Execution -> FINISH
    if needs_calc and not (needs_research and "research" in p_lower):
        if not has_execution:
            return "execution"
        return "FINISH"

    # General Workflow Sequencing
    if needs_research and not has_research:
        return "research"
    if needs_analysis and not has_analysis:
        return "analysis"
    if needs_calc and not has_execution:
        return "execution"

    return "FINISH"

def supervisor_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Supervisor Agent Node:
    Coordinates specialized agents, determines the next routing step,
    and synthesizes the final result when complete.
    """
    problem = state.get("problem", "").strip()
    step_count = state.get("step_count", 0)
    research_result = state.get("research_result")
    analysis_result = state.get("analysis_result")
    execution_result = state.get("execution_result")
    observations = state.get("observations", [])

    # Handle empty problem error case
    if not problem:
        return {
            "current_agent": "supervisor",
            "next_agent": "FINISH",
            "final_result": "Error: Empty problem provided. Please specify a valid task.",
            "error": "Empty problem",
            "step_count": step_count + 1,
        }

    # Safety limit to prevent infinite loops
    if step_count >= 6:
        next_agent = "FINISH"
    else:
        has_research = research_result is not None
        has_analysis = analysis_result is not None
        has_execution = execution_result is not None
        next_agent = decide_next_agent(problem, has_research, has_analysis, has_execution)

    # When finishing, synthesize the final result
    final_result: Optional[str] = None
    if next_agent == "FINISH":
        obs_text_blocks = []
        for obs in observations:
            agent = obs.get("agent", "Agent")
            val = obs.get("observation", "")
            obs_text_blocks.append(f"[{agent.upper()} AGENT OBSERVATION]:\n{val}")
        
        obs_context = "\n\n".join(obs_text_blocks) if obs_text_blocks else "No observations recorded."

        synthesis_content = (
            f"User Problem: {problem}\n\n"
            f"Collected Agent Observations:\n{obs_context}\n\n"
            "Provide the final comprehensive result for the user."
        )

        try:
            llm = get_llm()
            synthesis_res = llm.invoke([
                SystemMessage(content=SUPERVISOR_SYNTHESIS_PROMPT),
                HumanMessage(content=synthesis_content),
            ])
            final_result = synthesis_res.content.strip()
        except Exception as e:
            final_result = f"Final Result Synthesis:\nProblem: {problem}\nObservations:\n{obs_context}"

    return {
        "current_agent": "supervisor",
        "next_agent": next_agent,
        "final_result": final_result,
        "step_count": step_count + 1,
    }

def route_supervisor(state: MultiAgentState) -> str:
    """
    LangGraph conditional edge router function.
    Reads the supervisor's decision from state['next_agent'].
    """
    return state.get("next_agent", "FINISH")
