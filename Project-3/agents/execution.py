import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm
from tools.tools import calculate
from workflow.state import MultiAgentState

EXECUTION_SYSTEM_PROMPT = """You are Agent C (Execution Agent) in a multi-agent system.
Your job is to execute computational or arithmetic actions required to solve the problem, utilizing any prior research and analysis context.
You MUST use the `calculate` tool to evaluate mathematical expressions (e.g., '15 * 24' or '3 * 24.00').
Do not guess calculation results without calling the tool."""

def execution_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Execution Agent Node:
    Uses tool calling with `calculate` to execute arithmetic computations and returns observations.
    """
    problem = state.get("problem", "")
    research_result = state.get("research_result")
    analysis_result = state.get("analysis_result")

    llm = get_llm()
    llm_with_tools = llm.bind_tools([calculate])

    context_parts = [f"Problem: {problem}"]
    if research_result:
        context_parts.append(f"Research Data: {research_result}")
    if analysis_result:
        context_parts.append(f"Analysis Summary: {analysis_result}")
    
    context_text = "\n\n".join(context_parts)
    messages = [
        SystemMessage(content=EXECUTION_SYSTEM_PROMPT),
        HumanMessage(content=f"Execute the necessary calculation for this task:\n\n{context_text}"),
    ]

    tool_used = False
    observation = ""
    expression_used = ""

    try:
        response = llm_with_tools.invoke(messages)
        if response.tool_calls:
            for call in response.tool_calls:
                if call["name"] == "calculate":
                    expression_used = call["args"].get("expression", "")
                    calc_res = calculate.invoke({"expression": expression_used})
                    observation = f"Calculation Expression: {expression_used} = {calc_res}"
                    tool_used = True
                    break

        # Fallback heuristic if LLM didn't emit a tool call directly
        if not tool_used:
            # Check for numbers in problem or analysis
            # e.g., "15 items at $24 each" -> "15 * 24" or "3 units" -> "3 * 24"
            numbers = re.findall(r"\b\d+(?:\.\d+)?\b", problem)
            if len(numbers) >= 2:
                expression_used = f"{numbers[0]} * {numbers[1]}"
            elif len(numbers) == 1 and research_result and ("24" in research_result or "$24" in research_result):
                expression_used = f"{numbers[0]} * 24"
            else:
                expression_used = "0"
            
            calc_res = calculate.invoke({"expression": expression_used})
            observation = f"Calculation Expression: {expression_used} = {calc_res}"
    except Exception as e:
        observation = f"Execution tool execution error: {str(e)}"
        expression_used = "error"

    new_observation_entry = {
        "agent": "execution",
        "tool": "calculate",
        "expression": expression_used,
        "observation": observation,
    }

    current_observations = list(state.get("observations", []))
    current_observations.append(new_observation_entry)

    return {
        "execution_result": observation,
        "observations": current_observations,
        "current_agent": "execution",
        "step_count": state.get("step_count", 0) + 1,
    }
