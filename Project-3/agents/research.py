from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from config import get_llm
from tools.tools import research_lookup
from workflow.state import MultiAgentState

RESEARCH_SYSTEM_PROMPT = """You are Agent A (Research Agent) in a multi-agent system.
Your job is to gather and retrieve accurate factual information or technical data needed to understand the user's problem.
You MUST use the `research_lookup` tool to retrieve data from the knowledge base.
Do not guess or hallucinate facts. Query the research tool with an appropriate search term."""

def research_node(state: MultiAgentState) -> Dict[str, Any]:
    """
    Research Agent Node:
    Uses tool calling with `research_lookup` to gather facts and returns observations.
    """
    problem = state.get("problem", "")
    llm = get_llm()
    llm_with_tools = llm.bind_tools([research_lookup])

    messages = [
        SystemMessage(content=RESEARCH_SYSTEM_PROMPT),
        HumanMessage(content=f"Problem to research: {problem}"),
    ]

    tool_used = False
    observation = ""
    query_used = ""

    try:
        response = llm_with_tools.invoke(messages)
        if response.tool_calls:
            for call in response.tool_calls:
                if call["name"] == "research_lookup":
                    query_used = call["args"].get("query", problem)
                    observation = research_lookup.invoke({"query": query_used})
                    tool_used = True
                    break
        
        # Fallback if LLM didn't emit a tool call directly
        if not tool_used:
            query_used = problem
            observation = research_lookup.invoke({"query": query_used})
    except Exception as e:
        observation = f"Research tool execution error: {str(e)}"
        query_used = problem

    new_observation_entry = {
        "agent": "research",
        "tool": "research_lookup",
        "query": query_used,
        "observation": observation,
    }

    current_observations = list(state.get("observations", []))
    current_observations.append(new_observation_entry)

    return {
        "research_result": observation,
        "observations": current_observations,
        "current_agent": "research",
        "step_count": state.get("step_count", 0) + 1,
    }
