"""Backend runner function for invoking the LangGraph Agent."""

from typing import Dict, Any
from agent.graph import agent_app
from agent.state import AgentState


def run_agent(query: str) -> Dict[str, Any]:
    """Invokes the LangGraph Agent workflow with a user query.

    Args:
        query: The user query or command string.

    Returns:
        The resulting AgentState dictionary after workflow completion.
    """
    initial_state: AgentState = {
        "user_query": query,
        "query_analysis": "",
        "selected_action": "",
        "retrieved_context": None,
        "tool_result": None,
        "processed_result": None,
        "validation_status": None,
        "result": None,
        "final_response": None,
        "error": None,
    }

    try:
        final_state = agent_app.invoke(initial_state)
        return final_state
    except Exception as e:
        return {
            "user_query": query,
            "query_analysis": "ERROR",
            "selected_action": "ERROR",
            "retrieved_context": None,
            "tool_result": None,
            "processed_result": {
                "success": False,
                "source": "ERROR",
                "result": None,
                "error": str(e),
                "raw_data": None,
            },
            "validation_status": "INVALID",
            "result": None,
            "final_response": f"Error during workflow execution: {str(e)}",
            "error": str(e),
        }
