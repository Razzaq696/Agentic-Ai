from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from workflow.state import MultiAgentState
from agents.supervisor import supervisor_node, route_supervisor
from agents.research import research_node
from agents.analysis import analysis_node
from agents.execution import execution_node

def create_multi_agent_graph():
    """
    Constructs and compiles the LangGraph multi-agent workflow.
    Supervisor routes dynamically to Research, Analysis, or Execution agents,
    and specialized agents return their observations back to the Supervisor.
    """
    builder = StateGraph(MultiAgentState)

    # 1. Register Nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research_agent", research_node)
    builder.add_node("analysis_agent", analysis_node)
    builder.add_node("execution_agent", execution_node)

    # 2. Add entry point: START -> supervisor
    builder.add_edge(START, "supervisor")

    # 3. Add conditional routing edges from Supervisor
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "research": "research_agent",
            "analysis": "analysis_agent",
            "execution": "execution_agent",
            "FINISH": END,
        },
    )

    # 4. Route back to Supervisor from all specialized agents
    builder.add_edge("research_agent", "supervisor")
    builder.add_edge("analysis_agent", "supervisor")
    builder.add_edge("execution_agent", "supervisor")

    return builder.compile()

# Singleton compiled graph instance
multi_agent_app = create_multi_agent_graph()

def run_workflow(problem: str) -> MultiAgentState:
    """
    Executes the multi-agent problem-solving workflow for a given user problem.
    """
    initial_state: MultiAgentState = {
        "problem": problem,
        "current_agent": "supervisor",
        "next_agent": "",
        "research_result": None,
        "analysis_result": None,
        "execution_result": None,
        "observations": [],
        "final_result": None,
        "step_count": 0,
        "error": None,
    }

    result_state = multi_agent_app.invoke(initial_state)
    return result_state
