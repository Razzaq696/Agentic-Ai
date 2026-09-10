"""LangGraph Agent Workflow graph construction for Phase 2."""

from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.analyzer import (
    ACTION_LLM_REASONING,
    ACTION_RAG_KNOWLEDGE,
    ACTION_TOOL_API,
    ACTION_ERROR,
)
from agent.nodes import (
    analyze_query_node,
    decide_action_node,
    llm_reasoning_node,
    rag_knowledge_node,
    tools_api_node,
    process_result_node,
    validate_result_node,
    generate_final_reply_node,
)


def route_decision(state: AgentState) -> str:
    """Evaluates state.selected_action and determines the target branch node."""
    selected_action = state.get("selected_action")

    if selected_action == ACTION_LLM_REASONING:
        return "llm_reasoning"
    elif selected_action == ACTION_RAG_KNOWLEDGE:
        return "rag_knowledge"
    elif selected_action == ACTION_TOOL_API:
        return "tools_api"
    else:
        # For ERROR or invalid queries, proceed directly to process_result
        return "process_result"


def create_agent_graph():
    """Builds and compiles the complete Phase 2 LangGraph agent workflow."""
    workflow = StateGraph(AgentState)

    # 1. Add Workflow Nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("decide_action", decide_action_node)
    workflow.add_node("llm_reasoning", llm_reasoning_node)
    workflow.add_node("rag_knowledge", rag_knowledge_node)
    workflow.add_node("tools_api", tools_api_node)
    workflow.add_node("process_result", process_result_node)
    workflow.add_node("validate_result", validate_result_node)
    workflow.add_node("generate_final_reply", generate_final_reply_node)

    # 2. Add Standard Entry Edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "decide_action")

    # 3. Add Conditional Routing Edge
    workflow.add_conditional_edges(
        "decide_action",
        route_decision,
        {
            "llm_reasoning": "llm_reasoning",
            "rag_knowledge": "rag_knowledge",
            "tools_api": "tools_api",
            "process_result": "process_result",
        },
    )

    # 4. Connect Branch Outputs to Result Processing Node
    workflow.add_edge("llm_reasoning", "process_result")
    workflow.add_edge("rag_knowledge", "process_result")
    workflow.add_edge("tools_api", "process_result")

    # 5. Connect Process -> Validate -> Generate Final Reply -> END
    workflow.add_edge("process_result", "validate_result")
    workflow.add_edge("validate_result", "generate_final_reply")
    workflow.add_edge("generate_final_reply", END)

    # Compile the graph
    app = workflow.compile()
    return app


# Pre-compiled agent graph instance
agent_app = create_agent_graph()
