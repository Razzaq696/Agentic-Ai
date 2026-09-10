"""Complete LangGraph workflow for the Knowledge-Based Decision Agent."""

import re
import sys
from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, START, END

from agent.state import AgentState
from agent.reasoning import reason_and_decide
from rag.retriever import retrieve_context


def extract_search_keywords(raw_query: str) -> str:
    """Normalize user query and extract core retrieval keywords without conversational noise."""
    text = raw_query.strip()
    if not text:
        return ""

    prefix_patterns = [
        r"^(please\s+tell\s+me|can\s+you\s+explain|could\s+you\s+explain)\s+",
        r"^(what\s+is\s+the|what\s+are\s+the|what\s+is|what\s+are)\s+",
        r"^(tell\s+me\s+about|how\s+do\s+i|how\s+can\s+a\s+student|how\s+can\s+i)\s+",
        r"^(who\s+is\s+eligible\s+for\s+the|who\s+is\s+eligible\s+for|who\s+can\s+apply\s+for\s+the|who\s+can\s+apply\s+for|who\s+can)\s+",
        r"^(when\s+is\s+the|when\s+are\s+the|when\s+is|when\s+are)\s+",
        r"^(explain\s+the|describe\s+the|give\s+details\s+on)\s+",
    ]
    for pattern in prefix_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    text = re.sub(r"[?!.,;:]+$", "", text).strip()
    return text if text else raw_query.strip()


def analyze_query_node(state: AgentState) -> Dict[str, Any]:
    """Node 1: Query Analysis - Prepares user query for knowledge base search."""
    raw_query = state.get("query", "").strip()
    if not raw_query:
        return {"retrieval_query": "", "error": "Empty query received."}

    retrieval_query = extract_search_keywords(raw_query)
    return {"retrieval_query": retrieval_query}


# Alias for Phase 2 compatibility
analyze_query = analyze_query_node


def retrieve_context_node(state: AgentState) -> Dict[str, Any]:
    """Node 2: Retriever - Queries Chroma vector database using analyzed search keywords."""
    search_query = state.get("retrieval_query", "").strip() or state.get("query", "").strip()
    if not search_query:
        return {"context": [], "error": state.get("error") or "No query available for retrieval."}

    try:
        chunks = retrieve_context(query=search_query, k=3)
        return {"context": chunks}
    except Exception as e:
        print(f"[Error] Retriever Node failed: {e}")
        return {"context": [], "error": f"Retrieval error: {str(e)}"}


def generate_answer_node(state: AgentState) -> Dict[str, Any]:
    """Node 3: Agent Reasoning - Generates grounded answer/decision/recommendation via LLM."""
    return reason_and_decide(state)


def build_decision_agent_graph():
    """Construct and compile the full Phase 3 LangGraph Decision Agent workflow.

    Topology:
        START -> analyze_query -> retrieve_context -> generate_answer -> END
    """
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve_context", retrieve_context_node)
    workflow.add_node("generate_answer", generate_answer_node)

    # 2. Add Edges
    workflow.add_edge(START, "analyze_query")
    workflow.add_edge("analyze_query", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    # 3. Compile Graph
    return workflow.compile()


# Global compiled graph instance
decision_agent_graph = build_decision_agent_graph()

# Aliases for backwards compatibility with Phase 2
build_retrieval_graph = build_decision_agent_graph
retrieval_graph = decision_agent_graph


def run_decision_agent(query: str, k: int = 3) -> AgentState:
    """Execute the full Knowledge-Based Decision Agent pipeline for a user query.

    Pipeline:
        USER QUERY -> QUERY ANALYSIS -> RETRIEVER -> CHROMA -> CONTEXT -> REASONING -> ANSWER

    Args:
        query: User prompt/question.
        k: Number of chunks to retrieve.

    Returns:
        Final AgentState containing query, retrieval_query, context, answer, sources, and error.
    """
    initial_state: AgentState = {
        "query": query,
        "retrieval_query": "",
        "context": [],
        "answer": "",
        "sources": [],
        "error": None,
    }

    result = decision_agent_graph.invoke(initial_state)
    return result


# Alias for Phase 2 compatibility
run_retrieval_graph = run_decision_agent


def cli_main():
    """Command-line runner for interactive/testing Knowledge-Based Decision Agent."""
    if len(sys.argv) < 2:
        test_query = "What are the attendance requirements for students?"
    else:
        test_query = " ".join(sys.argv[1:])

    print("=" * 70)
    print(f"USER QUERY:\n{test_query}\n")

    result = run_decision_agent(test_query)

    print(f"ANALYZED RETRIEVAL QUERY:\n{result.get('retrieval_query')}\n")

    sources = result.get("sources", [])
    print(f"SOURCES REFERENCED: {', '.join(sources) if sources else 'None'}\n")

    print("AGENT DECISION / ANSWER:")
    print("=" * 70)
    print(result.get("answer", "No answer generated."))
    print("=" * 70)


if __name__ == "__main__":
    cli_main()
