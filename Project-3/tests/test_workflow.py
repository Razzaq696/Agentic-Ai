import pytest
from workflow.graph import create_multi_agent_graph, run_workflow

def test_langgraph_construction():
    """Verify that the LangGraph compiles with all nodes and edges."""
    app = create_multi_agent_graph()
    assert app is not None
    # Verify nodes
    node_names = set(app.nodes.keys())
    assert "supervisor" in node_names
    assert "research_agent" in node_names
    assert "analysis_agent" in node_names
    assert "execution_agent" in node_names

def test_test1_research_and_analysis_workflow():
    """
    Test 1 — Research + Analysis:
    'Compare Python and JavaScript for beginner web development.'
    """
    problem = "Compare Python and JavaScript for beginner web development."
    result = run_workflow(problem)

    assert result["final_result"] is not None
    assert result["research_result"] is not None
    assert result["analysis_result"] is not None
    # Execution should not be triggered for pure comparison
    assert len(result["observations"]) >= 2
    
    agents_involved = [obs["agent"] for obs in result["observations"]]
    assert "research" in agents_involved
    assert "analysis" in agents_involved

    # Verify tool calling was recorded
    research_obs = [obs for obs in result["observations"] if obs["agent"] == "research"][0]
    assert research_obs["tool"] == "research_lookup"
    assert "Python" in research_obs["observation"] or "JavaScript" in research_obs["observation"]

def test_test2_execution_workflow():
    """
    Test 2 — Execution:
    'Calculate the total cost of 15 items at $24 each and explain the result.'
    """
    problem = "Calculate the total cost of 15 items at $24 each and explain the result."
    result = run_workflow(problem)

    assert result["final_result"] is not None
    assert result["execution_result"] is not None
    assert "360" in result["execution_result"]

    execution_obs = [obs for obs in result["observations"] if obs["agent"] == "execution"][0]
    assert execution_obs["tool"] == "calculate"
    assert "360" in execution_obs["observation"]

def test_test3_multi_agent_collaboration_workflow():
    """
    Test 3 — Multi-Agent Collaboration:
    'Research the average price information provided by the available research data, analyze it, and calculate the total for 3 units.'
    """
    problem = "Research the average price information provided by the available research data, analyze it, and calculate the total for 3 units."
    result = run_workflow(problem)

    assert result["final_result"] is not None
    assert result["research_result"] is not None
    assert result["analysis_result"] is not None
    assert result["execution_result"] is not None

    agents_involved = [obs["agent"] for obs in result["observations"]]
    assert "research" in agents_involved
    assert "analysis" in agents_involved
    assert "execution" in agents_involved

    # Execution calculation should evaluate 3 * 24 = 72
    assert "72" in result["execution_result"] or "72" in result["final_result"]

def test_tool_calling_verification():
    """Verify that real tool calls are made and recorded across agents."""
    problem = "Research pricing catalog data and calculate 5 * 24."
    result = run_workflow(problem)

    tools_used = [obs.get("tool") for obs in result["observations"] if "tool" in obs]
    assert "research_lookup" in tools_used or "calculate" in tools_used

def test_error_handling_empty_problem():
    """Verify graceful failure on empty problem."""
    result = run_workflow("   ")
    assert result["error"] == "Empty problem"
    assert "Error" in result["final_result"]
