import pytest
from agents.research import research_node
from agents.analysis import analysis_node
from agents.execution import execution_node
from agents.supervisor import supervisor_node
from workflow.state import MultiAgentState

def test_research_agent_execution():
    state: MultiAgentState = {
        "problem": "Compare Python and JavaScript for beginner web development.",
        "current_agent": "supervisor",
        "next_agent": "research",
        "research_result": None,
        "analysis_result": None,
        "execution_result": None,
        "observations": [],
        "final_result": None,
        "step_count": 1,
        "error": None,
    }
    result = research_node(state)
    assert result["research_result"] is not None
    assert "Python" in result["research_result"] or "JavaScript" in result["research_result"]
    assert len(result["observations"]) == 1
    assert result["observations"][0]["agent"] == "research"
    assert result["observations"][0]["tool"] == "research_lookup"

def test_analysis_agent_execution():
    state: MultiAgentState = {
        "problem": "Compare Python and JavaScript for beginner web development.",
        "current_agent": "research",
        "next_agent": "analysis",
        "research_result": "Python is great for backend; JavaScript is standard for web browsers.",
        "analysis_result": None,
        "execution_result": None,
        "observations": [
            {"agent": "research", "observation": "Python is great for backend; JavaScript is standard for web browsers."}
        ],
        "final_result": None,
        "step_count": 2,
        "error": None,
    }
    result = analysis_node(state)
    assert result["analysis_result"] is not None
    assert len(result["analysis_result"]) > 10
    assert len(result["observations"]) == 2
    assert result["observations"][1]["agent"] == "analysis"

def test_execution_agent_tool_calling():
    state: MultiAgentState = {
        "problem": "Calculate the total cost of 15 items at $24 each.",
        "current_agent": "supervisor",
        "next_agent": "execution",
        "research_result": None,
        "analysis_result": None,
        "execution_result": None,
        "observations": [],
        "final_result": None,
        "step_count": 1,
        "error": None,
    }
    result = execution_node(state)
    assert result["execution_result"] is not None
    assert "360" in result["execution_result"]
    assert len(result["observations"]) == 1
    assert result["observations"][0]["agent"] == "execution"
    assert result["observations"][0]["tool"] == "calculate"

def test_supervisor_agent_empty_problem():
    state: MultiAgentState = {
        "problem": "   ",
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
    result = supervisor_node(state)
    assert result["next_agent"] == "FINISH"
    assert result["error"] == "Empty problem"
    assert "Error" in result["final_result"]
