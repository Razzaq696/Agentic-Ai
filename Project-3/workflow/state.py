from typing import TypedDict, List, Dict, Any, Optional

class MultiAgentState(TypedDict):
    """
    Shared state schema for the Multi-Agent Problem Solving System.
    """
    problem: str
    current_agent: str
    next_agent: str
    research_result: Optional[str]
    analysis_result: Optional[str]
    execution_result: Optional[str]
    observations: List[Dict[str, Any]]
    final_result: Optional[str]
    step_count: int
    error: Optional[str]
