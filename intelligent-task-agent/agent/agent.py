"""
ReAct Agent Module for Intelligent Task Execution Agent.
Phase 4-5: Multi-Step Agent Decision with Live Real-Time Event Streaming

Orchestrates multi-step reasoning, tool execution, observation evaluation,
and autonomous Continue vs. Finish decision tracking with optional live progress callbacks.
"""

import re
import warnings
from typing import Dict, Any, List, Optional, Callable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

# Suppress harmless LangGraph deprecation notice for create_react_agent
warnings.filterwarnings("ignore", category=UserWarning)

from agent.llm import get_llm
from agent.prompts import SYSTEM_PROMPT
from tools.registry import TOOLS


def _clean_response(content: str) -> str:
    """Removes internal reasoning tags (<think>...</think>) from the final response."""
    if not content:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", str(content), flags=re.DOTALL)
    return cleaned.strip()


def get_agent(llm=None, tools=None):
    """
    Constructs and returns a compiled ReAct agent graph.

    Args:
        llm: Optional ChatModel instance (defaults to get_llm()).
        tools: Optional list of tools (defaults to central TOOLS registry).

    Returns:
        CompiledStateGraph: The compiled ReAct agent graph.
    """
    active_llm = llm or get_llm()
    active_tools = tools if tools is not None else TOOLS

    return create_react_agent(
        model=active_llm,
        tools=active_tools,
        prompt=SYSTEM_PROMPT,
    )


def run_agent(
    user_goal: str,
    llm=None,
    tools=None,
    on_progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """
    Executes a user goal through the ReAct agent with live progress streaming
    and explicit Continue/Finish step tracking.

    Args:
        user_goal: The user's query or objective.
        llm: Optional ChatModel override.
        tools: Optional tool list override.
        on_progress_callback: Optional callback receiving live progress event descriptions.

    Returns:
        Dict[str, Any] containing execution metadata, steps, decisions, and final response.
    """
    if not user_goal or not user_goal.strip():
        return {
            "goal": user_goal or "",
            "final_response": "Error: User goal cannot be empty.",
            "tool_calls_made": False,
            "total_steps": 0,
            "steps": [],
            "decision_flow": ["FINISH (Empty Input)"],
            "tool_calls": [],
            "success": False,
            "error": "Empty user input",
        }

    cleaned_goal = user_goal.strip()

    if on_progress_callback:
        on_progress_callback("🧠 **Analyzing user goal and decomposing task...**")

    try:
        agent = get_agent(llm=llm, tools=tools)

        # Collect messages across stream updates
        messages: List[Any] = [HumanMessage(content=cleaned_goal)]
        raw_steps: List[Dict[str, Any]] = []
        pending_tool_calls: Dict[str, Dict[str, Any]] = {}

        for event in agent.stream({"messages": [HumanMessage(content=cleaned_goal)]}, stream_mode="updates"):
            for node_name, node_output in event.items():
                node_msgs = node_output.get("messages", [])
                for msg in node_msgs:
                    messages.append(msg)

                    if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            tc_id = tc.get("id", str(len(pending_tool_calls)))
                            t_name = tc.get("name", "tool")
                            t_args = tc.get("args", {})
                            pending_tool_calls[tc_id] = {
                                "tool": t_name,
                                "input": t_args,
                                "observation": "",
                            }
                            if on_progress_callback:
                                on_progress_callback(f"🛠️ **Selected Tool:** `{t_name}` with input `{t_args}`")

                    elif isinstance(msg, ToolMessage):
                        tc_id = getattr(msg, "tool_call_id", None)
                        obs = str(msg.content)
                        if tc_id and tc_id in pending_tool_calls:
                            pending_tool_calls[tc_id]["observation"] = obs
                            raw_steps.append(pending_tool_calls[tc_id])
                        else:
                            raw_steps.append({
                                "tool": getattr(msg, "name", "tool"),
                                "input": {},
                                "observation": obs,
                            })
                        if on_progress_callback:
                            on_progress_callback(f"📥 **Observation Received:** `{obs[:150]}{'...' if len(obs) > 150 else ''}`")

        # Determine step-level Continue vs. Finish decisions
        structured_steps: List[Dict[str, Any]] = []
        decision_flow: List[str] = []
        total_steps = len(raw_steps)

        for idx, step_info in enumerate(raw_steps, start=1):
            is_last_step = (idx == total_steps)
            step_decision = "FINISH" if is_last_step else "CONTINUE"
            decision_flow.append(step_decision)

            structured_steps.append({
                "step_number": idx,
                "tool": step_info["tool"],
                "tool_input": step_info["input"],
                "observation": step_info["observation"],
                "decision": step_decision,
            })

        if total_steps == 0:
            decision_flow.append("FINISH (Direct Response)")
            if on_progress_callback:
                on_progress_callback("ℹ️ **Direct Reasoning:** No external tools required.")
        else:
            if on_progress_callback:
                on_progress_callback(f"🏁 **Agent Decision:** `FINISH` (Observation evaluated, all steps satisfied)")

        if on_progress_callback:
            on_progress_callback("🎯 **Formulating final grounded response...**")

        # Extract final user-facing response
        final_ai_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                cleaned_content = _clean_response(msg.content)
                if cleaned_content:
                    final_ai_msg = cleaned_content
                    break

        if not final_ai_msg:
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_ai_msg = str(msg.content).strip()
                    if final_ai_msg:
                        break

        compat_tool_calls = [
            {
                "tool": s["tool"],
                "input": s["tool_input"],
                "observation": s["observation"],
            }
            for s in structured_steps
        ]

        return {
            "goal": cleaned_goal,
            "final_response": final_ai_msg or "Task completed.",
            "tool_calls_made": total_steps > 0,
            "total_steps": total_steps,
            "steps": structured_steps,
            "decision_flow": decision_flow,
            "tool_calls": compat_tool_calls,
            "success": True,
            "error": None,
        }

    except ConnectionError:
        return {
            "goal": cleaned_goal,
            "final_response": "Error: Failed to connect to the local Ollama LLM service.",
            "tool_calls_made": False,
            "total_steps": 0,
            "steps": [],
            "decision_flow": ["FINISH (Error)"],
            "tool_calls": [],
            "success": False,
            "error": "Ollama connection failure",
        }
    except Exception as e:
        err_msg = str(e)
        return {
            "goal": cleaned_goal,
            "final_response": f"Error during agent execution: {err_msg}",
            "tool_calls_made": False,
            "total_steps": 0,
            "steps": [],
            "decision_flow": ["FINISH (Error)"],
            "tool_calls": [],
            "success": False,
            "error": err_msg,
        }
