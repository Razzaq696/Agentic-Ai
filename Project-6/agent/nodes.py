"""LangGraph Node implementations for Project 6 Phase 2."""

from typing import Dict, Any
from agent.state import AgentState
from agent.analyzer import (
    classify_query,
    ACTION_LLM_REASONING,
    ACTION_RAG_KNOWLEDGE,
    ACTION_TOOL_API,
    ACTION_ERROR,
)
from agent.llm import execute_llm_reasoning, execute_rag_reasoning
from agent.rag import retrieve_knowledge
from agent.tools import execute_tool_by_query
from agent.validation import validate_execution_result


def analyze_query_node(state: AgentState) -> Dict[str, Any]:
    """Inspects the incoming query and determines what kind of request it is."""
    query = state.get("user_query", "")
    if not query or not query.strip():
        return {
            "query_analysis": ACTION_ERROR,
            "error": "Query is empty or invalid. Please provide a valid query.",
        }

    analysis = classify_query(query)
    return {
        "query_analysis": analysis,
        "error": None,
    }


def decide_action_node(state: AgentState) -> Dict[str, Any]:
    """Uses the query analysis to select the next workflow action."""
    analysis = state.get("query_analysis", ACTION_ERROR)

    if analysis == ACTION_ERROR or state.get("error"):
        return {
            "selected_action": ACTION_ERROR,
        }

    return {
        "selected_action": analysis,
    }


def llm_reasoning_node(state: AgentState) -> Dict[str, Any]:
    """Executes the functional LLM / Reasoning path."""
    query = state.get("user_query", "")
    try:
        response_text = execute_llm_reasoning(query)
        return {
            "result": response_text,
            "error": None,
        }
    except Exception as e:
        return {
            "result": None,
            "error": f"LLM reasoning failed: {str(e)}",
        }


def rag_knowledge_node(state: AgentState) -> Dict[str, Any]:
    """Retrieves relevant knowledge from ChromaDB and generates a grounded response."""
    query = state.get("user_query", "")
    try:
        context = retrieve_knowledge(query, top_k=2)
        if not context:
            return {
                "retrieved_context": [],
                "result": None,
                "error": "No relevant documents found in knowledge base.",
            }

        grounded_answer = execute_rag_reasoning(query, context)
        return {
            "retrieved_context": context,
            "result": grounded_answer,
            "error": None,
        }
    except Exception as e:
        return {
            "retrieved_context": [],
            "result": None,
            "error": f"RAG knowledge retrieval error: {str(e)}",
        }


def tools_api_node(state: AgentState) -> Dict[str, Any]:
    """Selects and executes domain-appropriate tool and captures structured outcome."""
    query = state.get("user_query", "")
    try:
        tool_output = execute_tool_by_query(query)
        status = tool_output.get("status")
        if status == "SUCCESS":
            return {
                "tool_result": tool_output,
                "result": str(tool_output.get("result")),
                "error": None,
            }
        else:
            return {
                "tool_result": tool_output,
                "result": None,
                "error": tool_output.get("error", "Tool execution failed."),
            }
    except Exception as e:
        failure_dict = {
            "tool_name": "tool_executor",
            "status": "FAILURE",
            "result": None,
            "error": f"Tool execution failed: {str(e)}",
        }
        return {
            "tool_result": failure_dict,
            "result": None,
            "error": failure_dict["error"],
        }


def process_result_node(state: AgentState) -> Dict[str, Any]:
    """Normalizes the intermediate output from LLM, RAG, or Tools into a standardized structure."""
    action = state.get("selected_action", "UNKNOWN")
    error = state.get("error")
    result = state.get("result")
    tool_result = state.get("tool_result")
    retrieved_context = state.get("retrieved_context")

    if error:
        processed = {
            "success": False,
            "source": action,
            "result": None,
            "error": error,
            "raw_data": tool_result or retrieved_context,
        }
    else:
        processed = {
            "success": True,
            "source": action,
            "result": result,
            "error": None,
            "raw_data": tool_result or retrieved_context,
        }

    return {
        "processed_result": processed,
    }


def validate_result_node(state: AgentState) -> Dict[str, Any]:
    """Validates the processed result prior to final reply generation."""
    processed = state.get("processed_result")
    action = state.get("selected_action", "")
    retrieved_context = state.get("retrieved_context")

    validation_status, validation_error = validate_execution_result(
        processed=processed,
        action=action,
        retrieved_context=retrieved_context,
    )

    if validation_status == "INVALID":
        return {
            "validation_status": "INVALID",
            "error": validation_error or state.get("error", "Validation failed."),
        }

    return {
        "validation_status": "VALID",
        "error": None,
    }


def generate_final_reply_node(state: AgentState) -> Dict[str, Any]:
    """Produces the final user-facing response from the validated result."""
    validation_status = state.get("validation_status")
    error = state.get("error")
    processed = state.get("processed_result", {})
    action = state.get("selected_action")

    # If an error occurred or validation failed
    if error or validation_status == "INVALID" or not processed.get("success"):
        clean_error = error or "An unexpected issue occurred while processing your request."
        return {
            "final_response": f"I couldn't complete that request because: {clean_error}",
        }

    result_text = processed.get("result", "")

    # Clean formatting depending on the source action
    if action == ACTION_TOOL_API:
        final_text = f"The calculated result is: {result_text}"
    else:
        final_text = result_text

    return {
        "final_response": final_text,
    }
