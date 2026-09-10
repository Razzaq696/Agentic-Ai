"""Validation module using Pydantic for result verification."""

from typing import Optional, Any, Dict, List, Tuple
from pydantic import BaseModel, Field


class ProcessedResultSchema(BaseModel):
    """Pydantic model representing normalized internal result structure."""

    success: bool = Field(..., description="Whether the branch execution was successful")
    source: str = Field(..., description="The source node (LLM_REASONING, RAG_KNOWLEDGE, TOOL_API, ERROR)")
    result: Optional[str] = Field(None, description="The processed response text or output")
    error: Optional[str] = Field(None, description="Error message if execution or validation failed")
    raw_data: Optional[Any] = Field(None, description="Optional raw payload from tool or retriever")


def validate_execution_result(
    processed: Optional[Dict[str, Any]],
    action: str,
    retrieved_context: Optional[List[str]] = None,
) -> Tuple[str, Optional[str]]:
    """Validates the processed result against system constraints.

    Args:
        processed: The normalized processed result dictionary.
        action: The selected action route.
        retrieved_context: The context retrieved if RAG was invoked.

    Returns:
        Tuple of (validation_status: "VALID" | "INVALID", error_message: Optional[str])
    """
    if not processed:
        return "INVALID", "Processed result is missing or null."

    try:
        validated_model = ProcessedResultSchema(**processed)
    except Exception as ve:
        return "INVALID", f"Result schema validation failed: {str(ve)}"

    # 1. Handle error cases
    if not validated_model.success:
        error_msg = validated_model.error or "Execution failed without a specific error message."
        return "INVALID", error_msg

    # 2. Check for result existence on expected success
    if not validated_model.result or not validated_model.result.strip():
        return "INVALID", f"Expected valid result from {action}, but received empty content."

    # 3. Specific validation for RAG branch
    if action == "RAG_KNOWLEDGE":
        if retrieved_context is not None and len(retrieved_context) == 0:
            return "INVALID", "RAG retrieval yielded no relevant knowledge documents."

    # 4. Specific validation for Tool branch
    if action == "TOOL_API":
        if validated_model.raw_data and isinstance(validated_model.raw_data, dict):
            if validated_model.raw_data.get("status") == "FAILURE":
                return "INVALID", validated_model.raw_data.get("error", "Tool failed execution.")

    return "VALID", None
