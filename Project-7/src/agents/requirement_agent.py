"""LLM-powered Requirement Analysis Agent."""

from typing import Any, Optional
from langchain_core.messages import HumanMessage, SystemMessage

from src.guardrails.validators import validate_structured_output, GuardrailValidationError
from src.llm.factory import get_llm
from src.llm.prompts import (
    REQUIREMENT_ANALYSIS_SYSTEM_PROMPT,
    REQUIREMENT_ANALYSIS_USER_TEMPLATE,
)
from src.models.schemas import BudgetInfo, RequirementAnalysisOutput
from src.utils.logger import logger


class RequirementAnalysisAgent:
    """Agent responsible for parsing, interpreting, and structuring shopping requirements."""

    def __init__(self, llm: Optional[Any] = None):
        """Initialize the agent with an LLM instance or use the default factory."""
        self.llm = llm or get_llm()

    def analyze(self, user_request: str) -> RequirementAnalysisOutput:
        """Analyze a user's shopping request and return structured requirements.

        Args:
            user_request: Natural language query describing shopping desire.

        Returns:
            Structured RequirementAnalysisOutput.
        """
        logger.info(f"Analyzing shopping request: {user_request!r}")

        prompt_messages = [
            SystemMessage(content=REQUIREMENT_ANALYSIS_SYSTEM_PROMPT),
            HumanMessage(
                content=REQUIREMENT_ANALYSIS_USER_TEMPLATE.format(user_request=user_request)
            ),
        ]

        try:
            # Bind structured output schema if available on LLM
            if hasattr(self.llm, "with_structured_output"):
                runnable = self.llm.with_structured_output(RequirementAnalysisOutput)
            else:
                runnable = self.llm

            raw_result = runnable.invoke(prompt_messages)
            logger.debug(f"Raw LLM output received: {raw_result}")

            # Guardrail validation of the structured output
            structured_output = validate_structured_output(raw_result)
            logger.info(
                f"Successfully parsed requirements for category '{structured_output.product_category}'"
            )
            return structured_output

        except GuardrailValidationError as gve:
            logger.error(f"Guardrail validation failure: {gve}")
            return RequirementAnalysisOutput(
                product_category=None,
                budget=BudgetInfo(is_specified=False, is_flexible=True),
                required_features=[],
                preferences=[],
                priorities=[],
                intended_use=None,
                is_clear=False,
                ambiguities_or_missing_info=[f"Validation Error: {str(gve)}"],
                summary="The agent encountered a validation issue while structuring your request.",
            )

        except Exception as e:
            logger.error(f"Unexpected error during requirement analysis: {e}", exc_info=True)
            return RequirementAnalysisOutput(
                product_category=None,
                budget=BudgetInfo(is_specified=False, is_flexible=True),
                required_features=[],
                preferences=[],
                priorities=[],
                intended_use=None,
                is_clear=False,
                ambiguities_or_missing_info=[f"Processing Error: {str(e)}"],
                summary="Failed to parse shopping request due to an internal error.",
            )
