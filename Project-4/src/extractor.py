"""LLM Extraction Agent for extracting structured information from retrieved context."""
import json
import logging
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.config import OLLAMA_MODEL, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT_TEMPLATE = """You are an expert Document Information Extraction Agent.
Your task is to accurately extract structured invoice data from the provided document context.

Target JSON Schema Structure:
{{
  "document_type": "Invoice",
  "invoice_number": "string (e.g. INV-1001)",
  "date": "string (e.g. YYYY-MM-DD)",
  "customer_name": "string",
  "items": [
    {{
      "description": "string",
      "quantity": 1,
      "unit_price": 10.0,
      "total": 10.0
    }}
  ],
  "total_amount": 10.0
}}

Guidelines:
1. Extract numerical fields (quantity, unit_price, total, total_amount) as pure numbers (integers or floats), removing currency symbols like $.
2. Do not fabricate missing information.
3. Return ONLY a valid JSON object matching the schema above. No markdown code fences, no extra text.

{feedback_section}

Document Context:
{context}
"""


class LLMExtractor:
    """LLM Extraction Agent component."""

    def __init__(self, model_name: str = OLLAMA_MODEL, temperature: float = LLM_TEMPERATURE):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature,
            format="json"
        )
        self.prompt = PromptTemplate(
            template=EXTRACTION_PROMPT_TEMPLATE,
            input_variables=["context", "feedback_section"]
        )

    def extract(
        self,
        retrieved_docs: List[Document] | str,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured JSON data from retrieved document context.

        Args:
            retrieved_docs: List of retrieved Document chunks or a combined text string.
            feedback: Optional feedback from previous failed validation for re-processing.

        Returns:
            Dict[str, Any]: Parsed JSON dictionary from LLM extraction.

        Raises:
            ValueError: If context is empty or LLM output is not valid JSON.
        """
        if isinstance(retrieved_docs, list):
            context_text = "\n\n---\n\n".join(doc.page_content for doc in retrieved_docs)
        else:
            context_text = str(retrieved_docs)

        if not context_text.strip():
            raise ValueError("Cannot extract information from empty context.")

        feedback_section = ""
        if feedback:
            feedback_section = (
                f"IMPORTANT - PREVIOUS VALIDATION FAILED WITH ERRORS:\n"
                f"{feedback}\n"
                f"Please correct the errors and re-extract carefully."
            )

        formatted_prompt = self.prompt.format(
            context=context_text,
            feedback_section=feedback_section
        )

        response = self.llm.invoke(formatted_prompt)
        raw_content = response.content.strip()

        # Parse JSON
        try:
            data = json.loads(raw_content)
            if not isinstance(data, dict):
                raise ValueError(f"Expected JSON object, got {type(data).__name__}")
            return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode JSON from LLM response: {raw_content}") from e
