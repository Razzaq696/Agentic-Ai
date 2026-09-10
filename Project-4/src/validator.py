"""Pydantic validation component for structured extraction output."""
from typing import Any, Dict
from pydantic import ValidationError
from src.schema import InvoiceSchema, ValidationResult


class DataValidator:
    """Validates extracted JSON dictionaries against the Pydantic schema."""

    @staticmethod
    def validate(raw_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate dictionary data against the InvoiceSchema.

        Args:
            raw_data: Dictionary containing extracted fields.

        Returns:
            ValidationResult: Result object containing validity status, validated model or error list.
        """
        if not isinstance(raw_data, dict):
            return ValidationResult(
                is_valid=False,
                data=None,
                errors=[f"Input must be a dictionary, got {type(raw_data).__name__}"]
            )

        try:
            validated_model = InvoiceSchema.model_validate(raw_data)
            return ValidationResult(
                is_valid=True,
                data=validated_model,
                errors=[]
            )
        except ValidationError as e:
            error_messages = []
            for err in e.errors():
                loc = " -> ".join(str(l) for l in err.get("loc", []))
                msg = err.get("msg", "Invalid value")
                error_messages.append(f"Field '{loc}': {msg}")
            return ValidationResult(
                is_valid=False,
                data=None,
                errors=error_messages
            )
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                data=None,
                errors=[f"Validation exception: {str(e)}"]
            )
