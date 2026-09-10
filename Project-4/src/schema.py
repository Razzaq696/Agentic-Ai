"""Pydantic schemas for document extraction, validation, and pipeline results."""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class InvoiceItem(BaseModel):
    """Line item in an invoice."""
    description: str = Field(description="Description of the product or service")
    quantity: int = Field(description="Quantity purchased", ge=1)
    unit_price: float = Field(description="Price per unit", ge=0.0)
    total: float = Field(description="Total price for the line item", ge=0.0)

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Item description cannot be empty")
        return v.strip()


class InvoiceSchema(BaseModel):
    """Structured schema for extracted invoice documents."""
    document_type: str = Field(description="Type of document, e.g. 'Invoice'")
    invoice_number: str = Field(description="Unique invoice number/identifier")
    date: str = Field(description="Invoice date, e.g. YYYY-MM-DD")
    customer_name: str = Field(description="Customer or client company name")
    items: List[InvoiceItem] = Field(description="List of purchased line items", min_length=1)
    total_amount: float = Field(description="Grand total amount due", ge=0.0)

    @field_validator("document_type", "invoice_number", "customer_name", "date")
    @classmethod
    def validate_non_empty_strings(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    @field_validator("total_amount")
    @classmethod
    def validate_positive_total(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Total amount must be greater than 0")
        return v

    @model_validator(mode="after")
    def validate_items_and_total(self) -> "InvoiceSchema":
        if not self.items:
            raise ValueError("Invoice must contain at least one item")
        return self


class ValidationResult(BaseModel):
    """Result of Pydantic validation step."""
    is_valid: bool
    data: Optional[InvoiceSchema] = None
    errors: List[str] = Field(default_factory=list)


class PipelineResult(BaseModel):
    """End-to-end document processing pipeline output."""
    status: str = Field(description="'SUCCESS' or 'FAILED'")
    document_source: str
    extracted_data: Optional[InvoiceSchema] = None
    attempts: int = 1
    reprocessed: bool = False
    errors: List[str] = Field(default_factory=list)
    retrieved_chunks: List[str] = Field(default_factory=list)
