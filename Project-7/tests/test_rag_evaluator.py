"""Tests for Agentic RAG evaluator and structured RAG output schemas."""

import pytest
from src.models.schemas import (
    BudgetInfo,
    RAGDecision,
    RetrievedProduct,
    StructuredRAGOutput,
)
from src.rag.evaluator import AgenticRAGEvaluator
from src.guardrails.validators import validate_structured_rag_output


class TestAgenticRAGEvaluator:
    """Unit tests for Agentic RAG decision making."""

    def test_sufficient_retrieval(self, rag_evaluator):
        """Verify matching product within budget yields SUFFICIENT decision."""
        sample_products = [
            RetrievedProduct(
                product_id="test-1",
                product_name="Keychron Q1 Pro",
                category="Mechanical Keyboard",
                brand="Keychron",
                price=199.0,
                features=["Tactile switches", "Wireless"],
                source="Keychron Official",
            )
        ]
        budget = BudgetInfo(max_amount=250.0, is_specified=True, is_flexible=False)
        output = rag_evaluator.evaluate(
            retrieved_products=sample_products,
            category="Mechanical Keyboard",
            budget=budget,
            requirements=["Tactile switches"],
        )

        assert isinstance(output, StructuredRAGOutput)
        assert output.decision == RAGDecision.SUFFICIENT
        assert output.retrieval_sufficient is True
        assert output.research_needed is False
        assert output.retrieval_confidence >= 0.8
        assert len(output.relevant_information) > 0
        assert "Keychron Q1 Pro" in output.relevant_information[0]

    def test_empty_retrieval_triggers_external_research(self, rag_evaluator):
        """Verify empty retrieval automatically triggers EXTERNAL_RESEARCH_NEEDED."""
        output = rag_evaluator.evaluate(
            retrieved_products=[],
            category="Electric Scooter",
            budget=BudgetInfo(),
            requirements=["Foldable"],
        )

        assert output.decision == RAGDecision.EXTERNAL_RESEARCH_NEEDED
        assert output.retrieval_sufficient is False
        assert output.research_needed is True
        assert output.retrieval_confidence == 0.0
        assert "No product knowledge exists" in output.reasoning

    def test_category_mismatch_triggers_refinement_then_external(self, rag_evaluator):
        """Verify category mismatch triggers REFINE_QUERY on first retry, then EXTERNAL_RESEARCH_NEEDED."""
        sample_products = [
            RetrievedProduct(
                product_id="test-h1",
                product_name="Sony WH-1000XM5",
                category="Headphones",
                brand="Sony",
                price=398.0,
                source="Sony",
            )
        ]
        # Request was for a Laptop, but got Headphones
        # Attempt 0: Should trigger REFINE_QUERY
        out_retry0 = rag_evaluator.evaluate(
            retrieved_products=sample_products,
            category="Laptop",
            retry_count=0,
        )
        assert out_retry0.decision == RAGDecision.REFINE_QUERY
        assert out_retry0.retrieval_sufficient is False
        assert out_retry0.research_needed is True

        # Attempt 1 (at limit): Should trigger EXTERNAL_RESEARCH_NEEDED
        out_retry1 = rag_evaluator.evaluate(
            retrieved_products=sample_products,
            category="Laptop",
            retry_count=1,
        )
        assert out_retry1.decision == RAGDecision.EXTERNAL_RESEARCH_NEEDED
        assert out_retry1.retrieval_sufficient is False
        assert out_retry1.research_needed is True

    def test_budget_exceeded_triggers_research_needed(self, rag_evaluator):
        """Verify all products strictly exceeding budget leads to external research."""
        sample_products = [
            RetrievedProduct(
                product_id="test-expensive",
                product_name="High-End Workstation",
                category="Laptop",
                brand="Dell",
                price=3500.0,
                source="Dell Catalog",
            )
        ]
        budget = BudgetInfo(max_amount=1000.0, is_specified=True, is_flexible=False)
        output = rag_evaluator.evaluate(
            retrieved_products=sample_products,
            category="Laptop",
            budget=budget,
            retry_count=1,  # at max retries
        )

        assert output.decision == RAGDecision.EXTERNAL_RESEARCH_NEEDED
        assert output.retrieval_sufficient is False
        assert output.research_needed is True
        assert "exceed the specified budget ceiling" in output.reasoning

    def test_structured_rag_output_validation(self):
        """Verify Pydantic model serialization, deserialization, and validator."""
        valid_data = {
            "retrieved_products": [],
            "relevant_information": ["Info bullet 1"],
            "retrieval_confidence": 0.9,
            "retrieval_sufficient": True,
            "research_needed": False,
            "decision": "sufficient",
            "reasoning": "Valid test reasoning",
        }
        validated = validate_structured_rag_output(valid_data)
        assert isinstance(validated, StructuredRAGOutput)
        assert validated.decision == RAGDecision.SUFFICIENT
        assert validated.retrieval_sufficient is True

        # Malformed input fallback
        fallback = validate_structured_rag_output("invalid string")
        assert fallback.decision == RAGDecision.EXTERNAL_RESEARCH_NEEDED
        assert fallback.retrieval_sufficient is False
