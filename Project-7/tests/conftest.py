"""Pytest fixtures and configuration."""

import pytest
from src.agents.requirement_agent import RequirementAnalysisAgent
from src.llm.factory import MockRequirementLLM
from src.models.schemas import BudgetInfo, RequirementAnalysisOutput
from src.workflow.graph import build_shopping_graph


@pytest.fixture
def mock_llm():
    """Default mock LLM with heuristic rule parsing."""
    return MockRequirementLLM()


@pytest.fixture
def failing_llm():
    """Mock LLM that simulates an API error or outage."""
    return MockRequirementLLM(should_fail=True)


@pytest.fixture
def requirement_agent(mock_llm):
    """Requirement agent initialized with default mock LLM."""
    return RequirementAnalysisAgent(llm=mock_llm)


@pytest.fixture
def default_workflow(mock_llm):
    """Workflow compiled with the deterministic mock agent."""
    agent = RequirementAnalysisAgent(llm=mock_llm)
    return build_shopping_graph(agent=agent)


@pytest.fixture
def sample_structured_output():
    """Sample valid RequirementAnalysisOutput."""
    return RequirementAnalysisOutput(
        product_category="Laptop",
        budget=BudgetInfo(max_amount=1200.0, is_specified=True, is_flexible=False),
        required_features=["16GB RAM", "14-inch display"],
        preferences=["Lightweight"],
        priorities=["Programming productivity", "Battery endurance"],
        intended_use="Coding and coursework",
        is_clear=True,
        ambiguities_or_missing_info=[],
        summary="A portable 14-inch coding laptop under $1200.",
    )


# =====================================================================
# PHASE 2: RAG & CHROMADB FIXTURES
# =====================================================================

from src.llm.factory import DeterministicMockEmbeddings
from src.rag.loader import load_product_dataset, prepare_documents
from src.rag.vectorstore import ChromaVectorStoreManager
from src.rag.retriever import ProductRetriever
from src.rag.evaluator import AgenticRAGEvaluator


@pytest.fixture
def mock_embeddings():
    """Lightweight deterministic embeddings for testing."""
    return DeterministicMockEmbeddings(dimension=64)


import uuid


@pytest.fixture
def ephemeral_vectorstore(mock_embeddings):
    """Isolated, ephemeral in-memory Chroma instance with sample products pre-indexed."""
    unique_col = f"test_ephemeral_{uuid.uuid4().hex[:8]}"
    manager = ChromaVectorStoreManager(
        embeddings=mock_embeddings,
        collection_name=unique_col,
        is_ephemeral=True,
    )
    products, _ = load_product_dataset("data/products.json")
    if products:
        docs = prepare_documents(products)
        manager.initialize_with_documents(docs)
    return manager


@pytest.fixture
def product_retriever(ephemeral_vectorstore):
    """ProductRetriever wired to the ephemeral vector store."""
    return ProductRetriever(vector_store_manager=ephemeral_vectorstore, top_k=3)


@pytest.fixture
def rag_evaluator():
    """Default AgenticRAGEvaluator instance."""
    return AgenticRAGEvaluator(max_retries=1)


@pytest.fixture
def rag_workflow(requirement_agent, product_retriever, rag_evaluator):
    """End-to-end LangGraph workflow configured with mock agent and ephemeral vector store."""
    return build_shopping_graph(
        agent=requirement_agent,
        retriever=product_retriever,
        evaluator=rag_evaluator,
    )
