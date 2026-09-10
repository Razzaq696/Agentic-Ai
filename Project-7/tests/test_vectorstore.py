"""Tests for Chroma vector store manager and embeddings integration."""

import pytest
from langchain_core.documents import Document
from src.llm.factory import DeterministicMockEmbeddings
from src.rag.vectorstore import ChromaVectorStoreManager


class TestChromaVectorStoreManager:
    """Unit tests for Chroma lifecycle and vector search."""

    def test_ephemeral_initialization(self, mock_embeddings):
        """Verify ephemeral Chroma manager starts with 0 items."""
        manager = ChromaVectorStoreManager(
            embeddings=mock_embeddings,
            collection_name="test_init_col",
            is_ephemeral=True,
        )
        assert manager.is_ephemeral is True
        assert manager.get_collection_count() == 0

    def test_indexing_documents(self, mock_embeddings):
        """Verify documents can be indexed and count increases."""
        manager = ChromaVectorStoreManager(
            embeddings=mock_embeddings,
            collection_name="test_index_col",
            is_ephemeral=True,
        )
        docs = [
            Document(page_content="Keychron mechanical keyboard tactile", metadata={"category": "Mechanical Keyboard"}),
            Document(page_content="Sony noise canceling headphones", metadata={"category": "Headphones"}),
        ]
        indexed = manager.initialize_with_documents(docs)
        assert indexed == 2
        assert manager.get_collection_count() == 2

    def test_similarity_search(self, ephemeral_vectorstore):
        """Verify similarity search returns relevant documents."""
        results = ephemeral_vectorstore.similarity_search("mechanical keyboard tactile typing", k=2)
        assert len(results) > 0
        top_content = results[0].page_content.lower()
        assert "keyboard" in top_content or "keychron" in top_content

    def test_similarity_search_with_metadata_filter(self, ephemeral_vectorstore):
        """Verify metadata filtering restricts results to specific category."""
        results = ephemeral_vectorstore.similarity_search(
            query="portable device for coding",
            k=3,
            filter_dict={"category": "Laptop"},
        )
        assert len(results) > 0
        for doc in results:
            assert doc.metadata.get("category") == "Laptop"

    def test_similarity_search_with_score(self, ephemeral_vectorstore):
        """Verify similarity_search_with_score returns documents and numeric scores."""
        results_with_scores = ephemeral_vectorstore.similarity_search_with_score("laptop 16gb", k=2)
        assert len(results_with_scores) > 0
        doc, score = results_with_scores[0]
        assert isinstance(doc, Document)
        assert isinstance(score, (float, int))

    def test_empty_query_returns_empty_list(self, ephemeral_vectorstore):
        """Verify empty or whitespace query safely returns empty list."""
        assert ephemeral_vectorstore.similarity_search("") == []
        assert ephemeral_vectorstore.similarity_search("   ") == []
        assert ephemeral_vectorstore.similarity_search_with_score("") == []

    def test_deterministic_mock_embeddings(self):
        """Verify DeterministicMockEmbeddings returns non-zero, consistent vectors."""
        emb = DeterministicMockEmbeddings(dimension=64)
        v1 = emb.embed_query("apple macbook air")
        v2 = emb.embed_query("apple macbook air")
        assert len(v1) == 64
        assert v1 == v2  # Strictly deterministic

        # Non-empty strings should produce non-zero vector
        assert any(x != 0.0 for x in v1)
