"""Automated Test Suite for Project 4 Phase 1 Document Processing Pipeline.

Tests:
- Test 1 — Document Loading (TXT & PDF)
- Test 2 — Chunking (Recursive Splitter & Metadata)
- Test 3 — Chroma DB (Embedding & Storage)
- Test 4 — Retrieval (Semantic Search via Retriever)
- Test 5 — LLM Extraction (Structured JSON Extraction)
- Test 6 — Pydantic Validation (Valid Data Acceptance)
- Test 7 — Invalid Data (Pydantic Rejection & Error Reporting)
- Test 8 — Re-process (Retry Loop & Max Retry Limit)
"""
import pytest
from pathlib import Path
from src.document_loader import DocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import VectorStoreManager
from src.extractor import LLMExtractor
from src.validator import DataValidator
from src.schema import InvoiceSchema, ValidationResult, InvoiceItem
from src.pipeline import process_document_pipeline
from src.config import SAMPLE_DOCS_DIR


@pytest.fixture
def sample_txt_path():
    return SAMPLE_DOCS_DIR / "sample_invoice.txt"


@pytest.fixture
def sample_pdf_path():
    return SAMPLE_DOCS_DIR / "sample_invoice.pdf"


# =====================================================================
# Test 1 — Document Loading
# =====================================================================
def test_1_document_loading_txt(sample_txt_path):
    """Verify TXT document loads successfully with metadata."""
    docs = DocumentLoader.load(sample_txt_path)
    assert len(docs) >= 1
    assert "ACME CORPORATION INVOICE" in docs[0].page_content
    assert docs[0].metadata["filename"] == "sample_invoice.txt"
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["page"] == 1


def test_1_document_loading_pdf(sample_pdf_path):
    """Verify PDF document loads successfully with metadata."""
    docs = DocumentLoader.load(sample_pdf_path)
    assert len(docs) >= 1
    assert "INVOICE" in docs[0].page_content
    assert docs[0].metadata["filename"] == "sample_invoice.pdf"
    assert docs[0].metadata["file_type"] == "pdf"


def test_1_document_loading_nonexistent():
    """Verify loading nonexistent document raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        DocumentLoader.load("non_existent_file.txt")


def test_1_document_loading_unsupported(tmp_path):
    """Verify loading unsupported format raises ValueError."""
    fake_csv = tmp_path / "test.csv"
    fake_csv.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported document format"):
        DocumentLoader.load(fake_csv)


# =====================================================================
# Test 2 — Chunking
# =====================================================================
def test_2_chunking(sample_txt_path):
    """Verify document is split into meaningful chunks with preserved metadata."""
    docs = DocumentLoader.load(sample_txt_path)
    splitter = DocumentSplitter(chunk_size=200, chunk_overlap=30)
    chunks = splitter.split_documents(docs)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) > 0
        assert "chunk_index" in chunk.metadata
        assert "filename" in chunk.metadata


# =====================================================================
# Test 3 — Chroma DB Storage
# =====================================================================
def test_3_chroma_storage(sample_txt_path, tmp_path):
    """Verify chunks are stored and accessible in Chroma DB."""
    docs = DocumentLoader.load(sample_txt_path)
    splitter = DocumentSplitter()
    chunks = splitter.split_documents(docs)

    vec_mgr = VectorStoreManager(persist_dir=tmp_path / "test_chroma")
    vectorstore = vec_mgr.create_vectorstore(chunks, collection_name="test_storage_col")

    assert vectorstore is not None
    # Check that collection has documents stored
    collection = vectorstore._collection
    assert collection.count() == len(chunks)


# =====================================================================
# Test 4 — Retrieval
# =====================================================================
def test_4_retrieval(sample_txt_path, tmp_path):
    """Verify Chroma retriever retrieves relevant document content."""
    docs = DocumentLoader.load(sample_txt_path)
    splitter = DocumentSplitter()
    chunks = splitter.split_documents(docs)

    vec_mgr = VectorStoreManager(persist_dir=tmp_path / "test_retrieval")
    vectorstore = vec_mgr.create_vectorstore(chunks, collection_name="test_retrieval_col")

    retrieved = vec_mgr.retrieve_relevant_context(
        vectorstore,
        query="Cloud Server Hosting monthly tier",
        k=2
    )

    assert len(retrieved) > 0
    joined_text = " ".join(d.page_content for d in retrieved)
    assert "Cloud Server Hosting" in joined_text or "ACME" in joined_text


# =====================================================================
# Test 5 — LLM Extraction
# =====================================================================
def test_5_llm_extraction():
    """Verify the LLM produces structured extraction dictionary from context."""
    extractor = LLMExtractor()
    context = """
    Document Type: Invoice
    Invoice Number: INV-2024-TEST
    Date: 2024-05-10
    Customer Name: Testing Solutions LLC
    Items:
    1. Premium Consulting - Qty: 2, Unit Price: $100.00, Total: $200.00
    Total Amount Due: $200.00
    """
    raw_output = extractor.extract(context)
    assert isinstance(raw_output, dict)
    assert "invoice_number" in raw_output
    assert "customer_name" in raw_output
    assert "items" in raw_output
    assert "total_amount" in raw_output
    assert len(raw_output["items"]) >= 1


# =====================================================================
# Test 6 — Pydantic Validation (Valid Data)
# =====================================================================
def test_6_pydantic_validation_valid():
    """Verify valid extracted data passes Pydantic validation."""
    valid_data = {
        "document_type": "Invoice",
        "invoice_number": "INV-1001",
        "date": "2024-01-15",
        "customer_name": "Acme Corp",
        "items": [
            {
                "description": "Web Development",
                "quantity": 1,
                "unit_price": 500.0,
                "total": 500.0
            }
        ],
        "total_amount": 500.0
    }

    result = DataValidator.validate(valid_data)
    assert result.is_valid is True
    assert isinstance(result.data, InvoiceSchema)
    assert result.data.invoice_number == "INV-1001"
    assert result.data.total_amount == 500.0
    assert len(result.errors) == 0


# =====================================================================
# Test 7 — Invalid Data (Pydantic Rejection)
# =====================================================================
def test_7_pydantic_validation_invalid():
    """Verify intentionally invalid data is rejected by Pydantic."""
    # Missing required invoice_number, empty items, and negative total_amount
    invalid_data = {
        "document_type": "Invoice",
        "customer_name": "Test Customer",
        "items": [],
        "total_amount": -50.0
    }

    result = DataValidator.validate(invalid_data)
    assert result.is_valid is False
    assert result.data is None
    assert len(result.errors) > 0


# =====================================================================
# Test 8 — Re-process Flow
# =====================================================================
def test_8_reprocess_retry_limit(sample_txt_path, monkeypatch):
    """Verify invalid extraction enters the re-processing path and respects max retry limit."""
    call_count = 0

    def mock_invalid_extract(self, retrieved_docs, feedback=None):
        nonlocal call_count
        call_count += 1
        # Always return invalid data to trigger re-process attempts
        return {
            "document_type": "Invoice",
            "invoice_number": "",  # invalid empty string
            "customer_name": "",
            "items": [],
            "total_amount": -10.0
        }

    monkeypatch.setattr(LLMExtractor, "extract", mock_invalid_extract)

    max_retries = 2
    res = process_document_pipeline(sample_txt_path, max_retries=max_retries, collection_name="test_reprocess_col")

    # Initial attempt (1) + max_retries (2) = 3 total attempts
    assert call_count == max_retries + 1
    assert res.status == "FAILED"
    assert res.attempts == max_retries
    assert res.reprocessed is True
    assert len(res.errors) > 0


# =====================================================================
# End-to-End Pipeline Integration Test
# =====================================================================
def test_end_to_end_pipeline_success(sample_txt_path):
    """Verify end-to-end processing of a sample document from loading to structured output."""
    result = process_document_pipeline(sample_txt_path, collection_name="e2e_txt_test")
    assert result.status == "SUCCESS"
    assert result.extracted_data is not None
    assert result.extracted_data.document_type.lower() == "invoice"
    assert "INV-2024-8842" in result.extracted_data.invoice_number
    assert result.extracted_data.total_amount == 900.0
    assert len(result.extracted_data.items) >= 2
