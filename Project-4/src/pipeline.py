"""End-to-end backend pipeline coordinating Document Processing, Chroma RAG, LLM Extraction, and Validation."""
import logging
from pathlib import Path
from typing import Optional

from src.config import MAX_RETRIES, RETRIEVER_K
from src.document_loader import DocumentLoader
from src.text_splitter import DocumentSplitter
from src.vector_store import VectorStoreManager
from src.extractor import LLMExtractor
from src.validator import DataValidator
from src.schema import PipelineResult, ValidationResult

logger = logging.getLogger(__name__)


def process_document_pipeline(
    file_path: str | Path,
    max_retries: int = MAX_RETRIES,
    collection_name: Optional[str] = None
) -> PipelineResult:
    """
    Execute the complete Phase 1 automated document processing pipeline.

    Flow:
        Document -> DocumentLoader -> DocumentSplitter -> Chroma DB VectorStore
        -> Retriever -> LLM Extraction Agent -> Pydantic Validation
        -> [VALID: Output | INVALID: Re-process loop up to max_retries]

    Args:
        file_path: Path to the local document (.txt or .pdf).
        max_retries: Maximum number of re-processing retry attempts on validation failure.
        collection_name: Optional custom Chroma collection name (useful for isolated tests).

    Returns:
        PipelineResult: Standardized pipeline execution result.
    """
    file_str = str(file_path)
    file_name = Path(file_path).name

    try:
        # Step 1: Document Processing (Load document)
        docs = DocumentLoader.load(file_str)

        # Step 2: Chunk / Process (Split text)
        splitter = DocumentSplitter()
        chunks = splitter.split_documents(docs)

        # Step 3: Chroma DB Indexing
        vec_mgr = VectorStoreManager()
        suffix_clean = Path(file_path).suffix.replace(".", "")
        stem_clean = Path(file_path).stem.replace("-", "_").replace(" ", "_")
        col_name = collection_name or f"doc_{stem_clean}_{suffix_clean}"
        vectorstore = vec_mgr.create_vectorstore(chunks, collection_name=col_name)

        # Step 4: Retrieval via Chroma Retriever
        retrieved_docs = vec_mgr.retrieve_relevant_context(
            vectorstore,
            query="Extract document type, invoice number, date, customer name, line items with description, quantity, price, total, and grand total amount.",
            k=RETRIEVER_K
        )
        retrieved_texts = [d.page_content for d in retrieved_docs]

        # Step 5 & 6: LLM Extraction and Pydantic Validation with Re-processing Loop
        extractor = LLMExtractor()
        validator = DataValidator()

        feedback: Optional[str] = None
        validation_result: Optional[ValidationResult] = None
        attempt_count = 0

        # Initial attempt + retries
        while attempt_count <= max_retries:
            attempt_count += 1
            try:
                # LLM Extraction
                raw_extracted = extractor.extract(retrieved_docs=retrieved_docs, feedback=feedback)

                # Pydantic Schema Validation
                validation_result = validator.validate(raw_extracted)

                if validation_result.is_valid:
                    # VALID -> OUTPUT
                    return PipelineResult(
                        status="SUCCESS",
                        document_source=file_name,
                        extracted_data=validation_result.data,
                        attempts=attempt_count,
                        reprocessed=(attempt_count > 1),
                        errors=[],
                        retrieved_chunks=retrieved_texts
                    )
                else:
                    # INVALID -> Prepare feedback for re-process
                    error_feedback = "; ".join(validation_result.errors)
                    feedback = error_feedback

            except Exception as extract_err:
                feedback = f"Extraction error: {str(extract_err)}"
                validation_result = ValidationResult(
                    is_valid=False,
                    data=None,
                    errors=[str(extract_err)]
                )

        # If loop exhausts without success -> Validation Failure
        final_errors = validation_result.errors if validation_result else ["Extraction failed"]
        return PipelineResult(
            status="FAILED",
            document_source=file_name,
            extracted_data=None,
            attempts=attempt_count - 1,
            reprocessed=True,
            errors=final_errors,
            retrieved_chunks=retrieved_texts
        )

    except Exception as e:
        # Document loading / system level error
        return PipelineResult(
            status="FAILED",
            document_source=file_name,
            extracted_data=None,
            attempts=0,
            reprocessed=False,
            errors=[str(e)],
            retrieved_chunks=[]
        )
