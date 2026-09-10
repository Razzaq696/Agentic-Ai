"""CLI Runner for Project 4 Phase 1 Document Processing Pipeline."""
import sys
import json
from pathlib import Path
from src.pipeline import process_document_pipeline
from src.config import SAMPLE_DOCS_DIR


def main():
    if len(sys.argv) > 1:
        doc_path = Path(sys.argv[1])
    else:
        doc_path = SAMPLE_DOCS_DIR / "sample_invoice.txt"

    print(f"\n=======================================================")
    print(f" PROJECT 4 - PHASE 1: DOCUMENT PROCESSING PIPELINE")
    print(f" Processing Target: {doc_path.name}")
    print(f" Full Path: {doc_path.resolve()}")
    print(f"=======================================================\n")

    if not doc_path.exists():
        print(f"[ERROR] Document not found: {doc_path}")
        sys.exit(1)

    print("[1/5] Loading Document and Splitting Chunks...")
    print("[2/5] Creating Chroma Vector Embeddings & Indexing...")
    print("[3/5] Performing RAG Context Retrieval...")
    print("[4/5] Extracting Data with LLM Extraction Agent...")
    print("[5/5] Validating Output with Pydantic Schema...\n")

    result = process_document_pipeline(doc_path)

    print("-------------------------------------------------------")
    print(" PIPELINE EXECUTION RESULT")
    print("-------------------------------------------------------")
    print(f" Status:        {result.status}")
    print(f" Source:        {result.document_source}")
    print(f" Total Attempts:{result.attempts}")
    print(f" Reprocessed:   {result.reprocessed}")

    if result.status == "SUCCESS" and result.extracted_data:
        print("\n[VALID STRUCTURED OUTPUT]")
        print(json.dumps(result.extracted_data.model_dump(), indent=2))
    else:
        print("\n[ERRORS / VALIDATION FAILURES]")
        for err in result.errors:
            print(f" - {err}")

    print("\n-------------------------------------------------------")
    print(" RETRIEVED CONTEXT CHUNKS")
    print("-------------------------------------------------------")
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk.strip())
    print("\n=======================================================\n")


if __name__ == "__main__":
    main()
