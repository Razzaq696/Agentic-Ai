"""Vector store and indexing pipeline using Chroma."""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.loader import load_documents
from rag.splitter import split_documents
from rag.embeddings import get_embedding_model


def get_default_persist_directory() -> Path:
    """Return default path to the local Chroma persistence directory."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "chroma_db"


def get_vector_store(
    persist_directory: Optional[str | Path] = None,
    collection_name: str = "university_knowledge_base"
) -> Chroma:
    """Load an existing Chroma vector store or initialize an empty instance.

    Args:
        persist_directory: Path to Chroma storage folder.
        collection_name: Name of the vector store collection.

    Returns:
        Chroma vector store instance.
    """
    persist_dir = Path(persist_directory) if persist_directory else get_default_persist_directory()
    embedding_model = get_embedding_model(persist_directory=persist_dir)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=str(persist_dir)
    )
    return vector_store


def build_knowledge_base(
    data_directory: Optional[str | Path] = None,
    persist_directory: Optional[str | Path] = None,
    collection_name: str = "university_knowledge_base",
    force_rebuild: bool = True
) -> Chroma:
    """Run the complete indexing pipeline from local text files to Chroma.

    Pipeline:
        Local Documents -> Document Loader -> Text Splitter -> Embeddings -> Chroma

    Args:
        data_directory: Path to source data directory.
        persist_directory: Path to Chroma database directory.
        collection_name: Name of Chroma collection.
        force_rebuild: If True, clears existing collection to prevent duplicate chunks.

    Returns:
        Populated Chroma vector store instance.
    """
    persist_dir = Path(persist_directory) if persist_directory else get_default_persist_directory()
    
    # 1. Load documents
    raw_docs = load_documents(data_dir=data_directory)
    if not raw_docs:
        print("[Warning] No documents found to index.")
        return get_vector_store(persist_directory=persist_dir, collection_name=collection_name)

    # 2. Split documents into chunks
    chunks = split_documents(raw_docs, chunk_size=400, chunk_overlap=50)
    print(f"[Indexing] Loaded {len(raw_docs)} documents -> {len(chunks)} chunks.")

    # 3. Fit embedding vocabulary across chunks
    embedding_model = get_embedding_model(persist_directory=persist_dir)
    chunk_texts = [c.page_content for c in chunks]
    embedding_model.fit(chunk_texts)

    # 4. Safe rebuild to avoid duplicate chunks
    if force_rebuild and persist_dir.exists():
        # Clear collection cleanly or reinitialize Chroma
        try:
            temp_store = Chroma(
                collection_name=collection_name,
                embedding_function=embedding_model,
                persist_directory=str(persist_dir)
            )
            temp_store.delete_collection()
        except Exception as e:
            # If collection did not exist, continue
            pass

    # 5. Populate Chroma with chunk IDs
    chunk_ids = [c.metadata.get("chunk_id", f"chunk_{i}") for i, c in enumerate(chunks)]
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        ids=chunk_ids,
        collection_name=collection_name,
        persist_directory=str(persist_dir)
    )

    print(f"[Indexing] Successfully indexed {len(chunks)} chunks into Chroma collection '{collection_name}'.")
    return vector_store


if __name__ == "__main__":
    db = build_knowledge_base()
    print("Knowledge base indexing complete!")
