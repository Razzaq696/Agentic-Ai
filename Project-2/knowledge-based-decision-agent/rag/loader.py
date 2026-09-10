"""Document loader module for the private knowledge base."""

import os
from pathlib import Path
from typing import List, Optional
from langchain_core.documents import Document


def get_data_directory() -> Path:
    """Return the absolute path to the data directory."""
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    return project_root / "data"


def load_documents(data_dir: Optional[str | Path] = None) -> List[Document]:
    """Load text documents from the local knowledge base directory.

    Args:
        data_dir: Optional custom path to data folder.

    Returns:
        List of LangChain Document objects with preserved metadata.
    """
    directory = Path(data_dir) if data_dir else get_data_directory()
    if not directory.exists() or not directory.is_dir():
        print(f"[Warning] Knowledge base directory does not exist: {directory}")
        return []

    documents: List[Document] = []
    text_files = sorted(list(directory.glob("*.txt")))

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                print(f"[Warning] Skipping empty document: {file_path.name}")
                continue

            # Extract title from first markdown heading or filename
            lines = content.splitlines()
            title = lines[0].lstrip("# ").strip() if lines and lines[0].startswith("#") else file_path.stem

            metadata = {
                "source": file_path.name,
                "title": title,
                "file_path": str(file_path),
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
        except Exception as e:
            print(f"[Error] Failed to read document {file_path.name}: {e}")

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents:")
    for doc in docs:
        print(f" - {doc.metadata['source']} ({len(doc.page_content)} chars)")
