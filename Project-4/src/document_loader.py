"""Document loader module supporting TXT and PDF formats with metadata preservation."""
import os
from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader


class DocumentLoader:
    """Document Agent component responsible for loading documents."""

    SUPPORTED_EXTENSIONS = {".txt", ".pdf"}

    @classmethod
    def load(cls, file_path: str | Path) -> List[Document]:
        """
        Load a document from a given file path.

        Args:
            file_path: Path to the local document (.txt or .pdf).

        Returns:
            List[Document]: Extracted LangChain Document objects with preserved metadata.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If file type is unsupported or document is empty.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"Document file not found at: {path.resolve()}")

        ext = path.suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported document format '{ext}'. Supported formats: {cls.SUPPORTED_EXTENSIONS}"
            )

        if os.path.getsize(path) == 0:
            raise ValueError(f"Document at '{path.name}' is empty (0 bytes).")

        documents: List[Document] = []

        if ext == ".txt":
            loader = TextLoader(str(path.resolve()), encoding="utf-8")
            documents = loader.load()
            for doc in documents:
                doc.metadata["filename"] = path.name
                doc.metadata["file_type"] = "txt"
                doc.metadata["page"] = 1
        elif ext == ".pdf":
            loader = PyPDFLoader(str(path.resolve()))
            documents = loader.load()
            for doc in documents:
                doc.metadata["filename"] = path.name
                doc.metadata["file_type"] = "pdf"
                # PyPDFLoader usually populates 'page' in metadata (0-indexed)
                if "page" in doc.metadata:
                    doc.metadata["page"] = doc.metadata["page"] + 1
                else:
                    doc.metadata["page"] = 1

        # Validate extracted content
        total_text = "".join(doc.page_content.strip() for doc in documents)
        if not total_text:
            raise ValueError(f"Extracted document content from '{path.name}' is empty.")

        return documents
