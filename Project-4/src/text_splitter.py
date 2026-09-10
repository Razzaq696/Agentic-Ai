"""Text splitter module for chunking documents while preserving metadata."""
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentSplitter:
    """Chunking component for splitting documents into meaningful chunks."""

    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into chunked documents.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List[Document]: List of chunked Document objects with metadata preserved.
        """
        if not documents:
            return []

        chunks = self.splitter.split_documents(documents)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            # Ensure essential metadata keys exist
            if "source" not in chunk.metadata and "filename" in chunk.metadata:
                chunk.metadata["source"] = chunk.metadata["filename"]

        return chunks
