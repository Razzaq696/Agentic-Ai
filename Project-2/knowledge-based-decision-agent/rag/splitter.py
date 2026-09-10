"""Text splitting and chunking module for RAG pipeline."""

from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Document],
    chunk_size: int = 600,
    chunk_overlap: int = 80
) -> List[Document]:
    """Split a list of LangChain documents into clean semantic chunks while preserving metadata.

    Args:
        documents: List of input Document objects.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        List of chunked Document objects with source metadata and chunk IDs.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks: List[Document] = []
    for doc in documents:
        doc_chunks = text_splitter.split_documents([doc])
        for idx, chunk in enumerate(doc_chunks):
            clean_content = chunk.page_content.strip()
            if not clean_content:
                continue

            chunk.page_content = clean_content
            chunk.metadata = {
                **doc.metadata,
                "chunk_index": idx,
                "chunk_id": f"{doc.metadata.get('source', 'unknown')}_chunk_{idx}",
            }
            chunks.append(chunk)

    return chunks


if __name__ == "__main__":
    from rag.loader import load_documents
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} clean chunks.")
    if chunks:
        print("\nSample Chunk:")
        print(f"Source: {chunks[0].metadata['source']} (Chunk {chunks[0].metadata['chunk_index']})")
        print(f"Content:\n{chunks[0].page_content}")
