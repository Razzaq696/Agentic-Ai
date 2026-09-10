"""Retriever module for querying the Chroma private knowledge base."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.vector_store import get_vector_store, build_knowledge_base, get_default_persist_directory


class PolicyRetriever:
    """Retriever wrapper for semantic search over private policy documents in Chroma."""

    def __init__(
        self,
        persist_directory: Optional[str | Path] = None,
        collection_name: str = "university_knowledge_base",
        k: int = 3
    ):
        self.persist_directory = Path(persist_directory) if persist_directory else get_default_persist_directory()
        self.collection_name = collection_name
        self.k = k
        self._vector_store: Optional[Chroma] = None

    @property
    def vector_store(self) -> Chroma:
        """Lazy load or initialize the vector store."""
        if self._vector_store is None:
            # If database does not exist or is empty, index automatically
            state_file = self.persist_directory / "embedding_vocab.json"
            if not self.persist_directory.exists() or not state_file.exists():
                print("[Retriever] Chroma store not found. Building knowledge base...")
                self._vector_store = build_knowledge_base(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
            else:
                self._vector_store = get_vector_store(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name
                )
        return self._vector_store

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context chunks for a given query.

        Args:
            query: The user query string.
            k: Optional number of documents to retrieve (defaults to self.k).

        Returns:
            List of dictionaries containing content, source metadata, and similarity score.
        """
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        top_k = k if k is not None else self.k
        try:
            # similarity_search_with_score returns List[Tuple[Document, float]]
            results_with_scores = self.vector_store.similarity_search_with_score(
                clean_query,
                k=top_k
            )

            retrieved_items: List[Dict[str, Any]] = []
            for doc, score in results_with_scores:
                item = {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "title": doc.metadata.get("title", ""),
                    "chunk_id": doc.metadata.get("chunk_id", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                    "score": round(float(score), 4),
                }
                retrieved_items.append(item)

            return retrieved_items
        except Exception as e:
            print(f"[Error] Retrieval failed: {e}")
            return []

    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Retrieve standard LangChain Document objects."""
        clean_query = query.strip() if query else ""
        if not clean_query:
            return []

        top_k = k if k is not None else self.k
        return self.vector_store.similarity_search(clean_query, k=top_k)


def get_retriever(k: int = 3) -> PolicyRetriever:
    """Convenience helper to get a default configured PolicyRetriever."""
    return PolicyRetriever(k=k)


def retrieve_context(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Functional interface to retrieve context for a query."""
    retriever = get_retriever(k=k)
    return retriever.retrieve(query=query, k=k)


def cli_main():
    """Command-line interface for testing query retrieval."""
    if len(sys.argv) < 2:
        query = "What is the attendance policy?"
    else:
        query = " ".join(sys.argv[1:])

    print("=" * 60)
    print(f"Query:\n{query}\n")
    print("Retrieved Context:")
    print("=" * 60)

    results = retrieve_context(query, k=3)
    if not results:
        print("No matching documents found.")
        return

    for i, res in enumerate(results, 1):
        print(f"\n{i}. Source: {res['source']} (Score: {res['score']})")
        print(f"   Title: {res['title']}")
        print(f"   [Content]:\n{res['content']}")
        print("-" * 60)


if __name__ == "__main__":
    cli_main()
