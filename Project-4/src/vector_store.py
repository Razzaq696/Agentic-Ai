"""Chroma DB and Retriever vector store module."""
from typing import List, Optional
from pathlib import Path
import chromadb.utils.embedding_functions as ef
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from src.config import CHROMA_PERSIST_DIR, RETRIEVER_K


class ChromaDefaultEmbeddings(Embeddings):
    """Local ONNX-based embedding function wrapper matching LangChain Embeddings interface."""

    def __init__(self):
        self._ef = ef.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0]


class VectorStoreManager:
    """Manages Chroma vector database storage, indexing, and retrieval."""

    def __init__(self, persist_dir: Optional[str | Path] = None):
        self.persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
        self.embeddings = ChromaDefaultEmbeddings()

    def create_vectorstore(
        self,
        chunks: List[Document],
        collection_name: str = "document_chunks"
    ) -> Chroma:
        """
        Store document chunks and embeddings in local Chroma DB.

        Args:
            chunks: List of chunked Document objects.
            collection_name: Name of the Chroma collection.

        Returns:
            Chroma: The instantiated vector store.
        """
        if not chunks:
            raise ValueError("Cannot create vector store from empty document chunks.")

        import chromadb
        client = chromadb.PersistentClient(path=self.persist_dir)
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=collection_name,
            client=client
        )
        return vectorstore

    def get_retriever(
        self,
        vectorstore: Chroma,
        k: int = RETRIEVER_K
    ):
        """
        Create a retriever from the Chroma vector store.

        Args:
            vectorstore: Instantiated Chroma vector store.
            k: Number of relevant chunks to retrieve.

        Returns:
            VectorStoreRetriever: LangChain retriever instance.
        """
        return vectorstore.as_retriever(search_kwargs={"k": k})

    def retrieve_relevant_context(
        self,
        vectorstore: Chroma,
        query: str = "Extract all document and invoice details, line items, amounts, and dates",
        k: int = RETRIEVER_K
    ) -> List[Document]:
        """
        Retrieve relevant document chunks using the retriever.

        Args:
            vectorstore: Chroma vector store.
            query: Semantic search query.
            k: Number of chunks.

        Returns:
            List[Document]: Relevant chunks.
        """
        retriever = self.get_retriever(vectorstore, k=k)
        return retriever.invoke(query)
