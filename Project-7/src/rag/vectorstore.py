"""Chroma vector store manager supporting persistence, ephemeral testing, and similarity retrieval."""

import os
import shutil
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import settings
from src.llm.factory import get_embeddings
from src.utils.logger import logger


class ChromaVectorStoreManager:
    """Manages Chroma vector store lifecycle, collection indexing, and similarity queries."""

    def __init__(
        self,
        embeddings: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        is_ephemeral: bool = False,
    ):
        """Initialize Chroma manager.

        Args:
            embeddings: LangChain Embeddings instance. Defaults to get_embeddings().
            persist_directory: Directory for persistent database. Defaults to settings.chroma_persist_dir.
            collection_name: Chroma collection name. Defaults to settings.chroma_collection_name.
            is_ephemeral: If True, uses in-memory ephemeral client for fast, leak-free testing.
        """
        self.embeddings = embeddings or get_embeddings()
        self.collection_name = collection_name or settings.chroma_collection_name
        self.is_ephemeral = is_ephemeral
        self.persist_directory = None if is_ephemeral else (persist_directory or settings.chroma_persist_dir)

        self._vector_store: Optional[Chroma] = None
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        """Create or return Chromadb client."""
        if self._client is not None:
            return self._client

        if self.is_ephemeral:
            logger.debug("Initializing ephemeral in-memory Chroma client")
            self._client = chromadb.EphemeralClient()
        else:
            if self.persist_directory:
                os.makedirs(self.persist_directory, exist_ok=True)
            logger.debug(f"Initializing persistent Chroma client at '{self.persist_directory}'")
            self._client = chromadb.PersistentClient(path=self.persist_directory)

        return self._client

    def get_vector_store(self) -> Chroma:
        """Get or initialize the underlying LangChain Chroma instance."""
        if self._vector_store is None:
            client = self._get_client()
            self._vector_store = Chroma(
                client=client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )
        return self._vector_store

    def initialize_with_documents(self, documents: List[Document]) -> int:
        """Index a list of documents into the collection.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            Number of documents indexed.
        """
        if not documents:
            logger.warning("initialize_with_documents called with empty document list")
            return 0

        try:
            vector_store = self.get_vector_store()
            # In case collection exists, we add or re-index
            vector_store.add_documents(documents)
            count = len(documents)
            logger.info(f"Indexed {count} documents into Chroma collection '{self.collection_name}'")
            return count
        except Exception as e:
            logger.error(f"Error indexing documents into Chroma: {e}", exc_info=True)
            return 0

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform semantic similarity search on the Chroma collection.

        Args:
            query: Natural language search query.
            k: Maximum number of documents to return.
            filter_dict: Optional metadata filter dictionary (e.g. {'category': 'Laptop'}).

        Returns:
            List of matching Document objects.
        """
        if not query or not query.strip():
            logger.warning("Empty query passed to similarity_search")
            return []

        try:
            vector_store = self.get_vector_store()
            results = vector_store.similarity_search(
                query=query.strip(),
                k=k,
                filter=filter_dict,
            )
            logger.debug(f"similarity_search returned {len(results)} matches for query={query!r}")
            return results
        except Exception as e:
            logger.error(f"Chroma similarity search failed: {e}", exc_info=True)
            return []

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """Perform similarity search returning documents with their distance/similarity score.

        Args:
            query: Natural language search query.
            k: Top-k matches.
            filter_dict: Optional metadata filter.

        Returns:
            List of (Document, score) tuples.
        """
        if not query or not query.strip():
            return []

        try:
            vector_store = self.get_vector_store()
            results = vector_store.similarity_search_with_score(
                query=query.strip(),
                k=k,
                filter=filter_dict,
            )
            return results
        except Exception as e:
            logger.error(f"Chroma similarity_search_with_score failed: {e}", exc_info=True)
            return []

    def get_collection_count(self) -> int:
        """Return the total number of items in the Chroma collection."""
        try:
            client = self._get_client()
            collection = client.get_or_create_collection(self.collection_name)
            return collection.count()
        except Exception as e:
            logger.warning(f"Failed to get collection count: {e}")
            return 0

    def reset_collection(self) -> None:
        """Delete and recreate the active collection."""
        try:
            client = self._get_client()
            try:
                client.delete_collection(self.collection_name)
                logger.info(f"Deleted Chroma collection '{self.collection_name}'")
            except Exception:
                pass
            self._vector_store = None
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
