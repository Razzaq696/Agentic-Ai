"""RAG Knowledge Base & Retriever module using ChromaDB."""

import os
import json
from typing import List, Optional
import chromadb

# Knowledge file path
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "knowledge_docs.json")

# Default in-memory Chroma Client & Collection
_chroma_client: Optional[chromadb.ClientAPI] = None
_collection = None


def get_or_create_knowledge_collection(collection_name: str = "project6_knowledge"):
    """Initializes ChromaDB and indexes the local knowledge documents."""
    global _chroma_client, _collection

    if _collection is not None:
        return _collection

    _chroma_client = chromadb.Client()

    # Try to delete if exists to ensure fresh load
    try:
        _chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    _collection = _chroma_client.create_collection(name=collection_name)

    # Load local documents
    docs = []
    ids = []
    metadatas = []

    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, "r", encoding="utf-8") as f:
                doc_list = json.load(f)
                for item in doc_list:
                    docs.append(f"{item.get('title', '')}: {item.get('content', '')}")
                    ids.append(str(item.get("id", len(ids) + 1)))
                    metadatas.append({
                        "title": item.get("title", ""),
                        "category": item.get("category", "general"),
                    })
        except Exception:
            pass

    # Fallback default knowledge if file reading fails
    if not docs:
        docs = [
            "Telegram Agentic AI Assistant Architecture: Built using LangGraph stateful workflow with query analysis and dynamic routing.",
            "Project 6 System Features: Supports arithmetic calculations, RAG knowledge retrieval, and LLM reasoning.",
            "Supported Bot Commands: /start, /help, /status, and automated multi-branch query processing.",
            "Data Governance Policy: Strict data privacy with no API keys or raw internal traces exposed.",
        ]
        ids = ["doc_1", "doc_2", "doc_3", "doc_4"]
        metadatas = [{"category": "architecture"}, {"category": "features"}, {"category": "commands"}, {"category": "policy"}]

    _collection.add(
        documents=docs,
        ids=ids,
        metadatas=metadatas,
    )

    return _collection


def retrieve_knowledge(query: str, top_k: int = 2) -> List[str]:
    """Retrieves the most relevant knowledge documents for a given query.

    Args:
        query: The user query string.
        top_k: Number of relevant chunks to retrieve.

    Returns:
        List of relevant text snippets.
    """
    if not query or not query.strip():
        return []

    try:
        collection = get_or_create_knowledge_collection()
        results = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
        )

        documents = results.get("documents", [[]])
        if documents and len(documents) > 0:
            return documents[0]
        return []
    except Exception:
        return []
