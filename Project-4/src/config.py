"""Configuration module for Project 4 Phase 1 Document Processing Pipeline."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "sample_documents"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# LLM & Embedding Settings
OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_TEMPERATURE = 0.0

# Text Splitter Settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Retriever Settings
RETRIEVER_K = 3

# Pipeline Settings
MAX_RETRIES = 2
