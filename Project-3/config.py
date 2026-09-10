import os
from typing import Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

# Centralized LLM Configuration
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

def get_llm(
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
) -> ChatOllama:
    """
    Returns a configured ChatOllama instance.
    Centralizes all LLM initialization across the multi-agent system.
    """
    return ChatOllama(
        model=model or DEFAULT_MODEL,
        base_url=base_url or DEFAULT_BASE_URL,
        temperature=DEFAULT_TEMPERATURE if temperature is None else temperature,
    )
