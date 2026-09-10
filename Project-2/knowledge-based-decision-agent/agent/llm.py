"""LLM interface module for Ollama integration."""

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
    num_predict: Optional[int] = 180
) -> ChatOllama:
    """Get configured local LLM instance via Ollama.

    Args:
        model_name: Optional model override (defaults to OLLAMA_MODEL env or llama3.2:3b/qwen3:4b).
        temperature: Sampling temperature (default 0.0 for deterministic analysis).
        base_url: Optional base URL for Ollama service.
        num_predict: Maximum token generation limit.

    Returns:
        ChatOllama instance.
    """
    model = model_name or os.getenv("OLLAMA_MODEL", "llama3.2:3b")
    host = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    kwargs = {
        "model": model,
        "base_url": host,
        "temperature": temperature,
    }
    if num_predict is not None:
        kwargs["num_predict"] = num_predict

    return ChatOllama(**kwargs)


if __name__ == "__main__":
    llm = get_llm()
    print(f"Initialized LLM with model: {llm.model}")
    try:
        res = llm.invoke("Respond with one word: ready.")
        print(f"Test response: {res.content.strip()}")
    except Exception as e:
        print(f"LLM invocation failed: {e}")
