"""
LLM Connection Module for Intelligent Task Execution Agent.
Phase 1-5: LangChain + Ollama LLM Integration with Multi-Core Optimization

Connects to the local Ollama instance using LangChain's ChatOllama wrapper.
Configures optimal thread count and context length for high CPU inference performance.
"""

import os
from typing import Optional
from langchain_ollama import ChatOllama
from config import OLLAMA_BASE_URL, MODEL_NAME, TEMPERATURE, REQUEST_TIMEOUT


def get_llm(
    model_name: Optional[str] = None,
    temperature: Optional[float] = None,
    base_url: Optional[str] = None,
    request_timeout: Optional[int] = None,
) -> ChatOllama:
    """
    Initializes and returns a ChatOllama instance configured with centralized settings
    and multi-threaded CPU acceleration.

    Args:
        model_name: Optional model identifier override (e.g., 'qwen3:4b' or 'llama3.2:3b').
        temperature: Optional sampling temperature override (defaults to config.TEMPERATURE).
        base_url: Optional Ollama API endpoint override (defaults to config.OLLAMA_BASE_URL).
        request_timeout: Optional timeout in seconds (defaults to config.REQUEST_TIMEOUT).

    Returns:
        ChatOllama: Configured LangChain Chat Model instance.
    """
    selected_model = model_name or MODEL_NAME
    selected_temp = TEMPERATURE if temperature is None else temperature
    selected_url = base_url or OLLAMA_BASE_URL
    selected_timeout = request_timeout or REQUEST_TIMEOUT

    cpu_cores = os.cpu_count() or 8

    return ChatOllama(
        model=selected_model,
        temperature=selected_temp,
        base_url=selected_url,
        timeout=selected_timeout,
        num_ctx=2048,
        num_thread=cpu_cores,
    )
