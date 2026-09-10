"""LangSmith observability setup and environment synchronization."""

import os
from typing import Any, Optional
from src.config import settings
from src.utils.logger import logger


def configure_langsmith(
    enabled: Optional[bool] = None,
    api_key: Optional[str] = None,
    project: Optional[str] = None,
    endpoint: Optional[str] = None,
    config: Optional[Any] = None,
) -> bool:
    """Configure LangSmith tracing in the environment if enabled.

    Args:
        enabled: Override tracing enabled flag.
        api_key: Override LangSmith API key.
        project: Override project name.
        endpoint: Override endpoint URL.
        config: Optional Settings instance.

    Returns:
        True if LangSmith tracing was configured and enabled, False otherwise.
    """
    cfg = config or settings
    is_enabled = enabled if enabled is not None else getattr(cfg, "langchain_tracing_v2", False)
    key = api_key if api_key is not None else getattr(cfg, "langchain_api_key", "")
    proj = project if project is not None else getattr(cfg, "langchain_project", "shopping-decision-agent")
    endp = endpoint if endpoint is not None else getattr(cfg, "langchain_endpoint", "https://api.smith.langchain.com")

    if not is_enabled:
        logger.debug("LangSmith observability: Tracing is disabled.")
        # Ensure tracing is not accidentally on if disabled
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    if not key or not key.strip():
        logger.warning("LangSmith observability: LANGCHAIN_TRACING_V2 is true, but LANGCHAIN_API_KEY is not set. Tracing disabled.")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    # Apply configuration safely to environment for LangChain/LangGraph
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = key
    os.environ["LANGCHAIN_PROJECT"] = proj
    os.environ["LANGCHAIN_ENDPOINT"] = endp

    logger.info(f"LangSmith observability: Tracing enabled for project '{proj}' at {endp}")
    return True
