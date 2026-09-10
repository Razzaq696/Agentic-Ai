"""Structured logging configuration for AI Smart Shopping Decision Agent."""

import logging
import sys
from typing import Optional


def setup_logger(name: str = "shopping_agent", level: Optional[str] = None) -> logging.Logger:
    """Set up and return a configured logger instance.

    Args:
        name: The name of the logger.
        level: Optional log level override (e.g. 'DEBUG', 'INFO', 'WARNING').

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level_str = (level or "INFO").upper()
        numeric_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(numeric_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(numeric_level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logger()
