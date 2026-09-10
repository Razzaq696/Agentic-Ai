"""
Configuration and Environment Management.
Loads environment variables and determines mock/real execution modes.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


class Config:
    """Central configuration class for credentials and runtime settings."""

    # SendGrid
    SENDGRID_API_KEY: Optional[str] = os.getenv("SENDGRID_API_KEY")
    SENDGRID_FROM_EMAIL: str = os.getenv("SENDGRID_FROM_EMAIL", "alerts@communication-assistant.local")

    # Pushover
    PUSHOVER_API_TOKEN: Optional[str] = os.getenv("PUSHOVER_API_TOKEN")
    PUSHOVER_USER_KEY: Optional[str] = os.getenv("PUSHOVER_USER_KEY")

    # LLM Keys
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

    # Mock mode default flag
    _default_mock: str = os.getenv("DEFAULT_MOCK_MODE", "true").strip().lower()
    DEFAULT_MOCK_MODE: bool = _default_mock in ("true", "1", "yes")

    @classmethod
    def is_sendgrid_configured(cls) -> bool:
        """Check if real SendGrid credentials are provided."""
        return bool(cls.SENDGRID_API_KEY and cls.SENDGRID_API_KEY.strip() and not cls.SENDGRID_API_KEY.startswith("your_"))

    @classmethod
    def is_pushover_configured(cls) -> bool:
        """Check if real Pushover credentials are provided."""
        return bool(
            cls.PUSHOVER_API_TOKEN and cls.PUSHOVER_USER_KEY and
            not cls.PUSHOVER_API_TOKEN.startswith("your_") and
            not cls.PUSHOVER_USER_KEY.startswith("your_")
        )

    @classmethod
    def is_llm_configured(cls) -> bool:
        """Check if any LLM API key is configured."""
        return bool(
            (cls.GEMINI_API_KEY and not cls.GEMINI_API_KEY.startswith("your_")) or
            (cls.OPENAI_API_KEY and not cls.OPENAI_API_KEY.startswith("your_"))
        )
