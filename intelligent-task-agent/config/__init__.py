"""
Configuration package for Intelligent Task Execution Agent.
Contains environment loading, application settings, and model configurations.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load environment variables if .env exists
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE)
else:
    load_dotenv()

# Application Settings defaults
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen3:4b")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
APP_NAME = os.getenv("APP_NAME", "Intelligent Task Execution Agent")
APP_ENV = os.getenv("APP_ENV", "development")
