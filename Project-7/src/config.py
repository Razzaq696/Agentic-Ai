"""Configuration management using Pydantic Settings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Provider Configuration
    llm_provider: str = Field(
        default="openai",
        description="LLM provider name: 'openai', 'google', 'groq', 'ollama', or 'mock'",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Target LLM model identifier",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for structured reasoning",
    )

    # API Keys
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    google_api_key: Optional[str] = Field(default=None, description="Google Gemini API key")
    groq_api_key: Optional[str] = Field(default=None, description="Groq API key")

    # Local Ollama config
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama service",
    )

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    # Embeddings & Chroma Vector Store
    embedding_provider: str = Field(
        default="auto",
        description="Embeddings provider: 'openai', 'google', 'mock', or 'auto' to match llm_provider",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        description="Directory for persistent Chroma storage",
    )
    chroma_collection_name: str = Field(
        default="product_knowledge_base",
        description="Collection name inside Chroma",
    )
    product_dataset_path: str = Field(
        default="./data/products.json",
        description="Path to local product knowledge base JSON",
    )
    rag_top_k: int = Field(
        default=3,
        description="Number of top product matches to retrieve",
    )
    rag_max_retries: int = Field(
        default=1,
        description="Maximum refinement attempts for Agentic RAG",
    )

    # Phase 3: Web Search Configuration
    search_provider: str = Field(
        default="mock",
        description="Search provider: 'mock', 'duckduckgo', 'tavily', or 'serper'",
    )
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="Tavily Search API key",
    )
    serper_api_key: Optional[str] = Field(
        default=None,
        description="Serper.dev Search API key",
    )
    search_max_results: int = Field(
        default=3,
        description="Maximum search result items to return per query",
    )

    # Phase 3: Playwright Browser Configuration
    browser_headless: bool = Field(
        default=True,
        description="Run Playwright browser in headless mode",
    )
    browser_timeout_ms: int = Field(
        default=10000,
        description="Playwright page navigation timeout in milliseconds",
    )
    max_react_iterations: int = Field(
        default=2,
        description="Hard maximum number of tool calling iterations for the ReAct research loop",
    )

    # Phase 5: Observability & Automation (LangSmith, n8n, SendGrid, Pushover)
    # LangSmith
    langchain_tracing_v2: bool = Field(
        default=False,
        description="Enable LangSmith tracing for LangChain/LangGraph",
    )
    langchain_api_key: Optional[str] = Field(
        default=None,
        description="LangSmith API key",
    )
    langchain_project: str = Field(
        default="shopping-decision-agent",
        description="LangSmith project name for tracing",
    )
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint",
    )

    # n8n Automation Webhook
    n8n_enabled: bool = Field(
        default=False,
        description="Enable forwarding final decision to n8n webhook",
    )
    n8n_webhook_url: Optional[str] = Field(
        default=None,
        description="n8n webhook URL to receive structured decision JSON",
    )
    n8n_timeout_sec: float = Field(
        default=5.0,
        description="Timeout in seconds for n8n webhook HTTP request",
    )

    # SendGrid Email Reporting
    sendgrid_enabled: bool = Field(
        default=False,
        description="Enable email delivery of recommendation report via SendGrid",
    )
    sendgrid_api_key: Optional[str] = Field(
        default=None,
        description="SendGrid API key",
    )
    sendgrid_from_email: Optional[str] = Field(
        default=None,
        description="Sender email address verified in SendGrid",
    )
    sendgrid_to_email: Optional[str] = Field(
        default=None,
        description="Recipient email address for shopping recommendations",
    )
    sendgrid_timeout_sec: float = Field(
        default=5.0,
        description="Timeout in seconds for SendGrid API calls",
    )

    # Pushover Push Notifications
    pushover_enabled: bool = Field(
        default=False,
        description="Enable push notifications via Pushover for key shopping events",
    )
    pushover_user_key: Optional[str] = Field(
        default=None,
        description="Pushover User or Group key",
    )
    pushover_api_token: Optional[str] = Field(
        default=None,
        description="Pushover Application API token",
    )
    pushover_timeout_sec: float = Field(
        default=5.0,
        description="Timeout in seconds for Pushover notification requests",
    )


# Global singleton settings instance
settings = Settings()
