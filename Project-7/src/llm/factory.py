"""LLM factory supporting multiple providers and offline deterministic mocking."""

import re
from typing import Any, Dict, List, Optional, Type
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.config import settings
from src.models.schemas import BudgetInfo, RequirementAnalysisOutput
from src.utils.logger import logger


class MockRequirementLLM:
    """Mock LLM used for deterministic unit testing and offline execution."""

    def __init__(self, forced_output: Optional[RequirementAnalysisOutput] = None, should_fail: bool = False):
        self.forced_output = forced_output
        self.should_fail = should_fail

    def with_structured_output(self, schema: Type[Any], **kwargs):
        return self

    def invoke(self, messages: Any) -> RequirementAnalysisOutput:
        """Simulate LLM extraction for testing and offline fallback."""
        if self.should_fail:
            raise ValueError("Simulated LLM service failure or invalid JSON generation")

        if self.forced_output is not None:
            return self.forced_output

        # Extract text from input messages (focusing on HumanMessage / user input)
        user_text = ""
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, HumanMessage):
                    user_text += f" {m.content}"
                elif isinstance(m, BaseMessage) and not isinstance(m, SystemMessage):
                    user_text += f" {m.content}"
                elif isinstance(m, tuple) and len(m) == 2 and m[0] in ("user", "human"):
                    user_text += f" {m[1]}"
                elif isinstance(m, dict) and m.get("role") in ("user", "human"):
                    user_text += f" {m.get('content', '')}"
            if not user_text:
                # Fallback if no specific human message was isolated
                for m in messages:
                    if hasattr(m, "content"):
                        user_text += f" {m.content}"
        elif isinstance(messages, str):
            user_text = messages

        return self._heuristic_mock_parse(user_text)

    def _heuristic_mock_parse(self, text: str) -> RequirementAnalysisOutput:
        """Provide realistic mock structured output based on keywords for offline testing."""
        clean = text.lower()

        # Check for vague / ambiguous input
        if any(phrase in clean for phrase in ["something good", "anything", "buy something", "recommend something", "help me buy"]):
            return RequirementAnalysisOutput(
                product_category=None,
                budget=BudgetInfo(is_specified=False, is_flexible=True),
                required_features=[],
                preferences=[],
                priorities=[],
                intended_use=None,
                is_clear=False,
                ambiguities_or_missing_info=["Product category is completely missing or ambiguous"],
                summary="The user's shopping request lacks a specific product category.",
            )

        # Category detection
        category = None
        if "laptop" in clean or "notebook" in clean or "macbook" in clean:
            category = "Laptop"
        elif "keyboard" in clean:
            category = "Mechanical Keyboard"
        elif "headphone" in clean or "earbud" in clean or "earphones" in clean:
            category = "Headphones"
        elif "phone" in clean or "smartphone" in clean or "iphone" in clean:
            category = "Smartphone"
        elif "monitor" in clean or "display" in clean:
            category = "Monitor"
        elif "espresso" in clean or "coffee" in clean:
            category = "Espresso Machine"
        elif "shoes" in clean or "sneakers" in clean:
            category = "Shoes"
        else:
            # Fallback category extraction
            words = [w for w in clean.split() if len(w) > 2 and w not in ("need", "want", "for", "with", "under", "buy", "produce", "structured", "requirement", "analysis.", "analysis")]
            category = words[-1].capitalize() if words else "Product"

        # Budget extraction
        budget_match = re.search(r"(?:under|below|less than|\$)\s*(\d+(?:,\d+)*(?:\.\d+)?)", clean)
        range_match = re.search(r"\$?(\d+)\s*(?:-|to)\s*\$?(\d+)", clean)
        
        budget = BudgetInfo(is_specified=False, is_flexible=True)
        if range_match:
            min_val = float(range_match.group(1).replace(",", ""))
            max_val = float(range_match.group(2).replace(",", ""))
            budget = BudgetInfo(min_amount=min_val, max_amount=max_val, is_specified=True, is_flexible=False, raw_text=range_match.group(0))
        elif budget_match:
            val = float(budget_match.group(1).replace(",", ""))
            budget = BudgetInfo(max_amount=val, is_specified=True, is_flexible=False, raw_text=budget_match.group(0))

        # Features & specs
        features = []
        if "16gb" in clean or "16 gb" in clean:
            features.append("16GB RAM")
        if "32gb" in clean:
            features.append("32GB RAM")
        if "anc" in clean or "noise cancel" in clean:
            features.append("Active Noise Cancellation")
        if "wireless" in clean or "bluetooth" in clean:
            features.append("Wireless")
        if "mechanical" in clean:
            features.append("Mechanical switches")
        if "4k" in clean:
            features.append("4K Display")
        if "lightweight" in clean:
            features.append("Lightweight")

        # Preferences & Priorities
        preferences = []
        priorities = []
        if "coding" in clean or "programming" in clean:
            priorities.append("Optimized for software development")
        if "gaming" in clean:
            priorities.append("Gaming performance")
        if "battery" in clean:
            priorities.append("Long battery life")
        if "black" in clean or "white" in clean:
            preferences.append("Color preference noted")

        # Ambiguities
        ambiguities = []
        if not features and category in ["Laptop", "Monitor"]:
            ambiguities.append("Specific performance requirements or screen specs not detailed")

        return RequirementAnalysisOutput(
            product_category=category,
            budget=budget,
            required_features=features,
            preferences=preferences,
            priorities=priorities,
            intended_use="General / specified use case",
            is_clear=True,
            ambiguities_or_missing_info=ambiguities,
            summary=f"Looking for {category} with extracted specifications.",
        )


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    **kwargs,
) -> Any:
    """Factory to instantiate LLM instances based on environment or configuration."""
    active_provider = (provider or settings.llm_provider).lower()
    active_model = model or settings.llm_model
    temp = temperature if temperature is not None else settings.llm_temperature

    logger.debug(f"Initializing LLM with provider={active_provider}, model={active_model}")

    if active_provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = settings.openai_api_key
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. Falling back to MockRequirementLLM for offline stability.")
            return MockRequirementLLM()
        return ChatOpenAI(
            model=active_model,
            temperature=temp,
            api_key=api_key,
            **kwargs,
        )

    elif active_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = settings.google_api_key
        if not api_key:
            logger.warning("GOOGLE_API_KEY is not set. Falling back to MockRequirementLLM for offline stability.")
            return MockRequirementLLM()
        return ChatGoogleGenerativeAI(
            model=active_model,
            temperature=temp,
            google_api_key=api_key,
            **kwargs,
        )

    elif active_provider == "groq":
        from langchain_groq import ChatGroq
        api_key = settings.groq_api_key
        if not api_key:
            logger.warning("GROQ_API_KEY is not set. Falling back to MockRequirementLLM for offline stability.")
            return MockRequirementLLM()
        return ChatGroq(
            model_name=active_model,
            temperature=temp,
            groq_api_key=api_key,
            **kwargs,
        )

    elif active_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=active_model,
            base_url=settings.ollama_base_url,
            temperature=temp,
            **kwargs,
        )

    elif active_provider == "mock":
        return MockRequirementLLM()

    else:
        logger.warning(f"Unknown LLM provider '{active_provider}'. Using MockRequirementLLM.")
        return MockRequirementLLM()


# =====================================================================
# PHASE 2: EMBEDDINGS FACTORY & DETERMINISTIC MOCK EMBEDDINGS
# =====================================================================

import hashlib
import math
from langchain_core.embeddings import Embeddings


class DeterministicMockEmbeddings(Embeddings):
    """Deterministic, lightweight local embeddings provider for offline testing."""

    def __init__(self, dimension: int = 128):
        self.dimension = dimension

    def _embed_text(self, text: str) -> List[float]:
        tokens = text.lower().replace("-", " ").replace("_", " ").split()
        vec = [0.0] * self.dimension
        if not tokens:
            return vec
        for token in tokens:
            h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dimension
            sign = 1.0 if ((h >> 8) & 1) else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text)


def get_embeddings(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    **kwargs,
) -> Embeddings:
    """Factory to instantiate Embeddings instances based on settings or environment."""
    raw_provider = (provider or settings.embedding_provider).lower()
    active_provider = settings.llm_provider.lower() if raw_provider == "auto" else raw_provider
    active_model = model or settings.embedding_model

    logger.debug(f"Initializing Embeddings with provider={active_provider}, model={active_model}")

    if active_provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        api_key = settings.openai_api_key
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. Falling back to DeterministicMockEmbeddings.")
            return DeterministicMockEmbeddings()
        return OpenAIEmbeddings(
            model=active_model,
            api_key=api_key,
            **kwargs,
        )

    elif active_provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        api_key = settings.google_api_key
        if not api_key:
            logger.warning("GOOGLE_API_KEY is not set. Falling back to DeterministicMockEmbeddings.")
            return DeterministicMockEmbeddings()
        emb_model = active_model if "embedding" in active_model else "models/embedding-001"
        return GoogleGenerativeAIEmbeddings(
            model=emb_model,
            google_api_key=api_key,
            **kwargs,
        )

    elif active_provider == "mock":
        return DeterministicMockEmbeddings()

    else:
        logger.info(f"Using DeterministicMockEmbeddings for provider '{active_provider}'.")
        return DeterministicMockEmbeddings()
