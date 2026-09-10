"""End-to-end integration tests for Telegram Bot Interface and LangGraph Agent."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, Message, User, Chat
from bot import start_command, help_command, handle_message, error_handler, create_telegram_app


def create_mock_update(text: str, user_id: int = 12345):
    """Helper to construct a realistic mocked Telegram Update object."""
    mock_update = MagicMock(spec=Update)
    mock_message = MagicMock(spec=Message)
    mock_user = MagicMock(spec=User)
    mock_chat = MagicMock(spec=Chat)

    mock_user.id = user_id
    mock_chat.id = user_id
    mock_message.text = text
    mock_message.reply_text = AsyncMock()

    mock_update.message = mock_message
    mock_update.effective_user = mock_user
    mock_update.effective_chat = mock_chat

    mock_context = MagicMock()
    return mock_update, mock_context


class TestTelegramEndToEnd:
    """Tests full end-to-end flows connecting Telegram handlers to the LangGraph Agent."""

    def test_start_command(self):
        """TEST 1 — /start: Verify welcome message is sent."""
        update, context = create_mock_update("/start")
        asyncio.run(start_command(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert "Welcome to the Telegram Agentic AI Assistant" in sent_text or "Agentic AI Assistant" in sent_text

    def test_help_command(self):
        """TEST 5 — /help: Verify help documentation is sent."""
        update, context = create_mock_update("/help")
        asyncio.run(help_command(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert "Capabilities" in sent_text or "LangGraph" in sent_text

    def test_telegram_llm_query(self):
        """TEST 2 — LLM REASONING: Verify Telegram message routes through LangGraph LLM reasoning."""
        query = "Explain why Python is commonly used in AI development."
        update, context = create_mock_update(query)

        asyncio.run(handle_message(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert len(sent_text) > 0
        assert "python" in sent_text.lower()

    def test_telegram_rag_query(self):
        """TEST 3 — RAG: Verify Telegram message routes through LangGraph RAG and returns grounded knowledge."""
        query = "What are the core features and knowledge base of the Telegram Agentic AI Assistant?"
        update, context = create_mock_update(query)

        asyncio.run(handle_message(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert len(sent_text) > 0
        assert "knowledge" in sent_text.lower() or "assistant" in sent_text.lower() or "features" in sent_text.lower()

    def test_telegram_tool_query(self):
        """TEST 4 — TOOL: Verify Telegram message routes through LangGraph Tool calculation."""
        query = "Calculate 125 × 8."
        update, context = create_mock_update(query)

        asyncio.run(handle_message(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert "1,000" in sent_text or "1000" in sent_text

    def test_telegram_tool_failure(self):
        """TEST 6 — ERROR: Verify tool division by zero returns friendly error to Telegram user."""
        query = "Calculate 10 / 0"
        update, context = create_mock_update(query)

        asyncio.run(handle_message(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert "division by zero" in sent_text.lower() or "couldn't complete" in sent_text.lower()

    def test_telegram_empty_query(self):
        """Verify empty message triggers polite prompt without crashing."""
        update, context = create_mock_update("   ")

        asyncio.run(handle_message(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert "valid query" in sent_text.lower()

    def test_telegram_unexpected_workflow_exception(self, monkeypatch):
        """Verify unhandled workflow exception is caught gracefully and returns friendly error."""
        import bot

        def mock_failing_agent(q):
            raise RuntimeError("Database connection suddenly dropped")

        monkeypatch.setattr(bot, "run_agent", mock_failing_agent)

        update, context = create_mock_update("Test error query")
        asyncio.run(handle_message(update, context))

        update.message.reply_text.assert_called_once()
        sent_text = update.message.reply_text.call_args[0][0]
        assert "couldn't process your request" in sent_text.lower()


class TestTelegramAppBuilder:
    """Verifies Telegram application initialization."""

    def test_app_creation(self):
        """Ensures create_telegram_app registers all handlers without error."""
        app = create_telegram_app("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
        assert app is not None
