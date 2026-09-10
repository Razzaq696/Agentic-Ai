"""Telegram Bot Interface for Project 6 Agentic AI Assistant."""

import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from agent.runner import run_agent

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("TelegramAgentBot")


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /start command."""
    welcome_text = (
        "🤖 Welcome to the **Telegram Agentic AI Assistant**!\n\n"
        "I can help you with:\n"
        "• 🧠 **LLM Reasoning**: Ask conceptual, technical, or analytical questions.\n"
        "• 📚 **Knowledge Retrieval**: Query project documentation and policies.\n"
        "• 🧮 **Tools & Calculations**: Send math expressions (e.g. `Calculate 125 × 8`).\n\n"
        "Simply send any question or request to get started!"
    )
    if update.message:
        await update.message.reply_text(welcome_text, parse_mode="Markdown")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handles the /help command."""
    help_text = (
        "📖 **Assistant Help & Capabilities**\n\n"
        "This bot is powered by an autonomous LangGraph agent workflow that analyzes your query "
        "and routes it to the optimal subsystem:\n\n"
        "1. **LLM / Reasoning**: For general explanations, coding, and logical thinking.\n"
        "2. **RAG / Knowledge**: For factual inquiries about project specifications and architecture.\n"
        "3. **Tools / APIs**: For arithmetic evaluations and utilities.\n\n"
        "Commands:\n"
        "/start - Welcome message and introduction\n"
        "/help - Display this help guide"
    )
    if update.message:
        await update.message.reply_text(help_text, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Receives user message, passes query to LangGraph workflow, and replies with final response."""
    if not update.message or not update.message.text:
        return

    user_query = update.message.text.strip()

    if not user_query:
        await update.message.reply_text("Please provide a valid query or request.")
        return

    logger.info(f"Received query from user {update.effective_user.id if update.effective_user else 'unknown'}: {user_query}")

    try:
        # Route query through the existing LangGraph Agent workflow
        agent_state = run_agent(user_query)
        final_reply = agent_state.get("final_response")

        if not final_reply:
            final_reply = "Sorry, I couldn't process your request right now."

        await update.message.reply_text(final_reply)

    except Exception as e:
        logger.error(f"Error processing Telegram request: {str(e)}", exc_info=True)
        await update.message.reply_text("Sorry, I couldn't process your request right now.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Global error handler for Telegram application exceptions."""
    logger.error(f"Unhandled Telegram exception: {context.error}", exc_info=context.error)
    if isinstance(update, Update) and update.message:
        try:
            await update.message.reply_text("Sorry, an internal error occurred. Please try again later.")
        except Exception:
            pass


from telegram.request import HTTPXRequest


def create_telegram_app(token: str):
    """Builds and configures the Telegram Bot application with custom request timeouts and proxy support."""
    proxy_url = os.getenv("TELEGRAM_PROXY_URL") or os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")

    request_kwargs = {
        "connect_timeout": 30.0,
        "read_timeout": 30.0,
        "write_timeout": 30.0,
    }

    if proxy_url:
        request_kwargs["proxy"] = proxy_url
        logger.info(f"Using proxy for Telegram Bot: {proxy_url}")

    request = HTTPXRequest(**request_kwargs)
    app = ApplicationBuilder().token(token).request(request).build()

    # Register command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))

    # Register message handler for text queries
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Register error handler
    app.add_error_handler(error_handler)

    return app



def main():
    """Main entry point to run the Telegram Bot."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token or token == "your_telegram_bot_token_here":
        print("\n" + "=" * 70)
        print("[!] TELEGRAM_BOT_TOKEN is not set or using default placeholder.")
        print("Please configure a valid TELEGRAM_BOT_TOKEN in your .env file.")
        print("Example:")
        print("TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ")
        print("=" * 70 + "\n")
        return

    print("[*] Starting Telegram Agentic AI Assistant Bot (Polling mode)...")
    app = create_telegram_app(token)
    app.run_polling()



if __name__ == "__main__":
    main()
