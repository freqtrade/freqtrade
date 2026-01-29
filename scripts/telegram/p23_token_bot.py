#!/usr/bin/env python3
"""
P23 Telegram Token Bot
Simple polling bot to ingest session tokens.
Requires: pip install python-telegram-bot

Env Vars:
- TELEGRAM_BOT_TOKEN
- TELEGRAM_ALLOWED_CHAT_IDS (comma separated)
"""

import asyncio
import logging
import os
import sys

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# Logging setup - Strict: No tokens in logs
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

logger = logging.getLogger(__name__)

# Config
BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
ALLOWED_CHATS = os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "")
ALLOWED_CHAT_IDS = [int(x.strip()) for x in ALLOWED_CHATS.split(",") if x.strip()]


async def session_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in ALLOWED_CHAT_IDS:
        logger.warning(f"Unauthorized access attempt from {chat_id}")
        await update.message.reply_text("Unauthorized.")
        return

    # Expecting /session <token>
    if not context.args:
        await update.message.reply_text("Usage: /session <token>")
        return

    token = context.args[0]

    # Delegate to storage script via stdin using asyncio
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "scripts/p23_session_store.py",
            "--stdin",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_bytes, stderr_bytes = await proc.communicate(input=token.encode())

        stdout_str = stdout_bytes.decode().strip()
        stderr_str = stderr_bytes.decode().strip()

        if proc.returncode == 0:
            msg = "Session updated successfully."
            logger.info("Session token updated via Telegram.")
        else:
            msg = f"Failed to update session: {stderr_str or stdout_str}"
            logger.error(f"Session update failed: {msg}")

        await update.message.reply_text(msg)

    except Exception:
        logger.error("Error invoking storage script", exc_info=True)
        await update.message.reply_text("Internal Error.")


def main():
    if not BOT_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN missing.")
        sys.exit(1)

    if not ALLOWED_CHAT_IDS:
        logger.error("TELEGRAM_ALLOWED_CHAT_IDS missing.")
        sys.exit(1)

    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("session", session_command))

    logger.info("Bot started. Waiting for /session commands...")
    app.run_polling()


if __name__ == "__main__":
    main()
