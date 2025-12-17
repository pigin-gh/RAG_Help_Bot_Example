"""
Простой Telegram-бот для ответов на вопросы пользователей с использованием RAG.
Инициализация GigaChat и векторного индекса выполняется один раз при старте бота.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from rag_service import RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Глобальные ссылки, инициализируем один раз при запуске
model: GigaChat | None = None
rag_service: RAGService | None = None


def init_services() -> None:
    """Инициализирует GigaChat и RAG один раз при старте процесса."""
    global model, rag_service

    if model is not None and rag_service is not None:
        return

    logger.info("Загрузка переменных окружения")
    load_dotenv(find_dotenv())

    logger.info("Инициализация GigaChat")
    model = GigaChat(
        model="GigaChat-2",
        verify_ssl_certs=False,
    )

    knowledge_base_path = Path(__file__).parent / "baza-znanii-ecvi.txt"
    logger.info("Инициализация RAG сервиса (загружает/создает сохраненный индекс)")
    rag_service = RAGService(str(knowledge_base_path))


def build_prompt(question: str, context: str) -> list:
    """Формирует сообщения для LLM."""
    system_prompt = (
        "Ты — помощник, который отвечает на вопросы на основе базы знаний ECVI. "
        "Используй предоставленные фрагменты. Если информации нет, честно скажи об этом."
    )
    user_prompt = (
        f"Вопрос пользователя:\n{question}\n\n"
        f"Релевантные фрагменты из базы знаний:\n{context}\n\n"
        "Сформулируй короткий и полезный ответ. "
        "Если данных недостаточно, напиши об этом."
    )
    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    await update.message.reply_text(
        "Привет! Я бот ECVI. Задайте вопрос, я отвечу, опираясь на базу знаний."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    await update.message.reply_text("Просто отправьте вопрос текстом, и я отвечу.")


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает любое текстовое сообщение как вопрос пользователя."""
    if update.message is None or not update.message.text:
        return

    question = update.message.text.strip()
    if not question:
        await update.message.reply_text("Отправьте непустой вопрос.")
        return

    try:
        if rag_service is None or model is None:
            raise RuntimeError("Сервисы не инициализированы")

        # Получаем контекст из сохраненного индекса
        context_text = rag_service.get_relevant_context_as_text(question, k=3)
        messages = build_prompt(question, context_text)

        response = model.invoke(messages)
        await update.message.reply_text(response.content)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка при обработке вопроса: %s", exc)
        await update.message.reply_text(
            "Не удалось обработать запрос. Попробуйте позже."
        )


def main() -> None:
    """Точка входа для запуска Telegram-бота."""
    init_services()

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError(
            "Не задан TELEGRAM_BOT_TOKEN. Укажите токен в переменных окружения."
        )

    application = Application.builder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Бот запущен. Ожидаем сообщения...")
    application.run_polling()


if __name__ == "__main__":
    main()


