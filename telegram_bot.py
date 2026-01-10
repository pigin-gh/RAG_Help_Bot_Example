"""
Telegram-бот для ответов на вопросы пользователей с использованием RAG.
Инициализация GigaChat и векторного индекса выполняется один раз при старте бота.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
    CallbackQueryHandler,
)

from rag_service import RAGService
from user_context import user_context_manager
from rate_limiter import AsyncRateLimiter, RateLimitConfig, RateLimitError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Глобальные ссылки, инициализируем один раз при запуске
model: GigaChat | None = None
rag_service: RAGService | None = None
rate_limiter: AsyncRateLimiter | None = None

# Примеры вопросов для подсказки (макс 40 символов для кнопок)
EXAMPLE_QUESTIONS = [
    "Как создать заявку?",
    "Как оформить заезд?",
    "Как изменить период?",
    "Поиск по номеру телефона?",
    "Какие тарифы?",
]


def init_services() -> None:
    """Инициализирует GigaChat и RAG один раз при старте процесса."""
    global model, rag_service, rate_limiter

    if model is not None and rag_service is not None and rate_limiter is not None:
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
    rag_service = RAGService(
        str(knowledge_base_path),
        chunk_size=1000,
        chunk_overlap=200,
        use_hybrid_search=True
    )
    
    # Инициализация rate limiter
    rate_limit_config = RateLimitConfig(
        max_requests_per_minute=5,
        max_requests_per_hour=30,
        max_retries=3,
        retry_delay=1.0
    )
    rate_limiter = AsyncRateLimiter(rate_limit_config)
    logger.info("Rate limiter инициализирован")


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


def get_example_keyboard():
    """Создает клавиатуру с примерами вопросов"""
    keyboard = []
    for idx, question in enumerate(EXAMPLE_QUESTIONS):
        # Используем индекс вместо полного текста (ограничение 64 байта)
        keyboard.append([InlineKeyboardButton(text=f"❓ {question}", callback_data=f"ex_{idx}")])
    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    user_id = update.message.from_user.id if update.message and update.message.from_user else None
    
    if user_id:
        user_context_manager.update_context(user_id)
    
    welcome_text = (
        "👋 <b>Привет! Я бот ECVI</b>\n\n"
        "Я отвечаю на вопросы, используя базу знаний ECVI.\n\n"
        "💡 <b>Примеры вопросов:</b>\n"
        f"• Как создать заявку?\n"
        f"• Как оформить заезд?\n"
        f"• Как изменить период?\n"
        f"• Поиск по номеру телефона?\n"
        f"• Какие тарифы?\n\n"
        "👇 <b>Выберите пример вопроса или напишите свой:</b>"
    )
    
    await update.message.reply_text(
        welcome_text,
        parse_mode="HTML",
        reply_markup=get_example_keyboard()
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    help_text = (
        "🤖 <b>Помощь по использованию бота ECVI</b>\n\n"
        "<b>Как пользоваться:</b>\n"
        "1. Напишите ваш вопрос текстом\n"
        "2. Бот найдет релевантную информацию в базе знаний\n"
        "3. Получите ответ на основе найденных данных\n\n"
        "<b>Примеры вопросов:</b>\n"
        "• Как создать заявку?\n"
        "• Как оформить заезд?\n"
        "• Как изменить период?\n\n"
        "<b>Команды:</b>\n"
        "/start - начать работу\n"
        "/help - показать эту справку"
    )
    await update.message.reply_text(help_text, parse_mode="HTML")


async def example_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик нажатия на кнопку с примером вопроса."""
    query = update.callback_query
    await query.answer()
    
    # Извлекаем индекс вопроса из callback_data
    try:
        idx = int(query.data.replace("ex_", ""))
        question = EXAMPLE_QUESTIONS[idx]
    except (ValueError, IndexError):
        await query.message.reply_text("❌ Некорректный пример вопроса")
        return
    
    # Отправляем сообщение о начале обработки
    processing_msg = await query.message.reply_text(
        f"⚙️ <b>Обрабатываю вопрос:</b> {question}\n"
        "⏳ Пожалуйста, подождите...",
        parse_mode="HTML"
    )
    
    await process_question(query.message, question, processing_msg.message_id)


async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обрабатывает любое текстовое сообщение как вопрос пользователя."""
    if update.message is None or not update.message.text:
        return

    question = update.message.text.strip()
    if not question:
        await update.message.reply_text("Отправьте непустой вопрос.")
        return

    # Отправляем сообщение о начале обработки
    processing_msg = await update.message.reply_text(
        f"⚙️ <b>Обрабатываю вопрос:</b> {question}\n"
        "⏳ Пожалуйста, подождите...",
        parse_mode="HTML"
    )
    
    await process_question(update.message, question, processing_msg.message_id)


async def process_question(message, question: str, processing_msg_id: int) -> None:
    """Обрабатывает вопрос и отправляет ответ."""
    user_id = message.from_user.id if message and message.from_user else None
    if not user_id:
        await message.reply_text("Не удалось определить пользователя.")
        return

    # Проверка rate limit
    if rate_limiter:
        can_proceed, error_message = await rate_limiter.check_rate_limit(user_id)
        if not can_proceed:
            await message.reply_text(
                f"⚠️ {error_message}\n"
                f"Пожалуйста, подождите немного перед следующим запросом."
            )
            # Удаляем сообщение "обрабатываем"
            try:
                await message.bot.delete_message(chat_id=message.chat_id, message_id=processing_msg_id)
            except:
                pass
            return

    try:
        if rag_service is None or model is None:
            raise RuntimeError("Сервисы не инициализированы")

        # Получаем контекст пользователя
        user_context = user_context_manager.get_context(user_id)
        
        # Добавляем вопрос в историю
        user_context_manager.add_question(user_id, question)

        # Получаем контекст из сохраненного индекса с использованием улучшенного поиска
        context_text, has_results = rag_service.get_relevant_context_as_text(
            question, 
            k=3, 
            score_threshold=0.5,
            user_context=user_context
        )
        
        if not has_results:
            await message.reply_text(
                "❌ <b>Не удалось найти релевантную информацию</b>\n\n"
                "В базе знаний нет данных по вашему вопросу. "
                "Попробуйте переформулировать вопрос или задать другой.",
                parse_mode="HTML"
            )
            # Удаляем сообщение "обрабатываем"
            try:
                await message.bot.delete_message(chat_id=message.chat_id, message_id=processing_msg_id)
            except:
                pass
            return

        messages = build_prompt(question, context_text)
        
        # Регистрируем запрос в rate limiter
        if rate_limiter:
            await rate_limiter.get_limiter(user_id).acquire(user_id)

        # Обновляем сообщение о загрузке (имитация активности)
        try:
            await message.bot.edit_message_text(
                text=f"⚙️ <b>Обрабатываю вопрос:</b> {question}\n"
                     "⏳ Генерирую ответ...",
                chat_id=message.chat_id,
                message_id=processing_msg_id,
                parse_mode="HTML"
            )
        except:
            pass

        response = model.invoke(messages)
        
        # Удаляем сообщение "обрабатываем"
        try:
            await message.bot.delete_message(chat_id=message.chat_id, message_id=processing_msg_id)
        except:
            pass
        
        # Отправляем ответ
        await message.reply_text(response.content)
        
    except RateLimitError as e:
        logger.warning(f"Rate limit error for user {user_id}: {e}")
        await message.reply_text(
            "⚠️ <b>Превышен лимит запросов</b>\n\n"
            "Пожалуйста, подождите 1 минуту перед следующим запросом.",
            parse_mode="HTML"
        )
        # Удаляем сообщение "обрабатываем"
        try:
            await message.bot.delete_message(chat_id=message.chat_id, message_id=processing_msg_id)
        except:
            pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ошибка при обработке вопроса: %s", exc)
        await message.reply_text(
            "❌ <b>Ошибка при обработке запроса</b>\n\n"
            "Что-то пошло не так. Попробуйте позже или переформулируйте вопрос.",
            parse_mode="HTML"
        )
        # Удаляем сообщение "обрабатываем"
        try:
            await message.bot.delete_message(chat_id=message.chat_id, message_id=processing_msg_id)
        except:
            pass


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
    application.add_handler(CallbackQueryHandler(example_callback, pattern="^ex_"))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    logger.info("Бот запущен. Ожидаем сообщения...")
    application.run_polling()


if __name__ == "__main__":
    main()
