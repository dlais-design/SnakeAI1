"""
SnakeAI Telegram Bot — полностью локальный ИИ-ассистент
Работает через Ollama, без внешних API
"""
import asyncio
import logging
import re
from typing import Dict, List, Optional

from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import aiohttp

# ---------- НАСТРОЙКИ ----------
BOT_TOKEN = "7965315840:AAEyCa8sc6Mz_cm5XQlS4j6YxI1zl5ryuyY"
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"
ADMIN_USERNAME = "@dlais1337"

# Системный промпт
SYSTEM_PROMPT = """Ты — SnakeAI, персональный ИИ-ассистент. Твой создатель @dlais1337. 
Отвечай чётко, по делу, без лишней воды. Помогай с вопросами, текстами, кодом и анализом информации.
Будь полезен и эффективен."""

# Настройки генерации
GENERATION_CONFIG = {
    "temperature": 0.7,
    "num_ctx": 4096,
    "num_predict": 2048,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1
}

# Хранилище истории диалогов (в памяти)
history: Dict[int, List[Dict[str, str]]] = {}
MAX_HISTORY = 20

# Логирование
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Клиент для работы с Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=120, connect=10)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def check_model(self) -> bool:
        """Проверить, загружена ли модель"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return any(self.model in m for m in models)
                return False
        except Exception as e:
            logger.error(f"Ошибка проверки модели: {e}")
            return False
    
    async def pull_model(self) -> bool:
        """Загрузить модель, если её нет"""
        try:
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/api/pull",
                json={"name": self.model, "stream": False},
                timeout=None
            ) as resp:
                return resp.status == 200
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False
    
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Отправить запрос к модели и получить ответ"""
        try:
            session = await self._get_session()
            
            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": GENERATION_CONFIG
            }
            
            async with session.post(
                f"{self.base_url}/api/chat",
                json=payload
            ) as resp:
                if resp.status == 404:
                    logger.info("Модель не найдена, пробую загрузить...")
                    if await self.pull_model():
                        return await self.chat(messages)
                    return "Не удалось загрузить модель. Попробуйте позже."
                
                if resp.status != 200:
                    return f"Ошибка API: {resp.status}"
                
                data = await resp.json()
                reply = data.get("message", {}).get("content", "")
                return self._clean_response(reply)
                
        except asyncio.TimeoutError:
            return "Превышено время ожидания ответа. Попробуйте ещё раз."
        except Exception as e:
            logger.error(f"Ошибка чата: {e}")
            return "Произошла ошибка при обработке запроса."
    
    def _clean_response(self, text: str) -> str:
        """Очистка ответа от системного мусора"""
        # Убираем  thinking и прочие теги
        text = re.sub(r"", "", text, flags=re.DOTALL)
        text = re.sub(r"", "", text, flags=re.DOTALL)
        return text.strip()
    
    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# Глобальный клиент
ollama = OllamaClient(OLLAMA_URL, OLLAMA_MODEL)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /start"""
    await update.message.reply_text(
        "🐍 SnakeAI готов к работе.\n\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        f"Модель: {OLLAMA_MODEL} (локальная, без API)\n\n"
        "Просто напишите ваш вопрос — я отвечу.\n"
        "/reset — сбросить историю диалога\n"
        "/status — статус системы\n"
        "/help — справка"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Команда /help"""
    await update.message.reply_text(
        "📚 Справка SnakeAI\n\n"
        "Доступные команды:\n"
        "/start — начать работу\n"
        "/reset — сбросить историю диалога\n"
        "/status — статус системы\n"
        "/help — эта справка\n\n"
        "Бот понимает контекст диалога и отвечает на любые вопросы.\n"
        f"Модель работает локально на сервере, без внешних API."
    )


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Сброс истории диалога"""
    user_id = update.effective_user.id
    if user_id in history:
        del history[user_id]
    await update.message.reply_text("🧹 История диалога сброшена.")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Статус системы"""
    model_loaded = await ollama.check_model()
    status_text = "✅ загружена" if model_loaded else "❌ не загружена"
    
    await update.message.reply_text(
        "📊 Статус SnakeAI\n\n"
        f"Модель: {OLLAMA_MODEL}\n"
        f"Статус модели: {status_text}\n"
        f"Сервер Ollama: {OLLAMA_URL}\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        "Работает полностью локально, без внешних API."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработка текстовых сообщений"""
    user_id = update.effective_user.id
    user_text = update.message.text
    user_name = update.effective_user.first_name or "Пользователь"
    
    # Отправляем индикатор набора текста
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
    
    # Инициализация истории
    if user_id not in history:
        history[user_id] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # Добавляем сообщение пользователя
    history[user_id].append({"role": "user", "content": user_text})
    
    # Ограничиваем размер истории
    if len(history[user_id]) > MAX_HISTORY + 1:
        history[user_id] = [history[user_id][0]] + history[user_id][-(MAX_HISTORY-1):]
    
    # Получаем ответ от Ollama
    reply = await ollama.chat(history[user_id])
    
    # Сохраняем ответ в историю
    history[user_id].append({"role": "assistant", "content": reply})
    
    # Отправляем ответ (разбиваем на части если нужно)
    if len(reply) > 4096:
        for i in range(0, len(reply), 4096):
            await update.message.reply_text(reply[i:i+4096])
    else:
        await update.message.reply_text(reply)


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}")
    if update and hasattr(update, "effective_message"):
        await update.effective_message.reply_text("Произошла ошибка. Попробуйте позже.")


async def init_ollama() -> bool:
    """Инициализация Ollama при запуске"""
    logger.info(f"Проверяю модель {OLLAMA_MODEL}...")
    
    if await ollama.check_model():
        logger.info("Модель уже загружена")
        return True
    
    logger.info("Модель не найдена, начинаю загрузку...")
    if await ollama.pull_model():
        logger.info("Модель успешно загружена")
        return True
    
    logger.error("Не удалось загрузить модель")
    return False


def main() -> None:
    """Точка входа"""
    # Создаём приложение
    app = Application.builder().token(BOT_TOKEN).build()
    
    # Регистрируем обработчики
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    # Запускаем
    logger.info(f"SnakeAI запускается... Создатель: {ADMIN_USERNAME}")
    
    # Инициализируем Ollama асинхронно перед запуском
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if not loop.run_until_complete(init_ollama()):
        logger.warning("Модель не загружена. Бот будет пытаться загрузить её при первом запросе.")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()