import asyncio
import aiohttp
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.enums import ParseMode

# Конфигурация
BOT_TOKEN = "7965315840:AAEyCa8sc6Mz_cm5XQlS4j6YxI1zl5ryuyY"
GEMINI_API_KEY = "AIzaSyCb3C6dp2_yr9Y01fIxDsGpiSW-WC-WZ34"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
ADMIN_USERNAME = "@dlais1337"

# Системный промпт (официальный стиль)
SYSTEM_PROMPT = """Ты — SnakeAI, персональный ИИ-ассистент. Твой создатель @dlais1337. Отвечай чётко, по делу, без лишней воды. Помогай с вопросами, текстами, кодом и анализом информации. Будь полезен и эффективен."""

# Инициализация бота
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Хранилище истории диалогов
history = {}

async def ask_gemini(user_id: int, user_text: str) -> str:
    """Отправка запроса к Gemini API"""
    
    # Инициализация истории для нового пользователя
    if user_id not in history:
        history[user_id] = []
    
    # Добавляем системный промпт в начало, если история пуста
    if len(history[user_id]) == 0:
        history[user_id].append({"role": "user", "parts": [{"text": SYSTEM_PROMPT}]})
    
    # Добавляем сообщение пользователя
    history[user_id].append({"role": "user", "parts": [{"text": user_text}]})
    
    # Ограничиваем историю 10 последними сообщениями
    if len(history[user_id]) > 11:
        history[user_id] = history[user_id][:1] + history[user_id][-10:]
    
    # Формируем payload для Gemini
    payload = {
        "contents": history[user_id],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=payload
        ) as resp:
            data = await resp.json()
            
            # Проверка на ошибки
            if "error" in data:
                return f"Ошибка API: {data['error']['message']}"
            
            # Извлечение ответа
            reply = data["candidates"][0]["content"]["parts"][0]["text"]
            
            # Сохраняем ответ модели в историю
            history[user_id].append({"role": "model", "parts": [{"text": reply}]})
            
            return reply

@dp.message(Command("start"))
async def cmd_start(message: Message):
    await message.answer(
        "🐍 SnakeAI — персональный ИИ-ассистент.\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        "Модель: Gemini 1.5 Flash\n\n"
        "Я готов помочь с вопросами, текстами, кодом и анализом.\n"
        "/reset — сбросить историю диалога\n"
        "/status — статус системы"
    )

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    uid = message.from_user.id
    if uid in history:
        del history[uid]
    await message.answer("История диалога сброшена. Можем начинать заново.")

@dp.message(Command("status"))
async def cmd_status(message: Message):
    await message.answer(
        "📊 Статус SnakeAI:\n"
        "• Модель: Gemini 1.5 Flash\n"
        "• Провайдер: Google AI\n"
        "• Статус: онлайн\n"
        f"• Создатель: {ADMIN_USERNAME}\n"
        "• Время ответа: 2-5 секунд"
    )

@dp.message()
async def handle_message(message: Message):
    uid = message.from_user.id
    user_text = message.text
    
    # Отправляем индикатор набора текста
    await bot.send_chat_action(message.chat.id, "typing")
    
    try:
        reply = await ask_gemini(uid, user_text)
        # Разбиваем длинные сообщения если нужно
        if len(reply) > 4096:
            for i in range(0, len(reply), 4096):
                await message.answer(reply[i:i+4096])
        else:
            await message.answer(reply)
    except Exception as e:
        await message.answer(f"Произошла ошибка. Попробуйте позже.\nТехническая информация: {str(e)[:100]}")

async def main():
    print(f"SnakeAI запущен. Создатель: {ADMIN_USERNAME}")
    print("Модель: Gemini 1.5 Flash")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())