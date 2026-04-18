import asyncio
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message

BOT_TOKEN = "7965315840:AAEyCa8sc6Mz_cm5XQlS4j6YxI1zl5ryuyY"
GEMINI_KEY = "AIzaSyCb3C6dp2_yr9Y01fIxDsGpiSW-WC-WZ34"
# Актуальная модель (Gemini 2.5 Flash)
MODEL = "gemini-2.5-flash-preview-05-20"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

SYSTEM_PROMPT = """Ты — SnakeAI, персональный ИИ-ассистент. Твой создатель @dlais1337. Отвечай чётко, по делу, без лишней воды. Помогай с вопросами, текстами, кодом и анализом информации."""

# Хранилище истории
history = {}

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "🐍 SnakeAI готов к работе.\n"
        f"Создатель: @dlais1337\n"
        f"Модель: {MODEL}\n\n"
        "Задайте ваш вопрос."
    )

@dp.message(Command("reset"))
async def reset(message: Message):
    uid = message.from_user.id
    if uid in history:
        del history[uid]
    await message.answer("История диалога сброшена.")

@dp.message()
async def handle(message: Message):
    uid = message.from_user.id
    user_text = message.text
    
    await bot.send_chat_action(message.chat.id, "typing")
    
    # Инициализация истории
    if uid not in history:
        history[uid] = []
    
    # Добавляем системный промпт если история пуста
    if len(history[uid]) == 0:
        history[uid].append({"role": "user", "parts": [{"text": SYSTEM_PROMPT}]})
        history[uid].append({"role": "model", "parts": [{"text": "Понял. Я готов помогать."}]})
    
    # Добавляем сообщение пользователя
    history[uid].append({"role": "user", "parts": [{"text": user_text}]})
    
    # Ограничиваем историю
    if len(history[uid]) > 12:
        history[uid] = history[uid][:2] + history[uid][-10:]
    
    payload = {
        "contents": history[uid],
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 2048
        }
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{GEMINI_URL}?key={GEMINI_KEY}", json=payload) as resp:
                data = await resp.json()
                
                if "error" in data:
                    await message.answer(f"Ошибка API: {data['error']['message']}")
                    return
                
                reply = data["candidates"][0]["content"]["parts"][0]["text"]
                history[uid].append({"role": "model", "parts": [{"text": reply}]})
                
                if len(reply) > 4096:
                    for i in range(0, len(reply), 4096):
                        await message.answer(reply[i:i+4096])
                else:
                    await message.answer(reply)
                    
    except Exception as e:
        await message.answer(f"Произошла ошибка: {str(e)[:100]}")

async def main():
    print(f"SnakeAI запущен с моделью: {MODEL}")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
