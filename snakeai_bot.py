import asyncio
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message

BOT_TOKEN = "7965315840:AAEyCa8sc6Mz_cm5XQlS4j6YxI1zl5ryuyY"
DEEPSEEK_KEY = "sk-159e3afa1f1846a4bfadf97a60937ec7"
MODEL = "deepseek-chat"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

SYSTEM_PROMPT = """Ты — SnakeAI, персональный ИИ-ассистент. Твой создатель @dlais1337. Отвечай чётко, по делу, без лишней воды. Помогай с вопросами, текстами, кодом и анализом информации."""

history = {}

@dp.message(Command("start"))
async def start(message: Message):
    await message.answer(
        "🐍 SnakeAI готов к работе.\n"
        f"Создатель: @dlais1337\n"
        f"Модель: {MODEL}\n\n"
        "Задайте ваш вопрос.\n"
        "/reset — сбросить историю диалога"
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
    
    if uid not in history:
        history[uid] = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    history[uid].append({"role": "user", "content": user_text})
    
    if len(history[uid]) > 12:
        history[uid] = history[uid][:1] + history[uid][-10:]
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": MODEL,
        "messages": history[uid],
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DEEPSEEK_URL, headers=headers, json=payload) as resp:
                data = await resp.json()
                
                if "error" in data:
                    await message.answer(f"Ошибка API: {data['error']['message']}")
                    return
                
                reply = data["choices"][0]["message"]["content"]
                history[uid].append({"role": "assistant", "content": reply})
                
                if len(reply) > 4096:
                    for i in range(0, len(reply), 4096):
                        await message.answer(reply[i:i+4096])
                else:
                    await message.answer(reply)
                    
    except Exception as e:
        await message.answer(f"Произошла ошибка. Попробуйте позже.")

async def main():
    print(f"SnakeAI запущен с моделью: {MODEL}")
    print(f"Создатель: @dlais1337")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
