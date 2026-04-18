"""
SnakeAI - Полностью автономный ИИ-ассистент
БЕЗ ВНЕШНИХ API. Обучение встроено в код.
Версия: 4.0 (Исправленная)
Создатель: @dlais1337
"""

import asyncio
import logging
import json
import os
import pickle
import re
import math
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ==================== КОНФИГУРАЦИЯ ====================
BOT_TOKEN = "7965315840:AAEyCa8sc6Mz_cm5XQlS4j6YxI1zl5ryuyY"
ADMIN_USERNAME = "@dlais1337"

# ==================== ЛОГИРОВАНИЕ ====================
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ==================== ТОКЕНИЗАТОР ====================
class SimpleTokenizer:
    """Простой токенизатор для русского и английского языков"""
    
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_counts = Counter()
        self.fitted = False
        
    def preprocess_text(self, text: str) -> str:
        """Очистка текста"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def fit(self, texts: List[str]):
        """Обучение токенизатора на корпусе текстов"""
        for text in texts:
            text = self.preprocess_text(text)
            words = text.split()
            self.word_counts.update(words)
        
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        for idx, (word, _) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.fitted = True
        logger.info(f"Токенизатор обучен, размер словаря: {len(self.word2idx)}")
    
    def encode(self, text: str, max_len: int = 64) -> List[int]:
        """Преобразование текста в последовательность индексов"""
        if not self.fitted:
            return [2, 3]
        
        text = self.preprocess_text(text)
        words = text.split()
        tokens = [2]  # <SOS>
        for word in words[:max_len-2]:
            tokens.append(self.word2idx.get(word, 1))
        tokens.append(3)  # <EOS>
        
        # Паддинг
        while len(tokens) < max_len:
            tokens.append(0)
        return tokens[:max_len]
    
    def decode(self, tokens: List[int]) -> str:
        """Преобразование индексов обратно в текст"""
        words = []
        for token in tokens:
            if token in [0, 1, 2, 3]:
                continue
            word = self.idx2word.get(token, "")
            if word:
                words.append(word)
        return ' '.join(words)

# ==================== ПРОСТАЯ НЕЙРОННАЯ СЕТЬ ====================
class SimpleNN:
    """Простая полносвязная сеть для классификации ответов"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Инициализация весов Xavier
        limit1 = math.sqrt(6 / (input_size + hidden_size))
        self.W1 = [[random.uniform(-limit1, limit1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0] * hidden_size
        
        limit2 = math.sqrt(6 / (hidden_size + output_size))
        self.W2 = [[random.uniform(-limit2, limit2) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [0.0] * output_size
    
    def _relu(self, x: float) -> float:
        return max(0.0, x)
    
    def _softmax(self, x: List[float]) -> List[float]:
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [ex / sum_exp for ex in exp_x]
    
    def forward(self, x: List[float]) -> List[float]:
        """Прямой проход"""
        # Слой 1
        hidden = [0.0] * self.hidden_size
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                if j < len(x):
                    hidden[i] += x[j] * self.W1[j][i]
            hidden[i] = self._relu(hidden[i] + self.b1[i])
        
        # Слой 2
        output = [0.0] * self.output_size
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                output[i] += hidden[j] * self.W2[j][i]
            output[i] += self.b2[i]
        
        return self._softmax(output)
    
    def train_step(self, x: List[float], y: int, lr: float = 0.01) -> float:
        """Один шаг обучения"""
        # Forward
        hidden = [0.0] * self.hidden_size
        hidden_raw = [0.0] * self.hidden_size
        for i in range(self.hidden_size):
            for j in range(self.input_size):
                if j < len(x):
                    hidden_raw[i] += x[j] * self.W1[j][i]
            hidden_raw[i] += self.b1[i]
            hidden[i] = self._relu(hidden_raw[i])
        
        output_raw = [0.0] * self.output_size
        for i in range(self.output_size):
            for j in range(self.hidden_size):
                output_raw[i] += hidden[j] * self.W2[j][i]
            output_raw[i] += self.b2[i]
        
        output = self._softmax(output_raw)
        
        # Loss (cross-entropy)
        loss = -math.log(max(output[y], 1e-10))
        
        # Backward
        grad_output = [output[i] - (1 if i == y else 0) for i in range(self.output_size)]
        
        grad_W2 = [[0.0] * self.output_size for _ in range(self.hidden_size)]
        grad_b2 = [0.0] * self.output_size
        for i in range(self.output_size):
            grad_b2[i] = grad_output[i]
            for j in range(self.hidden_size):
                grad_W2[j][i] = hidden[j] * grad_output[i]
        
        grad_hidden = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            for i in range(self.output_size):
                grad_hidden[j] += self.W2[j][i] * grad_output[i]
            if hidden_raw[j] <= 0:
                grad_hidden[j] = 0
        
        grad_W1 = [[0.0] * self.hidden_size for _ in range(self.input_size)]
        grad_b1 = [0.0] * self.hidden_size
        for j in range(self.hidden_size):
            grad_b1[j] = grad_hidden[j]
            for k in range(self.input_size):
                if k < len(x):
                    grad_W1[k][j] = x[k] * grad_hidden[j]
        
        # Update weights
        for j in range(self.hidden_size):
            self.b1[j] -= lr * grad_b1[j]
            for k in range(self.input_size):
                self.W1[k][j] -= lr * grad_W1[k][j]
        
        for i in range(self.output_size):
            self.b2[i] -= lr * grad_b2[i]
            for j in range(self.hidden_size):
                self.W2[j][i] -= lr * grad_W2[j][i]
        
        return loss

# ==================== ОБУЧАЮЩИЕ ДАННЫЕ ====================
TRAINING_DATA = [
    ("привет", "Здравствуйте! Я SnakeAI, ваш персональный ассистент."),
    ("здравствуй", "Добрый день! SnakeAI на связи."),
    ("ку", "Приветствую! SnakeAI готов к работе."),
    ("как дела", "У меня всё отлично! Я всегда готов помочь."),
    ("что ты умеешь", "Я отвечаю на вопросы, помогаю с текстами и кодом."),
    ("кто тебя создал", f"Мой создатель {ADMIN_USERNAME}."),
    ("как тебя зовут", "Меня зовут SnakeAI."),
    ("python", "Python — высокоуровневый язык программирования."),
    ("2+2", "2 + 2 = 4"),
    ("столица россии", "Столица России — Москва."),
    ("расскажи анекдот", "Почему программисты путают Хэллоуин и Рождество? 31 Oct == 25 Dec!"),
    ("пока", "До свидания! Обращайтесь ещё."),
]

# ==================== ОСНОВНОЙ КЛАСС ИИ ====================
class SnakeAI:
    """Основной класс ИИ-ассистента"""
    
    def __init__(self):
        self.tokenizer = SimpleTokenizer(vocab_size=2000)
        self.responses: List[str] = []
        self.model: Optional[SimpleNN] = None
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
        self.trained = False
        
        self._prepare_data()
        self._load_or_train()
    
    def _prepare_data(self):
        """Подготовка данных для обучения"""
        texts = [q for q, _ in TRAINING_DATA] + [a for _, a in TRAINING_DATA]
        self.tokenizer.fit(texts)
        
        # Собираем уникальные ответы
        self.responses = list(set(a for _, a in TRAINING_DATA))
        logger.info(f"Подготовлено {len(TRAINING_DATA)} примеров, {len(self.responses)} уникальных ответов")
    
    def _vectorize(self, text: str, max_len: int = 64) -> List[float]:
        """Преобразование текста в вектор"""
        tokens = self.tokenizer.encode(text, max_len)
        vec = [0.0] * self.tokenizer.vocab_size
        for t in tokens:
            if t > 0:
                vec[t] = 1.0
        return vec
    
    def _load_or_train(self):
        """Загрузка или обучение модели"""
        model_path = "snakeai_model.pkl"
        
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    saved = pickle.load(f)
                    self.model = saved['model']
                    self.tokenizer = saved['tokenizer']
                    self.responses = saved['responses']
                self.trained = True
                logger.info("Модель загружена из файла")
            except Exception as e:
                logger.error(f"Ошибка загрузки: {e}")
                self._train()
        else:
            self._train()
    
    def _train(self):
        """Обучение модели"""
        logger.info("Начинаю обучение...")
        
        input_size = self.tokenizer.vocab_size
        output_size = len(self.responses)
        self.model = SimpleNN(input_size, 64, output_size)
        
        epochs = 20
        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(TRAINING_DATA)
            
            for question, answer in TRAINING_DATA:
                x = self._vectorize(question)
                y = self.responses.index(answer)
                loss = self.model.train_step(x, y, lr=0.05)
                total_loss += loss
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Эпоха {epoch + 1}/{epochs}, loss: {total_loss/len(TRAINING_DATA):.4f}")
        
        # Сохранение
        try:
            with open("snakeai_model.pkl", 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'tokenizer': self.tokenizer,
                    'responses': self.responses
                }, f)
            logger.info("Модель сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения: {e}")
        
        self.trained = True
        logger.info("Обучение завершено")
    
    def _find_best_response(self, question: str) -> str:
        """Поиск лучшего ответа"""
        question_lower = question.lower()
        
        # Прямой поиск
        for q, a in TRAINING_DATA:
            if q in question_lower or question_lower in q:
                return a
        
        # Нейросеть
        if self.model:
            x = self._vectorize(question)
            probs = self.model.forward(x)
            best_idx = max(range(len(probs)), key=lambda i: probs[i])
            return self.responses[best_idx]
        
        return "Извините, я не понял вопрос. Переформулируйте, пожалуйста."
    
    def chat(self, user_id: int, message: str) -> str:
        """Основной метод общения"""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        response = self._find_best_response(message)
        
        self.conversation_history[user_id].append({"role": "user", "content": message})
        self.conversation_history[user_id].append({"role": "assistant", "content": response})
        
        if len(self.conversation_history[user_id]) > 20:
            self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
        
        return response
    
    def reset_history(self, user_id: int):
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]

# ==================== ИНИЦИАЛИЗАЦИЯ ====================
snake_ai = SnakeAI()

# ==================== TELEGRAM БОТ ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐍 SnakeAI — автономный ИИ-ассистент\n\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        f"Ответов в базе: {len(snake_ai.responses)}\n"
        "Работает без внешних API\n\n"
        "Просто напишите вопрос.\n"
        "/reset — сбросить историю\n"
        "/status — статус"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    snake_ai.reset_history(update.effective_user.id)
    await update.message.reply_text("🧹 История сброшена.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 Статус\n\n"
        f"Обучен: {'✅' if snake_ai.trained else '❌'}\n"
        f"Ответов: {len(snake_ai.responses)}\n"
        f"Словарь: {len(snake_ai.tokenizer.word2idx)}\n"
        f"Создатель: {ADMIN_USERNAME}"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    message = update.message.text
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
    
    response = snake_ai.chat(user_id, message)
    
    if len(response) > 4096:
        for i in range(0, len(response), 4096):
            await update.message.reply_text(response[i:i+4096])
    else:
        await update.message.reply_text(response)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {context.error}")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    logger.info(f"SnakeAI запущен. Создатель: {ADMIN_USERNAME}")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
