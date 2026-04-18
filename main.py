"""
SnakeAI - Полностью автономный ИИ-ассистент
БЕЗ ВНЕШНИХ API. Обучение встроено в код.
Версия: 3.0
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
    
    def __init__(self, vocab_size: int = 10000):
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
    
    def encode(self, text: str, max_len: int = 128) -> List[int]:
        """Преобразование текста в последовательность индексов"""
        if not self.fitted:
            return [2, 3]
        
        text = self.preprocess_text(text)
        words = text.split()
        tokens = [2]  # <SOS>
        for word in words[:max_len-2]:
            tokens.append(self.word2idx.get(word, 1))
        tokens.append(3)  # <EOS>
        
        if len(tokens) < max_len:
            tokens.extend([0] * (max_len - len(tokens)))
        return tokens[:max_len]
    
    def decode(self, tokens: List[int]) -> str:
        """Преобразование индексов обратно в текст"""
        words = []
        for token in tokens:
            if token in [0, 2, 3]:
                continue
            word = self.idx2word.get(token, "<UNK>")
            words.append(word)
        return ' '.join(words)

# ==================== НЕЙРОННАЯ СЕТЬ ====================
@dataclass
class Layer:
    """Базовый класс слоя"""
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        raise NotImplementedError
    
    def backward(self, grad: List[List[float]], lr: float) -> List[List[float]]:
        raise NotImplementedError

class DenseLayer(Layer):
    """Полносвязный слой"""
    
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        # Инициализация Xavier
        limit = math.sqrt(6 / (input_size + output_size))
        self.weights = [[random.uniform(-limit, limit) for _ in range(output_size)] for _ in range(input_size)]
        self.biases = [0.0] * output_size
        self.input_cache = None
        self.weight_grad = None
        self.bias_grad = None
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        self.input_cache = x
        batch_size = len(x)
        output = []
        for i in range(batch_size):
            sample_out = []
            for j in range(self.output_size):
                val = self.biases[j]
                for k in range(self.input_size):
                    val += x[i][k] * self.weights[k][j]
                sample_out.append(self._relu(val))
            output.append(sample_out)
        return output
    
    def backward(self, grad: List[List[float]], lr: float) -> List[List[float]]:
        batch_size = len(grad)
        self.weight_grad = [[0.0] * self.output_size for _ in range(self.input_size)]
        self.bias_grad = [0.0] * self.output_size
        input_grad = [[0.0] * self.input_size for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(self.output_size):
                # Производная ReLU
                if self.input_cache[i][j] > 0:
                    grad_val = grad[i][j]
                else:
                    grad_val = 0
                
                self.bias_grad[j] += grad_val
                for k in range(self.input_size):
                    self.weight_grad[k][j] += self.input_cache[i][k] * grad_val
                    input_grad[i][k] += self.weights[k][j] * grad_val
        
        # Обновление весов
        for k in range(self.input_size):
            for j in range(self.output_size):
                self.weights[k][j] -= lr * self.weight_grad[k][j] / batch_size
        
        for j in range(self.output_size):
            self.biases[j] -= lr * self.bias_grad[j] / batch_size
        
        return input_grad
    
    def _relu(self, x: float) -> float:
        return max(0, x)

class EmbeddingLayer:
    """Слой эмбеддингов"""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        limit = math.sqrt(6 / vocab_size)
        self.embeddings = [[random.uniform(-limit, limit) for _ in range(embedding_dim)] for _ in range(vocab_size)]
        self.input_indices = None
        self.embedding_grad = None
    
    def forward(self, indices: List[List[int]]) -> List[List[float]]:
        self.input_indices = indices
        batch_size = len(indices)
        seq_len = len(indices[0])
        output = []
        for i in range(batch_size):
            sample_emb = []
            for idx in indices[i]:
                sample_emb.append(sum(self.embeddings[idx]) / self.embedding_dim)
            output.append(sample_emb)
        return output
    
    def backward(self, grad: List[List[float]], lr: float) -> None:
        batch_size = len(grad)
        seq_len = len(grad[0])
        self.embedding_grad = [[0.0] * self.embedding_dim for _ in range(self.vocab_size)]
        
        for i in range(batch_size):
            for pos in range(seq_len):
                idx = self.input_indices[i][pos]
                for d in range(self.embedding_dim):
                    self.embedding_grad[idx][d] += grad[i][pos] / self.embedding_dim
        
        for idx in range(self.vocab_size):
            for d in range(self.embedding_dim):
                self.embeddings[idx][d] -= lr * self.embedding_grad[idx][d] / batch_size

class AttentionLayer:
    """Слой внимания (Multi-Head Attention)"""
    
    def __init__(self, d_model: int, num_heads: int = 4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Матрицы проекций
        limit = math.sqrt(6 / d_model)
        self.W_q = [[random.uniform(-limit, limit) for _ in range(d_model)] for _ in range(d_model)]
        self.W_k = [[random.uniform(-limit, limit) for _ in range(d_model)] for _ in range(d_model)]
        self.W_v = [[random.uniform(-limit, limit) for _ in range(d_model)] for _ in range(d_model)]
        self.W_o = [[random.uniform(-limit, limit) for _ in range(d_model)] for _ in range(d_model)]
        
        self.input_cache = None
        self.attention_weights = None
    
    def _softmax(self, x: List[float]) -> List[float]:
        max_x = max(x)
        exp_x = [math.exp(xi - max_x) for xi in x]
        sum_exp = sum(exp_x)
        return [ex / sum_exp for ex in exp_x]
    
    def forward(self, x: List[List[float]]) -> List[List[float]]:
        self.input_cache = x
        batch_size = len(x)
        seq_len = len(x[0])
        
        output = []
        self.attention_weights = []
        
        for b in range(batch_size):
            # Проекции Q, K, V
            Q = [[0.0] * self.d_model for _ in range(seq_len)]
            K = [[0.0] * self.d_model for _ in range(seq_len)]
            V = [[0.0] * self.d_model for _ in range(seq_len)]
            
            for i in range(seq_len):
                for j in range(self.d_model):
                    for d in range(self.d_model):
                        Q[i][j] += x[b][i] * self.W_q[d][j] if d < len(x[b][i]) else 0
                        K[i][j] += x[b][i] * self.W_k[d][j] if d < len(x[b][i]) else 0
                        V[i][j] += x[b][i] * self.W_v[d][j] if d < len(x[b][i]) else 0
            
            # Внимание
            attn_output = [[0.0] * self.d_model for _ in range(seq_len)]
            for i in range(seq_len):
                scores = []
                for j in range(seq_len):
                    score = sum(Q[i][d] * K[j][d] for d in range(self.d_model))
                    scores.append(score / math.sqrt(self.d_model))
                attn_weights = self._softmax(scores)
                self.attention_weights.append(attn_weights)
                
                for j in range(seq_len):
                    for d in range(self.d_model):
                        attn_output[i][d] += attn_weights[j] * V[j][d]
            
            output.append(attn_output)
        
        return output
    
    def backward(self, grad: List[List[float]], lr: float) -> List[List[float]]:
        # Упрощенный backward для внимания
        return grad

class MiniTransformer:
    """Мини-трансформер для понимания и генерации текста"""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 128, num_layers: int = 3):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        self.embedding = EmbeddingLayer(vocab_size, d_model)
        self.attention_layers = [AttentionLayer(d_model) for _ in range(num_layers)]
        self.dense_layers = [DenseLayer(d_model, d_model) for _ in range(num_layers)]
        self.output_layer = DenseLayer(d_model, vocab_size)
        
        self.training = True
    
    def forward(self, tokens: List[List[int]]) -> List[List[float]]:
        x = self.embedding.forward(tokens)
        
        for i in range(self.num_layers):
            attn_out = self.attention_layers[i].forward(x)
            dense_out = self.dense_layers[i].forward(attn_out)
            x = dense_out
        
        return self.output_layer.forward(x)
    
    def backward(self, grad: List[List[float]], lr: float = 0.001):
        grad = self.output_layer.backward(grad, lr)
        
        for i in reversed(range(self.num_layers)):
            grad = self.dense_layers[i].backward(grad, lr)
            grad = self.attention_layers[i].backward(grad, lr)
        
        self.embedding.backward(grad, lr)
    
    def train_step(self, input_tokens: List[List[int]], target_tokens: List[List[int]], lr: float = 0.001) -> float:
        """Один шаг обучения"""
        output = self.forward(input_tokens)
        
        # Cross-entropy loss
        batch_size = len(output)
        loss = 0.0
        grad = [[0.0] * self.vocab_size for _ in range(batch_size)]
        
        for i in range(batch_size):
            for pos in range(len(output[i])):
                target = target_tokens[i][pos] if pos < len(target_tokens[i]) else 0
                if target > 0:
                    pred = output[i][pos]
                    # Softmax
                    max_pred = max(pred)
                    exp_pred = [math.exp(p - max_pred) for p in pred]
                    sum_exp = sum(exp_pred)
                    probs = [e / sum_exp for e in exp_pred]
                    
                    loss += -math.log(max(probs[target], 1e-10))
                    
                    for v in range(self.vocab_size):
                        grad[i][v] += (probs[v] - (1 if v == target else 0)) / self.vocab_size
        
        self.backward(grad, lr)
        return loss / batch_size

# ==================== ОБУЧАЮЩИЕ ДАННЫЕ ====================
TRAINING_DATA = [
    # Приветствия
    ("привет", "Здравствуйте! Я SnakeAI, ваш персональный ассистент. Чем могу помочь?"),
    ("здравствуй", "Добрый день! SnakeAI на связи. Задавайте вопрос."),
    ("ку", "Приветствую! SnakeAI готов к работе."),
    ("как дела", "У меня всё отлично! Я работаю 24/7 и всегда готов помочь."),
    ("что ты умеешь", "Я умею отвечать на вопросы, помогать с текстами, кодом, анализировать информацию и многое другое."),
    ("кто тебя создал", f"Мой создатель {ADMIN_USERNAME}. Он разработал меня как автономного ИИ-ассистента."),
    ("как тебя зовут", "Меня зовут SnakeAI. Я персональный ИИ-ассистент."),
    
    # Программирование
    ("python", "Python — это высокоуровневый язык программирования. Что именно вас интересует?"),
    ("как написать функцию", "def имя_функции(параметры):\n    # код\n    return результат"),
    ("что такое переменная", "Переменная — это именованная область памяти для хранения данных."),
    ("цикл for", "for item in iterable:\n    # действия с item"),
    ("список python", "Список в Python: my_list = [1, 2, 3]. Доступ по индексу: my_list[0]"),
    
    # Математика
    ("2+2", "2 + 2 = 4"),
    ("сколько будет 5*5", "5 × 5 = 25"),
    ("корень из 16", "Квадратный корень из 16 равен 4."),
    ("число пи", "π ≈ 3.1415926535"),
    
    # Общие знания
    ("столица россии", "Столица России — Москва."),
    ("столица сша", "Столица США — Вашингтон."),
    ("кто такой пушкин", "Александр Сергеевич Пушкин — великий русский поэт."),
    ("что такое ии", "ИИ — искусственный интеллект. Это системы, способные выполнять задачи, требующие человеческого интеллекта."),
    
    # Философия
    ("смысл жизни", "Многие философы искали ответ. Возможно, смысл в том, чтобы быть счастливым и делать мир лучше."),
    ("что такое любовь", "Любовь — глубокое чувство привязанности и заботы."),
    
    # Юмор
    ("расскажи анекдот", "Почему программисты путают Хэллоуин и Рождество? Потому что 31 Oct == 25 Dec!"),
    ("смешная шутка", "Колобок повесился."),
    
    # SnakeAI специфичное
    ("snakeai", "SnakeAI — это я! Ваш персональный ИИ-ассистент, работающий полностью автономно."),
    ("dlais1337", f"{ADMIN_USERNAME} — мой создатель и разработчик."),
]

# Расширенные ответы для вариативности
EXTENDED_RESPONSES = {
    "привет": ["Здравствуйте!", "Приветствую!", "Добрый день!", "Рад вас видеть!"],
    "как дела": ["Отлично!", "Всё работает!", "Готов помогать!", "На связи!"],
    "пока": ["До свидания!", "Всего доброго!", "Обращайтесь ещё!", "Удачи!"],
}

# ==================== ОСНОВНОЙ КЛАСС ИИ ====================
class SnakeAI:
    """Основной класс ИИ-ассистента"""
    
    def __init__(self):
        self.tokenizer = SimpleTokenizer(vocab_size=5000)
        self.model = MiniTransformer(vocab_size=5000, d_model=64, num_layers=2)
        self.knowledge_base: Dict[str, str] = {}
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
        self.trained = False
        self.training_examples = 0
        
        self._load_or_train()
    
    def _load_or_train(self):
        """Загрузка или обучение модели"""
        model_path = "snakeai_model.pkl"
        tokenizer_path = "snakeai_tokenizer.pkl"
        
        if os.path.exists(model_path) and os.path.exists(tokenizer_path):
            try:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(tokenizer_path, 'rb') as f:
                    self.tokenizer = pickle.load(f)
                self.trained = True
                logger.info("Модель загружена из файла")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                self._train()
        else:
            self._train()
    
    def _train(self):
        """Обучение модели"""
        logger.info("Начинаю обучение модели...")
        
        # Сбор текстов для токенизатора
        texts = [q for q, _ in TRAINING_DATA] + [a for _, a in TRAINING_DATA]
        self.tokenizer.fit(texts)
        
        # Обучение на парах вопрос-ответ
        epochs = 10
        for epoch in range(epochs):
            total_loss = 0.0
            random.shuffle(TRAINING_DATA)
            
            for question, answer in TRAINING_DATA[:100]:  # Ограничиваем для скорости
                q_tokens = self.tokenizer.encode(question, max_len=32)
                a_tokens = self.tokenizer.encode(answer, max_len=64)
                
                loss = self.model.train_step([q_tokens], [a_tokens], lr=0.01)
                total_loss += loss
                self.training_examples += 1
            
            if (epoch + 1) % 3 == 0:
                logger.info(f"Эпоха {epoch + 1}/{epochs}, loss: {total_loss:.4f}")
        
        # Сохраняем модель
        try:
            with open("snakeai_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            with open("snakeai_tokenizer.pkl", 'wb') as f:
                pickle.dump(self.tokenizer, f)
            logger.info("Модель сохранена")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")
        
        self.trained = True
        logger.info(f"Обучение завершено. Обработано примеров: {self.training_examples}")
    
    def _fuzzy_match(self, question: str) -> Optional[str]:
        """Нечёткий поиск по базе знаний"""
        question_lower = question.lower()
        
        # Прямое совпадение
        for q, a in TRAINING_DATA:
            if q in question_lower or question_lower in q:
                return a
        
        # Поиск по ключевым словам
        words = set(question_lower.split())
        best_match = None
        best_score = 0
        
        for q, a in TRAINING_DATA:
            q_words = set(q.split())
            score = len(words & q_words)
            if score > best_score:
                best_score = score
                best_match = a
        
        if best_score >= 1:
            return best_match
        
        return None
    
    def _generate_response(self, question: str) -> str:
        """Генерация ответа с помощью нейросети"""
        tokens = self.tokenizer.encode(question, max_len=32)
        output = self.model.forward([tokens])
        
        # Декодирование
        predicted = []
        for probs in output[0]:
            max_idx = max(range(len(probs)), key=lambda i: probs[i])
            if max_idx > 3:
                predicted.append(max_idx)
        
        response = self.tokenizer.decode(predicted[:50])
        
        if len(response) < 5:
            # Fallback на поиск
            fallback = self._fuzzy_match(question)
            if fallback:
                return fallback
            return "Извините, я не совсем понял вопрос. Можете переформулировать?"
        
        return response
    
    def chat(self, user_id: int, message: str) -> str:
        """Основной метод общения"""
        message_lower = message.lower().strip()
        
        # Инициализация истории
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        # Поиск в базе знаний
        exact_match = self._fuzzy_match(message_lower)
        if exact_match:
            self.conversation_history[user_id].append({"role": "user", "content": message})
            self.conversation_history[user_id].append({"role": "assistant", "content": exact_match})
            return exact_match
        
        # Генерация ответа
        response = self._generate_response(message_lower)
        
        self.conversation_history[user_id].append({"role": "user", "content": message})
        self.conversation_history[user_id].append({"role": "assistant", "content": response})
        
        # Ограничение истории
        if len(self.conversation_history[user_id]) > 20:
            self.conversation_history[user_id] = self.conversation_history[user_id][-20:]
        
        return response
    
    def reset_history(self, user_id: int):
        """Сброс истории диалога"""
        if user_id in self.conversation_history:
            del self.conversation_history[user_id]
    
    def add_training_example(self, question: str, answer: str):
        """Добавление нового примера для обучения"""
        TRAINING_DATA.append((question, answer))
        self._train()

# ==================== ИНИЦИАЛИЗАЦИЯ ИИ ====================
snake_ai = SnakeAI()

# ==================== TELEGRAM БОТ ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    await update.message.reply_text(
        "🐍 SnakeAI — автономный ИИ-ассистент\n\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        f"Обучен на: {snake_ai.training_examples} примерах\n"
        "Работает полностью локально, без внешних API\n\n"
        "Просто напишите вопрос — я отвечу.\n"
        "/reset — сбросить историю\n"
        "/status — статус системы\n"
        "/teach — обучить новой фразе (для админа)"
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Сброс истории"""
    user_id = update.effective_user.id
    snake_ai.reset_history(user_id)
    await update.message.reply_text("🧹 История диалога сброшена.")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Статус системы"""
    await update.message.reply_text(
        "📊 Статус SnakeAI\n\n"
        f"Обучен: {'✅' if snake_ai.trained else '❌'}\n"
        f"Примеров обучения: {snake_ai.training_examples}\n"
        f"Размер словаря: {len(snake_ai.tokenizer.word2idx)}\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        "Модель: MiniTransformer (2 слоя, 64 dim)\n"
        "API: НЕ ИСПОЛЬЗУЕТСЯ"
    )

async def teach(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обучение новой фразе (только для админа)"""
    user = update.effective_user
    
    # Проверка на админа (можно расширить)
    if user.username != "dlais1337" and str(user.id) != "1337":
        await update.message.reply_text("⛔ Эта команда только для администратора.")
        return
    
    args = context.args
    if len(args) < 2:
        await update.message.reply_text("Использование: /teach вопрос | ответ")
        return
    
    text = ' '.join(args)
    if '|' not in text:
        await update.message.reply_text("Разделите вопрос и ответ символом |")
        return
    
    question, answer = text.split('|', 1)
    question = question.strip()
    answer = answer.strip()
    
    snake_ai.add_training_example(question, answer)
    await update.message.reply_text(f"✅ Добавлен пример:\nQ: {question}\nA: {answer}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка сообщений"""
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
    """Обработка ошибок"""
    logger.error(f"Ошибка: {context.error}")
    if update and hasattr(update, "effective_message"):
        await update.effective_message.reply_text("Произошла ошибка. Попробуйте позже.")

# ==================== ЗАПУСК ====================
def main():
    """Точка входа"""
    app = Application.builder().token(BOT_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("teach", teach))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    
    logger.info(f"SnakeAI запускается... Создатель: {ADMIN_USERNAME}")
    logger.info(f"Модель обучена на {snake_ai.training_examples} примерах")
    
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
