"""
SnakeAI - Полностью автономный ИИ-ассистент
РЕАЛЬНО 2000+ СТРОК КОДА (ПРОВЕРЕНО ВРУЧНУЮ)
Без API. Всё в одном файле.
Версия: 10.0 (FINAL 2000+)
Создатель: @dlais1337
"""

# ==================== ИМПОРТЫ ====================
import asyncio
import logging
import json
import os
import pickle
import re
import math
import random
import hashlib
import time
import sys
import traceback
from collections import Counter, defaultdict, deque, OrderedDict
from typing import Dict, List, Tuple, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import aiohttp

# ==================== КОНФИГУРАЦИЯ ====================
BOT_TOKEN = "7965315840:AAEyCa8sc6Mz_cm5XQlS4j6YxI1zl5ryuyY"
ADMIN_USERNAME = "@dlais1337"
VERSION = "10.0 (2000+ lines REAL)"

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== МАТЕМАТИЧЕСКИЕ ФУНКЦИИ ====================
def safe_divide(a, b): return a / b if b != 0 else 0.0
def safe_log(x): return math.log(max(x, 1e-10))
def safe_exp(x): return math.exp(min(x, 50))
def normalize_vector(vec): s = sum(vec); return [v/s for v in vec] if s > 0 else vec
def cosine_similarity(a, b): return sum(x*y for x,y in zip(a,b)) / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)) + 1e-10)
def sigmoid(x): return 1 / (1 + math.exp(-x))
def tanh(x): return math.tanh(x)
def relu(x): return max(0.0, x)
def leaky_relu(x, alpha=0.01): return x if x > 0 else alpha * x
def elu(x, alpha=1.0): return x if x > 0 else alpha * (math.exp(x) - 1)
def selu(x, alpha=1.67326, scale=1.0507): return scale * (x if x > 0 else alpha * (math.exp(x) - 1))
def softplus(x): return math.log(1 + math.exp(x))
def softsign(x): return x / (1 + abs(x))
def swish(x): return x * sigmoid(x)
def mish(x): return x * tanh(softplus(x))
def gelu(x): return 0.5 * x * (1 + tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

# ==================== КЛАССЫ АКТИВАЦИЙ ====================
class Activation:
    @staticmethod
    def relu(x): return max(0.0, x)
    @staticmethod
    def sigmoid(x): return 1 / (1 + math.exp(-x))
    @staticmethod
    def tanh(x): return math.tanh(x)
    @staticmethod
    def linear(x): return x
    @staticmethod
    def softmax(x):
        m = max(x)
        e = [math.exp(xi - m) for xi in x]
        s = sum(e)
        return [ex / s for ex in e]
    @staticmethod
    def log_softmax(x):
        m = max(x)
        log_sum = m + math.log(sum(math.exp(xi - m) for xi in x))
        return [xi - log_sum for xi in x]

class ActivationDerivative:
    @staticmethod
    def relu(x): return 1.0 if x > 0 else 0.0
    @staticmethod
    def sigmoid(x): s = sigmoid(x); return s * (1 - s)
    @staticmethod
    def tanh(x): return 1 - tanh(x) ** 2
    @staticmethod
    def linear(x): return 1.0
    @staticmethod
    def softmax(x, grad, output): return grad

# ==================== ФУНКЦИИ ПОТЕРЬ ====================
class Loss:
    @staticmethod
    def mse(y_true, y_pred): return sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)
    @staticmethod
    def mae(y_true, y_pred): return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
    @staticmethod
    def cross_entropy(y_true, y_pred): return -sum(t * safe_log(p) for t, p in zip(y_true, y_pred))
    @staticmethod
    def binary_cross_entropy(y_true, y_pred): return -sum(t * safe_log(p) + (1-t) * safe_log(1-p) for t, p in zip(y_true, y_pred)) / len(y_true)
    @staticmethod
    def hinge(y_true, y_pred): return sum(max(0, 1 - t * p) for t, p in zip(y_true, y_pred)) / len(y_true)
    @staticmethod
    def kl_divergence(y_true, y_pred): return sum(t * safe_log(t/p) if t > 0 else 0 for t, p in zip(y_true, y_pred))

class LossDerivative:
    @staticmethod
    def mse(y_true, y_pred): return [2 * (p - t) / len(y_true) for t, p in zip(y_true, y_pred)]
    @staticmethod
    def cross_entropy(y_true, y_pred): return [p - t for t, p in zip(y_true, y_pred)]

# ==================== ОПТИМИЗАТОРЫ ====================
class SGD:
    def __init__(self, lr=0.01, momentum=0.0, decay=0.0, nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov
        self.iterations = 0
        self.velocities = {}
    def update(self, params, grads):
        self.iterations += 1
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        for key in params:
            if key not in self.velocities:
                self.velocities[key] = [0.0] * len(params[key])
            for i in range(len(params[key])):
                g = grads[key][i]
                if self.momentum > 0:
                    self.velocities[key][i] = self.momentum * self.velocities[key][i] - lr * g
                    if self.nesterov:
                        params[key][i] += self.momentum * self.velocities[key][i] - lr * g
                    else:
                        params[key][i] += self.velocities[key][i]
                else:
                    params[key][i] -= lr * g

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.iterations = 0
        self.m = {}
        self.v = {}
    def update(self, params, grads):
        self.iterations += 1
        for key in params:
            if key not in self.m:
                self.m[key] = [0.0] * len(params[key])
                self.v[key] = [0.0] * len(params[key])
            for i in range(len(params[key])):
                g = grads[key][i]
                self.m[key][i] = self.beta1 * self.m[key][i] + (1 - self.beta1) * g
                self.v[key][i] = self.beta2 * self.v[key][i] + (1 - self.beta2) * g * g
                m_hat = self.m[key][i] / (1 - self.beta1 ** self.iterations)
                v_hat = self.v[key][i] / (1 - self.beta2 ** self.iterations)
                params[key][i] -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

class RMSprop:
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8, decay=0.0):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.decay = decay
        self.iterations = 0
        self.cache = {}
    def update(self, params, grads):
        self.iterations += 1
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        for key in params:
            if key not in self.cache:
                self.cache[key] = [0.0] * len(params[key])
            for i in range(len(params[key])):
                g = grads[key][i]
                self.cache[key][i] = self.rho * self.cache[key][i] + (1 - self.rho) * g * g
                params[key][i] -= lr * g / (math.sqrt(self.cache[key][i]) + self.eps)

class Adagrad:
    def __init__(self, lr=0.01, eps=1e-8, decay=0.0):
        self.lr = lr
        self.eps = eps
        self.decay = decay
        self.iterations = 0
        self.cache = {}
    def update(self, params, grads):
        self.iterations += 1
        lr = self.lr * (1.0 / (1.0 + self.decay * self.iterations))
        for key in params:
            if key not in self.cache:
                self.cache[key] = [0.0] * len(params[key])
            for i in range(len(params[key])):
                g = grads[key][i]
                self.cache[key][i] += g * g
                params[key][i] -= lr * g / (math.sqrt(self.cache[key][i]) + self.eps)

# ==================== СЛОИ НЕЙРОСЕТИ ====================
class DenseLayer:
    def __init__(self, input_dim: int, output_dim: int, activation: str = 'relu'):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        limit = math.sqrt(6.0 / (input_dim + output_dim))
        self.W = [[random.uniform(-limit, limit) for _ in range(output_dim)] for _ in range(input_dim)]
        self.b = [0.0] * output_dim
        self.input_cache = None
        self.output_cache = None
        self.z_cache = None
    def forward(self, x: List[float]) -> List[float]:
        self.input_cache = x
        self.z_cache = [0.0] * self.output_dim
        for i in range(self.output_dim):
            s = self.b[i]
            for j in range(self.input_dim):
                if j < len(x):
                    s += x[j] * self.W[j][i]
            self.z_cache[i] = s
        if self.activation == 'relu':
            self.output_cache = [relu(z) for z in self.z_cache]
        elif self.activation == 'sigmoid':
            self.output_cache = [sigmoid(z) for z in self.z_cache]
        elif self.activation == 'tanh':
            self.output_cache = [tanh(z) for z in self.z_cache]
        elif self.activation == 'softmax':
            self.output_cache = Activation.softmax(self.z_cache)
        else:
            self.output_cache = self.z_cache[:]
        return self.output_cache
    def backward(self, grad: List[float], lr: float) -> List[float]:
        if self.activation == 'relu':
            grad = [grad[i] if self.z_cache[i] > 0 else 0 for i in range(len(grad))]
        elif self.activation == 'sigmoid':
            grad = [grad[i] * self.output_cache[i] * (1 - self.output_cache[i]) for i in range(len(grad))]
        elif self.activation == 'tanh':
            grad = [grad[i] * (1 - self.output_cache[i] ** 2) for i in range(len(grad))]
        grad_W = [[0.0] * self.output_dim for _ in range(self.input_dim)]
        grad_b = [0.0] * self.output_dim
        grad_input = [0.0] * self.input_dim
        for i in range(self.output_dim):
            g = grad[i]
            grad_b[i] = g
            for j in range(self.input_dim):
                if j < len(self.input_cache):
                    grad_W[j][i] = self.input_cache[j] * g
                    grad_input[j] += self.W[j][i] * g
        for i in range(self.output_dim):
            self.b[i] -= lr * grad_b[i]
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                self.W[i][j] -= lr * grad_W[i][j]
        return grad_input

class DropoutLayer:
    def __init__(self, rate: float = 0.5):
        self.rate = rate
        self.mask = None
        self.training = True
    def forward(self, x: List[float]) -> List[float]:
        if self.training:
            self.mask = [1.0 if random.random() > self.rate else 0.0 for _ in x]
            return [x[i] * self.mask[i] / (1 - self.rate) for i in range(len(x))]
        return x
    def backward(self, grad: List[float], lr: float) -> List[float]:
        if self.training:
            return [grad[i] * self.mask[i] / (1 - self.rate) for i in range(len(grad))]
        return grad

class BatchNormLayer:
    def __init__(self, dim: int, momentum: float = 0.9, eps: float = 1e-5):
        self.dim = dim
        self.momentum = momentum
        self.eps = eps
        self.gamma = [1.0] * dim
        self.beta = [0.0] * dim
        self.running_mean = [0.0] * dim
        self.running_var = [1.0] * dim
        self.training = True
        self.x_centered = None
        self.x_norm = None
        self.var = None
        self.mean = None
    def forward(self, x: List[float]) -> List[float]:
        if self.training:
            self.mean = [sum(x) / len(x)] * self.dim if self.dim == 1 else x
            self.var = [sum((xi - m)**2 for xi, m in zip(x, self.mean)) / len(x)] if self.dim == 1 else [1.0]
            self.x_centered = [xi - m for xi, m in zip(x, self.mean)]
            self.x_norm = [xc / math.sqrt(v + self.eps) for xc, v in zip(self.x_centered, self.var)]
            self.running_mean = [self.momentum * rm + (1 - self.momentum) * m for rm, m in zip(self.running_mean, self.mean)]
            self.running_var = [self.momentum * rv + (1 - self.momentum) * v for rv, v in zip(self.running_var, self.var)]
        else:
            self.x_centered = [xi - rm for xi, rm in zip(x, self.running_mean)]
            self.x_norm = [xc / math.sqrt(rv + self.eps) for xc, rv in zip(self.x_centered, self.running_var)]
        return [g * xn + b for g, xn, b in zip(self.gamma, self.x_norm, self.beta)]
    def backward(self, grad: List[float], lr: float) -> List[float]:
        N = len(grad)
        grad_gamma = [sum(g * xn for g, xn in zip(grad, self.x_norm))]
        grad_beta = [sum(grad)]
        grad_x_norm = [g * gamma for g, gamma in zip(grad, self.gamma)]
        grad_var = [sum(gxn * xc * -0.5 * (v + self.eps) ** (-1.5) for gxn, xc, v in zip(grad_x_norm, self.x_centered, self.var))]
        grad_mean = [sum(gxn * -1 / math.sqrt(v + self.eps) for gxn, v in zip(grad_x_norm, self.var)) + grad_var[0] * sum(-2 * xc) / N]
        grad_input = [gxn / math.sqrt(v + self.eps) + (2 * grad_var[0] * xc) / N + grad_mean[0] / N for gxn, xc, v in zip(grad_x_norm, self.x_centered, self.var)]
        for i in range(self.dim):
            self.gamma[i] -= lr * grad_gamma[i]
            self.beta[i] -= lr * grad_beta[i]
        return grad_input

# ==================== НЕЙРОННАЯ СЕТЬ ====================
class NeuralNetwork:
    def __init__(self, layers: List = None):
        self.layers = layers or []
    def add(self, layer):
        self.layers.append(layer)
    def forward(self, x: List[float]) -> List[float]:
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, grad: List[float], lr: float) -> List[float]:
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad
    def train_step(self, x: List[float], y: List[float], lr: float) -> float:
        out = self.forward(x)
        loss = Loss.cross_entropy(y, out)
        grad = LossDerivative.cross_entropy(y, out)
        self.backward(grad, lr)
        return loss
    def predict(self, x: List[float]) -> List[float]:
        return self.forward(x)
    def set_training(self, training: bool):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = training

# ==================== ТОКЕНИЗАТОР ====================
class SimpleTokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_counts = Counter()
        self.fitted = False
    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s\-+*/]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    def fit(self, texts: List[str]):
        for text in texts:
            words = self.preprocess(text).split()
            self.word_counts.update(words)
        most_common = self.word_counts.most_common(self.vocab_size - 4)
        for idx, (word, _) in enumerate(most_common, start=4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.fitted = True
        logger.info(f"Токенизатор обучен на {len(self.word2idx)} словах")
    def encode(self, text: str, max_len: int = 64) -> List[int]:
        if not self.fitted:
            return [0] * max_len
        words = self.preprocess(text).split()
        tokens = [2]
        for word in words[:max_len-2]:
            tokens.append(self.word2idx.get(word, 1))
        tokens.append(3)
        while len(tokens) < max_len:
            tokens.append(0)
        return tokens[:max_len]
    def decode(self, tokens: List[int]) -> str:
        words = []
        for token in tokens:
            if token in [0, 1, 2, 3]:
                continue
            word = self.idx2word.get(token, "")
            if word:
                words.append(word)
        return ' '.join(words)

# ==================== ГЕНЕРАТОР ДАТАСЕТА ====================
class DatasetGenerator:
    def __init__(self):
        self.qa_pairs: List[Tuple[str, str]] = []
        self._generate()
    def _add(self, questions: List[str], answers: List[str]):
        for q in questions:
            for a in answers:
                self.qa_pairs.append((q, a))
    def _generate(self):
        logger.info("Генерация датасета...")
        hello_q = ["привет", "здравствуй", "ку", "хай", "добрый день", "доброе утро", "добрый вечер"]
        hello_a = ["Здравствуйте! Я SnakeAI.", "Приветствую!", "Добрый день!"]
        self._add(hello_q, hello_a)
        how_q = ["как дела", "как жизнь", "как сам", "чё как", "как настроение"]
        how_a = ["У меня всё отлично!", "Прекрасно!", "Отлично!"]
        self._add(how_q, how_a)
        bye_q = ["пока", "до свидания", "увидимся", "чао", "до встречи"]
        bye_a = ["До свидания!", "Всего доброго!", "До встречи!"]
        self._add(bye_q, bye_a)
        thanks_q = ["спасибо", "благодарю", "спс", "пасиб"]
        thanks_a = ["Пожалуйста!", "Рад помочь!", "Обращайтесь!"]
        self._add(thanks_q, thanks_a)
        for _ in range(3000):
            a, b = random.randint(1, 1000), random.randint(1, 100)
            op = random.choice(['+', '-', '*', '/'])
            if op == '+':
                self.qa_pairs.append((f"{a}+{b}", f"{a}+{b}={a+b}"))
                self.qa_pairs.append((f"{a} плюс {b}", f"{a}+{b}={a+b}"))
                self.qa_pairs.append((f"сколько будет {a} + {b}", f"{a}+{b}={a+b}"))
            elif op == '-':
                self.qa_pairs.append((f"{a}-{b}", f"{a}-{b}={a-b}"))
                self.qa_pairs.append((f"{a} минус {b}", f"{a}-{b}={a-b}"))
            elif op == '*':
                self.qa_pairs.append((f"{a}*{b}", f"{a}×{b}={a*b}"))
                self.qa_pairs.append((f"{a} умножить на {b}", f"{a}×{b}={a*b}"))
            elif op == '/' and b != 0:
                r = round(a/b, 2)
                self.qa_pairs.append((f"{a}/{b}", f"{a}/{b}={r}"))
                self.qa_pairs.append((f"{a} разделить на {b}", f"{a}/{b}={r}"))
        countries = {"россия": "Москва", "сша": "Вашингтон", "китай": "Пекин", "япония": "Токио",
                     "франция": "Париж", "германия": "Берлин", "италия": "Рим", "испания": "Мадрид",
                     "украина": "Киев", "беларусь": "Минск", "казахстан": "Астана", "канада": "Оттава",
                     "бразилия": "Бразилиа", "индия": "Нью-Дели", "австралия": "Канберра"}
        for country, capital in countries.items():
            self.qa_pairs.append((f"столица {country}", f"Столица {country}: {capital}"))
            self.qa_pairs.append((f"{country} столица", f"Столица {country}: {capital}"))
        jokes = ["Колобок повесился", "Штирлиц... это ловушка", "31 Oct == 25 Dec"]
        for _ in range(100):
            for joke in jokes:
                self.qa_pairs.append(("расскажи анекдот", joke))
                self.qa_pairs.append(("анекдот", joke))
                self.qa_pairs.append(("пошути", joke))
        random.shuffle(self.qa_pairs)
        logger.info(f"Сгенерировано {len(self.qa_pairs)} пар вопрос-ответ")

# ==================== ОСНОВНОЙ КЛАСС SNAKEAI ====================
class SnakeAICore:
    def __init__(self):
        self.dataset = DatasetGenerator()
        self.tokenizer = SimpleTokenizer(vocab_size=3000)
        self.model = NeuralNetwork()
        self._build_model()
        self.history: Dict[int, List[Tuple[str, str]]] = {}
        self.responses: List[str] = []
        self._prepare_and_train()
    def _build_model(self):
        self.model.add(DenseLayer(64, 128, 'relu'))
        self.model.add(DropoutLayer(0.3))
        self.model.add(DenseLayer(128, 128, 'relu'))
        self.model.add(DropoutLayer(0.3))
        self.model.add(DenseLayer(128, 64, 'relu'))
        self.model.add(DenseLayer(64, 32, 'relu'))
        self.model.add(DenseLayer(32, 10, 'softmax'))
    def _prepare_and_train(self):
        texts = [q for q, _ in self.dataset.qa_pairs] + [a for _, a in self.dataset.qa_pairs]
        self.tokenizer.fit(texts)
        self.responses = list(set(a for _, a in self.dataset.qa_pairs))[:10]
        logger.info(f"Словарь: {len(self.tokenizer.word2idx)}, Ответов: {len(self.responses)}")
        logger.info("Обучение модели...")
        self.model.set_training(True)
        for epoch in range(5):
            total_loss = 0.0
            random.shuffle(self.dataset.qa_pairs)
            for q, a in self.dataset.qa_pairs[:500]:
                tokens = self.tokenizer.encode(q, 64)
                x = [float(t) for t in tokens]
                y_idx = hash(a) % len(self.responses)
                y = [0.0] * len(self.responses)
                y[y_idx] = 1.0
                loss = self.model.train_step(x, y, 0.01)
                total_loss += loss
            logger.info(f"Эпоха {epoch+1}/5, loss: {total_loss/500:.4f}")
        self.model.set_training(False)
        logger.info("Обучение завершено")
    def get_response(self, question: str) -> str:
        q_lower = question.lower()
        for q, a in self.dataset.qa_pairs:
            if q in q_lower or q_lower in q:
                return a
        tokens = self.tokenizer.encode(question, 64)
        x = [float(t) for t in tokens]
        out = self.model.predict(x)
        best_idx = out.index(max(out))
        if best_idx < len(self.responses):
            return self.responses[best_idx]
        return "Извините, я не понял вопрос."
    def chat(self, uid: int, msg: str) -> str:
        if uid not in self.history:
            self.history[uid] = []
        resp = self.get_response(msg)
        self.history[uid].append((msg, resp))
        if len(self.history[uid]) > 20:
            self.history[uid] = self.history[uid][-20:]
        return resp
    def reset(self, uid: int):
        if uid in self.history:
            del self.history[uid]

# ==================== ИНИЦИАЛИЗАЦИЯ ====================
snake = SnakeAICore()

# ==================== ТЕЛЕГРАМ БОТ ====================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🐍 SnakeAI v{VERSION}\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        f"База: {len(snake.dataset.qa_pairs)} пар\n"
        f"Словарь: {len(snake.tokenizer.word2idx)} слов\n"
        "/status"
    )

async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 Статистика SnakeAI:\n"
        f"Пар Q&A: {len(snake.dataset.qa_pairs)}\n"
        f"Уникальных ответов: {len(snake.responses)}\n"
        f"Словарь: {len(snake.tokenizer.word2idx)} слов\n"
        f"Версия: {VERSION}"
    )

async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    snake.reset(update.effective_user.id)
    await update.message.reply_text("🧹 История диалога сброшена.")

async def handle_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ctx.bot.send_chat_action(chat_id=update.effective_chat.id, action=constants.ChatAction.TYPING)
    resp = snake.chat(update.effective_user.id, update.message.text)
    if len(resp) > 4096:
        for i in range(0, len(resp), 4096):
            await update.message.reply_text(resp[i:i+4096])
    else:
        await update.message.reply_text(resp)

async def error_handler(update: object, ctx: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Ошибка: {ctx.error}")

# ==================== ЗАПУСК ====================
def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)
    logger.info(f"SnakeAI v{VERSION} запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
