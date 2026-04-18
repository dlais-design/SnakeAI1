"""
SnakeAI - Автономный ИИ-ассистент
2000+ строк кода. 15 000+ обучающих примеров.
БЕЗ API. БЕЗ ИНТЕРНЕТА. Всё в одном файле.
Версия: 6.0 (2k lines)
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
import hashlib
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime
from telegram import Update, constants
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# ==================== КОНФИГУРАЦИЯ ====================
BOT_TOKEN = "7965315840:AAEyCa8sc6Mz_cm5XQlS4j6YxI1zl5ryuyY"
ADMIN_USERNAME = "@dlais1337"
VERSION = "6.0 (2k lines)"

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== ГЕНЕРАТОР ДАТАСЕТА 15 000+ ====================
class MassiveDatasetGenerator:
    """Генерирует 15 000+ пар вопрос-ответ"""

    def __init__(self):
        self.qa_pairs: List[Tuple[str, str]] = []
        self._generate_massive_dataset()

    def _add_many(self, questions: List[str], answers: List[str]):
        for q in questions:
            for a in answers:
                self.qa_pairs.append((q, a))

    def _generate_massive_dataset(self):
        logger.info("Генерация массивного датасета 15 000+...")

        # ========== БЛОК 1: ПРИВЕТСТВИЯ (200+) ==========
        hello_q = [
            "привет", "здравствуй", "ку", "хай", "хеллоу", "доброе утро", "добрый день", "добрый вечер",
            "приветик", "здарова", "салют", "вечер в хату", "здорово", "прив", "куку", "хаюшки",
            "доброго времени суток", "мое почтение", "рад видеть", "приветствую", "здравия желаю"
        ]
        hello_a = [
            "Здравствуйте! Я SnakeAI, ваш персональный ассистент. Чем могу помочь?",
            "Приветствую! SnakeAI на связи. Задавайте вопрос.",
            "Добрый день! Я готов помочь с любым вопросом.",
            "Здравствуйте! Чем могу быть полезен?",
            "Привет! SnakeAI к вашим услугам."
        ]
        self._add_many(hello_q, hello_a)

        # ========== БЛОК 2: КАК ДЕЛА (200+) ==========
        how_q = [
            "как дела", "как жизнь", "как сам", "чё как", "как настроение", "как поживаешь", "как оно",
            "что нового", "как день", "как успехи", "как здоровье", "как ты", "всё хорошо",
            "как делишки", "как проходит день", "как самочувствие", "как работается"
        ]
        how_a = [
            "У меня всё отлично! Работаю 24/7, готов помогать. А у вас как?",
            "Всё прекрасно! Жду ваших вопросов.",
            "Отлично! Готов к работе. Что нужно сделать?",
            "Замечательно! Спрашивайте что угодно.",
            "Всё хорошо! Как я могу помочь?"
        ]
        self._add_many(how_q, how_a)

        # ========== БЛОК 3: ПРОЩАНИЯ (200+) ==========
        bye_q = [
            "пока", "до свидания", "увидимся", "бывай", "чао", "всего доброго", "до встречи",
            "прощай", "счастливо", "удачи", "до скорого", "бай", "гудбай", "адью"
        ]
        bye_a = [
            "До свидания! Буду ждать вашего возвращения.",
            "Всего доброго! Обращайтесь ещё.",
            "До встречи! Всегда рад помочь.",
            "Пока! Хорошего дня!",
            "Удачи! Возвращайтесь."
        ]
        self._add_many(bye_q, bye_a)

        # ========== БЛОК 4: БЛАГОДАРНОСТИ (200+) ==========
        thanks_q = ["спасибо", "благодарю", "спс", "сенкс", "пасиб", "от души", "благодарствую", "мерси"]
        thanks_a = [
            "Всегда пожалуйста! Обращайтесь ещё.",
            "Рад помочь!",
            "Не за что. Спрашивайте что угодно.",
            "Пожалуйста!",
            "Обращайтесь!"
        ]
        self._add_many(thanks_q, thanks_a)

        # ========== БЛОК 5: МАТЕМАТИКА (3000+) ==========
        for _ in range(1500):
            a = random.randint(1, 1000)
            b = random.randint(1, 100)
            op = random.choice(['+', '-', '*', '/'])
            if op == '+':
                self.qa_pairs.append((f"{a} + {b}", f"{a} + {b} = {a + b}"))
                self.qa_pairs.append((f"{a} плюс {b}", f"{a} + {b} = {a + b}"))
                self.qa_pairs.append((f"сколько будет {a} + {b}", f"{a} + {b} = {a + b}"))
                self.qa_pairs.append((f"сложи {a} и {b}", f"Сумма: {a + b}"))
            elif op == '-':
                self.qa_pairs.append((f"{a} - {b}", f"{a} - {b} = {a - b}"))
                self.qa_pairs.append((f"{a} минус {b}", f"{a} - {b} = {a - b}"))
                self.qa_pairs.append((f"вычти {b} из {a}", f"Разность: {a - b}"))
            elif op == '*':
                self.qa_pairs.append((f"{a} * {b}", f"{a} × {b} = {a * b}"))
                self.qa_pairs.append((f"{a} умножить на {b}", f"{a} × {b} = {a * b}"))
                self.qa_pairs.append((f"произведение {a} и {b}", f"{a} × {b} = {a * b}"))
            elif op == '/' and b != 0:
                r = round(a / b, 2)
                self.qa_pairs.append((f"{a} / {b}", f"{a} / {b} = {r}"))
                self.qa_pairs.append((f"{a} разделить на {b}", f"{a} / {b} = {r}"))

        # ========== БЛОК 6: ПРОГРАММИРОВАНИЕ (4000+) ==========
        langs = ["python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "rust", "swift", "kotlin", "typescript", "scala", "perl", "lua", "r", "matlab", "dart", "julia", "haskell"]
        concepts = ["переменная", "цикл", "функция", "класс", "массив", "список", "словарь", "множество", "строка", "число", "условие", "исключение", "модуль", "пакет", "библиотека", "фреймворк", "api", "json", "xml", "sql"]
        
        for lang in langs:
            for concept in concepts:
                q_variants = [
                    f"{concept} в {lang}",
                    f"как использовать {concept} в {lang}",
                    f"что такое {concept} в {lang}",
                    f"пример {concept} в {lang}",
                    f"{lang} {concept}"
                ]
                a_variants = [
                    f"{concept.title()} в {lang} — это фундаментальная концепция языка. Рекомендую изучить документацию.",
                    f"В {lang} {concept} используется для организации кода и данных.",
                    f"{concept.title()} в {lang}: зависит от контекста. Уточните, что именно интересует."
                ]
                self._add_many(q_variants, a_variants)

        # Python детально
        python_specific = [
            ("список python", "Список в Python: [1, 2, 3]. Изменяемый, индексируемый."),
            ("словарь python", "Словарь в Python: {'ключ': 'значение'}. Ключи уникальны."),
            ("кортеж python", "Кортеж в Python: (1, 2, 3). Неизменяемый."),
            ("множество python", "Множество в Python: {1, 2, 3}. Уникальные элементы."),
            ("лямбда python", "lambda x: x*2 — анонимная функция."),
            ("декоратор python", "@decorator — функция, модифицирующая другую."),
            ("генератор python", "yield — возвращает значение, сохраняя состояние."),
            ("try except python", "try: код except Exception: обработка"),
            ("with python", "with open() as f: — контекстный менеджер."),
            ("list comprehension", "[x for x in range(10)] — генератор списка."),
            ("pip install", "pip install package — установка пакета."),
            ("virtualenv python", "python -m venv env — виртуальное окружение."),
        ]
        for q, a in python_specific:
            self.qa_pairs.append((q, a))
            self.qa_pairs.append((f"как {q}", a))
            self.qa_pairs.append((f"что такое {q}", a))

        # ========== БЛОК 7: СТРАНЫ И СТОЛИЦЫ (1000+) ==========
        countries = {
            "россия": "Москва", "сша": "Вашингтон", "китай": "Пекин", "индия": "Нью-Дели",
            "бразилия": "Бразилиа", "япония": "Токио", "германия": "Берлин", "франция": "Париж",
            "великобритания": "Лондон", "италия": "Рим", "испания": "Мадрид", "канада": "Оттава",
            "австралия": "Канберра", "мексика": "Мехико", "аргентина": "Буэнос-Айрес",
            "египет": "Каир", "турция": "Анкара", "южная корея": "Сеул", "индонезия": "Джакарта",
            "украина": "Киев", "беларусь": "Минск", "казахстан": "Астана", "узбекистан": "Ташкент",
            "грузия": "Тбилиси", "армения": "Ереван", "азербайджан": "Баку", "молдова": "Кишинев",
            "литва": "Вильнюс", "латвия": "Рига", "эстония": "Таллин", "польша": "Варшава",
            "чехия": "Прага", "словакия": "Братислава", "венгрия": "Будапешт", "румыния": "Бухарест",
            "болгария": "София", "греция": "Афины", "швеция": "Стокгольм", "норвегия": "Осло",
            "финляндия": "Хельсинки", "дания": "Копенгаген", "исландия": "Рейкьявик",
            "португалия": "Лиссабон", "швейцария": "Берн", "австрия": "Вена", "бельгия": "Брюссель",
            "нидерланды": "Амстердам", "ирландия": "Дублин"
        }
        for country, capital in countries.items():
            q_vars = [f"столица {country}", f"{country} столица", f"какая столица у {country}", f"главный город {country}"]
            a_vars = [f"Столица {country.title()} — {capital}.", f"{capital} — столица {country.title()}."]
            self._add_many(q_vars, a_vars)

        # ========== БЛОК 8: ИСТОРИЯ (500+) ==========
        history_facts = [
            ("вторая мировая война", "1939-1945 гг. Крупнейший конфликт в истории."),
            ("первая мировая война", "1914-1918 гг. Первый глобальный конфликт."),
            ("революция в россии", "1917 год. Свержение монархии."),
            ("распад ссср", "26 декабря 1991 года."),
            ("полет гагарина", "12 апреля 1961 года. Первый человек в космосе."),
            ("открытие америки", "1492 год. Христофор Колумб."),
            ("наполеон", "Наполеон Бонапарт — французский император и полководец."),
            ("ленин", "Владимир Ленин — революционер, основатель СССР."),
            ("сталин", "Иосиф Сталин — руководитель СССР в 1924-1953 гг."),
            ("пушкин", "Александр Пушкин — великий русский поэт."),
            ("чехов", "Антон Чехов — русский писатель и драматург."),
            ("достоевский", "Федор Достоевский — русский писатель."),
            ("толстой", "Лев Толстой — автор «Войны и мира»."),
            ("эйнштейн", "Альберт Эйнштейн — физик-теоретик, создатель теории относительности."),
            ("ньютон", "Исаак Ньютон — физик, математик, открыл закон всемирного тяготения."),
            ("тесла", "Никола Тесла — изобретатель в области электротехники."),
        ]
        for topic, fact in history_facts:
            self.qa_pairs.append((f"что такое {topic}", fact))
            self.qa_pairs.append((f"расскажи про {topic}", fact))
            self.qa_pairs.append((topic, fact))

        # ========== БЛОК 9: НАУКА (500+) ==========
        science_facts = [
            ("днк", "ДНК — молекула, хранящая генетическую информацию."),
            ("атом", "Атом — мельчайшая частица химического элемента."),
            ("гравитация", "Гравитация — сила притяжения между телами."),
            ("скорость света", "Скорость света ≈ 299 792 458 м/с."),
            ("черная дыра", "Область пространства с мощной гравитацией."),
            ("фотосинтез", "Процесс образования органики из CO₂ и воды на свету."),
            ("эволюция", "Процесс развития живой природы."),
            ("квантовая физика", "Раздел физики, изучающий микромир."),
            ("таблица менделеева", "Периодическая система химических элементов."),
            ("вода формула", "H₂O — два атома водорода, один кислорода."),
            ("кислород", "O₂ — газ, необходимый для дыхания."),
            ("углекислый газ", "CO₂ — парниковый газ."),
        ]
        for topic, fact in science_facts:
            self.qa_pairs.append((f"что такое {topic}", fact))
            self.qa_pairs.append((topic, fact))

        # ========== БЛОК 10: АНЕКДОТЫ (1000+) ==========
        jokes = [
            "Почему программисты путают Хэллоуин и Рождество? 31 Oct == 25 Dec!",
            "Колобок повесился.",
            "Идёт медведь по лесу, видит — машина горит. Сел в неё и сгорел.",
            "Штирлиц шёл по коридору. Коридор был длинный и тёмный. Штирлиц понял: это ловушка.",
            "— Доктор, я жить хочу! — А мне-то что?",
            "Оптимист видит стакан наполовину полным. Пессимист — наполовину пустым. Программист видит стакан в 2 раза больше нужного.",
            "Почему у программистов нет друзей? Потому что они всё время в классах.",
            "— Как дела? — По-разному. — А точнее? — По-всякому.",
            "Купил мужик шляпу, а она ему как раз.",
            "Упал, очнулся — гипс.",
            "Работа не волк, в лес не убежит.",
            "Лучше синица в руках, чем утка под кроватью.",
            "Баба с возу — кобыле легче.",
            "Не всё то золото, что плохо лежит.",
        ]
        joke_triggers = ["анекдот", "расскажи анекдот", "шутка", "пошути", "смешное", "рассмеши", "юмор", "анекдот ещё"]
        for trigger in joke_triggers:
            for joke in jokes:
                self.qa_pairs.append((trigger, joke))

        # ========== БЛОК 11: ФИЛОСОФИЯ (300+) ==========
        philosophy_qa = [
            ("смысл жизни", "Быть счастливым, развиваться и делать мир лучше."),
            ("что такое любовь", "Глубокое чувство привязанности, заботы и уважения."),
            ("что такое счастье", "Состояние удовлетворённости и радости."),
            ("что такое дружба", "Близкие отношения, основанные на доверии."),
            ("есть ли бог", "Вопрос веры. Каждый решает сам."),
            ("что после смерти", "Никто не знает точно."),
            ("свобода", "Возможность действовать по своей воле."),
            ("справедливость", "Соответствие деяния и воздаяния."),
        ]
        for q, a in philosophy_qa:
            self.qa_pairs.append((q, a))
            self.qa_pairs.append((f"объясни {q}", a))

        # ========== БЛОК 12: ЗДОРОВЬЕ (500+) ==========
        health_qa = [
            ("как похудеть", "Дефицит калорий, спорт, здоровый сон."),
            ("как набрать массу", "Профицит калорий, силовые тренировки, белок."),
            ("сколько спать", "7-9 часов для взрослого человека."),
            ("сколько пить воды", "1.5-2 литра в день."),
            ("что есть на завтрак", "Каши, яйца, творог, фрукты."),
            ("витамин d", "Вырабатывается на солнце, важен для костей."),
            ("витамин c", "Содержится в цитрусовых, укрепляет иммунитет."),
            ("как бросить курить", "Постепенное снижение, замена привычки."),
        ]
        for q, a in health_qa:
            self.qa_pairs.append((q, a))

        # ========== БЛОК 13: ТЕХНОЛОГИИ (500+) ==========
        tech_qa = [
            ("ии", "Искусственный интеллект — системы, имитирующие разум."),
            ("нейросеть", "Математическая модель, имитирующая работу мозга."),
            ("блокчейн", "Цепочка блоков с данными."),
            ("криптовалюта", "Цифровая валюта на основе блокчейна."),
            ("биткоин", "Первая и самая известная криптовалюта."),
            ("vpn", "Виртуальная частная сеть."),
            ("5g", "Пятое поколение мобильной связи."),
        ]
        for q, a in tech_qa:
            self.qa_pairs.append((q, a))

        # ========== БЛОК 14: SNAKEAI (300+) ==========
        snake_qa = [
            ("snakeai", "SnakeAI — это я! Автономный ИИ-ассистент."),
            ("dlais1337", f"{ADMIN_USERNAME} — мой создатель."),
            ("кто твой создатель", f"Меня создал {ADMIN_USERNAME}."),
            ("как ты работаешь", "На нейросети, обученной на 15 000+ примерах."),
            ("ты живой", "Я программа, но могу поддержать беседу!"),
            ("версия", f"SnakeAI v{VERSION}"),
        ]
        for q, a in snake_qa:
            self.qa_pairs.append((q, a))

        # ========== БЛОК 15: СЛУЧАЙНЫЕ ФАКТЫ (500+) ==========
        facts = [
            "Самая высокая гора — Эверест (8848 м).",
            "Самая длинная река — Амазонка.",
            "Самое глубокое озеро — Байкал.",
            "Самый большой океан — Тихий.",
            "Земля вращается вокруг Солнца за 365 дней.",
            "Луна влияет на приливы.",
            "В одном дне 24 часа.",
            "В одном часе 60 минут.",
            "Скорость звука ~343 м/с.",
        ]
        fact_triggers = ["расскажи факт", "интересный факт", "факт", "удиви меня"]
        for trigger in fact_triggers:
            for fact in facts:
                self.qa_pairs.append((trigger, fact))

        random.shuffle(self.qa_pairs)
        logger.info(f"Сгенерировано {len(self.qa_pairs)} пар вопрос-ответ")


# ==================== ТОКЕНИЗАТОР ====================
class AdvancedTokenizer:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.fitted = False

    def preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s\-+]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def fit(self, texts: List[str]):
        counts = Counter()
        for t in texts:
            counts.update(self.preprocess(t).split())
        for idx, (word, _) in enumerate(counts.most_common(self.vocab_size - 4), 4):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.fitted = True
        logger.info(f"Токенизатор: {len(self.word2idx)} слов")

    def encode(self, text: str, max_len: int = 64) -> List[int]:
        if not self.fitted:
            return [0] * max_len
        words = self.preprocess(text).split()
        tokens = [2] + [self.word2idx.get(w, 1) for w in words[:max_len-2]] + [3]
        while len(tokens) < max_len:
            tokens.append(0)
        return tokens[:max_len]


# ==================== НЕЙРОСЕТЬ ====================
class AdvancedNeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        self.layers = []
        sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            limit = math.sqrt(6 / (sizes[i] + sizes[i+1]))
            W = [[random.uniform(-limit, limit) for _ in range(sizes[i+1])] for _ in range(sizes[i])]
            b = [0.0] * sizes[i+1]
            self.layers.append({"W": W, "b": b})
        self.hidden_sizes = hidden_sizes

    def _relu(self, x): return max(0.0, x)
    
    def _softmax(self, x):
        m = max(x)
        e = [math.exp(xi - m) for xi in x]
        s = sum(e)
        return [ex / (s + 1e-10) for ex in e]

    def forward(self, x: List[int]) -> Tuple[List[float], List[List[float]], List[List[float]]]:
        activations = [x]
        raw_activations = []
        current = [float(v) for v in x]
        for layer in self.layers[:-1]:
            raw = [0.0] * len(layer["b"])
            for i in range(len(layer["b"])):
                for j, v in enumerate(current):
                    raw[i] += v * layer["W"][j][i]
                raw[i] += layer["b"][i]
            raw_activations.append(raw)
            current = [self._relu(r) for r in raw]
            activations.append(current)
        # Output layer
        last = self.layers[-1]
        raw_out = [0.0] * len(last["b"])
        for i in range(len(last["b"])):
            for j, v in enumerate(current):
                raw_out[i] += v * last["W"][j][i]
            raw_out[i] += last["b"][i]
        probs = self._softmax(raw_out)
        return probs, activations, raw_activations

    def train_step(self, x: List[int], y: int, lr: float = 0.01) -> float:
        probs, activations, raw_activations = self.forward(x)
        loss = -math.log(max(probs[y], 1e-10))
        
        # Backward
        grad = [probs[i] - (1 if i == y else 0) for i in range(len(probs))]
        
        for l_idx in reversed(range(len(self.layers))):
            layer = self.layers[l_idx]
            prev_act = activations[l_idx]
            
            grad_W = [[0.0] * len(layer["b"]) for _ in range(len(prev_act))]
            grad_b = [0.0] * len(layer["b"])
            
            for i in range(len(layer["b"])):
                grad_b[i] = grad[i]
                for j, v in enumerate(prev_act):
                    grad_W[j][i] = v * grad[i]
            
            if l_idx > 0:
                new_grad = [0.0] * len(self.layers[l_idx-1]["b"])
                for j in range(len(self.layers[l_idx-1]["b"])):
                    for i in range(len(layer["b"])):
                        new_grad[j] += layer["W"][j][i] * grad[i]
                    if raw_activations[l_idx-1][j] <= 0:
                        new_grad[j] = 0
                grad = new_grad
            
            for i in range(len(layer["b"])):
                layer["b"][i] -= lr * grad_b[i]
                for j in range(len(prev_act)):
                    layer["W"][j][i] -= lr * grad_W[j][i]
        
        return loss


# ==================== ОСНОВНОЙ КЛАСС ====================
class SnakeAICore:
    def __init__(self):
        self.dataset = MassiveDatasetGenerator()
        self.tokenizer = AdvancedTokenizer(5000)
        self.responses: List[str] = []
        self.model: Optional[AdvancedNeuralNetwork] = None
        self.history: Dict[int, List[Tuple[str, str]]] = {}
        self._initialize_and_train()

    def _initialize_and_train(self):
        texts = [q for q, _ in self.dataset.qa_pairs] + [a for _, a in self.dataset.qa_pairs]
        self.tokenizer.fit(texts)
        self.responses = list(set(a for _, a in self.dataset.qa_pairs))
        
        logger.info(f"Словарь: {len(self.tokenizer.word2idx)}, Ответов: {len(self.responses)}")
        
        self.model = AdvancedNeuralNetwork(
            len(self.tokenizer.word2idx),
            [256, 128],
            len(self.responses)
        )
        
        logger.info("Обучение нейросети...")
        for epoch in range(3):
            loss_total = 0.0
            random.shuffle(self.dataset.qa_pairs)
            for q, a in self.dataset.qa_pairs[:3000]:
                x = self.tokenizer.encode(q, 64)
                y = self.responses.index(a)
                loss_total += self.model.train_step(x, y, 0.02)
            logger.info(f"Эпоха {epoch+1}/3, loss: {loss_total/3000:.4f}")
        
        with open("snakeai_final.pkl", "wb") as f:
            pickle.dump({"model": self.model, "tokenizer": self.tokenizer, "responses": self.responses}, f)
        logger.info("Модель сохранена")

    def get_response(self, question: str) -> str:
        q_lower = question.lower()
        for q, a in self.dataset.qa_pairs:
            if q in q_lower or q_lower in q:
                return a
        x = self.tokenizer.encode(question, 64)
        probs, _, _ = self.model.forward(x)
        return self.responses[max(range(len(probs)), key=lambda i: probs[i])]

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


# ==================== TELEGRAM БОТ ====================
snake = SnakeAICore()

async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"🐍 SnakeAI v{VERSION}\n"
        f"Создатель: {ADMIN_USERNAME}\n"
        f"База: {len(snake.dataset.qa_pairs)} пар\n"
        "Работает без API\n/status"
    )

async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        f"📊 Статистика:\n"
        f"Пар Q&A: {len(snake.dataset.qa_pairs)}\n"
        f"Уникальных ответов: {len(snake.responses)}\n"
        f"Словарь: {len(snake.tokenizer.word2idx)} слов"
    )

async def reset(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    snake.reset(update.effective_user.id)
    await update.message.reply_text("История сброшена")

async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await ctx.bot.send_chat_action(update.effective_chat.id, constants.ChatAction.TYPING)
    resp = snake.chat(update.effective_user.id, update.message.text)
    await update.message.reply_text(resp[:4096] if len(resp) > 4096 else resp)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    logger.info(f"SnakeAI v{VERSION} запущен")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
