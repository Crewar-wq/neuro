# 🤖 Dostoevsky Neural Bot (LoRA + Mistral 7B)

Telegram-бот, основанный на модели [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1), дообученной с помощью LoRA на текстах Ф.М. Достоевского.

---

## 📁 Структура проекта

dostoevsky-texts/
├── bot.py # Telegram-бот с генерацией ответов
├── train.py # Скрипт дообучения модели
├── prepare_dataset.py # Преобразование и подготовка датасета
├── dostoevsky_dataset.jsonl # Обучающий датасет в формате JSONL
├── lora-mistral-dostoevsky/ # Каталог с LoRA-чекпойнтами
│ ├── checkpoint-500/
│ └── ...
├── project_structure.txt # Снимок структуры проекта (через tree)
├── .gitignore # Git-игнорируемые файлы
└── README.md # Документация (этот файл)

---

## ⚙️ Установка

> ⚠️ Требуется Linux-система с CUDA 11.8+, Python ≥ 3.10 и видеокарта с ≥12 GB VRAM

```bash
# 1. Клонируем репозиторий
git clone git@github.com:Crewar-wq/neuro.git
cd neuro  # либо cd dostoevsky-texts если ты уже там

# 2. Создаём виртуальное окружение
python3 -m venv training-env
source training-env/bin/activate

# 3. Устанавливаем зависимости
pip install --upgrade pip
pip install -r requirements.txt
Если файла requirements.txt нет, можно установить вручную:

pip install transformers datasets peft accelerate bitsandbytes aiogram
🧠 Обучение (LoRA Fine-tuning)

python prepare_dataset.py    # (если требуется собрать датасет)

python train.py              # запуск обучения
✅ Чекпоинты сохраняются в lora-mistral-dostoevsky/checkpoint-*

🤖 Запуск Telegram-бота
Укажи свой Telegram API токен в bot.py:

python
API_TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
Запусти бота:

python bot.py
Бот запустит inference на основе дообученной модели и будет отвечать на сообщения.

💡 Технические особенности
Базовая модель: mistralai/Mistral-7B-Instruct-v0.1

Обучение: LoRA с 4-bit квантованием (bitsandbytes)

Токенизация: max_length=512, padding=eos_token

Сохранение каждые N шагов (checkpoint-*)

Ответы очищаются от автодиалогов по шаблонам (User:, Bot:)

📦 Дополнительно
Используется PeftModel.from_pretrained для подгрузки LoRA весов

Для запуска нужен CUDA GPU, иначе сильно замедляется

Telegram-интерфейс через aiogram v3+ (async)
