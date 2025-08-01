## 🧠 Dostoevsky AI Telegram Bot

Интеллектуальный бот, дообученный на стиле Достоевского.

### 📂 Структура

- `bot.py` — Telegram-бот и генерация ответов
- `train.py` — обучение LoRA на датасете
- `prepare_dataset.py` — сбор и подготовка данных
- `db.py` — (если используется) логика базы
- `dostoevsky_dataset.jsonl` — корпус для обучения
- `lora-mistral-dostoevsky/` — директория с чекпоинтами LoRA

### ⚙️ Установка окружения

```bash
python3.12 -m venv training-env
source training-env/bin/activate
pip install -r requirements.txt
