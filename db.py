import psycopg2
import torch
import numpy as np
import os

from transformers import AutoTokenizer, AutoModel

# Настройки подключения к PostgreSQL
DB_PARAMS = {
    "dbname": "contextdb",
    "user": "botuser",
    "password": os.getenv("PGPASSWORD", ""),
    "host": "localhost",
    "port": 15432
}

# Загружаем модель и токенизатор
MODEL_NAME = "cointegrated/rubert-tiny"  # или другой, совместимый
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Устройство
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def get_conn():
    return psycopg2.connect(**DB_PARAMS)

@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]-токен
    return embeddings.cpu().numpy()[0].tolist()

def save_message(chat_id: str, text: str):
    embedding = get_embedding(text)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (chat_id, message, embedding)
                VALUES (%s, %s, %s)
            """, (chat_id, text, embedding))

def get_similar_messages(chat_id, text):
    embedding = get_embedding(text)

    # Преобразуем список embedding в строку для SQL (например: "-0.12,0.34,1.23,...")
    embedding_str = ','.join(map(str, embedding))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT message FROM messages
                WHERE chat_id = %s
                ORDER BY embedding <-> (ARRAY[%s]::vector)
                LIMIT 5
            """ % ("%s", embedding_str), (chat_id,))
            results = cur.fetchall()
            return [r[0] for r in results]





