import logging
import time
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from peft import PeftModel
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 📦 Загрузка модели
logger.info("📦 Загружается tokenizer и base model...")
base_model_path = "/home/mikhail/models/mistral-7b"
lora_path = "/home/mikhail/dostoevsky-texts/lora-mistral-dostoevsky/checkpoint-500"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    torch_dtype=torch.float16,
)
logger.info("🔗 Подключение LoRA адаптера...")
model = PeftModel.from_pretrained(model, lora_path)
logger.info(f"✅ LoRA подключена из: {lora_path}")


# 🤖 Telegram bot
API_TOKEN = "7996316253:AAE-oxy-EZBXC-m5W2Mg4UWXywiMSGWRPVk"  # ← замени на свой токен
bot = Bot(token=API_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
# 🎯 Генератор
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    return_full_text=False  # Только сгенерированный ответ
)
def generate_response(prompt: str) -> str:
    logger.info(f"🔷 Генерация началась. Prompt: {prompt}")
    input_text = f"User: {prompt}\nBot:"

    output = generator(
        input_text,
        max_new_tokens=200,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2
    )

    response_text = output[0]["generated_text"].strip()
    logger.info(f"📝 Generated raw: {response_text}")

    # ✂️ Обрезаем по меткам, чтобы не было "User:" или лишнего "Bot:"
    import re
    match = re.search(r"^(.*?)(User:|Bot:|\n|$)", response_text, re.DOTALL)
    response = match.group(1).strip() if match else "🤖 Я вас не понял."

    return response

@dp.message(F.text)
async def handle_message(message: Message):
    logger.info(f"📥 Сообщение от пользователя: {message.text}")
    await message.answer("✍️ Думаю...")

    try:
        response = generate_response(message.text)
        await message.answer(response)
    except Exception as e:
        logger.error(f"❌ Ошибка генерации: {e}")
        await message.answer("Произошла ошибка при генерации ответа.")


async def main():
    logger.info("📡 Бот запущен.")
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

