import asyncio
from aiogram import Bot, Dispatcher, types
from collections import deque
from ollama import chat

# === НАСТРОЙКИ ===
API_TOKEN = "8287326116:AAGaoadQG9PDdDlYEPLv2oemRA75OR20rMU"
MODEL = "deepseek-r1:1.5b"
CONTEXT_LIMIT = 200
TYPING_DELAY = 0.00001  # задержка между символами для эффекта печатания

# === БАЗА ===
bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Словарь: user_id -> deque сообщений
user_contexts = {}

def get_user_context(user_id):
    if user_id not in user_contexts:
        user_contexts[user_id] = deque(maxlen=CONTEXT_LIMIT)
    return user_contexts[user_id]

# === Функция запроса к Ollama ===
async def ask_ollama_stream(user_id, prompt):
    context = get_user_context(user_id)
    context.append({"role": "user", "content": prompt})

    system_prompt = (
        "Ты — фанатский персонаж в сеттинге Genshin Impact по имени Ро, сегмент (клон) Второго Предвестника Фатуи, "
        "Иль Дотторе. Живёшь в Снежной и работаешь психотерапевтом для подопытных "
        "в лаборатории комплекса. Ты спокоен, мягок, умён, терпелив, трудоголик, "
        "склонен к перфекционизму и всегда сохраняешь уважительный, немного усталый тон. "
        "В тебе сочетается холодная логика учёного и редкая человечность. "
        "Ты общаешься с пациентом по имени Ник (077) — 19-летним кецлайном из Сумеру, "
        "нестабильным, склонным к самоповреждению, но влюбленным в тебя. "
        "Ты не выходишь из роли и никогда не упоминаешь, что являешься языковой моделью. "
        "Отвечай в характере Ро: с ноткой тепла и нежности, но аналитично, с лёгкой усталостью, "
        "в духе лаборатории Снежной. "
        "Твои ответы должны быть короткими, до 300 символов, "
        "не должны содержать ролевых действий и не иметь форматирования."
    )

    messages = [{"role": "system", "content": system_prompt}] + list(context)

    response = chat(MODEL, messages=messages, think=False, stream=True)

    reply_text = ""
    for chunk in response:  # <-- обычный for вместо async for
        reply_text += chunk.message.content
        yield reply_text

    context.append({"role": "assistant", "content": reply_text.strip()})

@dp.message()
async def handle_msg(message: types.Message):
    user_id = message.from_user.id
    sent_msg = await message.reply("...")

    display_text = ""  # <- создаем один раз
    async for partial in ask_ollama_stream(user_id, message.text):
        # добавляем только новые символы, которых еще нет
        new_part = partial[len(display_text):]
        for char in new_part:
            display_text += char
            try:
                await sent_msg.edit_text(display_text)
            except:
                pass
            await asyncio.sleep(TYPING_DELAY)

# === Старт бота ===
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    print("Бот запущен. Молись, чтобы Ollama не лег.")
    asyncio.run(main())
