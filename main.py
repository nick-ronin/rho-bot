import asyncio
import tempfile
import json
from aiogram import Bot, Dispatcher, types, F
from ollama import Client
import asyncpg
import os
from collections import deque

# === НАСТРОЙКИ ===
API_TOKEN = os.environ.get("BOT_TOKEN")
MODEL = "gemma3:27b-cloud"
CONTEXT_LIMIT = 1000
DATABASE_URL = os.environ.get("DATABASE_URL")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

if not API_TOKEN:
    raise RuntimeError("Telegram BOT_TOKEN not set in environment variables")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in environment variables")
if not OLLAMA_API_KEY:
    raise RuntimeError("OLLAMA_API_KEY not set in environment variables")

# === БАЗА и Бот ===
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
db_pool = None
ollama_client = Client(host="https://ollama.com", headers={'Authorization': 'Bearer ' + os.environ.get('OLLAMA_API_KEY')})

# === Идентификация пользователей ===
NICK_ID = 823849772
CHANNEL_COMMENTS_NICK_ID = 1087968824
CHANNEL_NICK_ID = 777000
DANILIUM_ID = 779865230

NICK_PROMPT_FILE = "prompts/nick_prompt.txt"
DAN_PROMPT_FILE = "prompts/dan_prompt.txt"
COMMON_PROMPT_FILE = "prompts/common_prompt.txt"

BOT_USERNAME = "rho_segment_bot"

def load_prompt(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, encoding="utf-8") as f:
        return f.read().strip()

def is_nick(user_id: int) -> bool:
    return user_id in [NICK_ID, CHANNEL_COMMENTS_NICK_ID, CHANNEL_NICK_ID]

def is_danilium(user_id: int) -> bool:
    return user_id == DANILIUM_ID

def get_system_prompt(user_id: int) -> str:
    if is_nick(user_id):
        return load_prompt(NICK_PROMPT_FILE)
    elif is_danilium(user_id):
        return load_prompt(DAN_PROMPT_FILE)
    else:
        return load_prompt(COMMON_PROMPT_FILE)

# === Работа с контекстом через БД ===
async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    async with db_pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS contexts (
            id SERIAL PRIMARY KEY,
            chat_id BIGINT NOT NULL,
            user_id BIGINT NOT NULL,
            context_type TEXT NOT NULL,
            messages JSONB,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        CREATE UNIQUE INDEX IF NOT EXISTS contexts_unique_idx
        ON contexts (chat_id, user_id, context_type);
        """)

async def get_context(chat_id, user_id, context_type):
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT messages FROM contexts WHERE chat_id=$1 AND user_id=$2 AND context_type=$3",
            chat_id, user_id, context_type
        )
        if row:
            context = deque(json.loads(row["messages"]), maxlen=CONTEXT_LIMIT)
            return context
        return deque(maxlen=CONTEXT_LIMIT)

async def save_context(chat_id, user_id, context_type, context):
    if len(context) > CONTEXT_LIMIT:
        context = deque(list(context)[-CONTEXT_LIMIT:], maxlen=CONTEXT_LIMIT)
    async with db_pool.acquire() as conn:
        data = json.dumps(list(context))
        await conn.execute("""
            INSERT INTO contexts(chat_id, user_id, context_type, messages)
            VALUES($1, $2, $3, $4)
            ON CONFLICT(chat_id, user_id, context_type) DO UPDATE
            SET messages = $4,
            updated_at = CURRENT_TIMESTAMP
        """, chat_id, user_id, context_type, data)

# === Функции общения с Ollama ===
async def ask_ollama_stream(user_id, chat_id, prompt):
    personal = await get_context(chat_id, user_id, "personal")
    group = await get_context(chat_id, 0, "group")
    system_prompt = get_system_prompt(user_id)

    personal.append({"role": "user", "content": prompt})
    group.append({"role": "user", "user_id": user_id, "content": prompt})

    messages = [{"role": "system", "content": system_prompt}] + list(group) + list(personal)

    reply_text = ""
    response = ollama_client.chat(model=MODEL, messages=messages, stream=True)
    for chunk in response:
        delta = getattr(chunk.message, "content", "")
        if delta:
            reply_text += delta

    personal.append({"role": "assistant", "content": reply_text.strip()})
    group.append({"role": "assistant", "user_id": user_id, "content": reply_text.strip()})

    await save_context(chat_id, user_id, "personal", personal)
    await save_context(chat_id, 0, "group", group)

    return reply_text

# === Хэндлеры ===
@dp.message(F.text)
async def handle_msg(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    if message.chat.type == "private":
        respond = True
    else:
        respond = (
            user_id == CHANNEL_NICK_ID or
            (message.reply_to_message and message.reply_to_message.from_user.id == bot.id) or
            (f"@{BOT_USERNAME}" in message.text)
        )
    if not respond:
        return

    full_text = await ask_ollama_stream(user_id, chat_id, message.text)
    await message.reply(full_text, reply_to_message_id=message.message_id if message.chat.type != "private" else None)

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id

    photo = message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        await bot.download(photo, destination=tmp_path)
        prompt = message.caption or "Опиши изображение кратко и по делу."
        full_text = await ask_ollama_stream(user_id, chat_id, prompt)
        await message.reply(full_text)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# === Старт бота ===
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await init_db()
    await dp.start_polling(bot)

if __name__ == "__main__":
    print("Бот запущен. Персональные и групповые контексты сохраняются в PostgreSQL.")
    asyncio.run(main())
