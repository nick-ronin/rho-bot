import asyncio
import json
import os
import tempfile
import io
from collections import deque
from typing import List, Deque

import asyncpg
from aiogram import Bot, Dispatcher, types, F
from ollama import Client
from dotenv import load_dotenv
from PIL import Image

from asyncpg.types import Json

# ================== ENV ==================

load_dotenv()

API_TOKEN = os.getenv("BOT_TOKEN")
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not API_TOKEN or not DATABASE_URL or not OLLAMA_API_KEY:
    raise RuntimeError("Env vars missing")

# ================== MODELS ==================

MODEL = "gemma3:27b-cloud"

# ================== LIMITS ==================

PERSONAL_LIMIT = 30
GROUP_LIMIT = 30
NICK_GLOBAL_LIMIT = 20

MAX_PROMPT_CHARS = 12_000

BOT_USERNAME = "rho_segment_bot"

# ================== IDS ==================

NICK_ID = 823849772
CHANNEL_COMMENTS_NICK_ID = 1087968824
CHANNEL_NICK_ID = 777000
DANILIUM_ID = 779865230

# ================== PROMPTS ==================

def load_prompt(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()

BASE_SYSTEM_PROMPT = load_prompt("prompts/system_prompt.txt")
BASE_USER_PROMPT = load_prompt("prompts/base_user_prompt.txt")
NICK_PROMPT = load_prompt("prompts/nick_prompt.txt")
DAN_PROMPT = load_prompt("prompts/dan_prompt.txt")

# ================== BOT / CLIENT ==================

bot = Bot(API_TOKEN)
dp = Dispatcher()

ollama = Client(
    host="https://ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
)

db_pool: asyncpg.Pool | None = None

# ================== HELPERS ==================

def is_nick(user_id: int) -> bool:
    return user_id in {NICK_ID, CHANNEL_COMMENTS_NICK_ID, CHANNEL_NICK_ID}

def is_dan(user_id: int) -> bool:
    return user_id == DANILIUM_ID

def user_instruction(user_id: int) -> str | None:
    if is_nick(user_id):
        return NICK_PROMPT
    if is_dan(user_id):
        return DAN_PROMPT
    return BASE_USER_PROMPT or None

def context_limit(ctx_type: str) -> int:
    return {
        "personal": PERSONAL_LIMIT,
        "group": GROUP_LIMIT,
        "nick_global": NICK_GLOBAL_LIMIT,
    }.get(ctx_type, 20)

def trim_messages(messages: List[dict]) -> List[dict]:
    total = 0
    result = []
    for msg in reversed(messages):
        size = len(msg.get("content", ""))
        if total + size > MAX_PROMPT_CHARS:
            break
        result.append(msg)
        total += size
    return list(reversed(result))

# ================== IMAGE ==================

def resize_image(path: str) -> io.BytesIO | None:
    try:
        with Image.open(path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((512, 512))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            buf.seek(0)
            return buf
    except Exception:
        return None

# ================== DB ==================

async def init_db():
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL)
    async with db_pool.acquire() as conn:
        await conn.execute("""
        CREATE TABLE IF NOT EXISTS contexts (
            chat_id BIGINT,
            user_id BIGINT,
            context_type TEXT,
            messages JSONB,
            updated_at TIMESTAMPTZ DEFAULT now(),
            PRIMARY KEY (chat_id, user_id, context_type)
        )
        """)

async def load_context(chat_id: int, user_id: int, ctx_type: str) -> Deque:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT messages FROM contexts WHERE chat_id=$1 AND user_id=$2 AND context_type=$3",
            chat_id, user_id, ctx_type
        )
        limit = context_limit(ctx_type)
        if not row:
            return deque(maxlen=limit)
        return deque(Json(row["messages"]), maxlen=limit)

async def save_context(chat_id: int, user_id: int, ctx_type: str, ctx: Deque):
    async with db_pool.acquire() as conn:
        await conn.execute("""
        INSERT INTO contexts VALUES ($1,$2,$3,$4)
        ON CONFLICT (chat_id,user_id,context_type)
        DO UPDATE SET messages=$4, updated_at=now()
        """, chat_id, user_id, ctx_type, Json(list(ctx)))  

# ================== MESSAGE BUILD ==================

async def build_messages(user_id: int, chat_id: int, prompt: str) -> List[dict]:
    messages = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]

    instr = user_instruction(user_id)
    if instr:
        messages.append({"role": "user", "content": f"[Инструкция]\n{instr}"})

    if is_nick(user_id):
        nick_ctx = await load_context(0, NICK_ID, "nick_global")
        messages.extend(list(nick_ctx)[-5:])

    ctx_type = "personal" if chat_id == user_id else "group"
    chat_ctx = await load_context(chat_id, user_id, ctx_type)
    messages.extend(list(chat_ctx)[-5:])

    messages.append({"role": "user", "content": prompt})
    return trim_messages(messages)

# ================== TEXT ==================

async def ask_text(user_id: int, chat_id: int, prompt: str) -> str:
    messages = await build_messages(user_id, chat_id, prompt)

    response = ollama.chat(model=MODEL, messages=messages, stream=True)

    answer = ""
    for chunk in response:
        answer += getattr(chunk.message, "content", "")

    if not answer.strip():
        return "Ответ не сформирован."

    ctx_type = "personal" if chat_id == user_id else "group"
    ctx = await load_context(chat_id, user_id, ctx_type)
    ctx.append({"role": "user", "content": prompt})
    ctx.append({"role": "assistant", "content": answer})
    await save_context(chat_id, user_id, ctx_type, ctx)

    if is_nick(user_id):
        nick_ctx = await load_context(0, NICK_ID, "nick_global")
        nick_ctx.append({"role": "user", "content": prompt})
        nick_ctx.append({"role": "assistant", "content": answer})
        await save_context(0, NICK_ID, "nick_global", nick_ctx)

    return answer

# ================== IMAGE ==================

async def ask_image(user_id: int, chat_id: int, prompt: str, image_path: str) -> str:
    messages = await build_messages(user_id, chat_id, prompt)

    messages[-1] = {
        "role": "user",
        "content": prompt,
        "images": [image_path]
    }

    response = ollama.chat(model=MODEL, messages=messages, stream=True)

    answer = ""
    for chunk in response:
        answer += getattr(chunk.message, "content", "")

    if not answer.strip():
        return "Не удалось обработать изображение."

    ctx_type = "personal" if chat_id == user_id else "group"
    ctx = await load_context(chat_id, user_id, ctx_type)
    ctx.append({"role": "user", "content": "прислал изображение"})
    ctx.append({"role": "assistant", "content": answer})
    await save_context(chat_id, user_id, ctx_type, ctx)

    if is_nick(user_id):
        nick_ctx = await load_context(0, NICK_ID, "nick_global")
        nick_ctx.append({"role": "user", "content": "прислал изображение"})
        nick_ctx.append({"role": "assistant", "content": answer})
        await save_context(0, NICK_ID, "nick_global", nick_ctx)

    return answer

# ================== HANDLERS ==================

@dp.message(F.text)
async def handle_text(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    is_private = message.chat.type == "private"

    respond = (
        is_private
        or user_id == CHANNEL_NICK_ID
        or (message.reply_to_message and message.reply_to_message.from_user.id == bot.id)
        or f"@{BOT_USERNAME}" in message.text
    )
    if not respond:
        return

    await message.bot.send_chat_action(chat_id, "typing")
    answer = await ask_text(user_id, chat_id, message.text)
    await message.reply(answer)

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    is_private = message.chat.type == "private"

    respond = (
        is_private
        or user_id == CHANNEL_NICK_ID
        or (message.reply_to_message and message.reply_to_message.from_user.id == bot.id)
        or f"@{BOT_USERNAME}" in (message.caption or "")
    )
    if not respond:
        return

    photo = message.photo[-1]

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        raw_path = tmp.name

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as resized:
        resized_path = resized.name

    try:
        await bot.download(photo.file_id, destination=raw_path)
        buf = resize_image(raw_path)
        if not buf:
            await message.reply("Не удалось обработать изображение.")
            return

        with open(resized_path, "wb") as f:
            f.write(buf.getvalue())

        prompt = message.caption or "Опиши изображение кратко и по делу."
        await message.bot.send_chat_action(chat_id, "typing")
        answer = await ask_image(user_id, chat_id, prompt, resized_path)
        await message.reply(answer)

    finally:
        for p in (raw_path, resized_path):
            try:
                os.remove(p)
            except Exception:
                pass

# ================== START ==================

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await init_db()
    await dp.start_polling(bot)

if __name__ == "__main__":
    print("Бот запущен. Пока жив.")
    asyncio.run(main())
