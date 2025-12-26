import asyncio
import tempfile
import json
from aiogram import Bot, Dispatcher, types, F
from ollama import Client
import asyncpg
import os
from collections import deque
from dotenv import load_dotenv
from PIL import Image
import io
import base64

# === НАСТРОЙКИ ===
API_TOKEN = os.environ.get("BOT_TOKEN")
MODEL = "gemma3:27b-cloud"
PERSONAL_LIMIT = 30
GROUP_LIMIT = 50
DATABASE_URL = os.environ.get("DATABASE_URL")
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")

if not API_TOKEN:
    raise RuntimeError("Telegram BOT_TOKEN not set")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")
if not OLLAMA_API_KEY:
    raise RuntimeError("OLLAMA_API_KEY not set")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
db_pool = None
ollama_client = Client(
    host="https://ollama.com", 
    headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
)

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

def resize_image(image_path, max_size=(512, 512), quality=85):
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])
                else:
                    background.paste(img, (0, 0))
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')

            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality, optimize=True)
            buffer.seek(0)
            return buffer
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

def image_to_base64(image_path, max_size=(512, 512), quality=85):
    buffer = resize_image(image_path, max_size=max_size, quality=quality)
    if not buffer:
        return None
    # base64 без переносов
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii").replace("\n", "")
    return f"data:image/jpeg;base64,{encoded}"


def get_limit(context_type: str) -> int:
    if context_type == "personal":
        return PERSONAL_LIMIT
    if context_type == "group":
        return GROUP_LIMIT
    return 20

# === Очереди ===
group_text_queues = {}
group_text_locks = {}
group_image_queues = {}
group_image_locks = {}

async def process_group_queue(chat_id):
    if chat_id not in group_text_queues:
        return
    queue = group_text_queues[chat_id]
    lock = group_text_locks[chat_id]

    async with lock:
        while not queue.empty():
            user_id, prompt, message_id = await queue.get()
            try:
                reply_text, _ = await ask_ollama_text(user_id, chat_id, prompt, message_id)
                if reply_text:
                    await bot.send_message(chat_id, reply_text, reply_parameters=types.ReplyParameters(message_id=message_id))
            except Exception as e:
                print(f"Error processing group text {chat_id}: {e}")
            finally:
                queue.task_done()

async def process_group_image_queue(chat_id):
    if chat_id not in group_image_queues:
        return
    queue = group_image_queues[chat_id]
    lock = group_image_locks[chat_id]

    async with lock:
        while not queue.empty():
            try:
                user_id, prompt, message_id, image_path = await queue.get()
                full_text = await ask_ollama_image(user_id, chat_id, prompt, image_path)
                if full_text:
                    await bot.send_message(chat_id, full_text, reply_parameters=types.ReplyParameters(message_id=message_id))
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"Error processing group image {chat_id}: {e}")
            finally:
                queue.task_done()

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
    limit = get_limit(context_type)
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT messages FROM contexts WHERE chat_id=$1 AND user_id=$2 AND context_type=$3",
            chat_id, user_id, context_type
        )
        if row:
            data = json.loads(row["messages"])
            return deque(data, maxlen=limit)
        return deque(maxlen=limit)

async def save_context(chat_id, user_id, context_type, context):
    limit = get_limit(context_type)
    if len(context) > limit:
        context = deque(list(context)[-limit:], maxlen=limit)
    data = json.dumps(list(context))
    async with db_pool.acquire() as conn:
        await conn.execute("""
            INSERT INTO contexts(chat_id, user_id, context_type, messages)
            VALUES($1, $2, $3, $4)
            ON CONFLICT(chat_id, user_id, context_type) DO UPDATE
            SET messages=$4, updated_at=CURRENT_TIMESTAMP
        """, chat_id, user_id, context_type, data)

# === Текстовые сообщения ===
async def ask_ollama_text(user_id, chat_id, prompt, reply_to_message_id=None):
    is_private = chat_id == user_id
    system_prompt = get_system_prompt(user_id)
    reply_text = ""
    messages = [{"role": "system", "content": system_prompt}]

    try:
        history = await get_context(chat_id, user_id if is_private else 0, "personal" if is_private else "group")
        all_messages = messages + list(history) + [{"role": "user", "content": prompt}]
        response = ollama_client.chat(model=MODEL, messages=all_messages, stream=True)
        for chunk in response:
            delta = getattr(chunk.message, "content", "")
            if delta:
                reply_text += delta

        history.append({"role": "user", "content": prompt if is_private else "group user message"})
        history.append({"role": "assistant", "content": reply_text.strip()})
        await save_context(chat_id, user_id if is_private else 0, "personal" if is_private else "group", history)

    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"Ollama text error: {e}")

    return reply_text, reply_to_message_id

# === Изображения (с base64) ===
async def ask_ollama_image(user_id, chat_id, prompt, image_path):
    system_prompt = get_system_prompt(user_id)
    reply_text = ""
    messages = [{"role": "system", "content": system_prompt}]
    max_history_messages = 2

    try:
        history = await get_context(chat_id, user_id if chat_id == user_id else 0,
                                    "personal" if chat_id == user_id else "group")
        recent_history = list(history)[-max_history_messages:]

        # Передаём путь к файлу напрямую
        user_message = {"role": "user", "content": prompt, "images": [image_path]}
        all_messages = messages + recent_history + [user_message]

        response = ollama_client.chat(model=MODEL, messages=all_messages, stream=True)
        for chunk in response:
            delta = getattr(chunk.message, "content", "")
            if delta:
                reply_text += delta

        history.append({"role": "user", "content": "image received"})
        history.append({"role": "assistant", "content": reply_text.strip()})
        limit = PERSONAL_LIMIT if chat_id == user_id else GROUP_LIMIT
        if len(history) > limit:
            history = deque(list(history)[-limit:], maxlen=limit)
        await save_context(chat_id, user_id if chat_id == user_id else 0,
                           "personal" if chat_id == user_id else "group", history)

    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"Ollama image error: {e}")

    return reply_text


# === Хэндлеры ===
@dp.message(F.text)
async def handle_msg(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    is_private = message.chat.type == "private"
    respond = is_private or (user_id == CHANNEL_NICK_ID or (message.reply_to_message and message.reply_to_message.from_user.id == bot.id) or (f"@{BOT_USERNAME}" in message.text))
    if not respond:
        return

    if is_private:
        full_text, _ = await ask_ollama_text(user_id, chat_id, message.text, message.message_id)
        await message.reply(full_text)
    else:
        if chat_id not in group_text_queues:
            group_text_queues[chat_id] = asyncio.Queue()
            group_text_locks[chat_id] = asyncio.Lock()
        await group_text_queues[chat_id].put((user_id, message.text, message.message_id))
        asyncio.create_task(process_group_queue(chat_id))

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    user_id = message.from_user.id
    chat_id = message.chat.id
    is_private = message.chat.type == "private"
    respond = is_private or (user_id == CHANNEL_NICK_ID or (message.reply_to_message and message.reply_to_message.from_user.id == bot.id) or (f"@{BOT_USERNAME}" in (message.caption or "")))
    if not respond:
        return

    photo = message.photo[-1]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as resized_tmp:
        resized_path = resized_tmp.name

    try:
        await bot.download(photo.file_id, destination=tmp_path)
        buffer = resize_image(tmp_path)
        if not buffer:
            await message.reply("Не удалось обработать изображение.") if is_private else await bot.send_message(chat_id, "Не удалось обработать изображение.", reply_parameters=types.ReplyParameters(message_id=message.message_id))
            return

        with open(resized_path, 'wb') as f:
            f.write(buffer.getvalue())
        prompt = message.caption or "Опиши изображение кратко и по делу."
        await message.bot.send_chat_action(chat_id, "typing")

        if is_private:
            full_text = await ask_ollama_image(user_id, chat_id, prompt, resized_path)
            await message.reply(full_text)
        else:
            if chat_id not in group_image_queues:
                group_image_queues[chat_id] = asyncio.Queue()
                group_image_locks[chat_id] = asyncio.Lock()
            await group_image_queues[chat_id].put((user_id, prompt, message.message_id, resized_path))
            asyncio.create_task(process_group_image_queue(chat_id))
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
        if is_private:
            try: os.remove(resized_path)
            except Exception: pass

# === Старт бота ===
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await init_db()
    await dp.start_polling(bot)

if __name__ == "__main__":
    print("Бот запущен. Дай бог не ебнется")
    asyncio.run(main())
