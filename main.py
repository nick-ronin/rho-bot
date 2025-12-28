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
GROUP_LIMIT = 30
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
BASE_USER_PROMPT_FILE = "prompts/base_user_prompt.txt"
SYSTEM_PROMPT_FILE = "prompts/system_prompt.txt"

BOT_USERNAME = "rho_segment_bot"

def load_prompt(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, encoding="utf-8") as f:
        return f.read().strip()

BASE_SYSTEM_PROMPT = load_prompt(SYSTEM_PROMPT_FILE)

def is_nick(user_id: int) -> bool:
    return user_id in [NICK_ID, CHANNEL_COMMENTS_NICK_ID, CHANNEL_NICK_ID]

def is_danilium(user_id: int) -> bool:
    return user_id == DANILIUM_ID

BASE_USER_PROMPT_CONTENT = load_prompt(BASE_USER_PROMPT_FILE)
NICK_PROMPT_CONTENT = load_prompt(NICK_PROMPT_FILE)
DAN_PROMPT_CONTENT = load_prompt(DAN_PROMPT_FILE)

def get_user_instruction(user_id: int) -> str | None:
    """Возвращает промпт для особых пользователей"""
    if is_nick(user_id):
        return NICK_PROMPT_CONTENT
    elif is_danilium(user_id):
        return DAN_PROMPT_CONTENT
    else:
        return BASE_USER_PROMPT_CONTENT
    
def get_full_system_prompt(user_id: int) -> str:
    prompt = BASE_SYSTEM_PROMPT
    user_instr = get_user_instruction(user_id)
    if user_instr:
        prompt += f"\n\nДополнительная инструкция для пользователя {user_id}:\n{user_instr}"
    return prompt

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
                    await bot.send_message(chat_id, reply_text, 
                                         reply_parameters=types.ReplyParameters(message_id=message_id))
            except Exception as e:
                print(f"Error processing group text {chat_id} for user {user_id}: {e}")
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
                    await bot.send_message(chat_id, full_text, 
                                         reply_parameters=types.ReplyParameters(message_id=message_id))
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"Error processing group image {chat_id} for user {user_id}: {e}")
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

async def cleanup_all_contexts():
    """Очищает все контексты от мусорных сообщений"""
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT chat_id, user_id, context_type, messages FROM contexts")
        
        for row in rows:
            chat_id = row['chat_id']
            user_id = row['user_id']
            context_type = row['context_type']
            messages = json.loads(row['messages'])
            
            # Очищаем от мусора
            cleaned = []
            last_role = None
            
            for msg in messages:
                content = msg.get('content', '')
                
                # Пропускаем мусор
                if content in ["image received", "group user message", ""]:
                    continue
                
                # Пропускаем дублированные assistant
                if last_role == 'assistant' and msg.get('role') == 'assistant':
                    continue
                
                cleaned.append(msg)
                last_role = msg.get('role')
            
            # Сохраняем обратно
            data = json.dumps(cleaned)
            await conn.execute("""
                UPDATE contexts SET messages=$1 
                WHERE chat_id=$2 AND user_id=$3 AND context_type=$4
            """, data, chat_id, user_id, context_type)
            
            if len(messages) != len(cleaned):
                print(f"Cleaned {len(messages) - len(cleaned)} messages from {chat_id}/{user_id}/{context_type}")

# === Текстовые сообщения ===
async def ask_ollama_text(user_id, chat_id, prompt, reply_to_message_id=None):
    is_private = chat_id == user_id
    reply_text = ""
    
    # System prompt с УЧЕТОМ пользователя даже в группе
    system_prompt = BASE_SYSTEM_PROMPT
    user_instruction = get_user_instruction(user_id)
    if user_instruction:
        system_prompt += f"\n\n{user_instruction}"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    try:
        # КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: В группе тоже сохраняем по user_id, а не 0!
        if is_private:
            # Личка: контекст per пользователь
            history_user_id = user_id
            context_type = "personal"
        else:
            # Группа: контекст per пользователь в этом чате
            history_user_id = user_id  # НЕ 0!
            context_type = "group"
        
        history = await get_context(chat_id, history_user_id, context_type)
        
        # Очистка истории
        cleaned_history = deque(maxlen=history.maxlen)
        for msg in list(history):
            content = msg.get('content', '')
            role = msg.get('role', '')
            
            if not content or content.strip() == "":
                continue
            if content in ["image received", "group user message", "continue", "продолжай"]:
                continue
                
            # Проверяем последовательность
            if cleaned_history and cleaned_history[-1]['role'] == role:
                print(f"Skipping duplicate {role} message")
                continue
                
            cleaned_history.append(msg)
        
        # Добавляем историю
        all_messages = messages + list(cleaned_history)
        
        # Если последнее сообщение - assistant, добавляем continuation
        if all_messages and all_messages[-1]['role'] == 'assistant':
            print(f"Adding continuation for user {user_id} in {'private' if is_private else 'group'}")
            all_messages.append({"role": "user", "content": "продолжай"})
        
        # Добавляем текущий запрос
        all_messages.append({"role": "user", "content": prompt})
        
        # ДЕБАГ
        print(f"\n=== TEXT REQUEST ===")
        print(f"User: {user_id} ({'you' if is_nick(user_id) else 'other'})")
        print(f"Chat: {'private' if is_private else 'group'} {chat_id}")
        print(f"Messages: {len(all_messages)}, History: {len(cleaned_history)}")
        
        response = ollama_client.chat(model=MODEL, messages=all_messages, stream=True)
        
        collected_response = ""
        for chunk in response:
            delta = getattr(chunk.message, "content", "")
            if delta:
                collected_response += delta
                reply_text += delta
        
        if collected_response.strip():
            # Сохраняем с правильным user_id
            cleaned_history.append({"role": "user", "content": prompt})
            cleaned_history.append({"role": "assistant", "content": collected_response.strip()})
            
            await save_context(chat_id, history_user_id, context_type, cleaned_history)
            
            # Логируем
            print(f"Saved context for user {history_user_id} in {context_type}")
        else:
            reply_text = "Не получилось сформулировать ответ."
            
    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"Ollama text error: {e}")
        
        # При 500 сбрасываем контекст для ЭТОГО пользователя
        if "500" in str(e):
            print(f"Resetting context for user {history_user_id} in chat {chat_id}")
            await save_context(chat_id, history_user_id, context_type, deque(maxlen=history.maxlen))

    return reply_text, reply_to_message_id

# === Изображения (с base64) ===
async def ask_ollama_image(user_id, chat_id, prompt, image_path):
    reply_text = ""
    
    # System prompt с учетом пользователя
    system_prompt = BASE_SYSTEM_PROMPT
    user_instruction = get_user_instruction(user_id)
    if user_instruction:
        system_prompt += f"\n\n{user_instruction}"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    try:
        # Аналогично: в группе раздельный контекст
        if chat_id == user_id:
            # Личка
            history_user_id = user_id
            context_type = "personal"
        else:
            # Группа
            history_user_id = user_id  # НЕ 0!
            context_type = "group"
        
        history = await get_context(chat_id, history_user_id, context_type)
        
        # Очистка истории (пропускаем старые запросы на описание картинок)
        cleaned_history = deque(maxlen=history.maxlen)
        for msg in list(history):
            content = msg.get('content', '')
            role = msg.get('role', '')
            
            if not content or content.strip() == "":
                continue
            if content in ["image received", "group user message"]:
                continue
            if "опиши изображение" in content.lower():
                continue
                
            cleaned_history.append(msg)
        
        # Берём последние 2 сообщения
        max_history_messages = 2
        recent_history = list(cleaned_history)[-max_history_messages:]
        
        all_messages = messages + recent_history
        
        # Добавляем запрос с картинкой
        user_message = {"role": "user", "content": prompt, "images": [image_path]}
        all_messages.append(user_message)
        
        # ДЕБАГ
        print(f"\n=== IMAGE REQUEST ===")
        print(f"User: {user_id} ({'you' if is_nick(user_id) else 'other'})")
        print(f"Context: {context_type}")
        print(f"Recent history: {len(recent_history)} messages")
        
        response = ollama_client.chat(model=MODEL, messages=all_messages, stream=True)
        
        collected_response = ""
        for chunk in response:
            delta = getattr(chunk.message, "content", "")
            if delta:
                collected_response += delta
                reply_text += delta
        
        if collected_response.strip():
            # Сохраняем (но не дефолтные промпты "опиши изображение")
            save_prompt = "прислал изображение" if "опиши изображение" in prompt.lower() else prompt
            
            cleaned_history.append({"role": "user", "content": save_prompt})
            cleaned_history.append({"role": "assistant", "content": collected_response.strip()})
            
            limit = PERSONAL_LIMIT if context_type == "personal" else GROUP_LIMIT
            if len(cleaned_history) > limit:
                cleaned_history = deque(list(cleaned_history)[-limit:], maxlen=limit)
            
            await save_context(chat_id, history_user_id, context_type, cleaned_history)
            
            print(f"Saved image context for user {history_user_id}")
        else:
            reply_text = "Не получилось описать изображение."
            
    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"Ollama image error: {e}")
        
        # Сброс контекста для этого пользователя
        if "500" in str(e):
            print(f"Resetting image context for user {history_user_id}")
            await save_context(chat_id, history_user_id, context_type, deque(maxlen=history.maxlen))

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
    await cleanup_all_contexts()
    await dp.start_polling(bot)

if __name__ == "__main__":
    print("Бот запущен. Дай бог не ебнется")
    asyncio.run(main())
