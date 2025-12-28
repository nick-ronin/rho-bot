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
load_dotenv()

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
    
    # 1. Загружаем историю
    history = await get_context(chat_id, user_id if is_private else 0, 
                               "personal" if is_private else "group")
    
    # 2. ОЧИСТКА: удаляем мусор из истории
    cleaned_history = deque(maxlen=history.maxlen)
    for msg in list(history):
        content = msg.get('content', '')
        # Удаляем "image received", "group user message" и другие мусорные сообщения
        if content in ["image received", "group user message", "image received", ""]:
            continue
        # Удаляем дублированные assistant сообщения (следующие друг за другом)
        if cleaned_history and cleaned_history[-1]['role'] == 'assistant' and msg['role'] == 'assistant':
            print(f"WARNING: Skipping duplicate assistant message: {content[:50]}")
            continue
        cleaned_history.append(msg)
    
    # 3. Формируем system prompt
    system_prompt = BASE_SYSTEM_PROMPT
    user_instruction = get_user_instruction(user_id)
    if user_instruction:
        system_prompt += f"\n\n--- ИНСТРУКЦИЯ ДЛЯ ПОЛЬЗОВАТЕЛЯ {user_id} ---\n{user_instruction}\n--- КОНЕЦ ИНСТРУКЦИИ ---"
    
    messages = [{"role": "system", "content": system_prompt}]
    
    # 4. Добавляем ОЧИЩЕННУЮ историю
    all_messages = messages + list(cleaned_history)
    
    # 6. Добавляем текущий запрос
    all_messages.append({"role": "user", "content": prompt})
    
    # 7. ДЕБАГ-ВЫВОД
    print(f"\n=== CLEAN HISTORY ===")
    for i, msg in enumerate(list(cleaned_history)):
        role = msg['role']
        content = msg.get('content', 'NO CONTENT')
        print(f"{i:2}. [{role:10}]: {content[:80]}{'...' if len(content) > 80 else ''}")
    
    print(f"\n=== FINAL REQUEST TO OLLAMA ===")
    for i, msg in enumerate(all_messages):
        role = msg['role']
        content = msg.get('content', 'NO CONTENT')
        if role == 'system':
            content_preview = content[:100] + "..." if len(content) > 100 else content
        else:
            content_preview = content[:80] + "..." if len(content) > 80 else content
        print(f"{i:2}. [{role:10}]: {content_preview}")
    
    try:
        # 8. Отправляем запрос
        response = ollama_client.chat(model=MODEL, messages=all_messages, stream=True)
        
        collected_response = ""
        for chunk in response:
            delta = getattr(chunk.message, "content", "")
            if delta:
                collected_response += delta
                reply_text += delta
        
        # 9. ПРОВЕРКА: не сохраняем пустые или мусорные ответы
        if not collected_response.strip():
            print("WARNING: Empty response from Ollama!")
            reply_text = "Не получилось сформулировать ответ. Попробуй ещё раз."
        else:
            # 10. Сохраняем ТОЛЬКО если всё ок
            cleaned_history.append({"role": "user", "content": prompt})
            cleaned_history.append({"role": "assistant", "content": collected_response.strip()})
            
            # Ещё одна проверка на дубли
            if len(list(cleaned_history)) >= 2:
                last_two = list(cleaned_history)[-2:]
                if last_two[0]['role'] == 'assistant' and last_two[1]['role'] == 'assistant':
                    print("CRITICAL: Detected double assistant in cleaned history! Removing last one.")
                    cleaned_history.pop()
            
            await save_context(chat_id, user_id if is_private else 0, 
                              "personal" if is_private else "group", cleaned_history)
            
    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"Ollama text error: {e}")
        import traceback
        traceback.print_exc()
        
        # Если 500 - сбросим контекст для этого чата
        if "500" in str(e) or "Internal Server Error" in str(e):
            print(f"Resetting context for chat {chat_id}, user {user_id if is_private else 0}")
            await save_context(chat_id, user_id if is_private else 0, 
                              "personal" if is_private else "group", deque(maxlen=history.maxlen))

    return reply_text, reply_to_message_id

# === Изображения (с base64) ===
async def ask_ollama_image(user_id, chat_id, prompt, image_path):
    reply_text = ""
    
    # 1. Загружаем историю
    history = await get_context(chat_id, user_id if chat_id == user_id else 0,
                                "personal" if chat_id == user_id else "group")
    
    # 2. ОЧИСТКА: УДАЛЯЕМ ПРЕДЫДУЩИЕ ЗАПРОСЫ НА ОПИСАНИЕ КАРТИНОК
    cleaned_history = deque(maxlen=history.maxlen)
    for msg in list(history):
        content = msg.get('content', '')
        role = msg.get('role', '')
        
        # Пропускаем пустые
        if not content or content.strip() == "":
            continue
            
        # Пропускаем мусор
        if content in ["image received", "group user message"]:
            continue
            
        # КРИТИЧНО: пропускаем предыдущие запросы на описание картинок
        if "опиши изображение" in content.lower() or "что на изображении" in content.lower():
            print(f"Skipping previous image description request: {content[:50]}...")
            # Но сохраняем ответ на него, если он есть
            continue
            
        cleaned_history.append(msg)
    
    # 3. System prompt с нормальной личностью
    system_prompt = BASE_SYSTEM_PROMPT
    user_instruction = get_user_instruction(user_id)
    if user_instruction:
        system_prompt += f"\n\n{user_instruction}"
    
    max_history_messages = 2
    recent_history = []
    for msg in reversed(list(cleaned_history)):
        recent_history.insert(0, msg)
        if len(recent_history) >= max_history_messages:
            break
    
    # 5. Формируем запрос
    messages = [{"role": "system", "content": system_prompt}]
    all_messages = messages + recent_history
    
    # 7. Добавляем запрос с картинкой
    user_message = {"role": "user", "content": prompt, "images": [image_path]}
    all_messages.append(user_message)
    
    # 8. ДЕБАГ
    print(f"\n=== IMAGE REQUEST (PERSONALITY MODE) ===")
    print(f"System prompt length: {len(system_prompt)}")
    print(f"Recent history: {len(recent_history)} messages")
    print(f"Total messages: {len(all_messages)}")
    
    for i, msg in enumerate(all_messages):
        role = msg['role']
        if 'images' in msg:
            img_preview = msg.get('content', '')[:40]
            print(f"{i}. [{role}]: [IMAGE] '{img_preview}...'")
        elif role == 'system':
            preview = system_prompt[:100].replace('\n', ' ')
            print(f"{i}. [{role}]: {preview}...")
        else:
            content = msg.get('content', '')[:60]
            print(f"{i}. [{role}]: {content}...")
    
    try:
        # 9. Отправляем запрос с историей
        print(f"\nSending to {MODEL}...")
        response = ollama_client.chat(model=MODEL, messages=all_messages, stream=True)
        
        collected_response = ""
        for chunk in response:
            delta = getattr(chunk.message, "content", "")
            if delta:
                collected_response += delta
                reply_text += delta
        
        if collected_response.strip():
            print(f"Response length: {len(collected_response)} chars")
            print(f"Response preview: {collected_response[:150]}...")
            
            # 10. Сохраняем в историю, НО не сохраняем "Опиши изображение" если это дефолтный промпт
            save_prompt = prompt
            save_response = collected_response.strip()
            
            # Если это дефолтный промпт, сохраняем как "прислал изображение"
            if prompt.lower() in ["опиши изображение кратко и по делу.", "опиши изображение", "что на изображении"]:
                save_prompt = "прислал изображение"
            
            cleaned_history.append({"role": "user", "content": save_prompt})
            cleaned_history.append({"role": "assistant", "content": save_response})
            
            limit = PERSONAL_LIMIT if chat_id == user_id else GROUP_LIMIT
            if len(cleaned_history) > limit:
                cleaned_history = deque(list(cleaned_history)[-limit:], maxlen=limit)
            
            await save_context(chat_id, user_id if chat_id == user_id else 0,
                               "personal" if chat_id == user_id else "group", cleaned_history)
        else:
            reply_text = "Не получилось описать изображение."
            
    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"Ollama image error: {e}")
        
        # Альтернативный подход: без истории
        try:
            print("\n=== FALLBACK: Simple image request ===")
            simple_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt, "images": [image_path]}
            ]
            response = ollama_client.chat(model=MODEL, messages=simple_messages, stream=False)
            if response and hasattr(response, 'message'):
                reply_text = response.message.content
                print("Fallback worked!")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

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
