import asyncio
import tempfile
import json
from datetime import datetime
from aiogram import Bot, Dispatcher, types, F
from ollama import Client
import aiosqlite
import os
from collections import deque
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

# === НАСТРОЙКИ ===

API_TOKEN = os.environ.get("BOT_TOKEN")
MODEL = "gemma4:31b-cloud"
PERSONAL_LIMIT = 20
GROUP_LIMIT = 15
DATABASE_URL = os.environ.get("DATABASE_URL") or "data/bot.db"
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
DEBUG_UPDATES = os.environ.get("DEBUG_UPDATES", "0").strip().lower() in {"1", "true", "yes", "on"}

def normalize_id(raw_id: int | str | None) -> int | None:
    """
    Нормализует Telegram ID к внутреннему формату.
    Для chat/sender_chat Telegram присылает -100<id>, внутри используем <id>.
    """
    if raw_id is None:
        return None
    try:
        value = int(raw_id)
    except (TypeError, ValueError):
        return None

    value_str = str(value)
    if value_str.startswith("-100"):
        return int(value_str[4:])
    return value


def parse_required_int_env(name: str, *, normalize: bool = False) -> int:
    """Читает обязательную int-переменную окружения и валидирует формат."""
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value.strip() == "":
        raise RuntimeError(f"Переменная окружения {name} не установлена")

    if normalize:
        value_str = raw_value.strip()
        if value_str.startswith("-100"):
            value_str = value_str[4:]
        try:
            return int(value_str)
        except ValueError as exc:
            raise RuntimeError(f"Некорректное значение {name}: {raw_value}") from exc

    try:
        return int(raw_value)
    except ValueError as exc:
        raise RuntimeError(f"Некорректное значение {name}: {raw_value}") from exc


NICK_ID = parse_required_int_env("NICK_ID", normalize=True)
NICK_CHANNEL_IDS_STR = os.environ.get("NICK_CHANNEL_IDS", "")
NICK_CHANNEL_IDS = []
for cid_str in NICK_CHANNEL_IDS_STR.split(","):
    cid_str = cid_str.strip()
    if not cid_str:
        continue
    normalized = normalize_id(cid_str)
    if normalized is None:
        raise RuntimeError(f"Некорректное значение NICK_CHANNEL_IDS: {cid_str}")
    NICK_CHANNEL_IDS.append(normalized)

DANILIUM_ID = parse_required_int_env("DANILIUM_ID")
ALICE_ID = parse_required_int_env("ALICE_ID")

# === Маппинг каналов для группировки контекстов (из .env) ===
# Читаем ID каналов и комментариев из переменных окружения
def parse_channel_id(env_var: str) -> int:
    """Читает ID из .env и нормализует к внутреннему формату."""
    cid_str = os.environ.get(env_var, "0").strip()
    if not cid_str or cid_str == "0":
        raise RuntimeError(f"Переменная окружения {env_var} не установлена")
    normalized = normalize_id(cid_str)
    if normalized is None:
        raise RuntimeError(f"Некорректное значение {env_var}: {cid_str}")
    return normalized

SHITPOST_CHANNEL = parse_channel_id("SHITPOST_CHANNEL")
SHITPOST_COMMENTS = parse_channel_id("SHITPOST_COMMENTS")
MAIN_CHANNEL = parse_channel_id("MAIN_CHANNEL")
MAIN_COMMENTS = parse_channel_id("MAIN_COMMENTS")

CHANNEL_CONTEXT_MAP = {
    SHITPOST_CHANNEL: SHITPOST_COMMENTS,
    SHITPOST_COMMENTS: SHITPOST_COMMENTS,
    MAIN_CHANNEL: MAIN_COMMENTS,
    MAIN_COMMENTS: MAIN_COMMENTS,
}

# Множество ID комментариев (для определения, что это группа комментариев)
COMMENTS_GROUP_IDS = {SHITPOST_COMMENTS, MAIN_COMMENTS}

# Множество ID каналов (для определения, что это исходный канал)
CHANNEL_IDS = {SHITPOST_CHANNEL, MAIN_CHANNEL}


if not API_TOKEN:
    raise RuntimeError("Telegram BOT_TOKEN not set")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")
if not OLLAMA_API_KEY:
    raise RuntimeError("OLLAMA_API_KEY not set")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
db = None

db_path = os.path.abspath(DATABASE_URL)
db_dir = os.path.dirname(db_path)
os.makedirs(db_dir, exist_ok=True)

ollama_client = Client(
    host="https://ollama.com", 
    headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
)

# Логируем конфиг при запуске
print(f"[CONFIG] NICK_ID: {NICK_ID}")
print(f"[CONFIG] NICK_CHANNEL_IDS: {NICK_CHANNEL_IDS} (нормализованные ID)")
print(f"[CONFIG] DANILIUM_ID: {DANILIUM_ID}")
print(f"[CONFIG] ALICE_ID: {ALICE_ID}")
print(f"[CONFIG] SHITPOST_CHANNEL: {SHITPOST_CHANNEL}, SHITPOST_COMMENTS: {SHITPOST_COMMENTS}")
print(f"[CONFIG] MAIN_CHANNEL: {MAIN_CHANNEL}, MAIN_COMMENTS: {MAIN_COMMENTS}")
print(f"[CONFIG] DEBUG_UPDATES: {DEBUG_UPDATES}")

# === Идентификация пользователей ===

NICK_PROMPT_FILE = "prompts/nick_prompt.txt"
DAN_PROMPT_FILE = "prompts/dan_prompt.txt"
ALICE_PROMPT_FILE = "prompts/alice_prompt.txt"
BASE_USER_PROMPT_FILE = "prompts/base_user_prompt.txt"
SYSTEM_PROMPT_FILE = "prompts/system_prompt.txt"

BOT_USERNAME = "rho_segment_bot"

def load_prompt(file_path: str) -> str:
    if not os.path.exists(file_path):
        return ""
    with open(file_path, encoding="utf-8") as f:
        return f.read().strip()

BASE_SYSTEM_PROMPT = load_prompt(SYSTEM_PROMPT_FILE)

# === Нормализация ID ===
def extract_sender_id(message: types.Message) -> int | None:
    """
    Извлекает реальный ID отправителя сообщения.
    Если сообщение от канала (есть sender_chat), возвращает ID канала.
    Иначе возвращает ID пользователя из from_user.
    """
    if message.sender_chat:
        return normalize_id(message.sender_chat.id)
    elif message.from_user:
        return normalize_id(message.from_user.id)
    return None

def is_nick(user_id: int) -> bool:
    normalized_user_id = normalize_id(user_id)
    return normalized_user_id == NICK_ID or normalized_user_id in NICK_CHANNEL_IDS

def is_danilium(user_id: int) -> bool:
    return user_id == DANILIUM_ID

def is_alice(user_id: int) -> bool:
    return user_id == ALICE_ID

def get_context_user_id(sender_id: int) -> int:
    """
    Возвращает 'фокусный ID' для группировки контекста.
    Если sender_id это канал или его комменты, возвращает ID группы комментариев.
    Иначе возвращает оригинальный sender_id.
    """
    return CHANNEL_CONTEXT_MAP.get(sender_id, sender_id)

def is_reply_to_bot(message: types.Message) -> bool:
    """True, если текущее сообщение — прямой reply на сообщение бота."""
    if not message.reply_to_message:
        return False
    if message.reply_to_message.from_user and message.reply_to_message.from_user.id == bot.id:
        return True
    return False


def has_bot_mention(message: types.Message) -> bool:
    """True, если в тексте/подписи есть тег бота."""
    text_content = (message.text or message.caption or "").lower()
    return f"@{BOT_USERNAME.lower()}" in text_content


def should_respond_in_chat(message: types.Message) -> bool:
    """
    Определяет, должен ли бот отвечать на это сообщение.
    
    Логика:
    - ЛС: всегда отвечаем
    - Исходные каналы: НИКОГДА не отвечаем
    - Группы комментариев:
      * Если пост из исходного канала дублируется → ВСЕГДА отвечаем
      * Любое сообщение из комментариев (канал/юзер) → только на reply/mention,
        КРОМЕ дублированного поста из исходного канала
    """
    sender_id = extract_sender_id(message)
    if sender_id is None:
        return False

    chat_id = normalize_id(message.chat.id)
    is_private = message.chat.type == "private"
    
    # Личные сообщения - всегда отвечаем
    if is_private:
        return True
    
    # В исходных каналах вообще не отвечаем
    if chat_id in CHANNEL_IDS:
        return False
    
    # В группах комментариев
    if chat_id in COMMENTS_GROUP_IDS:
        if message.sender_chat:
            sender_chat_id = normalize_id(message.sender_chat.id)
            # Пост от исходного канала, продублированный в комментарии
            if sender_chat_id in CHANNEL_IDS:
                return True
        # Любые остальные сообщения в комментариях -> только reply/mention
        return is_reply_to_bot(message) or has_bot_mention(message)
    
    # В других чатах не отвечаем
    return False

BASE_USER_PROMPT_CONTENT = load_prompt(BASE_USER_PROMPT_FILE)
NICK_PROMPT_CONTENT = load_prompt(NICK_PROMPT_FILE)
DAN_PROMPT_CONTENT = load_prompt(DAN_PROMPT_FILE)
ALICE_PROMPT_CONTENT = load_prompt(ALICE_PROMPT_FILE)


def get_user_instruction(user_id: int) -> str | None:
    """Возвращает промпт для особых пользователей"""
    if is_nick(user_id):
        return NICK_PROMPT_CONTENT
    elif is_danilium(user_id):
        return DAN_PROMPT_CONTENT
    elif is_alice(user_id):
        return ALICE_PROMPT_CONTENT
    else:
        return BASE_USER_PROMPT_CONTENT
    
def get_full_system_prompt(user_id: int) -> str:
    prompt = BASE_SYSTEM_PROMPT
    user_instruction = get_user_instruction(user_id)
    if user_instruction:
        prompt += f"\n\nДополнительная инструкция для пользователя {user_id}:\n{user_instruction}"
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

def resolve_context_type(is_private: bool) -> str:
    return "personal" if is_private else "group"

# === Очереди ===
group_text_queues = {}
group_text_locks = {}
group_image_queues = {}
group_image_locks = {}

# === Буферы для групп фотографий (media_group_id) ===
media_group_buffers = {}  # media_group_id -> {"photos": [...], "sender_id": ..., "chat_id": ..., ...}
media_group_timers = {}   # media_group_id -> asyncio.Task

async def process_group_queue(api_chat_id):
    if api_chat_id not in group_text_queues:
        return
    queue = group_text_queues[api_chat_id]
    lock = group_text_locks[api_chat_id]

    async with lock:
        while not queue.empty():
            user_id, context_chat_id, prompt, message_id = await queue.get()
            try:
                reply_text, _ = await ask_ollama_text(user_id, context_chat_id, prompt, message_id, is_private=False)
                if reply_text:
                    await bot.send_message(api_chat_id, reply_text,
                                         reply_parameters=types.ReplyParameters(message_id=message_id))
            except Exception as e:
                print(f"Error processing group text api_chat={api_chat_id} ctx_chat={context_chat_id} for user {user_id}: {e}")
            finally:
                queue.task_done()

async def process_group_image_queue(api_chat_id):
    if api_chat_id not in group_image_queues:
        return
    queue = group_image_queues[api_chat_id]
    lock = group_image_locks[api_chat_id]

    async with lock:
        while not queue.empty():
            try:
                user_id, context_chat_id, prompt, message_id, image_data = await queue.get()
                
                # Проверяем, одно ли это изображение или несколько
                if isinstance(image_data, list):
                    # Несколько изображений
                    full_text = await ask_ollama_multiple_images(user_id, context_chat_id, prompt, image_data, is_private=False)
                    image_paths = image_data
                else:
                    # Одно изображение
                    full_text = await ask_ollama_image(user_id, context_chat_id, prompt, image_data, is_private=False)
                    image_paths = [image_data]
                
                if full_text:
                    await bot.send_message(api_chat_id, full_text,
                                         reply_parameters=types.ReplyParameters(message_id=message_id))
                
                # Удаляем все временные файлы
                for path in image_paths:
                    if os.path.exists(path):
                        os.remove(path)
            except Exception as e:
                print(f"Error processing group image api_chat={api_chat_id} ctx_chat={context_chat_id} for user {user_id}: {e}")
            finally:
                queue.task_done()

# === Работа с контекстом через БД ===

async def init_db():
    global db
    try:
        db = await aiosqlite.connect(db_path)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS contexts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                context_type TEXT NOT NULL,
                messages TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(chat_id, user_id, context_type)
            );
        """)
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")
        await db.commit()
        print(f"Database initialized at {db_path}")
    except Exception as e:
        print(f"Failed to initialize DB: {e}")
        raise

async def get_context(chat_id, user_id, context_type):
    limit = get_limit(context_type)
    async with db.execute(
        "SELECT messages FROM contexts WHERE chat_id=? AND user_id=? AND context_type=?",
        (chat_id, user_id, context_type)
    ) as cursor:
        row = await cursor.fetchone()

    if row:
        try:
            data = json.loads(row[0])
            if isinstance(data, list):
                return deque(data, maxlen=limit)
        except (TypeError, json.JSONDecodeError):
            print(f"Invalid context payload for {chat_id}/{user_id}/{context_type}, resetting")

    return deque(maxlen=limit)

async def save_context(chat_id, user_id, context_type, context):
    limit = get_limit(context_type)
    if len(context) > limit:
        context = deque(list(context)[-limit:], maxlen=limit)

    data = json.dumps(list(context))

    await db.execute("""
        INSERT INTO contexts(chat_id, user_id, context_type, messages)
        VALUES(?, ?, ?, ?)
        ON CONFLICT(chat_id, user_id, context_type)
        DO UPDATE SET
            messages=excluded.messages,
            updated_at=CURRENT_TIMESTAMP
    """, (chat_id, user_id, context_type, data))

    await db.commit()

async def cleanup_all_contexts():
    """Очищает все контексты от мусорных сообщений"""

    async with db.execute(
        "SELECT chat_id, user_id, context_type, messages FROM contexts"
    ) as cursor:
        rows = await cursor.fetchall()

    for row in rows:
        chat_id = row[0]
        user_id = row[1]
        context_type = row[2]
        try:
            messages = json.loads(row[3])
        except (TypeError, json.JSONDecodeError):
            print(f"Skipping invalid JSON context for {chat_id}/{user_id}/{context_type}")
            continue

        if not isinstance(messages, list):
            print(f"Skipping non-list context for {chat_id}/{user_id}/{context_type}")
            continue

        cleaned = []
        last_role = None

        for msg in messages:
            content = msg.get("content", "")

            if content in ["image received", "group user message", ""]:
                continue

            if last_role == "assistant" and msg.get("role") == "assistant":
                continue

            cleaned.append(msg)
            last_role = msg.get("role")

        if len(messages) != len(cleaned):
            data = json.dumps(cleaned)

            await db.execute(
                """
                UPDATE contexts
                SET messages = ?, updated_at = CURRENT_TIMESTAMP
                WHERE chat_id = ? AND user_id = ? AND context_type = ?
                """,
                (data, chat_id, user_id, context_type),
            )

            print(
                f"Cleaned {len(messages) - len(cleaned)} messages from {chat_id}/{user_id}/{context_type}"
            )

    await db.commit()

# === Текстовые сообщения ===
async def ask_ollama_text(user_id, chat_id, prompt, reply_to_message_id=None, is_private=False):
    reply_text = ""
    
    # System prompt с учетом индивидуальной инструкции пользователя
    system_prompt = get_full_system_prompt(user_id)
    
    messages = [{"role": "system", "content": system_prompt}]
    history_user_id = user_id
    context_type = resolve_context_type(is_private)
    history = deque(maxlen=get_limit(context_type))
    
    try:
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
        
        
        # Добавляем текущий запрос
        all_messages.append({"role": "user", "content": prompt})
        
        # ДЕБАГ
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] === TEXT REQUEST ===")
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
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Saved context for user {history_user_id} in {context_type}")
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
async def ask_ollama_image(user_id, chat_id, prompt, image_path, is_private=False):
    reply_text = ""
    
    # System prompt с учетом пользователя
    system_prompt = get_full_system_prompt(user_id)
    
    messages = [{"role": "system", "content": system_prompt}]
    history_user_id = user_id
    context_type = resolve_context_type(is_private)
    history = deque(maxlen=get_limit(context_type))
    
    try:
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
            if "посмотри на изображение и опиши свою реакцию на него." in content.lower():
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
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] === IMAGE REQUEST ===")
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
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Saved image context for user {history_user_id}")
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


# === Множественные изображения ===
async def ask_ollama_multiple_images(user_id, chat_id, prompt, image_paths: list, is_private=False):
    """
    Обрабатывает несколько изображений в одном сообщении.
    Отправляет все изображения LLM и получает синтезированный ответ.
    """
    reply_text = ""
    
    # System prompt с учетом пользователя
    system_prompt = get_full_system_prompt(user_id)
    
    messages = [{"role": "system", "content": system_prompt}]
    history_user_id = user_id
    context_type = resolve_context_type(is_private)
    
    try:
        history = await get_context(chat_id, history_user_id, context_type)
        
        # Очистка истории
        cleaned_history = deque(maxlen=history.maxlen)
        for msg in list(history):
            content = msg.get('content', '')
            role = msg.get('role', '')
            
            if not content or content.strip() == "":
                continue
            if content in ["image received", "group user message"]:
                continue
            if "посмотри на изображение и опиши свою реакцию на него." in content.lower():
                continue
                
            cleaned_history.append(msg)
        
        # Берём последние 2 сообщения
        max_history_messages = 2
        recent_history = list(cleaned_history)[-max_history_messages:]
        
        all_messages = messages + recent_history
        
        # Подготавливаем промпт для нескольких изображений
        num_images = len(image_paths)
        multi_prompt = f"Ты видишь {num_images} изображений. {prompt}\n\nПожалуйста, рассмотри все изображения и дай общий синтезированный ответ."
        
        # Добавляем запрос со всеми картинками
        user_message = {"role": "user", "content": multi_prompt, "images": image_paths}
        all_messages.append(user_message)
        
        # ДЕБАГ
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] === MULTIPLE IMAGES REQUEST ===")
        print(f"User: {user_id} ({'you' if is_nick(user_id) else 'other'})")
        print(f"Context: {context_type}")
        print(f"Images: {num_images}, Recent history: {len(recent_history)} messages")
        
        response = ollama_client.chat(model=MODEL, messages=all_messages, stream=True)
        
        collected_response = ""
        for chunk in response:
            delta = getattr(chunk.message, "content", "")
            if delta:
                collected_response += delta
                reply_text += delta
        
        if collected_response.strip():
            # Сохраняем в контекст
            save_prompt = f"прислал {num_images} изображения"
            
            cleaned_history.append({"role": "user", "content": save_prompt})
            cleaned_history.append({"role": "assistant", "content": collected_response.strip()})
            
            limit = PERSONAL_LIMIT if context_type == "personal" else GROUP_LIMIT
            if len(cleaned_history) > limit:
                cleaned_history = deque(list(cleaned_history)[-limit:], maxlen=limit)
            
            await save_context(chat_id, history_user_id, context_type, cleaned_history)
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Saved {num_images} images context for user {history_user_id}")
        else:
            reply_text = "Не получилось описать изображения."
            
    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"Ollama multiple images error: {e}")
        
        # Сброс контекста для этого пользователя
        if "500" in str(e):
            print(f"Resetting context for user {history_user_id}")
            await save_context(chat_id, history_user_id, context_type, deque(maxlen=history.maxlen))

    return reply_text


# === Обработка групп фотографий (media_group) ===
async def process_media_group_after_delay(media_group_id: str):
    """
    Ждет 0.5 секунды, чтобы собрать все фото из альбома,
    затем обрабатывает их как группу.
    """
    await asyncio.sleep(0.5)
    
    if media_group_id not in media_group_buffers:
        return
    
    buffer = media_group_buffers[media_group_id]
    sender_id = buffer["sender_id"]
    context_chat_id = buffer["context_chat_id"]
    api_chat_id = buffer["api_chat_id"]
    is_private = buffer["is_private"]
    message_id = buffer["message_id"]
    prompt = buffer["prompt"]
    photo_file_ids = buffer["photos"]
    
    resized_paths = []
    tmp_paths = []
    
    try:
        # Скачиваем и обрабатываем все фото
        for file_id in photo_file_ids:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            tmp_paths.append(tmp_path)
            
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as resized_tmp:
                resized_path = resized_tmp.name
            
            await bot.download(file_id, destination=tmp_path)
            buffer_img = resize_image(tmp_path)
            if not buffer_img:
                print(f"Failed to resize image from group {media_group_id}")
                try:
                    os.remove(resized_path)
                except Exception:
                    pass
                continue
            
            with open(resized_path, 'wb') as f:
                f.write(buffer_img.getvalue())
            resized_paths.append(resized_path)
        
        if not resized_paths:
            msg_text = "Не удалось обработать одно или несколько изображений."
            if is_private:
                await bot.send_message(api_chat_id, msg_text)
            else:
                await bot.send_message(api_chat_id, msg_text, reply_parameters=types.ReplyParameters(message_id=message_id))
            return
        
        await bot.send_chat_action(api_chat_id, "typing")
        
        # Обработка фото
        if is_private:
            full_text = await ask_ollama_multiple_images(sender_id, context_chat_id, prompt, resized_paths, is_private=True)
            await bot.send_message(api_chat_id, full_text)
        else:
            if api_chat_id not in group_image_queues:
                group_image_queues[api_chat_id] = asyncio.Queue()
                group_image_locks[api_chat_id] = asyncio.Lock()
            await group_image_queues[api_chat_id].put((sender_id, context_chat_id, prompt, message_id, resized_paths))
            asyncio.create_task(process_group_image_queue(api_chat_id))
    
    except Exception as e:
        print(f"Error processing media group {media_group_id}: {e}")
    
    finally:
        # Очищаем все временные файлы
        for path in tmp_paths:
            try: os.remove(path)
            except Exception: pass
        if is_private:
            for path in resized_paths:
                try: os.remove(path)
                except Exception: pass
        
        # Удаляем из буферов
        if media_group_id in media_group_buffers:
            del media_group_buffers[media_group_id]
        if media_group_id in media_group_timers:
            del media_group_timers[media_group_id]


# === Хэндлеры ===
@dp.message(F.text)
async def handle_msg(message: types.Message):
    sender_id = extract_sender_id(message)
    if not sender_id:
        return

    api_chat_id = message.chat.id
    context_chat_id = normalize_id(api_chat_id)
    if context_chat_id is None:
        return
    is_private = message.chat.type == "private"
    
    # Используем новую логику для определения, отвечать ли
    if not should_respond_in_chat(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SKIP] sender_id={sender_id}, chat_id={context_chat_id}, ignored")
        return

    # Получаем фокусный ID для группировки контекста
    context_user_id = get_context_user_id(sender_id)

    if is_private:
        full_text, _ = await ask_ollama_text(context_user_id, context_chat_id, message.text, message.message_id, is_private=True)
        await message.reply(full_text)
    else:
        if api_chat_id not in group_text_queues:
            group_text_queues[api_chat_id] = asyncio.Queue()
            group_text_locks[api_chat_id] = asyncio.Lock()
        await group_text_queues[api_chat_id].put((context_user_id, context_chat_id, message.text, message.message_id))
        asyncio.create_task(process_group_queue(api_chat_id))

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    sender_id = extract_sender_id(message)
    if not sender_id:
        return

    api_chat_id = message.chat.id
    context_chat_id = normalize_id(api_chat_id)
    if context_chat_id is None:
        return
    is_private = message.chat.type == "private"
    
    # Используем новую логику для определения, отвечать ли
    if not should_respond_in_chat(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SKIP] sender_id={sender_id}, chat_id={context_chat_id}, photo ignored")
        return
    
    # Получаем фокусный ID для группировки контекста
    context_user_id = get_context_user_id(sender_id)

    prompt = message.caption or "Опиши изображение кратко и по делу."
    photo = message.photo[-1]
    
    # Проверяем, является ли это частью альбома (media_group_id)
    if message.media_group_id:
        # Это часть группы фотографий
        media_group_id = message.media_group_id
        
        # Отменяем старый таймер если он был
        if media_group_id in media_group_timers:
            media_group_timers[media_group_id].cancel()
        
        # Инициализируем буфер если это первое фото из группы
        if media_group_id not in media_group_buffers:
            media_group_buffers[media_group_id] = {
                "photos": [],
                "sender_id": context_user_id,
                "context_chat_id": context_chat_id,
                "api_chat_id": api_chat_id,
                "is_private": is_private,
                "message_id": message.message_id,
                "prompt": prompt
            }
        
        # Добавляем фото в буфер
        media_group_buffers[media_group_id]["photos"].append(photo.file_id)
        
        # Планируем обработку группы после задержки
        task = asyncio.create_task(process_media_group_after_delay(media_group_id))
        media_group_timers[media_group_id] = task
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Media group {media_group_id}: added photo, total: {len(media_group_buffers[media_group_id]['photos'])}")
        return
    
    # Это одиночное фото (не альбом)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as resized_tmp:
        resized_path = resized_tmp.name

    try:
        await bot.download(photo.file_id, destination=tmp_path)
        buffer = resize_image(tmp_path)
        if not buffer:
            await message.reply("Не удалось обработать изображение.") if is_private else await bot.send_message(api_chat_id, "Не удалось обработать изображение.", reply_parameters=types.ReplyParameters(message_id=message.message_id))
            return

        with open(resized_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        await message.bot.send_chat_action(api_chat_id, "typing")

        if is_private:
            full_text = await ask_ollama_image(context_user_id, context_chat_id, prompt, resized_path, is_private=True)
            await message.reply(full_text)
        else:
            if api_chat_id not in group_image_queues:
                group_image_queues[api_chat_id] = asyncio.Queue()
                group_image_locks[api_chat_id] = asyncio.Lock()
            await group_image_queues[api_chat_id].put((context_user_id, context_chat_id, prompt, message.message_id, resized_path))
            asyncio.create_task(process_group_image_queue(api_chat_id))
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
        if is_private:
            try: os.remove(resized_path)
            except Exception: pass

# === Дебаг хэндлер для всех обновлений ===
@dp.message()
async def debug_all_messages(message: types.Message):
    """
    Ловит ВСЕ сообщения для отладки.
    Логирует информацию о каждом сообщении.
    """
    if not DEBUG_UPDATES:
        return

    sender_id = extract_sender_id(message)
    chat_id = normalize_id(message.chat.id)
    chat_type = message.chat.type
    
    # Определяем тип сообщения
    msg_type = "unknown"
    if message.text:
        msg_type = "text"
    elif message.photo:
        msg_type = "photo"
    elif message.video:
        msg_type = "video"
    elif message.document:
        msg_type = "document"
    
    # Логируем всё подробно
    log_msg = f"[DEBUG] Message from sender_id={sender_id}, chat_id={chat_id}, type={chat_type}, msg_type={msg_type}"
    if message.sender_chat:
        log_msg += f", sender_chat_id={message.sender_chat.id}"
    if message.media_group_id:
        log_msg += f", media_group_id={message.media_group_id}"
    if message.text:
        log_msg += f", text='{message.text[:50]}...'"
    
    print(log_msg)

# === Старт бота ===
async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await init_db()
    await cleanup_all_contexts()
    try:
        await dp.start_polling(bot)
    finally:
        if db:
            await db.close()
        await bot.session.close()

if __name__ == "__main__":
    print("Бот запущен")
    asyncio.run(main())