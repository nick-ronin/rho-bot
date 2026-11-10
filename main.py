import asyncio
import base64
import tempfile
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math
import random
import time
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
import aiohttp
import aiosqlite
import os
from collections import deque
from dotenv import load_dotenv
from PIL import Image
import io

load_dotenv()

# === НАСТРОЙКИ ===

API_TOKEN = os.environ.get("BOT_TOKEN")
PERSONAL_LIMIT = 20
GROUP_LIMIT = 15
IMAGE_HISTORY_MESSAGES = 2
DATABASE_URL = os.environ.get("DATABASE_URL") or "data/bot.db"
DEBUG_UPDATES = os.environ.get("DEBUG_UPDATES", "0").strip().lower() in {"1", "true", "yes", "on"}

# --- OpenRouter ---
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("OPENROUTER_MODEL", "google/gemma-4-31b-it:free")
# Запасные модели через запятую: если основная лежит/лимитится, OpenRouter сам пробует следующую
FALLBACK_MODELS = [m.strip() for m in os.environ.get("OPENROUTER_FALLBACK_MODELS", "").split(",") if m.strip()]
# Ризонинг жрёт токены и время. Для болталки в телеге обычно не нужен.
REASONING_ENABLED = os.environ.get("OPENROUTER_REASONING", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1500"))
_temp = os.environ.get("LLM_TEMPERATURE", "").strip()
TEMPERATURE = float(_temp) if _temp else None
# Число попыток для сетевых ошибок/5xx. После 429 повторов нет.
LLM_RETRIES = max(1, int(os.environ.get("LLM_RETRIES", "2")))
# Общие ограничения для ВСЕХ чатов и фотографий в одном процессе бота.
LLM_MIN_INTERVAL = max(6.0, float(os.environ.get("LLM_MIN_INTERVAL", "6")))
LLM_QUEUE_TIMEOUT = max(1.0, float(os.environ.get("LLM_QUEUE_TIMEOUT", "30")))
LLM_RATE_LIMIT_PAUSE = max(30.0, float(os.environ.get("LLM_RATE_LIMIT_PAUSE", "60")))

TG_MESSAGE_LIMIT = 4000  # у Telegram жёсткий лимит 4096, оставляем запас


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
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")

bot = Bot(token=API_TOKEN)
dp = Dispatcher()
db = None
http_session: aiohttp.ClientSession | None = None

db_path = os.path.abspath(DATABASE_URL)
db_dir = os.path.dirname(db_path)
os.makedirs(db_dir, exist_ok=True)

# Логируем конфиг при запуске
print(f"[CONFIG] MODEL: {MODEL} (fallbacks: {FALLBACK_MODELS or 'нет'}, reasoning: {REASONING_ENABLED})")
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

# === Отправка сообщений (с разбивкой под лимит Telegram) ===
def split_text(text: str, limit: int = TG_MESSAGE_LIMIT) -> list[str]:
    chunks = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks

async def send_reply(api_chat_id, text: str, reply_to_message_id: int | None = None):
    """Шлёт ответ, режет длинные тексты на части. Реплай только на первую часть."""
    for i, chunk in enumerate(split_text(text or "...")):
        reply_params = None
        if i == 0 and reply_to_message_id:
            reply_params = types.ReplyParameters(message_id=reply_to_message_id)
        await bot.send_message(api_chat_id, chunk, reply_parameters=reply_params)

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
                reply_text = await ask_llm(user_id, context_chat_id, prompt, is_private=False)
                if reply_text:
                    await send_reply(api_chat_id, reply_text, message_id)
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
            user_id, context_chat_id, prompt, message_id, image_data = await queue.get()
            image_paths = image_data if isinstance(image_data, list) else [image_data]
            try:
                full_text = await ask_llm_images(user_id, context_chat_id, prompt, image_paths, is_private=False)
                if full_text:
                    await send_reply(api_chat_id, full_text, message_id)
            except Exception as e:
                print(f"Error processing group image api_chat={api_chat_id} ctx_chat={context_chat_id} for user {user_id}: {e}")
            finally:
                # Удаляем все временные файлы
                for path in image_paths:
                    try: os.remove(path)
                    except Exception: pass
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

async def reset_context(chat_id, user_id, context_type):
    await db.execute(
        "DELETE FROM contexts WHERE chat_id=? AND user_id=? AND context_type=?",
        (chat_id, user_id, context_type),
    )
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

# === OpenRouter ===
class LLMError(Exception):
    def __init__(self, status: int, body: str = ""):
        super().__init__(f"OpenRouter error {status}: {body[:300]}")
        self.status = status
        self.body = body


# 429 обрабатываем отдельно: останавливаем запросы, а не повторяем их подряд.
RETRYABLE_STATUSES = {500, 502, 503, 504}
llm_request_lock = asyncio.Lock()
llm_next_request_at = 0.0
llm_cooldown_until = 0.0
llm_rate_limit_streak = 0


def parse_retry_after(value: str | None) -> float:
    """Retry-After: секунды либо HTTP-дата. Некорректный заголовок игнорируем."""
    if not value:
        return 0.0
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except ValueError:
        try:
            until = parsedate_to_datetime(value)
            if until.tzinfo is None:
                until = until.replace(tzinfo=timezone.utc)
            return max(0.0, (until - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def check_llm_cooldown() -> None:
    remaining = llm_cooldown_until - time.monotonic()
    if remaining > 0:
        raise LLMError(429, f"Общая пауза после лимита: ещё {math.ceil(remaining)} сек.")


async def call_llm(messages: list[dict]) -> str:
    """Общая очередь API: один запрос за раз, интервал и пауза после 429."""
    global llm_next_request_at, llm_cooldown_until, llm_rate_limit_streak
    check_llm_cooldown()
    # Не накапливаем бесконечную очередь при большом потоке сообщений.
    try:
        await asyncio.wait_for(llm_request_lock.acquire(), timeout=LLM_QUEUE_TIMEOUT)
    except asyncio.TimeoutError as exc:
        raise LLMError(0, "Очередь запросов занята. Попробуй немного позже.") from exc

    try:
        check_llm_cooldown()  # Пока ждали очередь, другой запрос мог получить 429.
        return await _call_llm_locked(messages)
    finally:
        llm_request_lock.release()


async def _call_llm_locked(messages: list[dict]) -> str:
    global llm_next_request_at, llm_cooldown_until, llm_rate_limit_streak
    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }
    if FALLBACK_MODELS:
        payload["models"] = list(dict.fromkeys([MODEL, *FALLBACK_MODELS]))
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    if REASONING_ENABLED:
        payload["reasoning"] = {"enabled": True}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "X-Title": BOT_USERNAME,
    }

    last_error = LLMError(0, "Запрос не выполнен")
    for attempt in range(LLM_RETRIES):
        check_llm_cooldown()
        delay = llm_next_request_at - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        # Считаем каждую попытку, включая неудачные запросы и повторы.
        llm_next_request_at = time.monotonic() + LLM_MIN_INTERVAL
        retry_after = 0.0
        try:
            async with http_session.post(OPENROUTER_URL, json=payload, headers=headers) as resp:
                raw = await resp.text()
                status = resp.status
                retry_after = parse_retry_after(resp.headers.get("Retry-After"))

            if status == 200:
                try:
                    data = json.loads(raw)
                    if isinstance(data, dict) and data.get("error"):
                        err = data["error"]
                        code = err.get("code", 500) if isinstance(err, dict) else 500
                        raise LLMError(int(code) if str(code).isdigit() else 500,
                                       json.dumps(err, ensure_ascii=False))
                    message = data["choices"][0]["message"]
                    content = (message.get("content") or "").strip()
                except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
                    raise LLMError(502, "Некорректный ответ OpenRouter") from exc
                llm_rate_limit_streak = 0
                return content

            last_error = LLMError(status, raw)
        except LLMError as exc:
            last_error = exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = LLMError(0, f"network: {exc}")

        if last_error.status == 429:
            # Учитываем Retry-After даже на последней попытке и на HTTP 200 + error.
            llm_rate_limit_streak += 1
            backoff = min(900.0, LLM_RATE_LIMIT_PAUSE * 2 ** min(llm_rate_limit_streak - 1, 5))
            pause = max(retry_after, backoff) + random.uniform(0, 3)
            llm_cooldown_until = time.monotonic() + pause
            print(f"[LLM] Получен 429. Общая пауза {math.ceil(pause)} сек.; без немедленного повтора.")
            raise last_error

        retryable = last_error.status in RETRYABLE_STATUSES or last_error.status == 0
        if retryable:
            delay = max(retry_after, 5.0 * 2 ** attempt) + random.uniform(0, 2)
            # Отложенный запуск сохраняется и после исчерпания попыток.
            llm_next_request_at = max(llm_next_request_at, time.monotonic() + delay)
            if attempt < LLM_RETRIES - 1:
                # Очень долгий Retry-After не держит очередь; бот попросит повторить позже.
                if delay > LLM_QUEUE_TIMEOUT:
                    llm_cooldown_until = max(llm_cooldown_until, llm_next_request_at)
                    break
                print(f"[LLM] {last_error}; retry in {math.ceil(delay)}s ({attempt + 1}/{LLM_RETRIES})")
                continue
            if retry_after > 0:
                llm_cooldown_until = max(llm_cooldown_until, llm_next_request_at)
        break

    raise last_error


def image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def build_user_content(prompt: str, image_paths: list[str] | None):
    """Для текста — строка, для картинок — OpenAI-формат с image_url (base64 data URL)."""
    if not image_paths:
        return prompt
    parts = [{"type": "text", "text": prompt}]
    for path in image_paths:
        parts.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    return parts


JUNK_CONTENT = {"image received", "group user message", "continue", "продолжай"}
IMAGE_DEFAULT_PROMPT_MARK = "посмотри на изображение и опиши свою реакцию на него."


def clean_history(history) -> deque:
    cleaned = deque(maxlen=history.maxlen)
    for msg in history:
        content = msg.get("content") or ""
        role = msg.get("role", "")

        if not content.strip() or content in JUNK_CONTENT:
            continue
        if IMAGE_DEFAULT_PROMPT_MARK in content.lower():
            continue
        if cleaned and cleaned[-1]["role"] == role:
            print(f"Skipping duplicate {role} message")
            continue

        cleaned.append(msg)
    return cleaned


# === Единая функция запроса к LLM (текст и картинки) ===
async def ask_llm(user_id, chat_id, prompt, *, image_paths=None, save_prompt=None, is_private=False) -> str:
    has_images = bool(image_paths)
    context_type = resolve_context_type(is_private)
    history_user_id = user_id
    save_prompt = save_prompt or prompt
    reply_text = ""

    try:
        history = await get_context(chat_id, history_user_id, context_type)
        cleaned_history = clean_history(history)

        recent = list(cleaned_history)
        if has_images:
            recent = recent[-IMAGE_HISTORY_MESSAGES:]
        # Некоторые шаблоны чата не любят, когда после system сразу идёт assistant
        while recent and recent[0]["role"] != "user":
            recent.pop(0)

        messages = [{"role": "system", "content": get_full_system_prompt(user_id)}]
        messages += [{"role": m["role"], "content": m["content"]} for m in recent]
        messages.append({"role": "user", "content": build_user_content(prompt, image_paths)})

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kind = f"IMAGES x{len(image_paths)}" if has_images else "TEXT"
        print(f"\n[{timestamp}] === {kind} REQUEST === user={user_id} ({'you' if is_nick(user_id) else 'other'}) "
              f"chat={chat_id} ({context_type}) messages={len(messages)} history={len(recent)}")

        reply_text = await call_llm(messages)

        if reply_text:
            cleaned_history.append({"role": "user", "content": save_prompt})
            cleaned_history.append({"role": "assistant", "content": reply_text})
            await save_context(chat_id, history_user_id, context_type, cleaned_history)
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved context for user {history_user_id} in {context_type}")
        else:
            reply_text = "Не получилось описать изображение." if has_images else "Не получилось сформулировать ответ."

    except LLMError as e:
        print(f"LLM error: {e}")
        if e.status == 429:
            remaining = max(1, math.ceil(llm_cooldown_until - time.monotonic()))
            reply_text = (f"Сервис временно ограничил запросы. Бот сделал паузу. "
                          f"Попробуй ещё раз примерно через {remaining} сек. "
                          "Если исчерпан суточный лимит, потребуется дождаться его сброса.")
        else:
            reply_text = "Сейчас бот не отвечает, попробуй позже."
        # 400 обычно значит, что контекст/запрос кривой — сбрасываем историю.
        # На 429/5xx контекст трогать не нужно: он тут ни при чём.
        if e.status == 400:
            print(f"Resetting context for user {history_user_id} in chat {chat_id}")
            await save_context(chat_id, history_user_id, context_type, deque(maxlen=get_limit(context_type)))
    except Exception as e:
        reply_text = "Сейчас бот не отвечает, попробуй позже."
        print(f"LLM unexpected error: {e}")

    return reply_text


async def ask_llm_images(user_id, chat_id, prompt, image_paths: list, is_private=False) -> str:
    """Одна или несколько картинок в одном запросе."""
    num_images = len(image_paths)
    if num_images > 1:
        llm_prompt = (f"Ты видишь {num_images} изображений. {prompt}\n\n"
                      "Пожалуйста, рассмотри все изображения и дай общий синтезированный ответ.")
        save_prompt = f"прислал {num_images} изображения"
    else:
        llm_prompt = prompt
        # Дефолтные промпты "опиши изображение" в историю не пишем
        save_prompt = "прислал изображение" if "опиши изображение" in prompt.lower() else prompt

    return await ask_llm(user_id, chat_id, llm_prompt, image_paths=image_paths,
                         save_prompt=save_prompt, is_private=is_private)


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
    queued = False

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
            await send_reply(api_chat_id, "Не удалось обработать одно или несколько изображений.",
                             None if is_private else message_id)
            return

        await bot.send_chat_action(api_chat_id, "typing")

        # Обработка фото
        if is_private:
            full_text = await ask_llm_images(sender_id, context_chat_id, prompt, resized_paths, is_private=True)
            await send_reply(api_chat_id, full_text)
        else:
            if api_chat_id not in group_image_queues:
                group_image_queues[api_chat_id] = asyncio.Queue()
                group_image_locks[api_chat_id] = asyncio.Lock()
            await group_image_queues[api_chat_id].put((sender_id, context_chat_id, prompt, message_id, resized_paths))
            queued = True  # файлы удалит сама очередь
            asyncio.create_task(process_group_image_queue(api_chat_id))

    except Exception as e:
        print(f"Error processing media group {media_group_id}: {e}")

    finally:
        # Очищаем временные файлы
        for path in tmp_paths:
            try: os.remove(path)
            except Exception: pass
        if not queued:
            for path in resized_paths:
                try: os.remove(path)
                except Exception: pass

        # Удаляем из буферов
        media_group_buffers.pop(media_group_id, None)
        media_group_timers.pop(media_group_id, None)


# === Хэндлеры ===
@dp.message(Command("reset"))
async def handle_reset(message: types.Message):
    """Сбрасывает контекст диалога (в ЛС и в группах комментариев)."""
    sender_id = extract_sender_id(message)
    chat_id = normalize_id(message.chat.id)
    if not sender_id or chat_id is None:
        return

    is_private = message.chat.type == "private"
    if not is_private and chat_id not in COMMENTS_GROUP_IDS:
        return

    await reset_context(chat_id, get_context_user_id(sender_id), resolve_context_type(is_private))
    await message.reply("Контекст сброшен. Начинаем с чистого листа.")


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

    if not should_respond_in_chat(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [SKIP] sender_id={sender_id}, chat_id={context_chat_id}, ignored")
        return

    # Получаем фокусный ID для группировки контекста
    context_user_id = get_context_user_id(sender_id)

    if is_private:
        await bot.send_chat_action(api_chat_id, "typing")
        full_text = await ask_llm(context_user_id, context_chat_id, message.text, is_private=True)
        await send_reply(api_chat_id, full_text, message.message_id)
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

    queued = False
    try:
        await bot.download(photo.file_id, destination=tmp_path)
        buffer = resize_image(tmp_path)
        if not buffer:
            await send_reply(api_chat_id, "Не удалось обработать изображение.", message.message_id)
            return

        with open(resized_path, 'wb') as f:
            f.write(buffer.getvalue())

        await bot.send_chat_action(api_chat_id, "typing")

        if is_private:
            full_text = await ask_llm_images(context_user_id, context_chat_id, prompt, [resized_path], is_private=True)
            await send_reply(api_chat_id, full_text, message.message_id)
        else:
            if api_chat_id not in group_image_queues:
                group_image_queues[api_chat_id] = asyncio.Queue()
                group_image_locks[api_chat_id] = asyncio.Lock()
            await group_image_queues[api_chat_id].put((context_user_id, context_chat_id, prompt, message.message_id, resized_path))
            queued = True  # файл удалит сама очередь
            asyncio.create_task(process_group_image_queue(api_chat_id))
    finally:
        try: os.remove(tmp_path)
        except Exception: pass
        if not queued:
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
    global http_session
    http_session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120))
    await bot.delete_webhook(drop_pending_updates=True)
    await init_db()
    await cleanup_all_contexts()
    try:
        await dp.start_polling(bot)
    finally:
        if db:
            await db.close()
        await http_session.close()
        await bot.session.close()

if __name__ == "__main__":
    print("Бот запущен")
    asyncio.run(main())