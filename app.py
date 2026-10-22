import asyncio
import logging
import os
import threading
import urllib.request
from pathlib import Path
from typing import Any

from llama_cpp import Llama
from maxrubika import Bot

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("qwen-rubika-bot")

# ----------------------------
# Model configuration
# ----------------------------
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/tmp/qwen-model"))
MODEL_PATH = MODEL_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

_model: Llama | None = None
_model_lock = threading.Lock()
_inference_lock = threading.Lock()

SYSTEM_PROMPT = (
    "You are a helpful assistant. Reply briefly in the same language as the user. "
    "If the user writes Persian, try to answer in Persian. Be honest if you do not know."
)

# Keep a short conversation history separately for each Rubika chat.
# Entries are protected by _inference_lock because model generation is serialized.
_conversations: dict[str, list[dict[str, str]]] = {}


def ensure_model_file() -> None:
    """Download the public GGUF model to temporary disk, without loading it into RAM."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 400_000_000:
        logger.info("Model file already exists: %s bytes", MODEL_PATH.stat().st_size)
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
    try:
        logger.info("Downloading Qwen GGUF model from Hugging Face; this is about 491 MB.")
        req = urllib.request.Request(
            MODEL_URL,
            headers={"User-Agent": "qwen-local-rubika-render/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)

        size = partial.stat().st_size
        if size < 400_000_000:
            raise RuntimeError(f"The downloaded model looks incomplete ({size} bytes).")
        partial.replace(MODEL_PATH)
        logger.info("Model download complete: %s bytes", size)
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                logger.warning("Could not delete incomplete model download: %s", partial)


def get_model() -> Llama:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        ensure_model_file()
        logger.info("Loading Qwen model into memory. The free 512 MB plan may run out of RAM.")
        _model = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=128,
            n_threads=1,
            n_threads_batch=1,
            n_batch=64,
            n_ubatch=8,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        logger.info("Qwen model loaded successfully")
        return _model


def generate_reply_sync(chat_id: str, user_text: str) -> str:
    """Run the synchronous llama.cpp inference outside the Rubika event loop."""
    # One inference at a time: this is safer for the tiny CPU/RAM plan.
    with _inference_lock:
        history = _conversations.setdefault(chat_id, [])
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_text[:700]})

        model = get_model()
        result: dict[str, Any] = model.create_chat_completion(
            messages=messages,
            max_tokens=48,
            temperature=0.7,
            top_p=0.85,
            repeat_penalty=1.1,
        )
        reply = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        if not reply:
            reply = "جواب قابل‌نمایشی تولید نشد؛ لطفاً سؤال کوتاه‌تری بپرس."

        history.extend(
            [
                {"role": "user", "content": user_text[:700]},
                {"role": "assistant", "content": reply},
            ]
        )
        # Retain only two recent user/assistant turns per chat.
        _conversations[chat_id] = history[-4:]
        return reply


async def process_and_reply(bot: Bot, event: Any, user_text: str) -> None:
    """Acknowledge immediately, then generate and send the final answer."""
    try:
        await event.reply(
            "پیامت رسید ✅\nدارم جواب رو با مدل محلی Qwen تولید می‌کنم. "
            "بار اول ممکنه دانلود مدل حدود ۴۹۱ مگابایتی زمان ببره."
        )
    except Exception:
        logger.exception("Could not send the initial acknowledgement to Rubika")

    try:
        chat_id = str(getattr(event, "chat_id", "") or "unknown-chat")
        reply = await asyncio.to_thread(generate_reply_sync, chat_id, user_text)
        # Rubika messages have practical length limits; the model is configured for short answers.
        await event.reply(reply[:3500])
        logger.info("Replied successfully to chat %s", chat_id)
    except Exception:
        logger.exception("Local Qwen inference failed")
        try:
            await event.reply(
                "متأسفانه مدل نتونست جواب بسازه. لطفاً Logs در Render رو بررسی کن؛ "
                "ممکنه مدل دانلود نشده باشه یا RAM هاست کافی نباشه."
            )
        except Exception:
            logger.exception("Could not send the inference-error message to Rubika")


async def main() -> None:
    token = os.environ.get("BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "Missing BOT_TOKEN environment variable. Add BOT_TOKEN in Render Environment."
        )

    bot = Bot(token, show_welcome=False)
    active_tasks: set[asyncio.Task] = set()

    @bot.on_message()
    async def on_message(bot_instance: Bot, event: Any) -> None:
        try:
            if getattr(event, "update_type", None) not in (None, "NewMessage"):
                return
            # Avoid responding to other bots if the library supplies sender_type.
            if getattr(event, "sender_type", None) == "Bot":
                return

            text = (getattr(event, "text", None) or "").strip()
            if not text:
                return

            normalized = text.casefold()
            if normalized in {"/start", "start", "/شروع", "شروع"}:
                await event.reply(
                    "سلام! من چت‌بات Qwen هستم و پاسخ‌ها را با یک مدل محلی تولید می‌کنم.\n"
                    "هر سؤال متنی بفرستی، آن را به مدل می‌دهم.\n\n"
                    "توجه: روی هاست رایگان با ۵۱۲ مگابایت RAM ممکن است مدل کند باشد یا به کمبود حافظه بخورد."
                )
                return
            if normalized in {"/help", "help", "راهنما"}:
                await event.reply("سؤالت رو به صورت پیام متنی بفرست. برای شروع هم /start رو ارسال کن.")
                return

            task = asyncio.create_task(process_and_reply(bot_instance, event, text))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)
        except Exception:
            logger.exception("Failed while handling a Rubika update")

    port = int(os.environ.get("PORT", "10000"))
    public_url = (
        os.environ.get("RENDER_EXTERNAL_URL")
        or os.environ.get("PUBLIC_URL")
        or "https://ai-model-c2f5.onrender.com"
    ).rstrip("/")

    logger.info("Starting MAXRubika webhook server on port %s", port)
    logger.info("Registering Rubika webhook at %s/wk", public_url)
    # MAXRubika registers the Rubika endpoints and serves incoming updates on /wk.
    # Webhook mode suits a Render Web Service better than a polling-only background loop.
    await bot.start(
        webhook_url=public_url,
        webhook_path="/wk",
        host="0.0.0.0",
        port=port,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped")
    except Exception:
        logger.exception("The Rubika bot could not start")
        raise
