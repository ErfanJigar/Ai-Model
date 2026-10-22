import os
import threading
import urllib.request
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from llama_cpp import Llama

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/tmp/qwen-model"))
MODEL_PATH = MODEL_DIR / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
MODEL_URL = (
    "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/"
    "qwen2.5-0.5b-instruct-q4_k_m.gguf"
)

app = Flask(__name__)
_model = None
_model_lock = threading.Lock()

PAGE = r'''<!doctype html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen محلی</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{margin:0;background:#10131b;color:#edf0f7;font-family:Tahoma,Arial,sans-serif;display:flex;justify-content:center;min-height:100vh;padding:18px}.app{width:min(780px,100%);display:flex;flex-direction:column;gap:14px}.top{padding:18px;border:1px solid #303748;border-radius:16px;background:#171c27}.top h1{font-size:20px;margin:0 0 8px}.top p{margin:0;color:#aab3c5;font-size:13px;line-height:1.9}.chat{border:1px solid #303748;border-radius:16px;background:#141923;min-height:42vh;max-height:62vh;overflow:auto;padding:16px;display:flex;flex-direction:column;gap:12px}.msg{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.9;padding:12px 14px;border-radius:12px;max-width:94%}.user{align-self:flex-start;background:#263b61}.bot{align-self:flex-end;background:#202938}.status{font-size:12px;color:#c7d1e4;min-height:20px;line-height:1.8}.form{display:flex;gap:8px}.form textarea{flex:1;resize:vertical;min-height:66px;max-height:180px;background:#171c27;color:#fff;border:1px solid #394257;border-radius:12px;padding:12px;font:inherit;line-height:1.7}.form button{border:0;border-radius:12px;padding:0 18px;background:#8aa4ff;color:#071022;font-weight:bold;cursor:pointer}.form button:disabled{opacity:.5;cursor:wait}.foot{font-size:11px;color:#8993a6;line-height:1.9}@media(max-width:520px){body{padding:10px}.form{flex-direction:column}.form button{padding:12px}}
</style></head>
<body><main class="app">
<section class="top"><h1>چت‌بات Qwen محلی</h1><p>بدون API هوش مصنوعی؛ مدل از Hugging Face دانلود و روی همین سرویس اجرا می‌شود. دانلود مدل به اینترنت نیاز دارد. روی پلن رایگان ۵۱۲ مگابایتی ممکن است به علت کمبود RAM اجرا نشود.</p></section>
<section id="chat" class="chat" aria-live="polite"><div class="msg bot">سلام! یک سؤال کوتاه بپرس. بار اول ممکن است دانلود مدل زمان ببرد.</div></section>
<div id="status" class="status"></div>
<form id="form" class="form"><textarea id="input" maxlength="1000" placeholder="پیامت را بنویس..." required></textarea><button id="send" type="submit">ارسال</button></form>
<div class="foot">مدل: Qwen2.5-0.5B-Instruct Q4_K_M · بدون API هوش مصنوعی · پاسخ‌ها ممکن است کند یا نادقیق باشند. سرویس رایگان Render در زمان بیکاری خاموش می‌شود و فایل مدل ممکن است مجدداً دانلود شود.</div>
</main><script>
const form=document.getElementById('form'),input=document.getElementById('input'),chat=document.getElementById('chat'),statusEl=document.getElementById('status'),send=document.getElementById('send');
const history=[];
function add(text,who){const el=document.createElement('div');el.className='msg '+who;el.textContent=text;chat.appendChild(el);chat.scrollTop=chat.scrollHeight;}
form.addEventListener('submit',async e=>{e.preventDefault();const text=input.value.trim();if(!text)return;add(text,'user');history.push({role:'user',content:text});input.value='';send.disabled=true;statusEl.textContent='در صورت اولین اجرا، فایل حدود ۴۹۱ مگابایتی دانلود می‌شود. لطفاً منتظر بمان؛ اگر سرویس کمبود حافظه داشته باشد ممکن است متوقف شود.';try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({messages:history.slice(-4)})});let data={};try{data=await r.json()}catch(_){}if(!r.ok)throw new Error(data.error||'پاسخی از سرویس دریافت نشد');add(data.reply||'پاسخی تولید نشد.','bot');history.push({role:'assistant',content:data.reply||''});}catch(err){add('خطا: '+err.message,'bot');statusEl.textContent='بخش Logs در داشبورد Render را بررسی کن؛ خطای Out of memory یعنی حافظه کافی نیست.';}finally{send.disabled=false;input.focus();if(statusEl.textContent.startsWith('در صورت اولین اجرا'))statusEl.textContent='';}});
</script></body></html>'''


def ensure_model_file():
    """Download the public GGUF file only when it is not already on this instance."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 400_000_000:
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    partial = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".part")
    try:
        req = urllib.request.Request(MODEL_URL, headers={"User-Agent": "qwen-local-render/1.0"})
        # urllib's timeout is a socket inactivity timeout; the loop streams the large file.
        with urllib.request.urlopen(req, timeout=120) as response, partial.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if partial.stat().st_size < 400_000_000:
            raise RuntimeError(f"فایل مدل ناقص دانلود شده است ({partial.stat().st_size} bytes).")
        partial.replace(MODEL_PATH)
    finally:
        if partial.exists():
            try:
                partial.unlink()
            except OSError:
                pass


def get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        ensure_model_file()
        # Intentionally conservative values for a tiny CPU/RAM plan.
        # These settings reduce memory use; they cannot make a 491 MB model fit in 512 MB reliably.
        _model = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=128,
            n_threads=1,
            n_batch=64,
            n_ubatch=8,
            n_gpu_layers=0,
            use_mmap=True,
            use_mlock=False,
            verbose=False,
        )
        return _model


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.get("/health")
def health():
    return jsonify({"ok": True, "model_loaded": _model is not None})


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    messages = payload.get("messages", [])
    if not isinstance(messages, list) or not messages:
        return jsonify({"error": "پیام معتبری دریافت نشد."}), 400

    cleaned = [{"role": "system", "content": "You are a helpful assistant. Reply briefly in the same language as the user. If the user writes Persian, try to answer in Persian. Be honest if you do not know."}]
    for item in messages[-4:]:
        if not isinstance(item, dict) or item.get("role") not in {"user", "assistant"}:
            continue
        content = str(item.get("content", ""))[:350].strip()
        if content:
            cleaned.append({"role": item["role"], "content": content})
    if len(cleaned) < 2 or cleaned[-1]["role"] != "user":
        return jsonify({"error": "آخرین پیام باید از طرف کاربر باشد."}), 400

    try:
        model = get_model()
        result = model.create_chat_completion(
            messages=cleaned,
            max_tokens=24,
            temperature=0.7,
            top_p=0.85,
            repeat_penalty=1.1,
        )
        reply = (result.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        if not reply:
            reply = "جواب قابل‌نمایشی تولید نشد؛ سؤال را کوتاه‌تر بپرس."
        return jsonify({"reply": reply})
    except Exception as exc:
        app.logger.exception("Local Qwen inference failed")
        return jsonify({"error": "مدل اجرا نشد. Logs در Render را بررسی کن. جزئیات: " + str(exc)[:220]}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
