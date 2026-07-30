import os
import json
import time
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv  # Imports env loader
from fastapi import FastAPI
from openai import OpenAI
from openai import APIStatusError
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# Load keys from local .env file if running locally
load_dotenv()

# --- CONFIGURATION (SECURE LAYER) ---
# os.environ.get reads from the .env file locally, or from Render's Settings on the web
TELEGRAM_BOT_TOKEN = os.environ.get("BOT_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
# Your own public server URL — set this to the Render URL after first deploy,
# e.g. BASE_URL=https://your-service.onrender.com. Defaults to localhost for
# local testing so nothing crashes if it's unset.
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000").rstrip("/")

GITHUB_USERNAME = "24f2001882"
GITHUB_REPO = "q5tds"
GITHUB_BRANCH = "main"

LOG_FILE = "run.jsonl"
LOG_URL = f"https://raw.githubusercontent.com/{GITHUB_USERNAME}/{GITHUB_REPO}/{GITHUB_BRANCH}/{LOG_FILE}"

# Initialize client using Google's official structural endpoint
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai",
    api_key=GEMINI_API_KEY
)

# Keeps the last few messages per chat, so multi-turn questions work
conversation_history = {}


def _push_log_to_github():
    """Runs in a background thread so git push never blocks the Telegram reply."""
    try:
        os.system(f'git add {LOG_FILE} && git commit -m "update log" && git push -q')
    except Exception as e:
        print(f"⚠️ Git push failed: {e}")


def log_event(event: dict):
    event["timestamp"] = time.time()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    # Push in the background so the git commit/push never blocks the reply
    threading.Thread(target=_push_log_to_github, daemon=True).start()


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    user_text = update.message.text
    log_event({"type": "incoming", "chat_id": chat_id, "text": user_text})

    history = conversation_history.setdefault(chat_id, [])
    history.append({"role": "user", "content": user_text})

    system_prompt = (
        "You are a careful data analyst. The user's LAST message asks a data-analysis "
        "question and tells you exactly what JSON shape the answer content should take. "
        "Work out the real answer (use any public data you know, e.g. MOSPI statistics, "
        "general world knowledge, or arithmetic on numbers given in the message). "
        "Your reply must ALWAYS be a single JSON object of the exact form "
        '{"answer": <the shape the question asked for>, "log_url": "..."} — the shape '
        'the question shows you (e.g. {"state": "<state name>"}) goes INSIDE the '
        '"answer" key, never at the top level. '
        "Reply with ONLY that JSON object and absolutely nothing else — no "
        "explanation, no markdown, no code fences, just the raw JSON."
    )

    # Real production model names valid on Google's OpenAI-compatible endpoint.
    # Each has a 5 RPM limit on the free tier — cascading on 429 spreads load across them.
    model_cascade = [
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ]

    response = None
    # Model fallback cascade loop
    for model_name in model_cascade:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}] + history[-6:],
            )
            # If successful, exit the fallback cascade loop
            break
        except APIStatusError as e:
            # Catch Rate Limits (429), Model Overload/Unavailable (503), or Deprecated Models (404)
            if e.status_code in [404, 429, 503]:
                print(f"⚠️ {model_name} failed (Status {e.status_code}). Cascading to next fallback...")
                time.sleep(1)  # brief buffer pause between cascade attempts
                continue
            else:
                # Re-raise critical authorization errors (like 401 Invalid Key)
                raise e
        except Exception as e:
            print(f"⚠️ Unexpected system exception with {model_name}: {e}")
            continue

    if not response or not response.choices:
        print("❌ All models in the Gemini cascade failed or were unavailable.")
        await update.message.reply_text("⚠️ System overloaded. Please try again in a moment.")
        return

    reply_text = response.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": reply_text})

    # Robust JSON extraction to prevent format_errors
    try:
        parsed = json.loads(reply_text)
    except json.JSONDecodeError:
        # If the model added code fences like ```json ... ```, strip and extract the text
        start, end = reply_text.find("{"), reply_text.rfind("}")
        if start != -1 and end != -1:
            try:
                parsed = json.loads(reply_text[start:end + 1])
            except Exception:
                parsed = {"error": "Invalid response format from model"}
        else:
            parsed = {"error": "Invalid response format from model"}

    # Enforce the required envelope: {"answer": <question shape>, "log_url": ...}
    # The model sometimes replies with the question's inner shape directly
    # (e.g. {"state": "Assam"}) instead of nesting it under "answer" — this
    # normalizes either case so the outer key is always present.
    if not isinstance(parsed, dict) or "answer" not in parsed:
        parsed = {"answer": parsed}

    # Inject your absolute working Raw GitHub link
    parsed["log_url"] = LOG_URL
    final_reply = json.dumps(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)


# ------------------------------------------------------------------
# Telegram application (long polling) — kept as its own object, separate
# from the FastAPI `app` below, since Render's `uvicorn bot:app` needs
# `app` to be the FastAPI instance.
# ------------------------------------------------------------------
telegram_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
telegram_app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))


def run_telegram_bot():
    # stop_signals=None: signal handlers can only be registered on the main
    # thread, and this runs in a background thread.
    telegram_app.run_polling(stop_signals=None)


def self_ping_loop():
    """Keeps Render's free-tier instance from spinning down after ~15 min idle."""
    while True:
        time.sleep(600)  # 10 minutes
        try:
            requests.get(f"{BASE_URL}/health", timeout=15)
        except Exception:
            pass


# ------------------------------------------------------------------
# FastAPI app — this is what Render/uvicorn actually binds to $PORT.
# Telegram polling and the self-ping run alongside it in background threads.
# ------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=run_telegram_bot, daemon=True).start()
    threading.Thread(target=self_ping_loop, daemon=True).start()
    print("Bot is running (Telegram polling + self-ping started)...")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.now(timezone.utc).isoformat()}


if __name__ == "__main__":
    # Local convenience: `python bot.py` works too, not just uvicorn.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))