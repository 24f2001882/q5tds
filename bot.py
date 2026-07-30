import os
import json
import time
import threading
from dotenv import load_dotenv  # Imports env loader
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
        "question and tells you exactly what JSON shape to reply with. Work out the "
        "real answer (use any public data you know, e.g. MOSPI statistics, general "
        "world knowledge, or arithmetic on numbers given in the message). "
        "Reply with ONLY that exact JSON object and absolutely nothing else — no "
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
                time.sleep(0.5)  # Brief buffer pause
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

    # Inject your absolute working Raw GitHub link
    parsed["log_url"] = LOG_URL
    final_reply = json.dumps(parsed)

    log_event({"type": "outgoing", "chat_id": chat_id, "text": final_reply})
    await update.message.reply_text(final_reply)


app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
print("Bot is running... (Ctrl+C to stop)")
app.run_polling()