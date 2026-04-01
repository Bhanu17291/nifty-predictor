import os
import asyncio
from telegram import Bot, Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

async def send_prediction(chat_id: str, direction: str, confidence: float, signal_label: str, driver: str):
    if not TOKEN:
        return
    bot = Bot(token=TOKEN)
    icon = "UP" if direction == "UP" else "DOWN"
    msg = (
        f"Nifty Prediction\n\n"
        f"Direction: {icon} {direction}\n"
        f"Confidence: {confidence:.1%}\n"
        f"Signal: {signal_label}\n"
        f"Key driver: {driver}\n\n"
        f"Powered by Nifty Intelligence"
    )
    await bot.send_message(chat_id=chat_id, text=msg)

async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await update.message.reply_text(
        f"Welcome to Nifty Intelligence Bot!\n\nYour chat ID: {chat_id}\n"
        "Add this to your .env as TELEGRAM_CHAT_ID to receive daily predictions."
    )

def run_bot():
    if not TOKEN:
        print("TELEGRAM_BOT_TOKEN not set — skipping bot.")
        return
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start_handler))
    print("Telegram bot running... press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    run_bot()