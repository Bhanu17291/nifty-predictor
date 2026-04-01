import os
import asyncio
from apscheduler.schedulers.blocking import BlockingScheduler
from dotenv import load_dotenv

load_dotenv()

from outcome_tracker import update_actuals, log_prediction, get_scorecard
from drift_monitor import log_drift
from live_predict_v2 import get_prediction   # your existing function
from bot import send_prediction

scheduler = BlockingScheduler(timezone="Asia/Kolkata")

@scheduler.scheduled_job("cron", hour=8, minute=45)
def morning_job():
    print("Running morning prediction job...")
    try:
        result = get_prediction()
        direction   = result["direction"]
        confidence  = result["confidence"]
        driver      = result.get("top_feature", "GIFT Nifty")
        signal      = result.get("signal_label", "Moderate")
        log_prediction(direction, confidence)
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if chat_id:
            asyncio.run(send_prediction(chat_id, direction, confidence, signal, driver))
        print(f"Prediction sent: {direction} @ {confidence:.1%}")
    except Exception as e:
        print(f"Morning job failed: {e}")

@scheduler.scheduled_job("cron", hour=16, minute=15)
def evening_job():
    print("Running evening actuals update...")
    try:
        updated = update_actuals()
        log_drift()
        score = get_scorecard(30)
        print(f"Updated {updated} actuals. 30-day accuracy: {score['accuracy']:.1%}")
    except Exception as e:
        print(f"Evening job failed: {e}")

if __name__ == "__main__":
    print("Scheduler started. Jobs: 8:45 AM and 4:15 PM IST.")
    scheduler.start()