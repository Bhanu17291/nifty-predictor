import json, os
import yfinance as yf
from datetime import datetime, timedelta

PRED_FILE = "data/predictions.json"

def load_predictions():
    if not os.path.exists(PRED_FILE):
        return []
    with open(PRED_FILE, "r") as f:
        return json.load(f)

def save_predictions(data):
    with open(PRED_FILE, "w") as f:
        json.dump(data, f, indent=2)

def log_prediction(direction: str, confidence: float, date: str = None):
    date = date or datetime.today().strftime("%Y-%m-%d")
    preds = load_predictions()
    preds = [p for p in preds if p["date"] != date]
    preds.append({"date": date, "prediction": direction, "confidence": round(confidence, 4), "actual": None})
    save_predictions(preds)

def update_actuals():
    preds = load_predictions()
    nifty = yf.download("^NSEI", period="10d", interval="1d", progress=False)
    updated = 0
    for i, p in enumerate(preds):
        if p["actual"] is not None:
            continue
        try:
            date = datetime.strptime(p["date"], "%Y-%m-%d")
            next_day = date + timedelta(days=1)
            mask = nifty.index.date == next_day.date()
            if mask.sum() == 0:
                continue
            row = nifty[mask].iloc[0]
            actual_dir = "UP" if row["Close"].item() >= row["Open"].item() else "DOWN"
            preds[i]["actual"] = actual_dir
            updated += 1
        except Exception:
            continue
    save_predictions(preds)
    return updated

def get_scorecard(days: int = 30):
    preds = load_predictions()
    cutoff = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    recent = [p for p in preds if p["date"] >= cutoff and p["actual"] is not None]
    if not recent:
        return {"total": 0, "correct": 0, "accuracy": 0.0}
    correct = sum(1 for p in recent if p["prediction"] == p["actual"])
    return {"total": len(recent), "correct": correct, "accuracy": round(correct / len(recent), 4)}