import pandas as pd
from datetime import datetime, date
import os

EVENTS_FILE = "data/events_calendar.csv"

def _is_expiry_day(today: date) -> bool:
    if today.weekday() != 3:
        return False
    last_thursday = max(
        d for d in [date(today.year, today.month, d) for d in range(25, 32)
                    if d <= 31 and date(today.year, today.month, d).weekday() == 3]
    )
    return today == last_thursday

def _days_to_event(today: date) -> int:
    if not os.path.exists(EVENTS_FILE):
        return 99
    df = pd.read_csv(EVENTS_FILE, parse_dates=["date"])
    future = df[df["date"].dt.date >= today]
    if future.empty:
        return 99
    return (future["date"].dt.date.min() - today).days

def compute_signal(confidence: float, vix: float, today: date = None) -> dict:
    today = today or date.today()
    base = confidence
    vix_penalty   = 0.08 if vix > 20 else (0.04 if vix > 15 else 0)
    expiry_penalty = 0.05 if _is_expiry_day(today) else 0
    event_penalty  = 0.05 if _days_to_event(today) <= 2 else 0
    final = max(0.0, base - vix_penalty - expiry_penalty - event_penalty)

    if final >= 0.70:
        label, color = "High conviction", "green"
    elif final >= 0.60:
        label, color = "Moderate — trade with caution", "orange"
    else:
        label, color = "Weak — consider sitting out", "red"

    return {
        "score": round(final, 4),
        "label": label,
        "color": color,
        "vix_penalty": vix_penalty,
        "expiry_penalty": expiry_penalty,
        "event_penalty": event_penalty,
        "is_expiry": _is_expiry_day(today),
        "days_to_event": _days_to_event(today),
    }