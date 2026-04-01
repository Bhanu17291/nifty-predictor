import pandas as pd
import streamlit as st
from datetime import date, timedelta
import os

EVENTS_FILE = "data/events_calendar.csv"

HARDCODED_EVENTS = [
    {"date": "2025-04-09", "event": "RBI Policy Decision",        "impact": "HIGH"},
    {"date": "2025-04-10", "event": "US CPI Data",                "impact": "HIGH"},
    {"date": "2025-04-17", "event": "F&O Expiry",                 "impact": "MEDIUM"},
    {"date": "2025-05-07", "event": "RBI Policy Decision",        "impact": "HIGH"},
    {"date": "2025-05-29", "event": "F&O Expiry",                 "impact": "MEDIUM"},
    {"date": "2025-06-06", "event": "US Jobs Report (NFP)",       "impact": "HIGH"},
    {"date": "2025-06-18", "event": "Fed Interest Rate Decision",  "impact": "HIGH"},
    {"date": "2025-06-26", "event": "F&O Expiry",                 "impact": "MEDIUM"},
    {"date": "2026-02-01", "event": "Union Budget",               "impact": "HIGH"},
]

def load_events() -> pd.DataFrame:
    if os.path.exists(EVENTS_FILE):
        try:
            df = pd.read_csv(EVENTS_FILE, parse_dates=["date"])
            if "impact" not in df.columns:
                df["impact"] = "MEDIUM"
            return df
        except Exception:
            pass
    df = pd.DataFrame(HARDCODED_EVENTS)
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_upcoming_events(days_ahead: int = 14) -> pd.DataFrame:
    df    = load_events()
    today = pd.Timestamp(date.today())
    end   = today + pd.Timedelta(days=days_ahead)
    return df[(df["date"] >= today) & (df["date"] <= end)].sort_values("date")

def render_economic_calendar():
    st.markdown('<p class="sec-label">Risk Events</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Economic Calendar — Next 14 Days</p>', unsafe_allow_html=True)

    upcoming = get_upcoming_events(14)

    if upcoming.empty:
        st.success("No major risk events in the next 14 days.")
        return

    for _, row in upcoming.iterrows():
        days_away = (row["date"].date() - date.today()).days
        impact    = row.get("impact", "MEDIUM")
        color     = "#ef4444" if impact == "HIGH" else "#f59e0b"
        label     = "Today" if days_away == 0 else (f"Tomorrow" if days_away == 1 else f"In {days_away} days")
        st.markdown(f"""
        <div style='background:rgba(15,23,42,.7);border:1px solid {color};border-left:4px solid {color};
                    border-radius:8px;padding:.8rem 1.2rem;margin:.4rem 0;display:flex;
                    justify-content:space-between;align-items:center;'>
            <div>
                <span style='font-family:IBM Plex Mono,monospace;font-size:.8rem;color:#64748b;'>{row['date'].strftime('%d %b %Y')}</span>
                <span style='font-family:IBM Plex Sans,sans-serif;font-size:1rem;color:#e2e8f0;margin-left:1rem;font-weight:500;'>{row['event']}</span>
            </div>
            <div>
                <span style='background:{"rgba(239,68,68,.2)" if impact=="HIGH" else "rgba(245,158,11,.2)"};
                             color:{color};padding:.2rem .7rem;border-radius:12px;
                             font-family:IBM Plex Mono,monospace;font-size:.75rem;font-weight:600;'>{impact}</span>
                <span style='color:#64748b;font-family:IBM Plex Mono,monospace;font-size:.8rem;margin-left:.8rem;'>{label}</span>
            </div>
        </div>""", unsafe_allow_html=True)