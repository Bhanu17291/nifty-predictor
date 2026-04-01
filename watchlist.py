import json, os, smtplib
import streamlit as st
import yfinance as yf
import pandas as pd
from email.mime.text import MIMEText

PREFS_FILE = "data/user_prefs.json"

def load_prefs():
    if not os.path.exists(PREFS_FILE):
        return {"tickers": [], "email": ""}
    with open(PREFS_FILE) as f:
        return json.load(f)

def save_prefs(prefs):
    with open(PREFS_FILE, "w") as f:
        json.dump(prefs, f, indent=2)

def compute_correlations(tickers: list) -> pd.DataFrame:
    nifty = yf.download("^NSEI", period="1y", interval="1d", progress=False)["Close"]
    nifty_dir = (nifty.diff() > 0).astype(int)
    rows = []
    for t in tickers:
        try:
            stock = yf.download(t + ".NS", period="1y", interval="1d", progress=False)["Close"]
            stock_dir = (stock.diff() > 0).astype(int)
            aligned = pd.concat([nifty_dir, stock_dir], axis=1).dropna()
            aligned.columns = ["nifty", "stock"]
            both_up = ((aligned["nifty"] == 1) & (aligned["stock"] == 1)).sum()
            nifty_up = (aligned["nifty"] == 1).sum()
            pct = round(both_up / nifty_up * 100, 1) if nifty_up else 0
            rows.append({"Ticker": t, "When Nifty UP, stock also UP": f"{pct}%"})
        except Exception:
            rows.append({"Ticker": t, "When Nifty UP, stock also UP": "N/A"})
    return pd.DataFrame(rows)

def send_email_alert(to_email: str, direction: str, confidence: float):
    sender = os.getenv("SENDER_EMAIL")
    password = os.getenv("SENDER_PASSWORD")
    if not sender or not password:
        return False
    try:
        msg = MIMEText(
            f"Nifty Intelligence Alert\n\nDirection: {direction}\nConfidence: {confidence:.1%}\n\nPowered by Nifty Intelligence"
        )
        msg["Subject"] = f"Nifty Prediction: {direction} today"
        msg["From"] = sender
        msg["To"] = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(sender, password)
            s.send_message(msg)
        return True
    except Exception:
        return False

def render_watchlist():
    st.subheader("My watchlist")
    prefs = load_prefs()
    raw = st.text_input("NSE tickers (comma-separated)", value=", ".join(prefs.get("tickers", [])))
    email = st.text_input("Alert email", value=prefs.get("email", ""))
    if st.button("Save & compute correlations"):
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        save_prefs({"tickers": tickers, "email": email})
        if tickers:
            with st.spinner("Computing correlations..."):
                df = compute_correlations(tickers)
            st.dataframe(df, use_container_width=True)