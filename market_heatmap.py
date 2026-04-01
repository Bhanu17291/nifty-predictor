import yfinance as yf
import streamlit as st

TICKERS = {
    "GIFT Nifty": "^NSEI",
    "S&P 500":    "^GSPC",
    "Nasdaq":     "^IXIC",
    "Crude Oil":  "CL=F",
    "DXY":        "DX-Y.NYB",
    "Gold":       "GC=F",
    "India VIX":  "^INDIAVIX",
}

def fetch_changes():
    results = {}
    for name, ticker in TICKERS.items():
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if len(data) >= 2:
                prev = data["Close"].iloc[-2].item()
                curr = data["Close"].iloc[-1].item()
                pct = ((curr - prev) / prev) * 100
                results[name] = round(pct, 2)
            else:
                results[name] = None
        except Exception:
            results[name] = None
    return results

def render_heatmap():
    st.subheader("Global Market Snapshot")
    changes = fetch_changes()
    cols = st.columns(len(TICKERS))
    for col, (name, pct) in zip(cols, changes.items()):
        with col:
            if pct is None:
                st.metric(name, "N/A")
            else:
                color = "normal" if pct >= 0 else "inverse"
                st.metric(label=name, value=f"{pct:+.2f}%", delta=f"{pct:+.2f}%", delta_color=color)