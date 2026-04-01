import yfinance as yf
import pandas as pd
import numpy as np
import pickle, os
import streamlit as st
import matplotlib.pyplot as plt

INDICES = {
    "Nifty 50"  : {"ticker": "^NSEI",     "model": "xgb_model_v2.pkl"},
    "BankNifty" : {"ticker": "^NSEBANK",  "model": "xgb_model_v2.pkl"},
    "FinNifty"  : {"ticker": "^CNXFIN",   "model": "xgb_model_v2.pkl"},
}

MODEL_DIR = "models"
DATA_DIR  = "data"

def fetch_index_data(ticker: str, days: int = 80) -> pd.DataFrame:
    from datetime import datetime, timedelta
    end   = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                         end=end.strftime("%Y-%m-%d"), progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return pd.DataFrame()

def compute_index_signal(ticker: str) -> dict:
    df = fetch_index_data(ticker)
    if df.empty or len(df) < 5:
        return {"available": False}
    try:
        closes  = df["Close"].dropna()
        ret_1d  = float((closes.iloc[-1] / closes.iloc[-2] - 1) * 100)
        ret_5d  = float((closes.iloc[-1] / closes.iloc[-6] - 1) * 100)
        ma20    = float(closes.rolling(20).mean().iloc[-1])
        ma50    = float(closes.rolling(50).mean().iloc[-1]) if len(closes) >= 50 else ma20
        price   = float(closes.iloc[-1])
        above_ma20 = price > ma20
        above_ma50 = price > ma50
        vol_5d  = float(closes.pct_change().rolling(5).std().iloc[-1] * 100)

        if ret_1d > 0.3 and above_ma20:
            bias, color = "Bullish", "green"
        elif ret_1d < -0.3 and not above_ma20:
            bias, color = "Bearish", "red"
        else:
            bias, color = "Neutral", "gray"

        return {
            "available"  : True,
            "price"      : round(price, 2),
            "ret_1d"     : round(ret_1d, 2),
            "ret_5d"     : round(ret_5d, 2),
            "ma20"       : round(ma20, 2),
            "ma50"       : round(ma50, 2),
            "above_ma20" : above_ma20,
            "above_ma50" : above_ma50,
            "vol_5d"     : round(vol_5d, 2),
            "bias"       : bias,
            "color"      : color,
            "closes"     : closes,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def render_multi_index():
    st.markdown('<p class="sec-label">Multi-Index Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Nifty · BankNifty · FinNifty</p>', unsafe_allow_html=True)

    cols = st.columns(3)
    signals = {}

    for col, (name, info) in zip(cols, INDICES.items()):
        with col:
            with st.spinner(f"Fetching {name}..."):
                sig = compute_index_signal(info["ticker"])
            signals[name] = sig

            if not sig["available"]:
                st.warning(f"{name} data unavailable")
                continue

            color_map = {"Bullish": "#22c55e", "Bearish": "#ef4444", "Neutral": "#64748b"}
            c = color_map[sig["bias"]]
            arrow = "▲" if sig["ret_1d"] > 0 else "▼" if sig["ret_1d"] < 0 else "—"

            st.markdown(f"""
            <div style='background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);
                        border-top:3px solid {c};border-radius:12px;padding:1.2rem 1.4rem;'>
                <p style='font-family:IBM Plex Mono,monospace;font-size:.75rem;color:#64748b;
                          text-transform:uppercase;letter-spacing:.1em;margin:0 0 .3rem;'>{name}</p>
                <p style='font-family:Playfair Display,serif;font-size:1.8rem;font-weight:800;
                          color:#e2e8f0;margin:0;'>₹{sig['price']:,.0f}</p>
                <p style='font-family:IBM Plex Mono,monospace;font-size:1rem;color:{c};margin:.2rem 0;'>
                  {arrow} {sig['ret_1d']:+.2f}% today &nbsp;|&nbsp; {sig['ret_5d']:+.2f}% 5d</p>
                <p style='font-family:IBM Plex Mono,monospace;font-size:.8rem;color:#64748b;margin:0;'>
                  MA20: {"above" if sig["above_ma20"] else "below"} &nbsp;·&nbsp;
                  MA50: {"above" if sig["above_ma50"] else "below"} &nbsp;·&nbsp;
                  Vol: {sig['vol_5d']:.2f}%</p>
                <span style='display:inline-block;margin-top:.5rem;padding:.2rem .8rem;
                             border-radius:12px;font-family:IBM Plex Mono,monospace;
                             font-size:.75rem;font-weight:600;
                             background:{"rgba(20,83,45,.3)" if sig["bias"]=="Bullish" else "rgba(127,29,29,.3)" if sig["bias"]=="Bearish" else "rgba(30,41,59,.5)"};
                             color:{c};'>{sig['bias']}</span>
            </div>""", unsafe_allow_html=True)

    st.divider()

    # Price chart comparison
    st.markdown('<p class="sec-label">30-Day Normalised Performance</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors_line = ["#3b82f6", "#22c55e", "#f59e0b"]
    for (name, sig), lc in zip(signals.items(), colors_line):
        if sig.get("available") and "closes" in sig:
            closes = sig["closes"].tail(30)
            norm   = (closes / closes.iloc[0] - 1) * 100
            ax.plot(norm.values, label=name, color=lc, linewidth=2)
    ax.axhline(0, color="#334155", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Return vs 30d ago (%)", fontsize=9)
    ax.legend(fontsize=9, framealpha=0, labelcolor="#94a3b8")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()