import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

SECTORS = {
    "Nifty Bank"   : "^NSEBANK",
    "Nifty IT"     : "^CNXIT",
    "Nifty Pharma" : "^CNXPHARMA",
    "Nifty Auto"   : "^CNXAUTO",
    "Nifty FMCG"   : "^CNXFMCG",
    "Nifty Metal"  : "^CNXMETAL",
    "Nifty Realty" : "^CNXREALTY",
    "Nifty Energy" : "^CNXENERGY",
}

def fetch_sector_returns() -> pd.DataFrame:
    rows = []
    for name, ticker in SECTORS.items():
        try:
            data = yf.download(ticker, period="5d", interval="1d", progress=False)
            if len(data) >= 2:
                d1 = float(((data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2]) * 100)
                d5 = float(((data["Close"].iloc[-1] - data["Close"].iloc[0])  / data["Close"].iloc[0])  * 100)
                rows.append({"sector": name, "1d": round(d1, 2), "5d": round(d5, 2)})
        except Exception:
            rows.append({"sector": name, "1d": 0.0, "5d": 0.0})
    return pd.DataFrame(rows)

def render_sector_heatmap():
    st.markdown('<p class="sec-label">Sector Rotation</p>', unsafe_allow_html=True)
    with st.spinner("Fetching sector data..."):
        df = fetch_sector_returns()

    if df.empty:
        st.warning("Sector data unavailable.")
        return

    period = st.radio("Period", ["1d", "5d"], horizontal=True, key="sector_period")
    vals   = df[period].values
    labels = df["sector"].values

    fig, ax = plt.subplots(figsize=(10, 2.5))
    norm    = mcolors.TwoSlopeNorm(vmin=min(vals.min(), -1), vcenter=0, vmax=max(vals.max(), 1))
    colors  = plt.cm.RdYlGn(norm(vals))

    for i, (label, val, color) in enumerate(zip(labels, vals, colors)):
        rect = plt.Rectangle((i, 0), 0.9, 1, color=color, alpha=0.85)
        ax.add_patch(rect)
        ax.text(i + 0.45, 0.62, label.replace("Nifty ", ""),
                ha="center", va="center", fontsize=8, color="#0f172a", fontweight="bold")
        ax.text(i + 0.45, 0.32, f"{val:+.2f}%",
                ha="center", va="center", fontsize=9, color="#0f172a")

    ax.set_xlim(0, len(labels))
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig)
    plt.close()

    best   = df.loc[df[period].idxmax(), "sector"]
    worst  = df.loc[df[period].idxmin(), "sector"]
    b1, b2 = st.columns(2)
    b1.metric("Leading sector",  best,  f"{df.loc[df[period].idxmax(), period]:+.2f}%")
    b2.metric("Lagging sector",  worst, f"{df.loc[df[period].idxmin(), period]:+.2f}%")