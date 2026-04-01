import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def fetch_fii_dii() -> pd.DataFrame:
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com"
        }, timeout=10)
        url  = "https://www.nseindia.com/api/fiidiiTradeReact"
        resp = session.get(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nseindia.com"
        }, timeout=10)
        data = resp.json()
        rows = []
        for item in data:
            rows.append({
                "date"     : item.get("date",""),
                "fii_net"  : float(str(item.get("fiiNet","0")).replace(",","")),
                "dii_net"  : float(str(item.get("diiNet","0")).replace(",","")),
            })
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        return df.dropna().sort_values("date").tail(20)
    except Exception as e:
        return pd.DataFrame()

def render_fii_dii():
    st.markdown('<p class="sec-label">Institutional Flows</p>', unsafe_allow_html=True)
    with st.spinner("Fetching FII/DII data from NSE..."):
        df = fetch_fii_dii()

    if df.empty:
        st.warning("FII/DII data unavailable — NSE may have blocked the request.")
        return

    latest = df.iloc[-1]
    f1, f2, f3 = st.columns(3)
    fii_delta = "normal" if latest["fii_net"] > 0 else "inverse"
    dii_delta = "normal" if latest["dii_net"] > 0 else "inverse"
    f1.metric("FII Net (latest day)", f"₹{latest['fii_net']:,.0f} Cr",
              delta_color=fii_delta, delta="Buying" if latest["fii_net"] > 0 else "Selling")
    f2.metric("DII Net (latest day)", f"₹{latest['dii_net']:,.0f} Cr",
              delta_color=dii_delta, delta="Buying" if latest["dii_net"] > 0 else "Selling")
    f3.metric("FII 5-day net",
              f"₹{df.tail(5)['fii_net'].sum():,.0f} Cr",
              delta="Bullish" if df.tail(5)["fii_net"].sum() > 0 else "Bearish")

    fig, ax = plt.subplots(figsize=(10, 3))
    x = range(len(df))
    ax.bar(x, df["fii_net"], color=["#22c55e" if v > 0 else "#ef4444" for v in df["fii_net"]],
           alpha=0.8, label="FII")
    ax.bar(x, df["dii_net"], color=["#3b82f6" if v > 0 else "#f97316" for v in df["dii_net"]],
           alpha=0.5, label="DII")
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.set_xticks(list(x)[::3])
    ax.set_xticklabels([str(d.date()) for d in df["date"].iloc[::3]], rotation=45, fontsize=7)
    ax.set_ylabel("Net Flow (₹ Cr)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0, labelcolor="#94a3b8")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()