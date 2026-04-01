import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf

def fetch_fii_dii() -> pd.DataFrame:
    """
    Try NSE API first, fall back to estimated proxy data from Nifty flow.
    """
    # ── Attempt 1: NSE API with better headers ────────────────────────────
    try:
        session = requests.Session()
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/market-data/fii-dii-activity",
            "Origin": "https://www.nseindia.com",
        }
        session.get("https://www.nseindia.com", headers=headers, timeout=10)
        session.get("https://www.nseindia.com/market-data/fii-dii-activity", headers=headers, timeout=10)
        resp = session.get(
            "https://www.nseindia.com/api/fiidiiTradeReact",
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if data and len(data) > 0:
                rows = []
                for item in data:
                    fii = str(item.get("fiiNet", item.get("FII_NET", "0"))).replace(",", "").replace(" ", "")
                    dii = str(item.get("diiNet", item.get("DII_NET", "0"))).replace(",", "").replace(" ", "")
                    date_str = item.get("date", item.get("Date", ""))
                    rows.append({
                        "date"   : date_str,
                        "fii_net": float(fii) if fii else 0.0,
                        "dii_net": float(dii) if dii else 0.0,
                    })
                df = pd.DataFrame(rows)
                df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
                df = df.dropna().sort_values("date").tail(20)
                if not df.empty and df["fii_net"].abs().sum() > 0:
                    return df
    except Exception:
        pass

    # ── Attempt 2: Moneycontrol/alternative source ────────────────────────
    try:
        url  = "https://priceapi.moneycontrol.com/pricefeed/notapplicable/fiidiidata/history?type=fii&period=1M"
        resp = requests.get(url, timeout=8,
                            headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            rows = []
            for item in data[:20]:
                rows.append({
                    "date"   : pd.to_datetime(item.get("date", ""), errors="coerce"),
                    "fii_net": float(str(item.get("netPurchase", 0)).replace(",", "") or 0),
                    "dii_net": 0.0,
                })
            df = pd.DataFrame(rows).dropna().sort_values("date")
            if not df.empty and df["fii_net"].abs().sum() > 0:
                return df
    except Exception:
        pass

    # ── Attempt 3: Generate proxy from Nifty price action ────────────────
    try:
        nifty = yf.download("^NSEI", period="1mo", interval="1d", progress=False)
        if not nifty.empty:
            nifty = nifty.tail(20).copy()
            nifty["ret"] = nifty["Close"].pct_change()
            nifty["volume"] = nifty["Volume"].fillna(0)
            # Proxy: positive return days → FII buying, negative → selling
            rows = []
            for date, row in nifty.iterrows():
                ret    = float(row["ret"]) if not pd.isna(row["ret"]) else 0
                vol    = float(row["volume"]) if not pd.isna(row["volume"]) else 1e6
                fii_est = round(ret * vol * 0.00005, 0)
                dii_est = round(-ret * vol * 0.00003, 0)
                rows.append({
                    "date"      : date,
                    "fii_net"   : fii_est,
                    "dii_net"   : dii_est,
                    "estimated" : True,
                })
            df = pd.DataFrame(rows).dropna()
            return df
    except Exception:
        pass

    return pd.DataFrame()


def render_fii_dii():
    st.markdown('<p class="sec-label">Institutional Flows</p>', unsafe_allow_html=True)
    with st.spinner("Fetching FII/DII data..."):
        df = fetch_fii_dii()

    if df.empty:
        st.info("FII/DII data temporarily unavailable. NSE API may be down.")
        return

    is_estimated = "estimated" in df.columns and df["estimated"].any()
    if is_estimated:
        st.info("⚠️ Showing estimated institutional flow proxy (NSE API blocked on cloud). Real data available locally.")

    latest  = df.iloc[-1]
    fii_5d  = df.tail(5)["fii_net"].sum()
    dii_5d  = df.tail(5)["dii_net"].sum()

    f1, f2, f3, f4 = st.columns(4)
    f1.metric(
        "FII Net (latest)",
        f"₹{latest['fii_net']:,.0f} Cr",
        delta="Buying" if latest["fii_net"] > 0 else "Selling",
        delta_color="normal" if latest["fii_net"] > 0 else "inverse"
    )
    f2.metric(
        "DII Net (latest)",
        f"₹{latest['dii_net']:,.0f} Cr",
        delta="Buying" if latest["dii_net"] > 0 else "Selling",
        delta_color="normal" if latest["dii_net"] > 0 else "inverse"
    )
    f3.metric(
        "FII 5-day Net",
        f"₹{fii_5d:,.0f} Cr",
        delta="Bullish" if fii_5d > 0 else "Bearish",
        delta_color="normal" if fii_5d > 0 else "inverse"
    )
    f4.metric(
        "DII 5-day Net",
        f"₹{dii_5d:,.0f} Cr",
        delta="Bullish" if dii_5d > 0 else "Bearish",
        delta_color="normal" if dii_5d > 0 else "inverse"
    )

    # ── Chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = range(len(df))
    ax.bar(x, df["fii_net"],
           color=["#22c55e" if v > 0 else "#ef4444" for v in df["fii_net"]],
           alpha=0.85, label="FII", zorder=2)
    ax.bar(x, df["dii_net"],
           color=["#3b82f6" if v > 0 else "#f97316" for v in df["dii_net"]],
           alpha=0.6, label="DII", zorder=2)
    ax.axhline(0, color="#64748b", linewidth=0.8)
    ax.set_xticks(list(x)[::2])
    ax.set_xticklabels(
        [str(d.date()) if hasattr(d, "date") else str(d)
         for d in df["date"].iloc[::2]],
        rotation=45, fontsize=7
    )
    ax.set_ylabel("Net Flow (₹ Cr)", fontsize=9)
    ax.legend(fontsize=8, framealpha=0, labelcolor="#94a3b8")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()