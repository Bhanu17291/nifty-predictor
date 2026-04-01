import requests
import pandas as pd
import streamlit as st

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com",
}

def fetch_options_data() -> dict:
    session = requests.Session()
    try:
        session.get("https://www.nseindia.com", headers=HEADERS, timeout=10)
        url  = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        resp = session.get(url, headers=HEADERS, timeout=10)
        data = resp.json()

        records = data["records"]["data"]
        expiry  = data["records"]["expiryDates"][0]

        rows = []
        for r in records:
            strike = r.get("strikePrice", 0)
            ce_oi  = r.get("CE", {}).get("openInterest", 0) if "CE" in r else 0
            pe_oi  = r.get("PE", {}).get("openInterest", 0) if "PE" in r else 0
            rows.append({"strike": strike, "ce_oi": ce_oi, "pe_oi": pe_oi})

        df = pd.DataFrame(rows)
        total_ce = df["ce_oi"].sum()
        total_pe = df["pe_oi"].sum()
        pcr      = round(total_pe / total_ce, 2) if total_ce > 0 else 1.0

        max_pain_row = df.copy()
        max_pain_row["pain"] = abs(df["strike"] - df["strike"].mean())
        max_pain = int(df.loc[df["ce_oi"] + df["pe_oi"] == (df["ce_oi"] + df["pe_oi"]).max(), "strike"].values[0])

        top_ce = df.nlargest(3, "ce_oi")[["strike","ce_oi"]].values.tolist()
        top_pe = df.nlargest(3, "pe_oi")[["strike","pe_oi"]].values.tolist()

        sentiment = "Bullish" if pcr > 1.2 else ("Bearish" if pcr < 0.8 else "Neutral")

        return {
            "pcr": pcr,
            "max_pain": max_pain,
            "expiry": expiry,
            "sentiment": sentiment,
            "top_ce_strikes": top_ce,
            "top_pe_strikes": top_pe,
            "available": True
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def render_options_signals():
    st.markdown('<p class="sec-label">Options Market</p>', unsafe_allow_html=True)
    with st.spinner("Fetching NSE options data..."):
        data = fetch_options_data()

    if not data["available"]:
        st.warning(f"Options data unavailable: {data.get('error','NSE blocked request')}")
        return

    o1, o2, o3 = st.columns(3)
    pcr_color = "green" if data["pcr"] > 1.2 else ("red" if data["pcr"] < 0.8 else "gray")
    o1.metric("Put-Call Ratio (PCR)", data["pcr"],
              delta=data["sentiment"], delta_color="normal" if data["pcr"] > 1 else "inverse")
    o2.metric("Max Pain Level",  f"₹{data['max_pain']:,}")
    o3.metric("Next Expiry",     data["expiry"])

    st.markdown("**Highest OI strikes:**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("Call OI (resistance)")
        for strike, oi in data["top_ce_strikes"]:
            st.markdown(f"- ₹{int(strike):,} — OI: {int(oi):,}")
    with c2:
        st.markdown("Put OI (support)")
        for strike, oi in data["top_pe_strikes"]:
            st.markdown(f"- ₹{int(strike):,} — OI: {int(oi):,}")