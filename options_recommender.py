import streamlit as st

STRATEGIES = {
    ("UP", "HIGH", "LOW"): {
        "strategy": "Buy ATM Call",
        "reason"  : "High conviction bullish signal with low volatility — directional call is optimal.",
        "risk"    : "Limited to premium paid.",
        "strikes" : "Buy 1 ATM CE expiring this week.",
        "color"   : "#22c55e",
    },
    ("UP", "HIGH", "HIGH"): {
        "strategy": "Bull Call Spread",
        "reason"  : "High conviction UP but elevated VIX makes naked calls expensive — spread reduces cost.",
        "risk"    : "Capped profit, defined risk.",
        "strikes" : "Buy ATM CE, Sell OTM CE (+100 points). Same expiry.",
        "color"   : "#22c55e",
    },
    ("UP", "MODERATE", "LOW"): {
        "strategy": "Buy ATM Call or Bull Call Spread",
        "reason"  : "Moderate bullish signal, low IV — single leg is fine but spread adds safety.",
        "risk"    : "Limited to premium.",
        "strikes" : "Buy ATM CE. Optional: sell OTM CE to reduce cost.",
        "color"   : "#86efac",
    },
    ("UP", "MODERATE", "HIGH"): {
        "strategy": "Bull Put Spread",
        "reason"  : "Moderate UP signal with high VIX — sell premium instead of buying it.",
        "risk"    : "Defined risk on both sides.",
        "strikes" : "Sell OTM PE, Buy deeper OTM PE. Collect credit.",
        "color"   : "#86efac",
    },
    ("DOWN", "HIGH", "LOW"): {
        "strategy": "Buy ATM Put",
        "reason"  : "High conviction bearish with low IV — directional put is optimal.",
        "risk"    : "Limited to premium paid.",
        "strikes" : "Buy 1 ATM PE expiring this week.",
        "color"   : "#ef4444",
    },
    ("DOWN", "HIGH", "HIGH"): {
        "strategy": "Bear Put Spread",
        "reason"  : "High conviction DOWN but high VIX — spread cuts the expensive premium.",
        "risk"    : "Capped profit, defined risk.",
        "strikes" : "Buy ATM PE, Sell OTM PE (-100 points). Same expiry.",
        "color"   : "#ef4444",
    },
    ("DOWN", "MODERATE", "LOW"): {
        "strategy": "Buy ATM Put or Bear Put Spread",
        "reason"  : "Moderate bearish, low IV — single put works, spread adds discipline.",
        "risk"    : "Limited to premium.",
        "strikes" : "Buy ATM PE. Optional: sell OTM PE to reduce cost.",
        "color"   : "#fca5a5",
    },
    ("DOWN", "MODERATE", "HIGH"): {
        "strategy": "Bear Call Spread",
        "reason"  : "Moderate DOWN + high VIX — sell inflated call premium.",
        "risk"    : "Defined risk.",
        "strikes" : "Sell OTM CE, Buy further OTM CE. Collect credit.",
        "color"   : "#fca5a5",
    },
    ("UP", "WEAK", "HIGH"): {
        "strategy": "Iron Condor",
        "reason"  : "Weak signal + high VIX — no clear direction, sell both sides and collect premium.",
        "risk"    : "Defined risk both sides.",
        "strikes" : "Sell OTM CE + OTM PE, Buy further OTM on both sides.",
        "color"   : "#f59e0b",
    },
    ("DOWN", "WEAK", "HIGH"): {
        "strategy": "Iron Condor",
        "reason"  : "Weak signal + high VIX — range-bound play is safest.",
        "risk"    : "Defined risk both sides.",
        "strikes" : "Sell OTM CE + OTM PE, Buy further OTM on both sides.",
        "color"   : "#f59e0b",
    },
    ("UP", "WEAK", "LOW"): {
        "strategy": "Sit Out / Paper Trade Only",
        "reason"  : "Weak signal + low IV — risk/reward not attractive enough.",
        "risk"    : "No position recommended.",
        "strikes" : "Wait for a stronger signal tomorrow.",
        "color"   : "#64748b",
    },
    ("DOWN", "WEAK", "LOW"): {
        "strategy": "Sit Out / Paper Trade Only",
        "reason"  : "Weak signal + low IV — no edge.",
        "risk"    : "No position recommended.",
        "strikes" : "Wait for a stronger signal tomorrow.",
        "color"   : "#64748b",
    },
}

def get_recommendation(direction: str, signal_label: str, vix: float, pcr: float = 1.0) -> dict:
    if "High" in signal_label or "Strong" in signal_label:
        conviction = "HIGH"
    elif "Moderate" in signal_label:
        conviction = "MODERATE"
    else:
        conviction = "WEAK"

    vix_level = "HIGH" if vix > 18 else "LOW"
    key       = (direction, conviction, vix_level)
    rec       = STRATEGIES.get(key, {
        "strategy": "Insufficient data",
        "reason"  : "Could not determine a clear strategy.",
        "risk"    : "N/A",
        "strikes" : "N/A",
        "color"   : "#64748b",
    })

    if pcr > 1.3:
        pcr_note = "PCR > 1.3 confirms strong put buying — supports bullish reversal thesis."
    elif pcr < 0.7:
        pcr_note = "PCR < 0.7 shows heavy call buying — supports bearish reversal thesis."
    else:
        pcr_note = "PCR at " + str(round(pcr, 2)) + " is neutral — no strong options flow confirmation."

    return {**rec, "conviction": conviction, "vix_level": vix_level, "pcr_note": pcr_note}

def render_options_recommender(direction: str, signal_label: str,
                                vix: float, pcr: float = 1.0,
                                current_price: float = 22000):
    st.markdown('<p class="sec-label">Options Strategy</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">AI Strategy Recommender</p>', unsafe_allow_html=True)

    rec   = get_recommendation(direction, signal_label, vix, pcr)
    color = rec["color"]

    card_html = (
        "<div style='background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);"
        "border-top:4px solid " + color + ";border-radius:14px;padding:1.8rem 2rem;'>"
        "<p style='font-family:IBM Plex Mono,monospace;font-size:.75rem;color:#64748b;"
        "text-transform:uppercase;letter-spacing:.12em;margin:0 0 .3rem;'>"
        "Recommended strategy</p>"
        "<p style='font-family:Playfair Display,serif;font-size:2rem;font-weight:800;"
        "color:" + color + ";margin:0 0 1rem;'>" + rec["strategy"] + "</p>"
        "<p style='font-family:IBM Plex Sans,sans-serif;font-size:1rem;color:#cbd5e1;"
        "line-height:1.8;margin:0 0 .8rem;'>" + rec["reason"] + "</p>"
        "<div style='display:flex;gap:1rem;flex-wrap:wrap;margin-top:1rem;'>"
        "<div style='background:rgba(30,58,138,.2);border:1px solid rgba(99,179,237,.2);"
        "border-radius:8px;padding:.7rem 1rem;flex:1;min-width:160px;'>"
        "<p style='font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#60a5fa;"
        "text-transform:uppercase;margin:0 0 .2rem;'>How to trade</p>"
        "<p style='font-family:IBM Plex Sans,sans-serif;font-size:.95rem;"
        "color:#e2e8f0;margin:0;'>" + rec["strikes"] + "</p>"
        "</div>"
        "<div style='background:rgba(30,58,138,.2);border:1px solid rgba(99,179,237,.2);"
        "border-radius:8px;padding:.7rem 1rem;flex:1;min-width:160px;'>"
        "<p style='font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#60a5fa;"
        "text-transform:uppercase;margin:0 0 .2rem;'>Risk profile</p>"
        "<p style='font-family:IBM Plex Sans,sans-serif;font-size:.95rem;"
        "color:#e2e8f0;margin:0;'>" + rec["risk"] + "</p>"
        "</div>"
        "<div style='background:rgba(30,58,138,.2);border:1px solid rgba(99,179,237,.2);"
        "border-radius:8px;padding:.7rem 1rem;flex:1;min-width:160px;'>"
        "<p style='font-family:IBM Plex Mono,monospace;font-size:.7rem;color:#60a5fa;"
        "text-transform:uppercase;margin:0 0 .2rem;'>Options flow</p>"
        "<p style='font-family:IBM Plex Sans,sans-serif;font-size:.95rem;"
        "color:#e2e8f0;margin:0;'>" + rec["pcr_note"] + "</p>"
        "</div>"
        "</div>"
        "<div style='margin-top:1rem;padding-top:.8rem;border-top:1px solid rgba(99,179,237,.1);'>"
        "<p style='font-family:IBM Plex Mono,monospace;font-size:.75rem;color:#475569;margin:0;'>"
        "Inputs: Direction=" + direction +
        " · Conviction=" + rec["conviction"] +
        " · VIX=" + str(round(vix, 1)) +
        " (" + rec["vix_level"] + ")" +
        " · PCR=" + str(round(pcr, 2)) +
        "</p></div></div>"
    )

    st.markdown(card_html, unsafe_allow_html=True)
    st.warning("This is an educational suggestion only — not financial advice. Always verify strikes, expiry, and lot sizes before trading.")