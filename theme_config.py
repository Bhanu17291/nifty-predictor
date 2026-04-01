import streamlit as st

DARK_CSS = """
<style>
html,body,[class*="css"]{background:#0b0f1a;color:#e2e8f0;}
.stApp{background:linear-gradient(160deg,#0b0f1a 0%,#0f1729 60%,#0b0f1a 100%);}
section[data-testid="stSidebar"]{background:#0d1117!important;}
[data-testid="metric-container"]{background:rgba(15,23,42,.85);border-top:3px solid #3b82f6;}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 50%,#0f172a 100%);}
</style>
"""

LIGHT_CSS = """
<style>
html,body,[class*="css"]{background:#f8fafc;color:#0f172a;}
.stApp{background:#f8fafc;}
section[data-testid="stSidebar"]{background:#ffffff!important;}
[data-testid="metric-container"]{background:#ffffff;border-top:3px solid #3b82f6;border:1px solid #e2e8f0;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#0f172a!important;}
[data-testid="metric-container"] label{color:#64748b!important;}
.hero{background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 50%,#dbeafe 100%);}
.hero-title{color:#0f172a;}
.hero-sub{color:#475569;}
.sec-title{color:#0f172a;}
.signal-up{background:rgba(220,252,231,.5);border-left:5px solid #22c55e;}
.signal-down{background:rgba(254,226,226,.5);border-left:5px solid #ef4444;}
.ticker-card{background:#ffffff;border:1px solid #e2e8f0;}
.chart-wrap{background:#ffffff;border:1px solid #e2e8f0;}
.commentary-card{background:#eff6ff;border-left:4px solid #3b82f6;}
.commentary-text{color:#1e40af;}
</style>
"""

def render_theme_toggle():
    if "dark_mode" not in st.session_state:
        st.session_state["dark_mode"] = True
    toggle = st.toggle("Dark mode", value=st.session_state["dark_mode"])
    st.session_state["dark_mode"] = toggle
    return toggle

def apply_theme():
    dark = st.session_state.get("dark_mode", True)
    st.markdown(DARK_CSS if dark else LIGHT_CSS, unsafe_allow_html=True)