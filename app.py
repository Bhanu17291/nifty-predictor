from multi_index import render_multi_index
from target_price import render_target_price, load_magnitude_model, train_magnitude_model
from options_recommender import render_options_recommender

from dotenv import load_dotenv
load_dotenv()

from voice_briefing import render_voice_briefing
from confidence_chart import render_confidence_chart
from theme_config import render_theme_toggle, apply_theme
from options_signals import render_options_signals
from fii_dii import render_fii_dii
from economic_calendar import render_economic_calendar
from sector_heatmap import render_sector_heatmap
from walk_forward import render_walk_forward

from outcome_tracker import log_prediction, get_scorecard, update_actuals
from market_heatmap import render_heatmap
from plain_reasoning import generate_reasoning
from calibration import load_calibrated_model, plot_reliability_diagram
from drift_monitor import render_drift_chart
from signal_scorer import compute_signal
from backtest_playground import render_playground
from sentiment import render_sentiment
from watchlist import render_watchlist

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, json, subprocess
from datetime import datetime, date
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, f1_score

st.set_page_config(page_title="Nifty Intelligence", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;background:#0b0f1a;color:#e2e8f0;font-size:18px;}
.stApp{background:linear-gradient(160deg,#0b0f1a 0%,#0f1729 60%,#0b0f1a 100%);}
#MainMenu,footer,header{visibility:hidden;}
.block-container{padding:0 3rem 3rem;max-width:1400px;}
hr{border:none;height:1px;background:rgba(99,179,237,.15);margin:2rem 0;}
section[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid rgba(99,179,237,.15)!important;}
section[data-testid="stSidebar"] *{color:#e2e8f0!important;}
[data-testid="metric-container"]{background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);border-top:3px solid #3b82f6;border-radius:12px;padding:1.4rem 1.8rem;transition:all .25s ease;box-shadow:0 4px 20px rgba(0,0,0,.4);}
[data-testid="metric-container"]:hover{border-top-color:#60a5fa;box-shadow:0 6px 28px rgba(59,130,246,.25);transform:translateY(-2px);}
[data-testid="metric-container"] label{font-family:'IBM Plex Mono',monospace!important;font-size:.85rem!important;color:#64748b!important;text-transform:uppercase;letter-spacing:.1em;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{font-family:'Playfair Display',serif!important;font-size:2.2rem!important;font-weight:700!important;color:#e2e8f0!important;}
[data-testid="stMetricDelta"]{font-family:'IBM Plex Mono',monospace!important;font-size:.85rem!important;}
.stButton>button{font-family:'IBM Plex Sans',sans-serif!important;font-weight:700!important;font-size:1rem!important;background:linear-gradient(135deg,#1d4ed8,#3b82f6)!important;color:#fff!important;border:none!important;border-radius:10px!important;padding:.85rem 2.2rem!important;transition:all .2s ease!important;box-shadow:0 4px 15px rgba(59,130,246,.4)!important;}
.stButton>button:hover{background:linear-gradient(135deg,#2563eb,#60a5fa)!important;box-shadow:0 6px 24px rgba(59,130,246,.55)!important;transform:translateY(-2px)!important;}
.stInfo{background:rgba(30,58,138,.25)!important;border:1px solid rgba(99,179,237,.3)!important;border-radius:10px!important;color:#93c5fd!important;}
.stSuccess{background:rgba(20,83,45,.3)!important;border:1px solid rgba(134,239,172,.3)!important;border-radius:10px!important;color:#86efac!important;}
.stError{background:rgba(127,29,29,.3)!important;border:1px solid rgba(252,165,165,.3)!important;border-radius:10px!important;color:#fca5a5!important;}
[data-testid="stDataFrame"]{border-radius:12px!important;border:1px solid rgba(99,179,237,.2)!important;overflow:hidden;}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e3a8a 50%,#0f172a 100%);border-bottom:1px solid rgba(99,179,237,.15);padding:3rem 0 2.5rem;margin-bottom:2rem;text-align:center;border-radius:0 0 24px 24px;}
.hero-eyebrow{font-family:'IBM Plex Mono',monospace;font-size:.85rem;color:#60a5fa;letter-spacing:.25em;text-transform:uppercase;}
.hero-title{font-family:'Playfair Display',serif;font-size:4rem;font-weight:800;color:#f1f5f9;letter-spacing:-.03em;line-height:1.1;margin:0;}
.hero-title span{color:#60a5fa;}
.hero-sub{font-family:'IBM Plex Sans',sans-serif;font-size:1.15rem;color:#94a3b8;margin:.7rem 0 0;font-weight:300;}
.hero-rule{width:56px;height:3px;background:linear-gradient(90deg,#3b82f6,#60a5fa);margin:1.2rem auto 0;border-radius:2px;}
.sec-label{font-family:'IBM Plex Mono',monospace;font-size:.85rem;color:#60a5fa;text-transform:uppercase;letter-spacing:.18em;margin-bottom:.2rem;}
.sec-title{font-family:'Playfair Display',serif;font-size:2.1rem;font-weight:700;color:#f1f5f9;margin:0 0 1.2rem;letter-spacing:-.02em;}
.signal-up{background:rgba(20,83,45,.25);border:1px solid rgba(134,239,172,.3);border-left:5px solid #22c55e;border-radius:14px;padding:2rem 2.2rem;margin:.5rem 0 1rem;}
.signal-down{background:rgba(127,29,29,.25);border:1px solid rgba(252,165,165,.3);border-left:5px solid #ef4444;border-radius:14px;padding:2rem 2.2rem;margin:.5rem 0 1rem;}
.signal-title{font-family:'Playfair Display',serif;font-size:2.5rem;font-weight:800;letter-spacing:-.03em;margin:0 0 .4rem;}
.signal-sub{font-family:'IBM Plex Sans',sans-serif;font-size:1.1rem;color:#94a3b8;margin:0;}
.up-text{color:#22c55e;} .down-text{color:#ef4444;}
.ticker-row{display:flex;gap:.7rem;margin:1rem 0 .5rem;flex-wrap:wrap;}
.ticker-card{background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);border-radius:12px;padding:.85rem 1.1rem;flex:1;min-width:85px;text-align:center;}
.ticker-label{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:#475569;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.2rem;}
.ticker-value{font-family:'Playfair Display',serif;font-size:1.4rem;font-weight:700;}
.up-val{color:#22c55e;} .down-val{color:#ef4444;} .neu-val{color:#e2e8f0;}
.regime-badge{display:inline-block;font-family:'IBM Plex Mono',monospace;font-size:.85rem;font-weight:600;padding:.4rem 1rem;border-radius:20px;letter-spacing:.08em;margin-bottom:.5rem;}
.regime-bull{background:rgba(20,83,45,.3);color:#86efac;border:1px solid rgba(134,239,172,.4);}
.regime-bear{background:rgba(127,29,29,.3);color:#fca5a5;border:1px solid rgba(252,165,165,.4);}
.regime-flat{background:rgba(30,58,138,.3);color:#93c5fd;border:1px solid rgba(99,179,237,.4);}
.tier-strong{background:rgba(113,63,18,.3);color:#fde68a;border:1px solid rgba(253,224,71,.4);}
.tier-moderate{background:rgba(20,83,45,.3);color:#86efac;border:1px solid rgba(134,239,172,.4);}
.tier-weak{background:rgba(154,52,18,.3);color:#fdba74;border:1px solid rgba(253,186,116,.4);}
.tier-unclear{background:rgba(30,41,59,.5);color:#64748b;border:1px solid rgba(100,116,139,.3);}
.chart-wrap{background:rgba(15,23,42,.7);border:1px solid rgba(99,179,237,.15);border-radius:14px;padding:1.5rem 1.5rem .8rem;margin-bottom:1rem;}
.chart-label{font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:#475569;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.8rem;}
.placeholder{background:rgba(15,23,42,.5);border:1.5px dashed rgba(99,179,237,.25);border-radius:14px;padding:3rem 2rem;text-align:center;}
.placeholder-text{font-family:'IBM Plex Sans',sans-serif;font-size:1.1rem;color:#475569;}
.hint-block{background:rgba(30,58,138,.2);border:1px solid rgba(99,179,237,.25);border-radius:10px;padding:1rem 1.2rem;margin-top:1rem;}
.hint-title{font-family:'IBM Plex Mono',monospace;font-size:.75rem;color:#60a5fa;text-transform:uppercase;letter-spacing:.12em;margin-bottom:.4rem;}
.hint-body{font-family:'IBM Plex Sans',sans-serif;font-size:1.05rem;color:#93c5fd;line-height:1.8;margin:0;}
.whatif-card{background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);border-radius:14px;padding:1.8rem;}
.commentary-card{background:rgba(30,58,138,.15);border:1px solid rgba(99,179,237,.25);border-left:4px solid #3b82f6;border-radius:12px;padding:1.5rem 2rem;margin:1rem 0;}
.commentary-text{font-family:'IBM Plex Sans',sans-serif;font-size:1.1rem;color:#cbd5e1;line-height:1.9;margin:0;}
.footer{text-align:center;font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:#334155;letter-spacing:.06em;padding:1.5rem 0 2rem;border-top:1px solid rgba(99,179,237,.1);margin-top:2rem;}
</style>
""", unsafe_allow_html=True)

DATA_DIR        = "data"
MODEL_DIR       = "models"
HISTORY_FILE    = "data/prediction_history.json"
INITIAL_CAPITAL = 100000
TRANSACTION     = 0.001

plt.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#0d1117",
    "axes.edgecolor":"#1e3a5f","axes.labelcolor":"#64748b",
    "xtick.color":"#475569","ytick.color":"#475569",
    "grid.color":"#1e293b","text.color":"#e2e8f0",
    "font.family":"monospace","axes.spines.top":False,
    "axes.spines.right":False,"axes.spines.left":False,
    "font.size":11,"axes.titlesize":12,"axes.labelsize":11,
})

USE_V2     = (os.path.exists(os.path.join(DATA_DIR,"features_v2.csv")) and
              os.path.exists(os.path.join(MODEL_DIR,"xgb_model_v2.pkl")))
FEAT_PATH  = os.path.join(DATA_DIR,"features_v2.csv"  if USE_V2 else "features.csv")
MODEL_PATH = os.path.join(MODEL_DIR,"xgb_model_v2.pkl" if USE_V2 else "xgb_model.pkl")

@st.cache_resource
def load_primary_model():
    with open(MODEL_PATH,"rb") as f: return pickle.load(f)

@st.cache_data
def load_feat_data():
    df = pd.read_csv(FEAT_PATH, parse_dates=True)
    date_col = "Date" if "Date" in df.columns else "date"
    df = df.set_index(date_col); df.index.name = "date"
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_resource
def load_all_models():
    models = {}
    for name in ["xgb","lgbm","rf"]:
        p = os.path.join(MODEL_DIR,f"{name}_model_v2.pkl")
        if os.path.exists(p):
            with open(p,"rb") as f: models[name] = pickle.load(f)
    return models if len(models)==3 else None

try:
    primary_model = load_primary_model()
    df            = load_feat_data()
except Exception as e:
    st.error(f"Startup error: {e}"); st.stop()

all_models   = load_all_models()
feature_cols = [c for c in df.columns if c != "target"]
test_df      = df[df.index.year >= 2023].copy()
X_test       = test_df[feature_cols]
y_test       = test_df["target"]

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ALIGNMENT — fixes the ValueError at startup
# ══════════════════════════════════════════════════════════════════════════════
def get_model_features(model):
    """Extract the exact feature list a trained model expects."""
    try:
        names = model.get_booster().feature_names
        if names:
            return list(names)
    except Exception:
        pass
    try:
        names = model.feature_names_in_
        if names is not None:
            return list(names)
    except Exception:
        pass
    return None


def align_features(model, X: pd.DataFrame) -> pd.DataFrame:
    """
    Reorder and/or fill X so it exactly matches what `model` was trained on.
    Safe to call at module load time — no st.warning calls here.
    """
    expected = get_model_features(model)
    if expected is None:
        return X
    missing = [c for c in expected if c not in X.columns]
    X_aligned = X.copy()
    for col in missing:
        X_aligned[col] = 0.0
    return X_aligned[expected]


def get_preds_proba(X: pd.DataFrame):
    def _xgb_predict(model, X_input):
        X_al = align_features(model, X_input)
        raw  = model.get_booster().inplace_predict(X_al.values, validate_features=False)
        return 1.0 / (1.0 + np.exp(-raw))

    def _skl_predict(model, X_input):
        X_al = align_features(model, X_input)
        return model.predict_proba(X_al)[:, 1]

    if all_models:
        W = {"xgb": .40, "lgbm": .40, "rf": .20}
        p = (_xgb_predict(all_models["xgb"],  X) * W["xgb"]  +
             _skl_predict(all_models["lgbm"], X) * W["lgbm"] +
             _skl_predict(all_models["rf"],   X) * W["rf"])
        return (p >= .5).astype(int), p

    p = _xgb_predict(primary_model, X)
    return (p >= .5).astype(int), p

# ══════════════════════════════════════════════════════════════════════════════

try:
    preds, proba = get_preds_proba(X_test)
except Exception as e:
    st.error(f"❌ Model prediction failed at startup: {e}")
    st.stop()

actual_rets = test_df["nifty_ret"].values/100
cap_s=cap_b=INITIAL_CAPITAL; s_curve=[INITIAL_CAPITAL]; b_curve=[INITIAL_CAPITAL]
wins=trades=0; monthly_pnl={}
for i,p in enumerate(preds):
    r=actual_rets[i]; cap_b*=(1+r)
    if p==1: cap_s*=(1+r-TRANSACTION); trades+=1
    if p==1 and r>0: wins+=1
    s_curve.append(cap_s); b_curve.append(cap_b)
    mo = test_df.index[i].strftime("%Y-%m")
    monthly_pnl[mo] = monthly_pnl.get(mo,0) + (r-TRANSACTION if p==1 else 0)

win_rate=(wins/trades*100) if trades else 0
s_ret=(cap_s/INITIAL_CAPITAL-1)*100; b_ret=(cap_b/INITIAL_CAPITAL-1)*100
acc=accuracy_score(y_test,preds); f1=f1_score(y_test,preds)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE,"r") as f: return json.load(f)
    return []

def save_history(h):
    os.makedirs("data",exist_ok=True)
    with open(HISTORY_FILE,"w") as f: json.dump(h,f,indent=2)

def add_to_history(r):
    history=load_history()
    entry={"date":r["as_of_date"],"prediction":r["prediction"],"pred_int":r["pred_int"],
           "confidence":r["confidence"],"up_prob":r["up_prob"],"regime":r.get("regime","N/A"),
           "tier":r.get("tier","N/A"),"actual":None,"correct":None,
           "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M")}
    if not any(h["date"]==entry["date"] for h in history):
        history.append(entry); save_history(history)
    return history

def plot_confidence_gauge(confidence,pred_int):
    fig,ax=plt.subplots(figsize=(4,2.2),subplot_kw={"projection":"polar"},facecolor="#0d1117")
    for t_start,t_end,color in [(np.pi,np.pi*1.2,"#7f1d1d"),(np.pi*1.2,np.pi*1.5,"#7c3d12"),(np.pi*1.5,np.pi*2,"#14532d")]:
        theta=np.linspace(t_start,t_end,50)
        ax.fill_between(theta,0.6,1.0,color=color,alpha=0.7)
    conf_norm=(confidence-50)/50
    angle=np.pi+conf_norm*np.pi
    ax.annotate("",xy=(angle,0.85),xytext=(0,0),
                arrowprops=dict(arrowstyle="->",color="#60a5fa",lw=2.5,mutation_scale=15))
    ax.set_ylim(0,1); ax.set_xlim(np.pi,2*np.pi)
    ax.set_theta_zero_location("W"); ax.set_theta_direction(-1); ax.axis("off")
    for ang,label in [(np.pi,"50%"),(np.pi*1.5,"75%"),(np.pi*2,"100%")]:
        ax.text(ang,0.45,label,ha="center",va="center",fontsize=8,color="#64748b",fontfamily="monospace")
    color="#22c55e" if pred_int==1 else "#ef4444"
    ax.text(np.pi*1.5,0.28,f"{confidence:.1f}%",ha="center",va="center",fontsize=20,
            fontweight="bold",color=color,fontfamily="monospace")
    ax.text(np.pi*1.5,0.1,"CONFIDENCE",ha="center",va="center",fontsize=7,
            color="#64748b",fontfamily="monospace")
    plt.tight_layout(pad=0); return fig

def run_whatif(sp500,nasdaq,gift,crude=0.0,usdinr=0.0):
    base=X_test.tail(1).copy()
    for col,val in [("sp500_ret",sp500),("nasdaq_ret",nasdaq),("gift_nifty_ret",gift),
                    ("crude_ret",crude),("usdinr_ret",usdinr),("sp500_lag1",sp500),
                    ("nasdaq_lag1",nasdaq),("us_avg_ret",(sp500+nasdaq)/2),("sp500_nasdaq_div",sp500-nasdaq)]:
        if col in base.columns: base[col]=val
    _,sim_p=get_preds_proba(base)
    up=float(sim_p[0])*100; pred=1 if up>=50 else 0; conf=up if pred==1 else 100-up
    return round(up,1),round(100-up,1),pred,round(conf,1)

def plot_accuracy_heatmap():
    correct=(preds==y_test.values).astype(int)
    dates=test_df.index
    df_acc=pd.DataFrame({"date":dates,"correct":correct})
    df_acc["month"]=df_acc["date"].dt.to_period("M")
    months=sorted(df_acc["month"].unique())
    fig,ax=plt.subplots(figsize=(14,4),facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for mi,month in enumerate(months):
        days=df_acc[df_acc["month"]==month]
        for di,(_,row) in enumerate(days.iterrows()):
            color="#22c55e" if row["correct"]==1 else "#ef4444"
            rect=plt.Rectangle((mi,di),0.9,0.9,color=color,alpha=0.8)
            ax.add_patch(rect)
    ax.set_xlim(0,len(months)); ax.set_ylim(0,max(df_acc.groupby("month").size())+1)
    ax.set_xticks(np.arange(len(months))+0.45)
    ax.set_xticklabels([str(m) for m in months],rotation=45,fontsize=7,color="#64748b")
    ax.set_yticks([]); ax.set_title("Daily Prediction Accuracy — Green=Correct  Red=Wrong",
                                    color="#e2e8f0",fontsize=11,pad=10)
    plt.tight_layout(); return fig

def plot_monthly_pnl():
    pnl_data={}
    for mo,val in monthly_pnl.items():
        yr,mn=mo.split("-")
        pnl_data.setdefault(yr,{})[int(mn)]=val*100
    years=sorted(pnl_data.keys())
    months_labels=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    data=np.full((len(years),12),np.nan)
    for yi,yr in enumerate(years):
        for mi in range(12):
            data[yi,mi]=pnl_data[yr].get(mi+1,np.nan)
    fig,ax=plt.subplots(figsize=(14,max(3,len(years)*1.5)),facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    for yi in range(len(years)):
        for mi in range(12):
            val=data[yi,mi]
            if np.isnan(val): continue
            color="#22c55e" if val>0 else "#ef4444"
            alpha=min(0.9,0.2+abs(val)*8)
            rect=plt.Rectangle((mi,yi),0.9,0.7,color=color,alpha=alpha)
            ax.add_patch(rect)
            ax.text(mi+0.45,yi+0.35,f"{val:+.1f}%",ha="center",va="center",
                    fontsize=7,color="#e2e8f0",fontweight="bold")
    ax.set_xlim(0,12); ax.set_ylim(0,len(years))
    ax.set_xticks(np.arange(12)+0.45); ax.set_xticklabels(months_labels,color="#64748b",fontsize=9)
    ax.set_yticks(np.arange(len(years))+0.35); ax.set_yticklabels(years,color="#64748b",fontsize=9)
    ax.set_title("Monthly Strategy P&L Heatmap",color="#e2e8f0",fontsize=11,pad=10)
    plt.tight_layout(); return fig

def generate_commentary(result):
    sp  = result.get("sp500_ret",0)
    nas = result.get("nasdaq_ret",0)
    crd = result.get("crude_ret",0)
    vix = result.get("vix",None)
    reg = result.get("regime","FLAT")
    conf= result.get("confidence",50)
    pred= result.get("pred_int",0)
    us_tone    = "positive" if (sp+nas)/2>0 else "negative"
    crude_tone = "rising" if crd>0 else "falling"
    vix_tone   = f"VIX at {vix:.1f} suggests {'elevated caution' if vix and vix>18 else 'calm markets'}." if vix else ""
    conf_tone  = "strong conviction" if conf>=65 else "moderate confidence" if conf>=55 else "low confidence — treat with caution"
    signal     = "gap-up open" if pred==1 else "gap-down open"
    regime_txt = {"BULL":"bullish trending","BEAR":"bearish trending","FLAT":"consolidating"}.get(reg,"neutral")
    lines=[
        f"US markets closed on a {us_tone} note — S&P500 {sp:+.2f}%, Nasdaq {nas:+.2f}%.",
        f"GIFT Nifty is indicating a {signal} with model confidence at {conf:.1f}% ({conf_tone}).",
        f"Crude oil is {crude_tone} ({crd:+.2f}%), and Indian markets are in a {regime_txt} regime. {vix_tone}",
    ]
    return " ".join(lines)

def send_telegram(token, chat_id, message):
    try:
        import urllib.request, urllib.parse
        url  = f"https://api.telegram.org/bot{token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id":chat_id,"text":message,"parse_mode":"HTML"}).encode()
        req  = urllib.request.Request(url,data=data)
        urllib.request.urlopen(req,timeout=10)
        return True
    except Exception as e:
        return str(e)

def format_telegram_message(result):
    emoji  = "🟢" if result["pred_int"]==1 else "🔴"
    signal = "BULLISH OPEN" if result["pred_int"]==1 else "BEARISH OPEN"
    tier   = result.get("tier_emoji","") + " " + result.get("tier_label","")
    msg = f"""📈 <b>NIFTY INTELLIGENCE</b> — {result['as_of_date']}

{emoji} <b>Prediction: {signal}</b>
🎯 Confidence: {result['confidence']}% | {tier}
📊 Regime: {result.get('regime_emoji','')} {result.get('regime','N/A')}

<b>Market Signals:</b>
• S&amp;P500: {result['sp500_ret']:+.2f}%
• Nasdaq: {result['nasdaq_ret']:+.2f}%
• GIFT Nifty: {result['nifty_ret']:+.2f}%
• Crude Oil: {result.get('crude_ret',0):+.2f}%
• USD/INR: {result.get('usdinr_ret',0):+.2f}%"""
    if result.get("vix"):
        msg += f"\n• India VIX: {result['vix']}"
    msg += "\n\n⚠️ <i>For educational purposes only. Not financial advice.</i>"
    return msg

def generate_pdf_report(result, commentary):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.units import inch
        import io
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4,
                                rightMargin=inch*0.8, leftMargin=inch*0.8,
                                topMargin=inch*0.8, bottomMargin=inch*0.8)
        styles = getSampleStyleSheet()
        story  = []
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=22, spaceAfter=6, textColor=colors.HexColor("#1e3a8a"))
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                     fontSize=10, textColor=colors.HexColor("#64748b"), spaceAfter=20)
        h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                     fontSize=14, textColor=colors.HexColor("#1e3a8a"), spaceBefore=16, spaceAfter=8)
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                     fontSize=11, leading=16, spaceAfter=10)
        story.append(Paragraph("📈 Nifty Intelligence — Morning Market Brief", title_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p IST')}", sub_style))
        pred_color = "#15803d" if result["pred_int"]==1 else "#b91c1c"
        pred_text  = "🟢 BULLISH OPEN EXPECTED" if result["pred_int"]==1 else "🔴 BEARISH OPEN EXPECTED"
        story.append(Paragraph("Prediction Signal", h2_style))
        story.append(Paragraph(f"<font color='{pred_color}'><b>{pred_text}</b></font>", body_style))
        signal_data = [["Metric","Value"],["Confidence", f"{result['confidence']}%"],
                       ["UP Probability", f"{result['up_prob']}%"],["DOWN Probability", f"{result['down_prob']}%"],
                       ["Regime", result.get("regime","N/A")],["Signal Tier", result.get("tier_label","N/A")]]
        t = Table(signal_data, colWidths=[3*inch, 3*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e3a8a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),colors.white]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),("PADDING",(0,0),(-1,-1),8),
        ]))
        story.append(t); story.append(Spacer(1,12))
        story.append(Paragraph("Market Signals", h2_style))
        market_data = [["Index","Change"],["S&P 500", f"{result['sp500_ret']:+.2f}%"],
                       ["Nasdaq", f"{result['nasdaq_ret']:+.2f}%"],["GIFT Nifty", f"{result['nifty_ret']:+.2f}%"],
                       ["Crude Oil", f"{result.get('crude_ret',0):+.2f}%"],["USD/INR", f"{result.get('usdinr_ret',0):+.2f}%"]]
        if result.get("vix"):
            market_data.append(["India VIX", str(result["vix"])])
        t2 = Table(market_data, colWidths=[3*inch,3*inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1e3a8a")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),10),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor("#f8fafc"),colors.white]),
            ("GRID",(0,0),(-1,-1),0.5,colors.HexColor("#e2e8f0")),("PADDING",(0,0),(-1,-1),8),
        ]))
        story.append(t2); story.append(Spacer(1,12))
        story.append(Paragraph("Market Commentary", h2_style))
        story.append(Paragraph(commentary, body_style))
        story.append(Spacer(1,20))
        story.append(Paragraph("<font color='#94a3b8'><i>⚠️ Educational purposes only. Not financial advice.</i></font>", body_style))
        doc.build(story); buf.seek(0)
        return buf.getvalue()
    except ImportError:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    render_theme_toggle()
    apply_theme()

    st.markdown("""
    <div style='text-align:center;padding:1rem 0 1.5rem;border-bottom:1px solid rgba(99,179,237,.15);margin-bottom:1.5rem;'>
        <p style='font-family:Playfair Display,serif;font-size:1.4rem;font-weight:800;color:#f1f5f9;margin:0;'>
            📈 Nifty Intel
        </p>
        <p style='font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#60a5fa;margin:.3rem 0 0;letter-spacing:.15em;text-transform:uppercase;'>
            Tier 3 · Ensemble
        </p>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Live Prediction",
        "🧪  What-If Simulator",
        "📊  Analytics",
        "📅  Heatmaps",
        "📈  Multi-Index",
        "📋  History",
        "⚙️  Settings & Retrain",
    ], label_visibility="collapsed")

    st.divider()
    render_economic_calendar()

    st.divider()
    render_sector_heatmap()

    st.markdown("""
    <div style='margin-top:2rem;padding-top:1rem;border-top:1px solid rgba(99,179,237,.1);
                font-family:IBM Plex Mono,monospace;font-size:.65rem;color:#334155;'>
        Model: XGB + LGBM + RF<br>
        Features: 65+<br>
        Train: 2019–2022<br>
        Test: 2023–2024
    </div>""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <p class="hero-eyebrow">NSE · Nifty50 · {'Tier 3 · Ensemble' if USE_V2 else 'Tier 1'}</p>
    <h1 class="hero-title">Nifty <span>Intelligence</span></h1>
    <p class="hero-sub">Predictive analytics for the Indian equity market opening session</p>
    <div class="hero-rule"></div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: LIVE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if "🏠" in page:
    # ── Scorecard bar ────────────────────────────────────────────────────────
    score = get_scorecard(30)
    if score["total"] > 0:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("30-day accuracy", f"{score['accuracy']:.1%}")
        sc2.metric("Correct calls",   score["correct"])
        sc3.metric("Total tracked",   score["total"])

    st.markdown("""
    <style>
    .dashboard-grid{display:grid;grid-template-columns:1fr 1fr 1fr;grid-template-rows:auto auto;gap:1rem;margin-bottom:1rem;}
    .dash-card{background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);border-radius:16px;padding:1.4rem 1.6rem;}
    .dash-card-wide{background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);border-radius:16px;padding:1.4rem 1.6rem;grid-column:span 2;}
    .dash-card-full{background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);border-radius:16px;padding:1.4rem 1.6rem;grid-column:span 3;}
    .card-label{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#60a5fa;text-transform:uppercase;letter-spacing:.15em;margin-bottom:.5rem;}
    .signal-big{font-family:'Playfair Display',serif;font-size:2.8rem;font-weight:800;letter-spacing:-.03em;margin:0;}
    .conf-num{font-family:'Playfair Display',serif;font-size:2.4rem;font-weight:700;margin:0;}
    .ticker-mini{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.4rem;}
    .tmini{background:rgba(30,41,59,.7);border:1px solid rgba(99,179,237,.15);border-radius:8px;padding:.4rem .7rem;text-align:center;flex:1;min-width:70px;}
    .tmini-label{font-family:'IBM Plex Mono',monospace;font-size:.65rem;color:#475569;text-transform:uppercase;}
    .tmini-val{font-family:'Playfair Display',serif;font-size:1.1rem;font-weight:700;}
    .reason-row{display:flex;align-items:center;gap:.6rem;padding:.45rem 0;border-bottom:1px solid rgba(99,179,237,.08);}
    .reason-row:last-child{border-bottom:none;}
    .reason-badge{font-family:'IBM Plex Mono',monospace;font-size:.75rem;padding:.2rem .55rem;border-radius:6px;font-weight:600;}
    .badge-up{background:rgba(20,83,45,.4);color:#22c55e;}
    .badge-dn{background:rgba(127,29,29,.4);color:#ef4444;}
    .reason-text{font-family:'IBM Plex Sans',sans-serif;font-size:.92rem;color:#cbd5e1;}
    .commentary-mini{font-family:'IBM Plex Sans',sans-serif;font-size:.95rem;color:#94a3b8;line-height:1.7;}
    .target-val{font-family:'Playfair Display',serif;font-size:2rem;font-weight:700;color:#60a5fa;margin:0;}
    .target-range{font-family:'IBM Plex Mono',monospace;font-size:.82rem;color:#64748b;margin:.3rem 0 0;}
    </style>
    """, unsafe_allow_html=True)

    # ── Generate button ───────────────────────────────────────────────────────
    col_btn, col_time = st.columns([1, 3])
    with col_btn:
        st.button("⚡ Generate Prediction", type="primary",
                  use_container_width=True, key="predict_btn")
    with col_time:
        st.markdown("""
        <div style='padding:.6rem 1rem;background:rgba(30,58,138,.2);border:1px solid rgba(99,179,237,.2);
                    border-radius:10px;font-family:IBM Plex Mono,monospace;font-size:.8rem;color:#60a5fa;'>
            ⏰ Best used after 11 PM IST &nbsp;·&nbsp; US markets closed &nbsp;·&nbsp; Before 9 AM NSE open
        </div>""", unsafe_allow_html=True)

    if st.session_state.get("predict_btn"):
        with st.spinner("Fetching live data & running ensemble..."):
            try:
                if USE_V2:
                    from live_predict_v2 import predict_tomorrow_v2
                    r = predict_tomorrow_v2(verbose=False)
                else:
                    from live_predict import predict_tomorrow
                    r = predict_tomorrow(verbose=False)

                st.session_state["last_result"] = r
                add_to_history(r)
                commentary  = generate_commentary(r)
                st.session_state["commentary"] = commentary

                pred_direction  = "UP" if r["pred_int"] == 1 else "DOWN"
                pred_confidence = r["confidence"] / 100
                log_prediction(pred_direction, pred_confidence)

                current_vix = r["vix"] if r.get("vix") else 15.0
                signal      = compute_signal(confidence=pred_confidence, vix=current_vix)
                mag         = r.get("magnitude", {})

                # ── helpers ──────────────────────────────────────────────────
                def vc(v): return "up-val" if v > 0 else "down-val" if v < 0 else "neu-val"
                def ar(v): return "▲" if v > 0 else "▼" if v < 0 else "—"

                is_up       = r["pred_int"] == 1
                sig_color   = "#22c55e" if is_up else "#ef4444"
                sig_label   = "🟢 Bullish Open" if is_up else "🔴 Bearish Open"
                sig_sub     = "Nifty expected to open HIGHER" if is_up else "Nifty expected to open LOWER"
                conf_color  = "#22c55e" if pred_confidence >= .65 else "#f97316" if pred_confidence >= .55 else "#ef4444"

                # SHAP reasons
                shap_top = []
                if r.get("explanation", {}).get("available"):
                    for reason in r["explanation"].get("reasons", [])[:4]:
                        if isinstance(reason, dict):
                            shap_top.append((reason.get("feature",""), reason.get("shap",0), reason.get("strength",0)))

                live_data = {
                    "SP500_return"     : r["sp500_ret"],
                    "Nasdaq_return"    : r["nasdaq_ret"],
                    "India_VIX"        : current_vix,
                    "USDINR_return"    : r.get("usdinr_ret", 0),
                    "Crude_return"     : r.get("crude_ret", 0),
                    "GIFT_Nifty_return": r["nifty_ret"],
                }
                reasons_text = generate_reasoning(
                    [(f, s) for f, s, _ in shap_top], live_data, pred_direction
                )
                if not reasons_text:
                    reasons_text = [
                        f"S&P 500 closed {r['sp500_ret']:+.2f}% overnight.",
                        f"Nasdaq closed {r['nasdaq_ret']:+.2f}% overnight.",
                        f"India VIX is at {current_vix:.1f}.",
                        f"USD/INR moved {r.get('usdinr_ret',0):+.2f}%.",
                    ]

                # target price strings
                if mag.get("available"):
                    target_str = f"₹{mag['pred_price']:,.0f}"
                    range_str  = f"₹{mag['low_price']:,.0f} – ₹{mag['high_price']:,.0f}"
                    ret_str    = f"{mag['predicted_ret']:+.2f}%"
                    ret_color  = "#22c55e" if mag["predicted_ret"] > 0 else "#ef4444"
                else:
                    target_str = "N/A"
                    range_str  = "Run prediction to calculate"
                    ret_str    = "—"
                    ret_color  = "#64748b"

                vix_html = f'<div class="tmini"><div class="tmini-label">VIX</div><div class="tmini-val neu-val">{r["vix"]}</div></div>' if r.get("vix") else ""

                reasons_html = ""
                for rt in reasons_text[:4]:
                    reasons_html += f'''
                    <div class="reason-row">
                        <span class="reason-badge {'badge-up' if is_up else 'badge-dn'}">
                            {'▲' if is_up else '▼'}
                        </span>
                        <span class="reason-text">{rt}</span>
                    </div>'''

                # ── MAIN DASHBOARD GRID ───────────────────────────────────────
                st.markdown(f"""
                <div class="dashboard-grid">

                  <!-- CARD 1: Signal -->
                  <div class="dash-card" style="border-top:3px solid {sig_color};">
                    <div class="card-label">Signal</div>
                    <p class="signal-big" style="color:{sig_color};">{sig_label}</p>
                    <p style="font-family:'IBM Plex Sans',sans-serif;font-size:.88rem;
                               color:#64748b;margin:.3rem 0 0;">{sig_sub}</p>
                    <p style="font-family:'IBM Plex Mono',monospace;font-size:.78rem;
                               color:#475569;margin:.6rem 0 0;">📅 {r['as_of_date']}</p>
                  </div>

                  <!-- CARD 2: Confidence -->
                  <div class="dash-card" style="border-top:3px solid {conf_color};">
                    <div class="card-label">Confidence</div>
                    <p class="conf-num" style="color:{conf_color};">{r['confidence']:.1f}%</p>
                    <div style="margin:.6rem 0 .3rem;background:rgba(30,41,59,.8);
                                border-radius:6px;height:8px;overflow:hidden;">
                      <div style="width:{r['up_prob']}%;height:100%;
                                  background:linear-gradient(90deg,#22c55e,#16a34a);border-radius:6px;"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:#64748b;">
                      <span>🟢 UP {r['up_prob']}%</span>
                      <span>🔴 DN {r['down_prob']}%</span>
                    </div>
                    <div style="margin-top:.6rem;">
                      <span class="regime-badge {'regime-bull' if r.get('regime')=='BULL' else 'regime-bear' if r.get('regime')=='BEAR' else 'regime-flat'}">
                        {r.get('regime_emoji','')} {r.get('regime','N/A')}
                      </span>
                      &nbsp;
                      <span class="regime-badge tier-{r.get('tier','moderate').lower()}">
                        {r.get('tier_emoji','')} {r.get('tier_label','')}
                      </span>
                    </div>
                  </div>

                  <!-- CARD 3: Target Price -->
                  <div class="dash-card" style="border-top:3px solid #3b82f6;">
                    <div class="card-label">Target Open Price</div>
                    <p class="target-val">{target_str}
                      <span style="font-size:1rem;color:{ret_color};"> {ret_str}</span>
                    </p>
                    <p class="target-range">Range: {range_str}</p>
                    <div style="margin-top:.8rem;font-family:'IBM Plex Mono',monospace;
                                font-size:.75rem;color:#475569;">
                      Signal Score: <strong style="color:#e2e8f0;">{signal['score']:.0%}</strong>
                      &nbsp;·&nbsp; Tier: <strong style="color:#e2e8f0;">{signal['label']}</strong>
                    </div>
                  </div>

                  <!-- CARD 4: Tickers (wide) -->
                  <div class="dash-card-wide" style="border-top:3px solid #1e3a8a;">
                    <div class="card-label">Market Signals</div>
                    <div class="ticker-mini">
                      <div class="tmini"><div class="tmini-label">S&P 500</div>
                        <div class="tmini-val {vc(r['sp500_ret'])}">{ar(r['sp500_ret'])} {r['sp500_ret']:+.2f}%</div></div>
                      <div class="tmini"><div class="tmini-label">Nasdaq</div>
                        <div class="tmini-val {vc(r['nasdaq_ret'])}">{ar(r['nasdaq_ret'])} {r['nasdaq_ret']:+.2f}%</div></div>
                      <div class="tmini"><div class="tmini-label">GIFT Nifty</div>
                        <div class="tmini-val {vc(r['nifty_ret'])}">{ar(r['nifty_ret'])} {r['nifty_ret']:+.2f}%</div></div>
                      <div class="tmini"><div class="tmini-label">Crude Oil</div>
                        <div class="tmini-val {vc(r.get('crude_ret',0))}">{ar(r.get('crude_ret',0))} {r.get('crude_ret',0):+.2f}%</div></div>
                      <div class="tmini"><div class="tmini-label">USD/INR</div>
                        <div class="tmini-val {vc(r.get('usdinr_ret',0))}">{ar(r.get('usdinr_ret',0))} {r.get('usdinr_ret',0):+.2f}%</div></div>
                      {vix_html}
                    </div>
                  </div>

                  <!-- CARD 5: Why -->
                  <div class="dash-card" style="border-top:3px solid #7c3aed;">
                    <div class="card-label">Why This Prediction</div>
                    {reasons_html}
                  </div>

                  <!-- CARD 6: Commentary (full width) -->
                  <div class="dash-card-full" style="border-top:3px solid #0891b2;">
                    <div class="card-label">🤖 AI Commentary</div>
                    <p class="commentary-mini">{commentary}</p>
                  </div>

                </div>
                """, unsafe_allow_html=True)

                # ── EXPANDABLE TABS for secondary content ─────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                    "📊 Options", "📈 FII/DII", "🎯 SHAP Details",
                    "🔊 Voice", "📰 Sentiment", "📋 Watchlist"
                ])

                with tab1:
                    pcr_val = 1.0
                    try:
                        from options_signals import fetch_options_data
                        opts_data = fetch_options_data()
                        if opts_data.get("available"):
                            pcr_val = opts_data["pcr"]
                    except Exception:
                        pass
                    render_options_signals()
                    st.divider()
                    render_options_recommender(
                        direction     = pred_direction,
                        signal_label  = signal["label"],
                        vix           = current_vix,
                        pcr           = pcr_val,
                        current_price = mag.get("current_price", 22000)
                    )

                with tab2:
                    render_fii_dii()

                with tab3:
                    if r.get("explanation", {}).get("available"):
                        st.markdown("**Top SHAP drivers:**")
                        for i, reason in enumerate(r["explanation"]["reasons"][:5], 1):
                            if isinstance(reason, dict):
                                arrow = "🟢 ▲" if reason["shap"] > 0 else "🔴 ▼"
                                st.markdown(f"`{i}. {reason['feature'][:28]:<28}` {arrow} strength **{reason['strength']:.4f}**")
                    else:
                        st.info("SHAP explanations not available for this prediction.")

                with tab4:
                    render_voice_briefing(commentary)

                with tab5:
                    render_sentiment()

                with tab6:
                    render_watchlist()

                # ── PDF download ──────────────────────────────────────────────
                pdf_bytes = generate_pdf_report(r, commentary)
                if pdf_bytes:
                    st.download_button(
                        "📄 Download Morning Brief PDF",
                        data      = pdf_bytes,
                        file_name = f"nifty_brief_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime      = "application/pdf"
                    )

                # expiry / event warnings
                if signal["is_expiry"]:
                    st.warning("⚠️ Today is F&O expiry — expect elevated volatility.")
                if signal["days_to_event"] <= 2:
                    st.warning(f"⚠️ Major market event in {signal['days_to_event']} day(s) — signal reliability reduced.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.markdown("""
        <div style='background:rgba(15,23,42,.6);border:1.5px dashed rgba(99,179,237,.25);
                    border-radius:16px;padding:4rem 2rem;text-align:center;margin-top:1rem;'>
            <p style='font-family:Playfair Display,serif;font-size:2rem;font-weight:700;
                      color:#334155;margin:0;'>Click ⚡ Generate Prediction</p>
            <p style='font-family:IBM Plex Sans,sans-serif;font-size:1rem;
                      color:#1e3a5f;margin:.5rem 0 0;'>to see today's market signal</p>
        </div>""", unsafe_allow_html=True)

    # ── Confidence gauge in sidebar ───────────────────────────────────────────
    if st.session_state.get("last_result"):
        with st.sidebar:
            st.divider()
            st.markdown('<p style="font-family:IBM Plex Mono,monospace;font-size:.72rem;color:#60a5fa;text-transform:uppercase;letter-spacing:.15em;">Confidence Meter</p>', unsafe_allow_html=True)
            r = st.session_state["last_result"]
            fig = plot_confidence_gauge(r["confidence"], r["pred_int"])
            st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "🧪" in page:
    st.markdown('<p class="sec-label">Scenario Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">What-If Simulator</p>', unsafe_allow_html=True)
    st.markdown("Drag the sliders to simulate different overnight market conditions.")

    s1,s2,s3 = st.columns([1.5,1.5,1], gap="large")
    with s1:
        sp500_sim  = st.slider("S&P 500 move (%)",  -4.0,4.0,0.0,0.1)
        nasdaq_sim = st.slider("Nasdaq move (%)",   -4.0,4.0,0.0,0.1)
        gift_sim   = st.slider("GIFT Nifty move (%)", -3.0,3.0,0.0,0.1)
    with s2:
        crude_sim  = st.slider("Crude Oil move (%)", -5.0,5.0,0.0,0.1)
        usdinr_sim = st.slider("USD/INR move (%)",   -1.5,1.5,0.0,0.05)
        up_p,dn_p,pred_i,conf = run_whatif(sp500_sim,nasdaq_sim,gift_sim,crude_sim,usdinr_sim)
    with s3:
        st.markdown('<div class="whatif-card">', unsafe_allow_html=True)
        st.markdown('<p class="sec-label">Simulated Result</p>', unsafe_allow_html=True)
        color = "#22c55e" if pred_i==1 else "#ef4444"
        label = "🟢 UP" if pred_i==1 else "🔴 DOWN"
        st.markdown(f"<p style='font-family:Playfair Display,serif;font-size:1.8rem;font-weight:800;color:{color};margin:0;'>{label}</p>", unsafe_allow_html=True)
        st.markdown(f"""<p style='font-family:IBM Plex Mono,monospace;font-size:.9rem;color:#64748b;margin:.5rem 0;line-height:2;'>
            UP Prob &nbsp;&nbsp; <strong style='color:#e2e8f0;'>{up_p}%</strong><br>
            DOWN Prob &nbsp;<strong style='color:#e2e8f0;'>{dn_p}%</strong><br>
            Confidence &nbsp;<strong style='color:{color};'>{conf}%</strong></p>""", unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(3,.5)); fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
        ax.barh(0,up_p/100,color="#22c55e",height=.6); ax.barh(0,dn_p/100,left=up_p/100,color="#ef4444",height=.6)
        ax.axvline(.5,color="#0d1117",linewidth=2); ax.set_xlim(0,1); ax.axis("off")
        plt.tight_layout(pad=0); st.pyplot(fig); plt.close()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="sec-label">Quick Scenarios</p>', unsafe_allow_html=True)
    sc1,sc2,sc3,sc4 = st.columns(4)
    scenarios = [
        ("🔥 US Rally",    2.0, 2.5,  0.8,  0.5, -0.2),
        ("💥 US Crash",   -2.5,-3.0, -1.0,  2.0,  0.5),
        ("⚡ Flat US",      0.2, 0.1,  0.3, -0.5, -0.1),
        ("🛢️ Oil Spike",   -0.5,-0.3, -0.2,  5.0,  0.3),
    ]
    for col,(label,sp,nq,gf,cr,usd) in zip([sc1,sc2,sc3,sc4],scenarios):
        up,dn,pi,cf = run_whatif(sp,nq,gf,cr,usd)
        c = "#22c55e" if pi==1 else "#ef4444"
        col.markdown(f"""<div style='background:rgba(15,23,42,.85);border:1px solid rgba(99,179,237,.2);
            border-radius:10px;padding:1rem;text-align:center;'>
            <p style='font-family:IBM Plex Sans,sans-serif;font-size:.95rem;color:#e2e8f0;margin:0 0 .3rem;font-weight:600;'>{label}</p>
            <p style='font-family:Playfair Display,serif;font-size:1.3rem;font-weight:800;color:{c};margin:0;'>
                {"🟢 UP" if pi==1 else "🔴 DOWN"} {cf:.0f}%</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif "📊" in page:
    st.markdown('<p class="sec-label">Backtest · 2023–2024</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Model Performance</p>', unsafe_allow_html=True)

    m1,m2,m3,m4,m5 = st.columns(5)
    m1.metric("Test Accuracy",    f"{acc*100:.1f}%",  delta=f"+{acc*100-50:.1f}% vs random")
    m2.metric("F1 Score",         f"{f1:.3f}")
    m3.metric("Win Rate",         f"{win_rate:.1f}%", delta="on traded days")
    m4.metric("Strategy Return",  f"{s_ret:+.1f}%")
    m5.metric("Benchmark Return", f"{b_ret:+.1f}%")

    if os.path.exists(os.path.join(MODEL_DIR,"model_comparison.png")):
        st.divider()
        st.markdown('<p class="sec-title">XGBoost vs LightGBM vs Random Forest vs Ensemble</p>', unsafe_allow_html=True)
        from PIL import Image
        st.image(Image.open(os.path.join(MODEL_DIR,"model_comparison.png")), use_container_width=True)

    st.divider()
    c1,c2 = st.columns(2,gap="medium")
    with c1:
        st.markdown('<div class="chart-wrap"><p class="chart-label">Portfolio Value</p>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(7,3.5))
        ax.plot(s_curve,color="#3b82f6",linewidth=2,label="Ensemble Strategy")
        ax.plot(b_curve,color="#64748b",linewidth=1.5,linestyle="--",label="Buy & Hold")
        ax.axhline(INITIAL_CAPITAL,color="#1e293b",linestyle=":",linewidth=1)
        ax.fill_between(range(len(s_curve)),s_curve,INITIAL_CAPITAL,alpha=.08,color="#3b82f6")
        ax.set_ylabel("Portfolio (₹)",fontsize=9); ax.set_xlabel("Trading Days",fontsize=9)
        ax.legend(fontsize=9,framealpha=0,labelcolor="#94a3b8"); ax.grid(True,alpha=.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"₹{x/1000:.0f}k"))
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="chart-wrap"><p class="chart-label">Rolling 30-Day Accuracy</p>', unsafe_allow_html=True)
        correct=(preds==y_test.values).astype(int)
        rolling_acc=pd.Series(correct).rolling(30).mean()*100
        fig,ax=plt.subplots(figsize=(7,3.5))
        ax.plot(rolling_acc.values,color="#3b82f6",linewidth=2)
        ax.axhline(50,color="#ef4444",linestyle="--",linewidth=1,alpha=.8,label="Random (50%)")
        ax.axhline(float(rolling_acc.mean()),color="#64748b",linestyle=":",linewidth=1,label=f"Avg {rolling_acc.mean():.1f}%")
        ax.fill_between(range(len(rolling_acc)),rolling_acc.values,50,where=(rolling_acc.values>50),alpha=.08,color="#3b82f6")
        ax.set_ylabel("Accuracy (%)",fontsize=9); ax.set_ylim(30,80)
        ax.legend(fontsize=9,framealpha=0,labelcolor="#94a3b8"); ax.grid(True,alpha=.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    c3,c4 = st.columns(2,gap="medium")
    with c3:
        st.markdown('<div class="chart-wrap"><p class="chart-label">Feature Importance — Top 15</p>', unsafe_allow_html=True)
        feat_src=all_models["xgb"] if all_models else primary_model
        imp=pd.Series(feat_src.feature_importances_,index=get_model_features(feat_src) or feature_cols).sort_values(ascending=False).head(15)
        colors_imp=["#3b82f6" if i<3 else "#1e3a5f" for i in range(len(imp))]
        fig,ax=plt.subplots(figsize=(7,4.5))
        imp[::-1].plot(kind="barh",ax=ax,color=colors_imp[::-1],edgecolor="none")
        ax.set_xlabel("Importance",fontsize=9); ax.grid(True,alpha=.3,axis="x"); ax.tick_params(labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c4:
        st.markdown('<div class="chart-wrap"><p class="chart-label">UP Probability — Test Period</p>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(7,4))
        ax.plot(proba,color="#3b82f6",alpha=.9,linewidth=1)
        ax.axhline(.5,color="#ef4444",linestyle="--",linewidth=1,alpha=.7)
        ax.fill_between(range(len(proba)),proba,.5,where=(proba>.5),alpha=.12,color="#22c55e")
        ax.fill_between(range(len(proba)),proba,.5,where=(proba<=.5),alpha=.12,color="#ef4444")
        ax.set_ylabel("P(Nifty UP)",fontsize=9); ax.set_ylim(0,1); ax.grid(True,alpha=.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    shap_path=os.path.join(MODEL_DIR,"shap_waterfall.png")
    if os.path.exists(shap_path):
        st.divider()
        st.markdown('<p class="sec-title">SHAP Explainability</p>', unsafe_allow_html=True)
        from PIL import Image
        st.image(Image.open(shap_path), use_container_width=True)

    st.divider()
    st.markdown('<p class="sec-label">Model Health</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Model Drift Monitor</p>', unsafe_allow_html=True)
    render_drift_chart()

    st.divider()
    render_playground()

    st.divider()
    st.markdown('<p class="sec-label">Calibration</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Reliability Diagram</p>', unsafe_allow_html=True)
    cal_model = load_calibrated_model()
    if cal_model:
        st.success("Calibrated model loaded.")
        try:
            y_prob_cal = cal_model.predict_proba(align_features(cal_model, X_test))[:,1]
            plot_reliability_diagram(y_test.values, y_prob_cal)
        except Exception as e:
            st.warning(f"Could not plot reliability diagram: {e}")
    else:
        st.info("No calibrated model found. Run calibration.py after retraining.")

    st.divider()
    st.markdown('<p class="sec-label">Prediction Tracking</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Confidence vs Accuracy Over Time</p>', unsafe_allow_html=True)
    render_confidence_chart()

    st.divider()
    render_walk_forward(df, feature_cols)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HEATMAPS
# ══════════════════════════════════════════════════════════════════════════════
elif "📅" in page:
    st.markdown('<p class="sec-label">Visual Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Accuracy & P&L Heatmaps</p>', unsafe_allow_html=True)

    st.markdown('<div class="chart-wrap"><p class="chart-label">Daily Prediction Accuracy Calendar</p>', unsafe_allow_html=True)
    fig = plot_accuracy_heatmap()
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-wrap"><p class="chart-label">Monthly Strategy P&L Heatmap</p>', unsafe_allow_html=True)
    fig = plot_monthly_pnl()
    st.pyplot(fig); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<p class="sec-label">Live Chart</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Nifty50 Live Chart</p>', unsafe_allow_html=True)
    st.components.v1.html("""
    <div class="tradingview-widget-container" style="height:450px;">
      <div id="tradingview_nifty"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({
        "width": "100%", "height": 450, "symbol": "NSE:NIFTY",
        "interval": "5", "timezone": "Asia/Kolkata", "theme": "dark",
        "style": "1", "locale": "en", "toolbar_bg": "#0d1117",
        "enable_publishing": false, "hide_top_toolbar": false,
        "hide_legend": false, "container_id": "tradingview_nifty"
      });
      </script>
    </div>""", height=460)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MULTI-INDEX
# ══════════════════════════════════════════════════════════════════════════════
elif "📈" in page:
    render_multi_index()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "📋" in page:
    st.markdown('<p class="sec-label">Prediction Log</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Prediction History</p>', unsafe_allow_html=True)

    history = load_history()
    if history:
        total     = len(history)
        correct_h = sum(1 for h in history if h.get("correct") is True)
        hist_acc  = (correct_h/total*100) if total else 0

        ha1,ha2,ha3,ha4 = st.columns(4)
        ha1.metric("Total Predictions", total)
        ha2.metric("Verified Correct",  correct_h)
        ha3.metric("History Accuracy",  f"{hist_acc:.1f}%" if total else "N/A")
        ha4.metric("Last Predicted",    history[-1]["date"])

        st.divider()
        hist_df = pd.DataFrame(history[::-1])
        show    = [c for c in ["date","prediction","confidence","regime","tier","actual","correct","timestamp"] if c in hist_df.columns]
        st.dataframe(hist_df[show].rename(columns={c:c.title() for c in show}),
                     use_container_width=True, height=350)

        st.divider()
        st.markdown('<p class="sec-label">Mark Yesterday\'s Actual Outcome</p>', unsafe_allow_html=True)
        last = history[-1]
        if last.get("actual") is None:
            st.write(f"**{last['date']}** — Predicted: {last['prediction']} ({last['confidence']}% confidence)")
            oc1,oc2,oc3 = st.columns(3)
            if oc1.button("✅ Nifty opened UP",   key="mark_up"):
                for h in history:
                    if h["date"]==last["date"]: h["actual"]="UP"; h["correct"]=(last["pred_int"]==1)
                save_history(history); st.rerun()
            if oc2.button("❌ Nifty opened DOWN", key="mark_dn"):
                for h in history:
                    if h["date"]==last["date"]: h["actual"]="DOWN"; h["correct"]=(last["pred_int"]==0)
                save_history(history); st.rerun()
            if oc3.button("⏭ Skip", key="mark_skip"):
                for h in history:
                    if h["date"]==last["date"]: h["actual"]="Skipped"
                save_history(history); st.rerun()
        else:
            st.success(f"✅ Yesterday's outcome already recorded: **{last['actual']}**")
    else:
        st.info("No predictions logged yet. Go to **🏠 Live Prediction** and click Generate to start tracking.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS & RETRAIN
# ══════════════════════════════════════════════════════════════════════════════
elif "⚙️" in page:
    st.markdown('<p class="sec-label">Settings</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Settings & Model Retraining</p>', unsafe_allow_html=True)

    si1,si2,si3,si4 = st.columns(4)
    si1.metric("Model Type",    "Ensemble" if USE_V2 else "XGBoost")
    si2.metric("Features",      f"{len(feature_cols)}")
    si3.metric("Test Accuracy", f"{acc*100:.1f}%")
    if os.path.exists(MODEL_PATH):
        mtime = os.path.getmtime(MODEL_PATH)
        last_train = datetime.fromtimestamp(mtime).strftime("%d %b %Y")
    else:
        last_train = "Unknown"
    si4.metric("Last Trained", last_train)

    st.divider()
    st.markdown('<p class="sec-label">One-Click Retrain</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Retrain All Models</p>', unsafe_allow_html=True)

    is_cloud = os.environ.get("STREAMLIT_SHARING_MODE") or not os.path.exists("data_fetch.py")

    if is_cloud:
        st.info("""
        ☁️ **Running on Streamlit Cloud**
        Retraining requires a local Python environment. To retrain:
        1. Clone the repo locally
        2. Run: `python data_fetch.py` → `python features_v2.py` → `python ensemble_model.py` → `python regime_detector.py` → `python explainer.py`
        3. Push new model files to GitHub — Streamlit Cloud will auto-redeploy.
        """)
    else:
        st.warning("⚠️ Retraining fetches fresh data and rebuilds all models. Takes 5–10 minutes.")
        if st.button("🔄 Start Full Retrain Pipeline", type="primary"):
            progress = st.progress(0)
            status   = st.empty()
            steps = [
                ("Fetching fresh market data...",  "python data_fetch.py",      20),
                ("Preprocessing data...",          "python data_preprocess.py", 40),
                ("Building V2 features...",        "python features_v2.py",     60),
                ("Training ensemble models...",    "python ensemble_model.py",  80),
                ("Running regime detection...",    "python regime_detector.py", 90),
                ("Building SHAP explanations...",  "python explainer.py",       100),
            ]
            success = True
            for msg, cmd, pct in steps:
                status.info(f"⏳ {msg}")
                try:
                    result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=300)
                    if result.returncode != 0:
                        st.error(f"Error in: {cmd}\n{result.stderr[:300]}")
                        success = False; break
                except Exception as e:
                    st.error(f"Failed: {e}"); success = False; break
                progress.progress(pct)
            if success:
                status.success("✅ Retraining complete! Refresh the page to use the new model.")
                st.cache_resource.clear(); st.cache_data.clear()
            else:
                status.error("❌ Retraining failed. Check errors above.")

    st.divider()
    st.markdown('<p class="sec-label">Recent Predictions</p>', unsafe_allow_html=True)
    recent        = test_df.tail(20).copy()
    recent_preds  = preds[-20:]
    recent_proba  = proba[-20:]
    recent_actual = y_test.values[-20:]
    correct_20    = sum(1 for p,a in zip(recent_preds,recent_actual) if p==a)

    st.markdown(f'<p class="sec-title">Last 20 Days &nbsp;<span style="font-family:IBM Plex Mono,monospace;font-size:.9rem;color:#60a5fa;">{correct_20}/20 correct</span></p>',
                unsafe_allow_html=True)
    table=pd.DataFrame({
        "Date"      : recent.index.strftime("%d %b %Y"),
        "S&P500"    : [f"{v:+.2f}%" for v in recent["sp500_ret"].values],
        "Nasdaq"    : [f"{v:+.2f}%" for v in recent["nasdaq_ret"].values],
        "Predicted" : ["🟢 UP" if p==1 else "🔴 DOWN" for p in recent_preds],
        "Confidence": [f"{(p if pred==1 else 1-p)*100:.1f}%" for p,pred in zip(recent_proba,recent_preds)],
        "Actual"    : ["🟢 UP" if a==1 else "🔴 DOWN" for a in recent_actual],
        "Result"    : ["✅" if p==a else "❌" for p,a in zip(recent_preds,recent_actual)],
    })
    st.dataframe(table.set_index("Date"), use_container_width=True, height=400)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
    Nifty Intelligence &nbsp;·&nbsp; {'Ensemble: XGBoost + LightGBM + RF' if USE_V2 else 'XGBoost'} &nbsp;·&nbsp;
    {len(feature_cols)} Features &nbsp;·&nbsp; Trained 2019–2022 &nbsp;·&nbsp; Tested 2023–2024<br><br>
    ⚠️&nbsp; Educational purposes only &nbsp;·&nbsp; Not financial advice &nbsp;·&nbsp; Always conduct independent research
</div>""", unsafe_allow_html=True)