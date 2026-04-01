import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import pickle, os

MODEL_DIR = "models"
DATA_DIR  = "data"

def train_magnitude_model(df: pd.DataFrame, feature_cols: list):
    df = df.copy()
    if "nifty_ret" not in df.columns:
        return None, None
    X = df[feature_cols].fillna(0)
    y = df["nifty_ret"]
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    model  = Ridge(alpha=1.0)
    model.fit(X_sc, y)
    try:
        pickle.dump((model, scaler), open(os.path.join(MODEL_DIR, "magnitude_model.pkl"), "wb"))
    except Exception:
        pass
    return model, scaler

def load_magnitude_model():
    path = os.path.join(MODEL_DIR, "magnitude_model.pkl")
    if os.path.exists(path):
        try:
            return pickle.load(open(path, "rb"))
        except Exception:
            pass
    return None, None

def predict_magnitude(X_live: pd.DataFrame, model, scaler) -> dict:
    try:
        X_sc    = scaler.transform(X_live.fillna(0))
        pred    = float(model.predict(X_sc)[0])
        std_err = 0.4
        low     = round(pred - std_err, 2)
        high    = round(pred + std_err, 2)
        return {
            "available"   : True,
            "predicted_ret": round(pred, 2),
            "range_low"   : low,
            "range_high"  : high,
            "direction"   : "UP" if pred > 0 else "DOWN",
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def render_target_price(X_live: pd.DataFrame, current_price: float,
                        df: pd.DataFrame, feature_cols: list):
    st.markdown('<p class="sec-label">Target Price</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Expected Open Range</p>', unsafe_allow_html=True)

    model, scaler = load_magnitude_model()
    if model is None:
        with st.spinner("Training magnitude model..."):
            model, scaler = train_magnitude_model(df, feature_cols)

    if model is None:
        st.warning("Magnitude model could not be trained — need nifty_ret in features.")
        return

    result = predict_magnitude(X_live, model, scaler)
    if not result["available"]:
        st.warning(f"Prediction failed: {result.get('error','')}")
        return

    pred_ret   = result["predicted_ret"]
    low_ret    = result["range_low"]
    high_ret   = result["range_high"]
    pred_price = round(current_price * (1 + pred_ret / 100), 0)
    low_price  = round(current_price * (1 + low_ret  / 100), 0)
    high_price = round(current_price * (1 + high_ret / 100), 0)

    color = "#22c55e" if pred_ret > 0 else "#ef4444"

    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Current Nifty",    f"₹{current_price:,.0f}")
    t2.metric("Predicted return", f"{pred_ret:+.2f}%", delta_color="normal" if pred_ret > 0 else "inverse")
    t3.metric("Target open",      f"₹{pred_price:,.0f}")
    t4.metric("Expected range",   f"₹{low_price:,.0f} – ₹{high_price:,.0f}")

    fig, ax = plt.subplots(figsize=(8, 1.2))
    ax.barh(0, high_ret - low_ret, left=low_ret,
            color=color, alpha=0.3, height=0.5)
    ax.axvline(pred_ret, color=color, linewidth=2.5)
    ax.axvline(0, color="#334155", linewidth=1, linestyle="--")
    ax.set_xlim(min(low_ret - 0.3, -1), max(high_ret + 0.3, 1))
    ax.set_yticks([])
    ax.set_xlabel("Expected open return (%)", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    for s in ax.spines.values(): s.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown(f"""
    <div style='background:rgba(15,23,42,.7);border:1px solid rgba(99,179,237,.15);
                border-left:4px solid {color};border-radius:10px;padding:1rem 1.4rem;margin-top:.5rem;'>
        <p style='font-family:IBM Plex Sans,sans-serif;font-size:1rem;color:#cbd5e1;margin:0;'>
        Nifty is expected to open around <strong style='color:{color};'>₹{pred_price:,.0f}</strong>
        ({pred_ret:+.2f}%), with a probable range of
        <strong>₹{low_price:,.0f} – ₹{high_price:,.0f}</strong>.
        This is a regression estimate — treat as a zone, not a precise level.
        </p>
    </div>""", unsafe_allow_html=True)