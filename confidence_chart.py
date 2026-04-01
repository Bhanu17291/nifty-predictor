import json, os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

PRED_FILE = "data/predictions.json"

def render_confidence_chart():
    if not os.path.exists(PRED_FILE):
        st.info("No predictions logged yet.")
        return
    with open(PRED_FILE) as f:
        preds = json.load(f)
    if not preds:
        st.info("No predictions logged yet.")
        return

    df = pd.DataFrame(preds)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["confidence_pct"] = df["confidence"] * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["confidence_pct"],
            color="#3b82f6", linewidth=2, label="Confidence %")

    if "actual" in df.columns:
        correct_mask  = df["prediction"] == df["actual"]
        wrong_mask    = (df["prediction"] != df["actual"]) & df["actual"].notna()
        ax.scatter(df["date"][correct_mask],  df["confidence_pct"][correct_mask],
                   color="#22c55e", s=60, zorder=5, label="Correct")
        ax.scatter(df["date"][wrong_mask],    df["confidence_pct"][wrong_mask],
                   color="#ef4444", s=60, zorder=5, label="Wrong")

    ax.axhline(65, color="#f59e0b", linestyle="--", linewidth=1, label="Strong threshold (65%)")
    ax.axhline(50, color="#ef4444", linestyle=":",  linewidth=1, label="Random baseline (50%)")
    ax.set_ylabel("Confidence (%)", fontsize=9)
    ax.set_ylim(40, 100)
    ax.legend(fontsize=8, framealpha=0, labelcolor="#94a3b8")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Stats
    if "actual" in df.columns:
        verified = df[df["actual"].notna()].copy()
        if not verified.empty:
            high_conf = verified[verified["confidence_pct"] >= 65]
            low_conf  = verified[verified["confidence_pct"] <  65]
            hc_acc = (high_conf["prediction"] == high_conf["actual"]).mean() if len(high_conf) else 0
            lc_acc = (low_conf["prediction"]  == low_conf["actual"]).mean()  if len(low_conf)  else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("High confidence accuracy", f"{hc_acc:.1%}", f"{len(high_conf)} predictions")
            c2.metric("Low confidence accuracy",  f"{lc_acc:.1%}", f"{len(low_conf)} predictions")
            c3.metric("Total verified",            len(verified))