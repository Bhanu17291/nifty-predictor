import csv, os
from datetime import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from outcome_tracker import get_scorecard

DRIFT_LOG = "data/drift_log.csv"
DRIFT_THRESHOLD = 0.50

def log_drift():
    score = get_scorecard(days=30)
    acc = score["accuracy"]
    date = datetime.today().strftime("%Y-%m-%d")
    file_exists = os.path.exists(DRIFT_LOG)
    with open(DRIFT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["date", "rolling_accuracy_30d"])
        writer.writerow([date, acc])
    return acc

def render_drift_chart():
    if not os.path.exists(DRIFT_LOG):
        st.info("No drift data yet. Predictions will populate this chart daily.")
        return
    try:
        df = pd.read_csv(DRIFT_LOG, parse_dates=["date"])
    except Exception:
        st.info("Drift log is empty — will populate as daily predictions are logged.")
        return
    if df.empty or len(df) < 1:
        st.info("Drift log is empty — will populate as daily predictions are logged.")
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["date"], df["rolling_accuracy_30d"], color="#1D9E75", linewidth=2)
    ax.axhline(DRIFT_THRESHOLD, color="#E24B4A", linestyle="--", linewidth=1, label="50% threshold")
    ax.set_ylabel("30-day rolling accuracy")
    ax.set_title("Model drift monitor")
    ax.legend()
    st.pyplot(fig)
    plt.close()
    latest = df["rolling_accuracy_30d"].iloc[-1]
    if latest < DRIFT_THRESHOLD:
        st.warning(f"Model accuracy dropped to {latest:.1%} in the last 30 days. Consider retraining.")
    else:
        st.success(f"Model is healthy — 30-day accuracy: {latest:.1%}")