import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from outcome_tracker import load_predictions

PRESETS = {
    "Last 30 days":      None,
    "COVID crash":       ("2020-02-01", "2020-04-30"),
    "Budget 2023":       ("2023-01-15", "2023-02-15"),
    "2024 election":     ("2024-05-01", "2024-06-15"),
}

def render_playground():
    st.subheader("Backtest playground")
    preset = st.selectbox("Quick preset", list(PRESETS.keys()))
    if PRESETS[preset]:
        d1 = pd.to_datetime(PRESETS[preset][0]).date()
        d2 = pd.to_datetime(PRESETS[preset][1]).date()
    else:
        import datetime
        d2 = datetime.date.today()
        d1 = d2 - datetime.timedelta(days=30)

    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=d1)
    end   = col2.date_input("End date", value=d2)

    preds = load_predictions()
    df = pd.DataFrame(preds)
    if df.empty or "actual" not in df.columns:
        st.info("No logged predictions yet.")
        return

    df["date"] = pd.to_datetime(df["date"])
    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end) & df["actual"].notna()
    df = df[mask].copy()

    if df.empty:
        st.info("No predictions with actuals in this date range.")
        return

    df["correct"] = df["prediction"] == df["actual"]
    acc = df["correct"].mean()
    st.metric("Accuracy in period", f"{acc:.1%}", f"{len(df)} predictions")

    df["equity"] = 100000 * (1 + df["correct"].map({True: 0.01, False: -0.01})).cumprod()
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(df["date"], df["equity"], color="#1D9E75", linewidth=2)
    ax.axhline(100000, color="#888780", linestyle="--", linewidth=1)
    ax.set_title(f"₹1 lakh equity curve ({start} → {end})")
    ax.set_ylabel("Portfolio value (₹)")
    st.pyplot(fig)
    plt.close()