"""
walk_forward.py
Provides render_walk_forward() for app.py
and can also be run standalone for validation.
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR  = "data"
MODEL_DIR = "models"


def render_walk_forward(df, feature_cols):
    import streamlit as st
    st.markdown('<p class="sec-label">Walk-Forward Validation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sec-title">Rolling Window Accuracy</p>', unsafe_allow_html=True)

    results_path = os.path.join(MODEL_DIR, "walk_forward_results.pkl")
    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
    else:
        st.info("Run wf_validate.py to generate walk-forward results.")
        return

    if not results:
        st.warning("No walk-forward results available.")
        return

    periods  = [r["test_start"] for r in results]
    accs     = [r["accuracy"] * 100 for r in results]
    f1s      = [r["f1"] for r in results]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))

    wc1, wc2, wc3, wc4 = st.columns(4)
    wc1.metric("Mean accuracy",   f"{mean_acc:.1f}%")
    wc2.metric("Best window",     f"{max(accs):.1f}%")
    wc3.metric("Worst window",    f"{min(accs):.1f}%")
    wc4.metric("Std (stability)", f"{std_acc:.1f}%")

    if std_acc > 5:
        st.warning("High variance across windows — consider retraining every 3 months.")
    else:
        st.success("Model is stable across rolling time windows.")

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, vals, ylabel, title, ylim, color in [
        (axes[0], accs, "Accuracy (%)", "Walk-Forward Test Accuracy", (40, 75), "#3b82f6"),
        (axes[1], f1s,  "F1 Score",     "Walk-Forward F1 Score",      (0.3, 0.75), "#0891b2"),
    ]:
        ax.plot(range(len(vals)), vals, "o-", color=color, linewidth=2, markersize=6)
        ax.axhline(float(np.mean(vals)), color="#64748b", linestyle="--", linewidth=1, label="Mean")
        if "Acc" in ylabel:
            ax.axhline(50, color="#ef4444", linestyle=":", linewidth=1, alpha=0.7, label="Random 50%")
        ax.fill_between(range(len(vals)), vals, float(np.mean(vals)), alpha=0.08, color=color)
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, rotation=45, fontsize=7)
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9, framealpha=0)
        ax.grid(True, alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


if __name__ == "__main__":
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score

    print("Walk-forward: 36mo train, 6mo test steps")
    print("-" * 75)

    df = pd.read_csv(os.path.join(DATA_DIR, "features_v2.csv"),
                     index_col=0, parse_dates=True).sort_index()
    feat_cols = [c for c in df.columns if c != "target"]

    spw = 1.0
    cw_path = os.path.join(DATA_DIR, "class_weights.pkl")
    if os.path.exists(cw_path):
        with open(cw_path, "rb") as f:
            spw = pickle.load(f).get("scale_pos_weight", 1.0)

    df["_ym"]  = df.index.to_period("M")
    all_months = sorted(df["_ym"].unique())
    results    = []

    for i in range(36, len(all_months), 6):
        tr_start = all_months[max(0, i - 36)]
        tr_end   = all_months[i - 1]
        te_start = all_months[i]
        te_end   = all_months[min(i + 5, len(all_months) - 1)]
        tr_mask  = (df["_ym"] >= tr_start) & (df["_ym"] <= tr_end)
        te_mask  = (df["_ym"] >= te_start) & (df["_ym"] <= te_end)
        X_tr, y_tr = df.loc[tr_mask, feat_cols], df.loc[tr_mask, "target"]
        X_te, y_te = df.loc[te_mask, feat_cols], df.loc[te_mask, "target"]
        if len(X_tr) < 100 or len(X_te) < 10:
            continue
        m1 = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.75,
                           scale_pos_weight=spw, random_state=42,
                           eval_metric="logloss", verbosity=0)
        m2 = LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                            subsample=0.8, is_unbalance=True, random_state=42, verbose=-1)
        m3 = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=10,
                                    class_weight="balanced", random_state=42, n_jobs=-1)
        m1.fit(X_tr, y_tr); m2.fit(X_tr, y_tr); m3.fit(X_tr, y_tr)
        p = (m1.predict_proba(X_te)[:,1]*0.40 + m2.predict_proba(X_te)[:,1]*0.40 +
             m3.predict_proba(X_te)[:,1]*0.20)
        preds = (p >= 0.5).astype(int)
        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, zero_division=0)
        print("Train: " + str(tr_start) + " to " + str(tr_end) +
              " | Test: " + str(te_start) + " to " + str(te_end) +
              " | acc=" + str(round(acc*100,1)) + "% f1=" + str(round(f1,3)))
        results.append({"test_start": str(te_start), "test_end": str(te_end),
                        "accuracy": round(acc,4), "f1": round(f1,4),
                        "n_train": len(X_tr), "n_test": len(X_te)})

    accs = [r["accuracy"] for r in results]
    print("-" * 75)
    print("Mean: " + str(round(float(np.mean(accs))*100,1)) + "%")
    print("Std : " + str(round(float(np.std(accs))*100,1)) + "%")
    with open(os.path.join(MODEL_DIR, "walk_forward_results.pkl"), "wb") as f:
        pickle.dump(results, f)
    print("Saved walk_forward_results.pkl")