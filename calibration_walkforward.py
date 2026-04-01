"""
STEP 9 — Probability Calibration (Platt Scaling)
STEP 10 — Walk-Forward Retraining
===================================================

Step 9:
  Wraps the ensemble in CalibratedClassifierCV(method='sigmoid')
  so that a 65% confidence actually means 65% historical accuracy.
  Saves calibrated wrapper to models/calibrated_ensemble.pkl
  Plots reliability diagram to show before/after calibration.

Step 10:
  Trains on rolling 36-month windows, stepping forward 6 months each time.
  Shows how accuracy evolves across time — reveals concept drift.
  Saves walk-forward results to models/walk_forward_results.pkl

Run AFTER optimise_weights.py:
    python calibration_walkforward.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, f1_score
from sklearn.base import BaseEstimator, ClassifierMixin

DATA_DIR  = "data"
MODEL_DIR = "models"


# ── Ensemble wrapper (needed for CalibratedClassifierCV) ─────────────────────
class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Wraps the 3-model ensemble into a single sklearn-compatible classifier.
    Required so CalibratedClassifierCV can wrap the whole ensemble at once
    rather than calibrating each model separately.
    """
    def __init__(self, models: dict, weights: dict):
        self.models  = models
        self.weights = weights
        self.classes_= np.array([0, 1])

    def fit(self, X, y):
        # Already trained — this is a post-hoc calibration wrapper
        return self

    def predict_proba(self, X):
        p_xgb  = self.models["xgb"].predict_proba(X)[:, 1]
        p_lgbm = self.models["lgbm"].predict_proba(X)[:, 1]
        p_rf   = self.models["rf"].predict_proba(X)[:, 1]
        p_up   = (p_xgb  * self.weights["xgb"]  +
                  p_lgbm * self.weights["lgbm"] +
                  p_rf   * self.weights["rf"])
        return np.column_stack([1 - p_up, p_up])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── Load helpers ──────────────────────────────────────────────────────────────
def load_models() -> dict:
    models = {}
    for name in ["xgb", "lgbm", "rf"]:
        path = os.path.join(MODEL_DIR, f"{name}_model_v2.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_v2.csv"), index_col=0, parse_dates=True)
    test  = pd.read_csv(os.path.join(DATA_DIR, "test_v2.csv"),  index_col=0, parse_dates=True)
    feat  = [c for c in train.columns if c != "target"]
    return train[feat], train["target"], test[feat], test["target"]


def load_optimal_weights() -> dict:
    path = os.path.join(MODEL_DIR, "optimal_weights.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {"global_weights": {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20},
            "best_threshold": 0.50}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Calibration
# ══════════════════════════════════════════════════════════════════════════════
def run_calibration(models: dict, weights: dict,
                    X_train: pd.DataFrame, y_train: pd.Series,
                    X_test:  pd.DataFrame, y_test:  pd.Series):
    """
    STEP 9 FIX — Platt scaling calibration.

    Problem before:
      The ensemble outputs raw probabilities from XGB/LGBM/RF.
      These are NOT calibrated — 70% confidence doesn't mean 70% historical win rate.
      Tree ensembles are known to be overconfident (probabilities cluster near 0/1).

    Fix:
      CalibratedClassifierCV(method='sigmoid') fits a logistic regression
      on top of the ensemble outputs using cross-validation.
      After calibration, if the model says 65%, it was actually right ~65% of the time.

    cv='prefit' means: the base estimator is already trained,
      just fit the calibration layer on the provided data.
    """
    print("\n  ── Step 9: Probability Calibration ────────────")

    ensemble = EnsembleClassifier(models, weights)

    # Calibrate on last 30% of training data (must not use test data)
    cal_size  = int(len(X_train) * 0.30)
    X_cal     = X_train.iloc[-cal_size:]
    y_cal     = y_train.iloc[-cal_size:]

    # cv=5 = fit calibration using 5-fold cross-validation on the calibration set
    # (cv='prefit' was removed in newer sklearn versions)
    calibrated = CalibratedClassifierCV(ensemble, method="sigmoid", cv=5)
    calibrated.fit(X_cal, y_cal)
    print(f"  Calibration fitted on {len(X_cal)} validation samples")

    # Compare raw vs calibrated on test set
    raw_proba = ensemble.predict_proba(X_test)[:, 1]
    cal_proba = calibrated.predict_proba(X_test)[:, 1]

    print(f"\n  {'':30} {'Raw':>10} {'Calibrated':>12}")
    print("  " + "-" * 55)
    for label, proba in [("Raw ensemble", raw_proba), ("Calibrated", cal_proba)]:
        preds = (proba >= 0.5).astype(int)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, zero_division=0)
        mean_conf = np.mean(np.maximum(proba, 1 - proba)) * 100
        print(f"  {label:<30} acc={acc*100:.1f}%  f1={f1:.3f}  mean_conf={mean_conf:.1f}%")

    # Reliability diagram (before and after)
    _plot_reliability(raw_proba, cal_proba, y_test.values)

    # Save calibrated model
    cal_path = os.path.join(MODEL_DIR, "calibrated_ensemble.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(calibrated, f)
    print(f"\n  💾 Saved → {cal_path}")
    print("  ────────────────────────────────────────────────\n")
    return calibrated


def _plot_reliability(raw_proba, cal_proba, y_true):
    """Reliability diagram: a well-calibrated model's points lie on the diagonal."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

    for ax, proba, title in [
        (axes[0], raw_proba, "Before calibration (raw ensemble)"),
        (axes[1], cal_proba, "After calibration (Platt scaling)"),
    ]:
        ax.set_facecolor("white")
        frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Perfect calibration")
        ax.plot(mean_pred, frac_pos, "o-", color="#1e3a8a", linewidth=2,
                markersize=6, label="Model")
        ax.fill_between(mean_pred, frac_pos, mean_pred,
                        alpha=0.1, color="#3b82f6")

        ax.set_title(title, fontsize=11, color="#0f172a", pad=10)
        ax.set_xlabel("Mean predicted probability", color="#64748b", fontsize=9)
        ax.set_ylabel("Fraction of positives", color="#64748b", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=9, framealpha=0)
        ax.grid(True, alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.tick_params(colors="#475569", labelsize=8)

    plt.suptitle("Reliability Diagram — Calibration Check",
                 fontsize=12, color="#0f172a", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "reliability_diagram.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  💾 Saved → models/reliability_diagram.png")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — Walk-Forward Retraining
# ══════════════════════════════════════════════════════════════════════════════
def run_walk_forward(train_months: int = 36, step_months: int = 6):
    """
    STEP 10 — Walk-forward validation with rolling retraining.

    Problem before:
      Model trained once on 2019–2022, never updated.
      Market regimes shift — a model trained pre-COVID behaves
      differently post-COVID. Silent accuracy decay over time.

    Fix:
      Train on a rolling 36-month window, step forward 6 months each time.
      For each window, record test accuracy.

    This answers: "Is my model getting worse over time?"
    If accuracy in 2024 is much lower than 2023, you need more recent training data.

    Results saved to models/walk_forward_results.pkl
    """
    print("  ── Step 10: Walk-Forward Retraining ───────────")

    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Load full feature dataset
    feat_path = os.path.join(DATA_DIR, "features_v2.csv")
    df = pd.read_csv(feat_path, index_col=0, parse_dates=True)
    df = df.sort_index()
    feat_cols = [c for c in df.columns if c != "target"]

    # Load class weights
    cw_path = os.path.join(DATA_DIR, "class_weights.pkl")
    spw = 1.0
    if os.path.exists(cw_path):
        with open(cw_path, "rb") as f:
            cw = pickle.load(f)
        spw = cw.get("scale_pos_weight", 1.0)

    # Build rolling windows
    df["year_month"] = df.index.to_period("M")
    all_months = sorted(df["year_month"].unique())

    results = []
    window  = []

    print(f"\n  Rolling window: {train_months}mo train → {step_months}mo test")
    print(f"  {'Window':<45} {'Train':>7} {'Test':>6} {'Acc':>7} {'F1':>6}")
    print("  " + "-" * 75)

    for i in range(train_months, len(all_months), step_months):
        train_start = all_months[max(0, i - train_months)]
        train_end   = all_months[i - 1]
        test_start  = all_months[i]
        test_end    = all_months[min(i + step_months - 1, len(all_months) - 1)]

        train_mask = (df["year_month"] >= train_start) & (df["year_month"] <= train_end)
        test_mask  = (df["year_month"] >= test_start)  & (df["year_month"] <= test_end)

        X_tr = df.loc[train_mask, feat_cols]
        y_tr = df.loc[train_mask, "target"]
        X_te = df.loc[test_mask,  feat_cols]
        y_te = df.loc[test_mask,  "target"]

        if len(X_tr) < 100 or len(X_te) < 10:
            continue

        # Train lightweight models for speed
        m_xgb = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.75, scale_pos_weight=spw,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        m_lgbm = LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, is_unbalance=True,
            random_state=42, verbose=-1,
        )
        m_rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

        m_xgb.fit(X_tr, y_tr)
        m_lgbm.fit(X_tr, y_tr)
        m_rf.fit(X_tr, y_tr)

        p = (m_xgb.predict_proba(X_te)[:, 1]  * 0.40 +
             m_lgbm.predict_proba(X_te)[:, 1] * 0.40 +
             m_rf.predict_proba(X_te)[:, 1]   * 0.20)
        preds = (p >= 0.5).astype(int)
        acc   = accuracy_score(y_te, preds)
        f1    = f1_score(y_te, preds, zero_division=0)

        label = f"{train_start}→{train_end} | test {test_start}→{test_end}"
        print(f"  {label:<45} {len(X_tr):>7} {len(X_te):>6} {acc*100:>6.1f}% {f1:>6.3f}")

        results.append({
            "train_start": str(train_start), "train_end": str(train_end),
            "test_start" : str(test_start),  "test_end"  : str(test_end),
            "n_train"    : len(X_tr),        "n_test"    : len(X_te),
            "accuracy"   : round(acc, 4),    "f1"        : round(f1, 4),
        })

    if not results:
        print("  ⚠️  Not enough data for walk-forward analysis")
        return []

    accs = [r["accuracy"] for r in results]
    print(f"\n  Walk-forward summary:")
    print(f"    Mean accuracy : {np.mean(accs)*100:.1f}%")
    print(f"    Min accuracy  : {np.min(accs)*100:.1f}%")
    print(f"    Max accuracy  : {np.max(accs)*100:.1f}%")
    print(f"    Std accuracy  : {np.std(accs)*100:.1f}%")

    if np.std(accs) > 0.05:
        print(f"\n  ⚠️  High variance across windows ({np.std(accs)*100:.1f}%)")
        print(f"     The model degrades significantly in some periods.")
        print(f"     Consider retraining more frequently (every 3–6 months).")
    else:
        print(f"\n  ✅ Model is stable across rolling windows")

    # Plot
    _plot_walk_forward(results)

    # Save
    wf_path = os.path.join(MODEL_DIR, "walk_forward_results.pkl")
    with open(wf_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\n  💾 Saved → {wf_path}")
    print("  ────────────────────────────────────────────────\n")
    return results


def _plot_walk_forward(results: list):
    periods = [f"{r['test_start']}" for r in results]
    accs    = [r["accuracy"] * 100   for r in results]
    f1s     = [r["f1"]               for r in results]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor="white")

    for ax, vals, ylabel, title, ylim, color in [
        (axes[0], accs, "Accuracy (%)",  "Walk-Forward Test Accuracy per Window",  (40, 75), "#1e3a8a"),
        (axes[1], f1s,  "F1 Score",      "Walk-Forward F1 Score per Window",        (0.3, 0.75), "#0891b2"),
    ]:
        ax.set_facecolor("white")
        ax.plot(range(len(vals)), vals, "o-", color=color, linewidth=2, markersize=6)
        ax.axhline(np.mean(vals), color="#64748b", linestyle="--", linewidth=1,
                   label=f"Mean: {np.mean(vals):.1f}{'%' if ylabel.startswith('Acc') else ''}")
        if ylabel.startswith("Acc"):
            ax.axhline(50, color="#f87171", linestyle=":", linewidth=1, alpha=0.7, label="Random (50%)")
        ax.fill_between(range(len(vals)), vals, np.mean(vals),
                        alpha=0.08, color=color)
        ax.set_title(title, fontsize=11, color="#0f172a", pad=10)
        ax.set_ylabel(ylabel, color="#64748b", fontsize=9)
        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels(periods, rotation=45, fontsize=7, color="#64748b")
        ax.set_ylim(*ylim)
        ax.legend(fontsize=9, framealpha=0, labelcolor="#94a3b8")
        ax.grid(True, alpha=0.3)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.tick_params(colors="#475569", labelsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "walk_forward.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  💾 Saved → models/walk_forward.png")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Steps 9 & 10")
    print("  Calibration + Walk-Forward Retraining")
    print("=" * 55 + "\n")

    models    = load_models()
    opt_data  = load_optimal_weights()
    weights   = opt_data["global_weights"]

    X_train, y_train, X_test, y_test = load_data()
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    # Step 9 — calibration
    run_calibration(models, weights, X_train, y_train, X_test, y_test)

    # Step 10 — walk-forward
    run_walk_forward(train_months=36, step_months=6)

    print("=" * 55)
    print("  ✅ All steps complete!")
    print()
    print("  Files saved:")
    print("    models/calibrated_ensemble.pkl  ← use in live_predict_v2.py")
    print("    models/reliability_diagram.png  ← calibration quality check")
    print("    models/walk_forward_results.pkl ← historical accuracy by period")
    print("    models/walk_forward.png         ← accuracy drift chart")
    print("=" * 55)
    print()
    print("  Full pipeline complete. Run order for future retraining:")
    print("    1. python data_fetch.py")
    print("    2. python features_v2.py")
    print("    3. python ensemble_model.py")
    print("    4. python regime_detector.py")
    print("    5. python optimise_weights.py")
    print("    6. python calibration_walkforward.py")


if __name__ == "__main__":
    main()