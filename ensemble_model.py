"""
ENSEMBLE MODEL
Trains XGBoost + LightGBM + Random Forest
Combines via soft voting weighted ensemble
Saves all 3 models + ensemble weights
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(DATA_DIR, "train_v2.csv")
TEST_PATH  = os.path.join(DATA_DIR, "test_v2.csv")

# Ensemble weights (must sum to 1.0)
WEIGHTS = {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20}


# ── Load ─────────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col="Date", parse_dates=True)
    test  = pd.read_csv(TEST_PATH,  index_col="Date", parse_dates=True)
    feat  = [c for c in train.columns if c != "target"]
    return (train[feat], train["target"],
            test[feat],  test["target"], feat)


# ── Train each model ──────────────────────────────────────────────────────────
def train_xgb(X_train, y_train):
    print("  Training XGBoost...")
    m = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        min_child_weight=5, gamma=0.1,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    m.fit(X_train, y_train)
    print("  ✅ XGBoost done")
    return m


def train_lgbm(X_train, y_train):
    print("  Training LightGBM...")
    m = LGBMClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        min_child_samples=20,
        random_state=42, verbose=-1,
    )
    m.fit(X_train, y_train)
    print("  ✅ LightGBM done")
    return m


def train_rf(X_train, y_train):
    print("  Training Random Forest...")
    m = RandomForestClassifier(
        n_estimators=200, max_depth=6,
        min_samples_leaf=10, max_features="sqrt",
        random_state=42, n_jobs=-1,
    )
    m.fit(X_train, y_train)
    print("  ✅ Random Forest done")
    return m


# ── Ensemble predict ──────────────────────────────────────────────────────────
def ensemble_proba(models, X, weights=WEIGHTS):
    p_xgb  = models["xgb"].predict_proba(X)[:, 1]
    p_lgbm = models["lgbm"].predict_proba(X)[:, 1]
    p_rf   = models["rf"].predict_proba(X)[:, 1]
    return (p_xgb  * weights["xgb"]  +
            p_lgbm * weights["lgbm"] +
            p_rf   * weights["rf"])


def ensemble_predict(models, X, weights=WEIGHTS, threshold=0.5):
    proba = ensemble_proba(models, X, weights)
    return (proba >= threshold).astype(int), proba


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_all(models, X_train, y_train, X_test, y_test):
    print("\n  ── Individual Model Performance ──")
    results = {}

    for name, m in models.items():
        tr_acc = accuracy_score(y_train, m.predict(X_train))
        te_acc = accuracy_score(y_test,  m.predict(X_test))
        te_f1  = f1_score(y_test, m.predict(X_test))
        results[name] = {"train_acc": tr_acc, "test_acc": te_acc, "f1": te_f1}
        print(f"  {name.upper():<8} Train: {tr_acc*100:.1f}%  Test: {te_acc*100:.1f}%  F1: {te_f1:.3f}")

    # Ensemble
    ens_preds, ens_proba = ensemble_predict(models, X_test)
    ens_train_preds, _   = ensemble_predict(models, X_train)
    ens_tr_acc = accuracy_score(y_train, ens_train_preds)
    ens_te_acc = accuracy_score(y_test,  ens_preds)
    ens_f1     = f1_score(y_test, ens_preds)
    results["ensemble"] = {"train_acc": ens_tr_acc, "test_acc": ens_te_acc,
                           "f1": ens_f1, "proba": ens_proba, "preds": ens_preds}

    print(f"\n  {'ENSEMBLE':<8} Train: {ens_tr_acc*100:.1f}%  Test: {ens_te_acc*100:.1f}%  F1: {ens_f1:.3f}")
    print()
    print("  ┌─────────────────────────────────────────┐")
    print(f"  │  Ensemble Test Accuracy : {ens_te_acc*100:.1f}%          │")
    print(f"  │  Ensemble F1 Score      : {ens_f1:.3f}             │")
    print(f"  │  Improvement over XGB   : +{(ens_te_acc - results['xgb']['test_acc'])*100:.1f}%         │")
    print("  └─────────────────────────────────────────┘")
    print()
    print("  Classification Report (Ensemble):")
    print(classification_report(y_test, ens_preds, target_names=["DOWN", "UP"]))
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_comparison(results, y_test):
    names  = ["XGBoost", "LightGBM", "Rand Forest", "Ensemble"]
    keys   = ["xgb", "lgbm", "rf", "ensemble"]
    accs   = [results[k]["test_acc"] * 100 for k in keys]
    f1s    = [results[k]["f1"] for k in keys]
    colors = ["#93c5fd", "#93c5fd", "#93c5fd", "#1e3a8a"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                             facecolor="white")

    # Accuracy bar
    ax = axes[0]
    ax.set_facecolor("white")
    bars = ax.bar(names, accs, color=colors, edgecolor="none", width=0.5)
    ax.axhline(50, color="#f87171", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_title("Test Accuracy by Model", fontsize=12, color="#0f172a", pad=10)
    ax.set_ylabel("Accuracy (%)", color="#64748b", fontsize=9)
    ax.set_ylim(45, 70)
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=8, color="#0f172a", fontweight="bold")

    # F1 bar
    ax = axes[1]
    ax.set_facecolor("white")
    bars = ax.bar(names, f1s, color=colors, edgecolor="none", width=0.5)
    ax.set_title("F1 Score by Model", fontsize=12, color="#0f172a", pad=10)
    ax.set_ylabel("F1 Score", color="#64748b", fontsize=9)
    ax.set_ylim(0.4, 0.75)
    ax.tick_params(colors="#475569", labelsize=8)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=8, color="#0f172a", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"), dpi=150,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  💾 Saved → models/model_comparison.png")


def plot_feature_importance(models, feat_cols):
    """Average feature importance across XGB and LGBM."""
    xgb_imp  = pd.Series(models["xgb"].feature_importances_,  index=feat_cols)
    lgbm_imp = pd.Series(models["lgbm"].feature_importances_, index=feat_cols)
    avg_imp  = ((xgb_imp / xgb_imp.sum()) + (lgbm_imp / lgbm_imp.sum())) / 2
    avg_imp  = avg_imp.sort_values(ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.set_facecolor("white")
    colors = ["#1e3a8a" if i < 5 else "#93c5fd" for i in range(len(avg_imp))]
    avg_imp[::-1].plot(kind="barh", ax=ax, color=colors[::-1], edgecolor="none")
    ax.set_title("Top 20 Feature Importances (Ensemble Average)",
                 fontsize=12, color="#0f172a", pad=10)
    ax.set_xlabel("Avg Importance Score", color="#64748b", fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(colors="#475569", labelsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "ensemble_feature_importance.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  💾 Saved → models/ensemble_feature_importance.png")

    print("\n  Top 10 Features (Ensemble):")
    for i, (feat, score) in enumerate(avg_imp.head(10).items(), 1):
        bar = "█" * int(score * 500)
        print(f"    {i:2}. {feat:<30} {bar} {score:.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
def save_models(models):
    for name, m in models.items():
        path = os.path.join(MODEL_DIR, f"{name}_model_v2.pkl")
        with open(path, "wb") as f:
            pickle.dump(m, f)
        print(f"  💾 Saved → {path}")

    # Save ensemble config
    config = {"weights": WEIGHTS, "threshold": 0.5}
    with open(os.path.join(MODEL_DIR, "ensemble_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"  💾 Saved → models/ensemble_config.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Ensemble Model Training")
    print("=" * 55 + "\n")

    X_train, y_train, X_test, y_test, feat_cols = load_data()
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Features: {len(feat_cols)}\n")

    models = {
        "xgb"  : train_xgb(X_train, y_train),
        "lgbm" : train_lgbm(X_train, y_train),
        "rf"   : train_rf(X_train, y_train),
    }

    results = evaluate_all(models, X_train, y_train, X_test, y_test)

    print("  Generating plots...")
    plot_comparison(results, y_test)
    plot_feature_importance(models, feat_cols)

    save_models(models)

    ens_acc = results["ensemble"]["test_acc"]
    print()
    print("=" * 55)
    print("  ✅ Ensemble training complete!")
    print(f"  Ensemble Test Accuracy: {ens_acc*100:.1f}%")
    print("=" * 55)
    print()
    print("  Next → run: python regime_detector.py")


if __name__ == "__main__":
    main()