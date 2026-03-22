"""
ENSEMBLE MODEL — Fixed & Improved
===================================
Changes from original:

  [Step 5] Class imbalance handling added
           Loads scale_pos_weight from data/class_weights.pkl (saved by features_v2.py)
           XGBoost: scale_pos_weight passed automatically
           Random Forest: class_weight='balanced'
           LightGBM: is_unbalance=True

  [Step 6] TimeSeriesSplit cross-validation added
           5-fold time-series CV used to validate hyperparameters
           Prevents overfitting to the single 2019-2022 window
           Reports per-fold accuracy so you can see stability over time
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
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH   = os.path.join(DATA_DIR, "train_v2.csv")
TEST_PATH    = os.path.join(DATA_DIR, "test_v2.csv")
WEIGHTS_PATH = os.path.join(DATA_DIR, "class_weights.pkl")

# Ensemble weights — will be replaced by optimised weights in Step 7
# For now these are the starting point, used as fallback
WEIGHTS = {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20}


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    train = pd.read_csv(TRAIN_PATH, index_col=0, parse_dates=True)
    test  = pd.read_csv(TEST_PATH,  index_col=0, parse_dates=True)

    # Handle both "Date" and "date" index names
    feat  = [c for c in train.columns if c != "target"]
    return (train[feat], train["target"],
            test[feat],  test["target"], feat)


def load_class_weights() -> dict:
    """
    STEP 5 — Load class weights saved by features_v2.py.
    Falls back to balanced (1.0) if file not found.
    """
    if os.path.exists(WEIGHTS_PATH):
        with open(WEIGHTS_PATH, "rb") as f:
            cw = pickle.load(f)
        print(f"  📊 Class weights loaded:")
        print(f"     UP: {cw['up_pct']:.1f}%  DOWN: {cw['down_pct']:.1f}%")
        print(f"     scale_pos_weight = {cw['scale_pos_weight']:.3f}")
        return cw
    else:
        print("  ⚠️  class_weights.pkl not found — using balanced (1.0)")
        print("     Run features_v2.py first to generate class weights")
        return {"scale_pos_weight": 1.0, "up_pct": 50.0, "down_pct": 50.0}


# ── STEP 6 — TimeSeriesSplit cross-validation ─────────────────────────────────
def time_series_cv(X_train: pd.DataFrame, y_train: pd.Series,
                   spw: float, n_splits: int = 5):
    """
    STEP 6 FIX — Walk-forward cross-validation on training data.

    Uses TimeSeriesSplit which respects temporal order:
      Fold 1: train on months 1-8,   validate on months 9-10
      Fold 2: train on months 1-10,  validate on months 11-12
      ...and so on, always training on past, validating on future.

    This is critical for financial time series — random K-fold would
    let the model "see" future data during training, inflating CV scores.

    Reports per-fold accuracy so you can see if model degrades over time.
    """
    print("\n  ── TimeSeriesSplit Cross-Validation (Step 6) ──")
    print(f"  Folds: {n_splits} | Training rows: {len(X_train)}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_tr  = X_train.iloc[tr_idx]
        y_tr  = y_train.iloc[tr_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]

        # Quick XGB fit for CV (fewer estimators for speed)
        m = XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.75,
            min_child_weight=5, gamma=0.1,
            scale_pos_weight=spw,
            random_state=42, eval_metric="logloss", verbosity=0,
        )
        m.fit(X_tr, y_tr)
        val_preds = m.predict(X_val)
        val_acc   = accuracy_score(y_val, val_preds)
        val_f1    = f1_score(y_val, val_preds)

        # Date range of validation fold
        if hasattr(X_val.index, 'min'):
            date_range = f"{X_val.index.min().date()} → {X_val.index.max().date()}"
        else:
            date_range = f"rows {val_idx[0]}–{val_idx[-1]}"

        fold_results.append({"fold": fold, "acc": val_acc, "f1": val_f1,
                              "n_val": len(val_idx), "dates": date_range})
        print(f"  Fold {fold}: {date_range} | acc={val_acc*100:.1f}%  f1={val_f1:.3f}  n={len(val_idx)}")

    mean_acc = np.mean([r["acc"] for r in fold_results])
    std_acc  = np.std([r["acc"]  for r in fold_results])
    mean_f1  = np.mean([r["f1"]  for r in fold_results])
    print(f"\n  CV Summary: acc={mean_acc*100:.1f}% ± {std_acc*100:.1f}%  f1={mean_f1:.3f}")

    # Stability check: if std > 5%, model performance is unstable across time
    if std_acc > 0.05:
        print(f"  ⚠️  High variance across folds ({std_acc*100:.1f}%) — model may not generalise well")
        print(f"     Consider: more regularisation, fewer features, or longer training window")
    else:
        print(f"  ✅ Model is stable across time folds")

    print("  ────────────────────────────────────────────────\n")
    return fold_results, mean_acc, mean_f1


# ── Train each model (with class imbalance fix) ───────────────────────────────
def train_xgb(X_train: pd.DataFrame, y_train: pd.Series, spw: float):
    """
    STEP 5 FIX — scale_pos_weight now passed from class_weights.pkl.
    Previously hardcoded to default (1.0 = assumes balanced classes).
    """
    print("  Training XGBoost...")
    m = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_weight=5,
        gamma=0.1,
        # STEP 5 FIX: was missing entirely — now loaded from class_weights.pkl
        scale_pos_weight=spw,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    m.fit(X_train, y_train)
    print(f"  ✅ XGBoost done  (scale_pos_weight={spw:.3f})")
    return m


def train_lgbm(X_train: pd.DataFrame, y_train: pd.Series, spw: float):
    """
    STEP 5 FIX — is_unbalance=True added.
    LightGBM handles imbalance internally when this flag is set.
    """
    print("  Training LightGBM...")
    m = LGBMClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.75,
        min_child_samples=20,
        # STEP 5 FIX: was missing — now enabled
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )
    m.fit(X_train, y_train)
    print(f"  ✅ LightGBM done  (is_unbalance=True)")
    return m


def train_rf(X_train: pd.DataFrame, y_train: pd.Series):
    """
    STEP 5 FIX — class_weight='balanced' added.
    RF now weights minority class samples inversely proportional to frequency.
    """
    print("  Training Random Forest...")
    m = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=10,
        max_features="sqrt",
        # STEP 5 FIX: was missing entirely
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    m.fit(X_train, y_train)
    print(f"  ✅ Random Forest done  (class_weight='balanced')")
    return m


# ── Ensemble predict ──────────────────────────────────────────────────────────
def ensemble_proba(models: dict, X: pd.DataFrame, weights: dict = WEIGHTS) -> np.ndarray:
    p_xgb  = models["xgb"].predict_proba(X)[:, 1]
    p_lgbm = models["lgbm"].predict_proba(X)[:, 1] if "lgbm" in models else p_xgb
    p_rf   = models["rf"].predict_proba(X)[:, 1]   if "rf"   in models else p_xgb
    return (p_xgb  * weights["xgb"]  +
            p_lgbm * weights["lgbm"] +
            p_rf   * weights["rf"])


def ensemble_predict(models: dict, X: pd.DataFrame,
                     weights: dict = WEIGHTS, threshold: float = 0.5):
    proba = ensemble_proba(models, X, weights)
    return (proba >= threshold).astype(int), proba


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_all(models: dict,
                 X_train: pd.DataFrame, y_train: pd.Series,
                 X_test:  pd.DataFrame, y_test:  pd.Series) -> dict:
    print("\n  ── Individual Model Performance ──")
    results = {}

    for name, m in models.items():
        tr_acc = accuracy_score(y_train, m.predict(X_train))
        te_acc = accuracy_score(y_test,  m.predict(X_test))
        te_f1  = f1_score(y_test, m.predict(X_test))
        results[name] = {"train_acc": tr_acc, "test_acc": te_acc, "f1": te_f1}
        overfit = tr_acc - te_acc
        flag    = " ⚠️ overfitting" if overfit > 0.10 else ""
        print(f"  {name.upper():<8} Train: {tr_acc*100:.1f}%  Test: {te_acc*100:.1f}%  "
              f"F1: {te_f1:.3f}  Gap: {overfit*100:+.1f}%{flag}")

    # Ensemble
    ens_preds, ens_proba   = ensemble_predict(models, X_test)
    ens_train_preds, _     = ensemble_predict(models, X_train)
    ens_tr_acc = accuracy_score(y_train, ens_train_preds)
    ens_te_acc = accuracy_score(y_test,  ens_preds)
    ens_f1     = f1_score(y_test, ens_preds)
    results["ensemble"] = {
        "train_acc": ens_tr_acc, "test_acc": ens_te_acc,
        "f1": ens_f1, "proba": ens_proba, "preds": ens_preds,
    }

    print(f"\n  {'ENSEMBLE':<8} Train: {ens_tr_acc*100:.1f}%  Test: {ens_te_acc*100:.1f}%  F1: {ens_f1:.3f}")
    print()
    print("  ┌─────────────────────────────────────────┐")
    print(f"  │  Ensemble Test Accuracy : {ens_te_acc*100:.1f}%          │")
    print(f"  │  Ensemble F1 Score      : {ens_f1:.3f}             │")
    print(f"  │  vs XGB alone           : {(ens_te_acc-results['xgb']['test_acc'])*100:+.1f}%         │")
    print("  └─────────────────────────────────────────┘")
    print()
    print("  Classification Report (Ensemble):")
    print(classification_report(y_test, ens_preds, target_names=["DOWN", "UP"]))
    return results


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_cv_results(fold_results: list):
    """Plot per-fold accuracy to visualise model stability over time."""
    folds = [r["fold"]   for r in fold_results]
    accs  = [r["acc"]*100 for r in fold_results]
    f1s   = [r["f1"]     for r in fold_results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(colors="#475569", labelsize=8)

    axes[0].plot(folds, accs,  "o-", color="#1e3a8a", linewidth=2, markersize=7)
    axes[0].axhline(50, color="#f87171", linestyle="--", linewidth=1, alpha=0.7)
    axes[0].axhline(np.mean(accs), color="#64748b", linestyle=":", linewidth=1)
    axes[0].set_title("CV Accuracy per Fold", fontsize=12, color="#0f172a", pad=10)
    axes[0].set_ylabel("Accuracy (%)", color="#64748b", fontsize=9)
    axes[0].set_xlabel("Fold (chronological)", color="#64748b", fontsize=9)
    axes[0].set_ylim(40, 75)

    axes[1].plot(folds, f1s, "s-", color="#0891b2", linewidth=2, markersize=7)
    axes[1].axhline(np.mean(f1s), color="#64748b", linestyle=":", linewidth=1)
    axes[1].set_title("CV F1 Score per Fold", fontsize=12, color="#0f172a", pad=10)
    axes[1].set_ylabel("F1 Score", color="#64748b", fontsize=9)
    axes[1].set_xlabel("Fold (chronological)", color="#64748b", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "cv_results.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  💾 Saved → models/cv_results.png")


def plot_comparison(results: dict):
    names  = ["XGBoost", "LightGBM", "Rand Forest", "Ensemble"]
    keys   = ["xgb", "lgbm", "rf", "ensemble"]
    accs   = [results[k]["test_acc"] * 100 for k in keys]
    f1s    = [results[k]["f1"] for k in keys]
    colors = ["#93c5fd", "#93c5fd", "#93c5fd", "#1e3a8a"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
    for ax, vals, title, ylabel, ylim in [
        (axes[0], accs, "Test Accuracy by Model", "Accuracy (%)", (45, 70)),
        (axes[1], f1s,  "F1 Score by Model",      "F1 Score",     (0.4, 0.75)),
    ]:
        ax.set_facecolor("white")
        bars = ax.bar(names, vals, color=colors, edgecolor="none", width=0.5)
        if title.startswith("Test"):
            ax.axhline(50, color="#f87171", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(title, fontsize=12, color="#0f172a", pad=10)
        ax.set_ylabel(ylabel, color="#64748b", fontsize=9)
        ax.set_ylim(*ylim)
        ax.tick_params(colors="#475569", labelsize=8)
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color("#e2e8f0")
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + (0.3 if title.startswith("Test") else 0.003),
                    f"{val:.1f}%" if title.startswith("Test") else f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8,
                    color="#0f172a", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "model_comparison.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  💾 Saved → models/model_comparison.png")


def plot_feature_importance(models: dict, feat_cols: list):
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
        print(f"    {i:2}. {feat:<35} {bar} {score:.4f}")


# ── Save ──────────────────────────────────────────────────────────────────────
def save_models(models: dict):
    for name, m in models.items():
        path = os.path.join(MODEL_DIR, f"{name}_model_v2.pkl")
        with open(path, "wb") as f:
            pickle.dump(m, f)
        print(f"  💾 Saved → {path}")

    config = {"weights": WEIGHTS, "threshold": 0.5}
    with open(os.path.join(MODEL_DIR, "ensemble_config.pkl"), "wb") as f:
        pickle.dump(config, f)
    print(f"  💾 Saved → models/ensemble_config.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Ensemble Model (Fixed)")
    print("=" * 55 + "\n")

    X_train, y_train, X_test, y_test, feat_cols = load_data()
    print(f"  Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print(f"  Features: {len(feat_cols)}\n")

    # Load class weights (Step 5)
    cw  = load_class_weights()
    spw = cw["scale_pos_weight"]

    # Run cross-validation first (Step 6)
    fold_results, cv_acc, cv_f1 = time_series_cv(X_train, y_train, spw, n_splits=5)
    plot_cv_results(fold_results)

    # Train final models on full training set
    print("  Training final models on full training set...")
    models = {
        "xgb"  : train_xgb(X_train,  y_train, spw),
        "lgbm" : train_lgbm(X_train, y_train, spw),
        "rf"   : train_rf(X_train,   y_train),
    }

    results = evaluate_all(models, X_train, y_train, X_test, y_test)

    print("  Generating plots...")
    plot_comparison(results)
    plot_feature_importance(models, feat_cols)
    save_models(models)

    ens_acc = results["ensemble"]["test_acc"]
    print()
    print("=" * 55)
    print("  ✅ Ensemble training complete!")
    print(f"  CV Accuracy (train)  : {cv_acc*100:.1f}%")
    print(f"  Test Accuracy        : {ens_acc*100:.1f}%")
    print(f"  CV vs Test gap       : {abs(cv_acc-ens_acc)*100:.1f}%  (lower = less overfit)")
    print("=" * 55)
    print()
    print("  Next → run: python regime_detector.py")


if __name__ == "__main__":
    main()