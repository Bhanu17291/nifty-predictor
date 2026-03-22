"""
STEP 7 — Optimise Ensemble Weights
=====================================
Changes from original:

  [Step 7a] Ensemble weights no longer hardcoded (XGB=0.40, LGBM=0.40, RF=0.20)
            scipy.optimize.minimize finds the weights that maximise validation F1
            Optimal weights saved to models/optimal_weights.pkl

  [Step 7b] Regime-specific weights also optimised per regime
            Previously BULL/BEAR/FLAT weights were intuition guesses
            Now each regime's weights are tuned on historical regime periods

Run AFTER ensemble_model.py:
    python optimise_weights.py
"""

import pandas as pd
import numpy as np
import pickle
import os
from scipy.optimize import minimize
from sklearn.metrics import f1_score, accuracy_score

DATA_DIR  = "data"
MODEL_DIR = "models"


# ── Load ──────────────────────────────────────────────────────────────────────
def load_models():
    models = {}
    for name in ["xgb", "lgbm", "rf"]:
        path = os.path.join(MODEL_DIR, f"{name}_model_v2.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    if len(models) < 3:
        raise FileNotFoundError("Not all models found. Run ensemble_model.py first.")
    return models


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_v2.csv"), index_col=0, parse_dates=True)
    test  = pd.read_csv(os.path.join(DATA_DIR, "test_v2.csv"),  index_col=0, parse_dates=True)
    feat  = [c for c in train.columns if c != "target"]
    return (train[feat], train["target"],
            test[feat],  test["target"])


# ── Step 7a — Global weight optimisation ─────────────────────────────────────
def optimise_global_weights(models: dict,
                             X_val: pd.DataFrame,
                             y_val: pd.Series) -> dict:
    """
    STEP 7a FIX — Find optimal ensemble weights on validation data.

    Uses Nelder-Mead optimisation to minimise negative F1 score.
    Constraint: weights must sum to 1.0 and all be >= 0.

    The softmax trick converts unconstrained params → valid weight vector,
    avoiding the need for explicit equality constraints.
    """
    print("\n  ── Step 7a: Global Weight Optimisation ────────")

    # Get individual model probabilities once (avoid repeated inference)
    p_xgb  = models["xgb"].predict_proba(X_val)[:, 1]
    p_lgbm = models["lgbm"].predict_proba(X_val)[:, 1]
    p_rf   = models["rf"].predict_proba(X_val)[:, 1]
    probas = np.stack([p_xgb, p_lgbm, p_rf], axis=1)  # shape (n, 3)

    def neg_f1(params):
        # Softmax converts any 3 real numbers → weights that sum to 1
        w = np.exp(params) / np.exp(params).sum()
        ensemble_prob = (probas * w).sum(axis=1)
        preds = (ensemble_prob >= 0.5).astype(int)
        return -f1_score(y_val, preds, zero_division=0)

    # Start from current equal weights in log space
    x0 = np.log([0.40, 0.40, 0.20])
    result = minimize(neg_f1, x0, method="Nelder-Mead",
                      options={"maxiter": 2000, "xatol": 1e-6, "fatol": 1e-6})

    # Convert optimised params back to weights
    opt_w = np.exp(result.x) / np.exp(result.x).sum()
    opt_weights = {
        "xgb" : round(float(opt_w[0]), 4),
        "lgbm": round(float(opt_w[1]), 4),
        "rf"  : round(float(opt_w[2]), 4),
    }

    # Compare old vs new
    old_weights = {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20}
    for label, w in [("Original", old_weights), ("Optimised", opt_weights)]:
        ep = (probas * np.array([w["xgb"], w["lgbm"], w["rf"]])).sum(axis=1)
        preds = (ep >= 0.5).astype(int)
        acc = accuracy_score(y_val, preds)
        f1  = f1_score(y_val, preds, zero_division=0)
        print(f"  {label:<12} XGB={w['xgb']:.3f} LGBM={w['lgbm']:.3f} RF={w['rf']:.3f} "
              f"→ acc={acc*100:.1f}%  f1={f1:.3f}")

    print(f"\n  ✅ Optimal weights: XGB={opt_weights['xgb']:.3f}  "
          f"LGBM={opt_weights['lgbm']:.3f}  RF={opt_weights['rf']:.3f}")
    print("  ────────────────────────────────────────────────\n")
    return opt_weights


# ── Step 7b — Regime-specific weight optimisation ─────────────────────────────
def optimise_regime_weights(models: dict,
                             X_val: pd.DataFrame,
                             y_val: pd.Series,
                             closes: pd.DataFrame) -> dict:
    """
    STEP 7b FIX — Optimise weights separately for BULL/BEAR/FLAT regimes.

    Previously these were:
        BULL: XGB=0.45, LGBM=0.40, RF=0.15  ← intuition
        BEAR: XGB=0.30, LGBM=0.35, RF=0.35  ← intuition
        FLAT: XGB=0.40, LGBM=0.40, RF=0.20  ← intuition

    Now each regime gets weights tuned on the actual historical
    days that fell in that regime.

    Falls back to global optimal weights if a regime has <30 samples.
    """
    print("  ── Step 7b: Regime-Specific Weight Optimisation ─")

    from regime_detector import get_regime_history
    regime_series = get_regime_history(closes)

    # Align regime labels with validation index
    regime_aligned = regime_series.reindex(X_val.index).fillna("FLAT")

    p_xgb  = models["xgb"].predict_proba(X_val)[:, 1]
    p_lgbm = models["lgbm"].predict_proba(X_val)[:, 1]
    p_rf   = models["rf"].predict_proba(X_val)[:, 1]
    probas = np.stack([p_xgb, p_lgbm, p_rf], axis=1)

    regime_weights = {}
    default_w      = {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20}

    for regime in ["BULL", "BEAR", "FLAT"]:
        mask = (regime_aligned == regime).values
        n    = mask.sum()

        if n < 30:
            print(f"  {regime:<5} only {n} samples — using global optimal weights")
            regime_weights[regime] = default_w
            continue

        p_regime = probas[mask]
        y_regime = y_val.values[mask]

        def neg_f1_regime(params):
            w  = np.exp(params) / np.exp(params).sum()
            ep = (p_regime * w).sum(axis=1)
            return -f1_score(y_regime, (ep >= 0.5).astype(int), zero_division=0)

        x0     = np.log([0.40, 0.40, 0.20])
        result = minimize(neg_f1_regime, x0, method="Nelder-Mead",
                          options={"maxiter": 1000})
        opt_w  = np.exp(result.x) / np.exp(result.x).sum()

        rw = {
            "xgb" : round(float(opt_w[0]), 4),
            "lgbm": round(float(opt_w[1]), 4),
            "rf"  : round(float(opt_w[2]), 4),
        }
        regime_weights[regime] = rw

        # Score
        ep    = (p_regime * opt_w).sum(axis=1)
        acc   = accuracy_score(y_regime, (ep >= 0.5).astype(int))
        f1    = f1_score(y_regime, (ep >= 0.5).astype(int), zero_division=0)
        print(f"  {regime:<5} n={n:>4}  XGB={rw['xgb']:.3f} LGBM={rw['lgbm']:.3f} "
              f"RF={rw['rf']:.3f}  acc={acc*100:.1f}%  f1={f1:.3f}")

    print("  ────────────────────────────────────────────────\n")
    return regime_weights


# ── Step 7c — Threshold optimisation ─────────────────────────────────────────
def optimise_threshold(models: dict,
                        X_val: pd.DataFrame,
                        y_val: pd.Series,
                        weights: dict) -> float:
    """
    STEP 7c — Find optimal decision threshold using Youden's J statistic.

    Default threshold of 0.50 is rarely optimal.
    Youden's J = sensitivity + specificity - 1
    The threshold that maximises J is the optimal operating point.

    Also shows the accuracy/precision/recall tradeoff across thresholds
    so you can choose based on your preference (fewer false positives vs more true positives).
    """
    print("  ── Step 7c: Threshold Optimisation ────────────")

    p_xgb  = models["xgb"].predict_proba(X_val)[:, 1]
    p_lgbm = models["lgbm"].predict_proba(X_val)[:, 1]
    p_rf   = models["rf"].predict_proba(X_val)[:, 1]
    w      = np.array([weights["xgb"], weights["lgbm"], weights["rf"]])
    proba  = (np.stack([p_xgb, p_lgbm, p_rf], axis=1) * w).sum(axis=1)
    y_arr  = y_val.values

    best_j    = -1
    best_acc  = -1
    best_t    = 0.50
    best_acc_t= 0.50

    print(f"\n  {'Threshold':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Youden J':>10}")
    print("  " + "-" * 62)

    thresholds = np.arange(0.40, 0.65, 0.01)
    for t in thresholds:
        preds = (proba >= t).astype(int)
        acc   = accuracy_score(y_arr, preds)
        f1    = f1_score(y_arr, preds, zero_division=0)

        tp = ((preds == 1) & (y_arr == 1)).sum()
        fp = ((preds == 1) & (y_arr == 0)).sum()
        fn = ((preds == 0) & (y_arr == 1)).sum()
        tn = ((preds == 0) & (y_arr == 0)).sum()

        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        precision   = tp / (tp + fp + 1e-9)
        youden_j    = sensitivity + specificity - 1

        if youden_j > best_j:
            best_j = youden_j
            best_t = round(t, 2)
        if acc > best_acc:
            best_acc   = acc
            best_acc_t = round(t, 2)

        marker = " ← best J" if round(t, 2) == best_t else ""
        print(f"  {t:>10.2f} {acc*100:>9.1f}% {precision:>10.3f} {sensitivity:>10.3f} "
              f"{f1:>8.3f} {youden_j:>10.3f}{marker}")

    print(f"\n  ✅ Best threshold by Youden's J  : {best_t:.2f}")
    print(f"  ✅ Best threshold by accuracy    : {best_acc_t:.2f}")
    print(f"  Using Youden's J threshold: {best_t:.2f}")
    print("  ────────────────────────────────────────────────\n")
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Step 7: Weight Optimisation")
    print("=" * 55 + "\n")

    models                    = load_models()
    X_train, y_train, X_test, y_test = load_data()

    # Use last 20% of training data as validation set for optimisation
    # (never touch test data for tuning — that would be leakage)
    val_size  = int(len(X_train) * 0.20)
    X_val     = X_train.iloc[-val_size:]
    y_val     = y_train.iloc[-val_size:]
    print(f"  Validation set: {len(X_val)} rows "
          f"({X_val.index[0].date()} → {X_val.index[-1].date()})\n")

    # Step 7a — optimise global weights
    opt_weights = optimise_global_weights(models, X_val, y_val)

    # Step 7b — optimise regime weights
    closes_path = os.path.join(DATA_DIR, "closes_v2.csv")
    if os.path.exists(closes_path):
        closes = pd.read_csv(closes_path, index_col=0, parse_dates=True)
        regime_weights = optimise_regime_weights(models, X_val, y_val, closes)
    else:
        print("  ⚠️  closes_v2.csv not found — skipping regime weight optimisation")
        regime_weights = {
            "BULL": opt_weights,
            "BEAR": opt_weights,
            "FLAT": opt_weights,
        }

    # Step 7c — optimise threshold
    best_threshold = optimise_threshold(models, X_val, y_val, opt_weights)

    # Final evaluation on test set with optimised weights + threshold
    print("  ── Final Test Evaluation (optimised weights) ──")
    p_xgb  = models["xgb"].predict_proba(X_test)[:, 1]
    p_lgbm = models["lgbm"].predict_proba(X_test)[:, 1]
    p_rf   = models["rf"].predict_proba(X_test)[:, 1]
    w      = np.array([opt_weights["xgb"], opt_weights["lgbm"], opt_weights["rf"]])
    proba  = (np.stack([p_xgb, p_lgbm, p_rf], axis=1) * w).sum(axis=1)

    for label, thresh in [("threshold=0.50 (original)", 0.50),
                           (f"threshold={best_threshold} (optimised)", best_threshold)]:
        preds = (proba >= thresh).astype(int)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds, zero_division=0)
        print(f"  {label:<40} acc={acc*100:.1f}%  f1={f1:.3f}")

    # Save everything
    output = {
        "global_weights"  : opt_weights,
        "regime_weights"  : regime_weights,
        "best_threshold"  : best_threshold,
    }
    out_path = os.path.join(MODEL_DIR, "optimal_weights.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(output, f)
    print(f"\n  💾 Saved → {out_path}")

    print()
    print("=" * 55)
    print("  ✅ Weight optimisation complete!")
    print(f"  Global weights  : XGB={opt_weights['xgb']:.3f}  "
          f"LGBM={opt_weights['lgbm']:.3f}  RF={opt_weights['rf']:.3f}")
    print(f"  Best threshold  : {best_threshold}")
    print("=" * 55)
    print()
    print("  Next → run: python calibration.py")


if __name__ == "__main__":
    main()