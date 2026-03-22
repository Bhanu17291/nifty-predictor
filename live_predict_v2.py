"""
LIVE PREDICT V2 — Fixed
=========================
Changes from original:

  [Step 8a] Missing feature fallback fixed
            WAS: X_live[col] = 0.0  ← zero is NOT neutral for return features
            NOW: X_live[col] = training_medians[col]  ← statistically neutral

  [Step 8b] Loads optimised weights from optimal_weights.pkl (Step 7)
            Falls back to regime_detector weights if file not found

  [Step 8c] Loads optimised threshold from optimal_weights.pkl (Step 7)
            Falls back to 0.50 if file not found
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

DATA_DIR  = "data"
MODEL_DIR = "models"

TICKERS = {
    "sp500"      : "^GSPC",
    "nasdaq"     : "^IXIC",
    "nifty"      : "^NSEI",
    # Step 1 fix carried through — NIFTYBEES.NS not ^NSEI
    "gift_nifty" : "NIFTYBEES.NS",
    "vix_india"  : "^INDIAVIX",
    "usdinr"     : "INR=X",
    "crude"      : "BZ=F",
}


def confidence_tier(conf: float) -> dict:
    if conf >= 65:
        return {"tier": "STRONG",   "label": "Strong Signal",   "emoji": "🔥"}
    elif conf >= 57:
        return {"tier": "MODERATE", "label": "Moderate Signal", "emoji": "✅"}
    elif conf >= 52:
        return {"tier": "WEAK",     "label": "Weak Signal",     "emoji": "⚠️"}
    else:
        return {"tier": "UNCLEAR",  "label": "No Clear Signal", "emoji": "⚪"}


def load_models() -> dict:
    models = {}
    for name in ["xgb", "lgbm", "rf"]:
        path = os.path.join(MODEL_DIR, f"{name}_model_v2.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def load_optimal_weights() -> dict:
    """
    STEP 8b — Load optimised weights and threshold from Step 7.
    Falls back gracefully if optimise_weights.py hasn't been run yet.
    """
    path = os.path.join(MODEL_DIR, "optimal_weights.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data
    # Fallback to default weights
    return {
        "global_weights" : {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20},
        "regime_weights" : {
            "BULL": {"xgb": 0.45, "lgbm": 0.40, "rf": 0.15},
            "BEAR": {"xgb": 0.30, "lgbm": 0.35, "rf": 0.35},
            "FLAT": {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20},
        },
        "best_threshold" : 0.50,
    }


# ── STEP 8a — Load training medians ──────────────────────────────────────────
def load_training_medians(feature_cols: list) -> pd.Series:
    """
    STEP 8a FIX — Load median of each feature from training data.

    When a live feature is missing (API down, market closed, new column),
    we fill with the training median instead of 0.0.

    Why median not mean:
      Return distributions are fat-tailed. Mean is pulled by outliers.
      Median is the true "average day" for financial return features.

    Saves medians to data/training_medians.pkl on first run,
    reloads from cache on subsequent runs.
    """
    cache_path = os.path.join(DATA_DIR, "training_medians.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            medians = pickle.load(f)
        return medians

    # Compute from training data if cache doesn't exist
    train_path = os.path.join(DATA_DIR, "train_v2.csv")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path, index_col=0, parse_dates=True)
        feat     = [c for c in feature_cols if c in train_df.columns]
        medians  = train_df[feat].median()
        with open(cache_path, "wb") as f:
            pickle.dump(medians, f)
        print(f"  💾 Training medians cached → {cache_path}")
        return medians

    # Last resort: return zeros (same as before, but now explicit)
    print("  ⚠️  Could not load training data for medians — using 0.0 fallback")
    return pd.Series({col: 0.0 for col in feature_cols})


def fetch_live(days_back: int = 120) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    closes = {}
    for name, sym in TICKERS.items():
        try:
            df = yf.download(sym,
                             start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False)
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes[name] = df["Close"]
        except Exception:
            pass
    df = pd.DataFrame(closes)
    df.index = pd.to_datetime(df.index)
    return df.ffill(limit=2)


def build_live_features(closes: pd.DataFrame) -> pd.DataFrame:
    """Rebuild features — must match features_v2.py exactly including z-scores."""
    from features_v2 import build_features
    df = build_features(closes)
    if df.empty:
        df = build_features(closes.ffill())
        df = df.ffill().dropna(how="all")
    return df


def predict_tomorrow_v2(verbose: bool = True) -> dict:
    if verbose:
        print("=" * 55)
        print("  NIFTY PREDICTOR V2 — Live Prediction (Fixed)")
        print("=" * 55 + "\n")
        print("  Fetching live market data...")

    closes   = fetch_live(days_back=120)
    features = build_live_features(closes)

    if features.empty:
        raise ValueError("Not enough live data to build features.")

    models = load_models()
    if not models:
        raise ValueError("No trained models found. Run ensemble_model.py first.")

    # Load feature columns from training data
    train_path   = os.path.join(DATA_DIR, "features_v2.csv")
    train_df     = pd.read_csv(train_path, index_col=0, nrows=2)
    feature_cols = [c for c in train_df.columns if c != "target"]

    # Get latest row
    latest      = features.tail(1)
    latest_date = latest.index[0]

    # ── STEP 8a FIX — Fill missing features with training medians ─────────────
    # WAS: X_live[col] = 0.0  ← implied "flat market" for ALL missing features
    # NOW: X_live[col] = training_medians[col]  ← statistically neutral value
    medians = load_training_medians(feature_cols)
    X_live  = pd.DataFrame(index=latest.index, columns=feature_cols, dtype=float)

    missing_cols = []
    for col in feature_cols:
        if col in latest.columns and not latest[col].isna().all():
            X_live[col] = float(latest[col].iloc[0])
        else:
            fill_val    = float(medians.get(col, 0.0))
            X_live[col] = fill_val
            missing_cols.append(col)

    if missing_cols and verbose:
        print(f"  ⚠️  {len(missing_cols)} features filled with training median "
              f"(not 0.0): {missing_cols[:5]}{'...' if len(missing_cols)>5 else ''}")

    X_live = X_live.astype(float)

    # ── Load optimised weights and threshold (Step 7) ─────────────────────────
    opt_data  = load_optimal_weights()
    threshold = opt_data.get("best_threshold", 0.50)

    # Detect regime and use regime-specific weights
    try:
        from regime_detector import detect_regime
        regime_info    = detect_regime(closes.dropna(subset=["nifty"]))
        regime         = regime_info["regime"]
        # STEP 8b — use optimised regime weights if available, else fall back
        regime_weights = opt_data.get("regime_weights", {})
        weights        = regime_weights.get(regime, opt_data["global_weights"])
    except Exception:
        regime_info = {"regime": "FLAT", "emoji": "🔵",
                       "description": "Regime detection failed",
                       "weights": opt_data["global_weights"]}
        weights     = opt_data["global_weights"]
        regime      = "FLAT"

    # Ensemble predict
    p_xgb  = float(models["xgb"].predict_proba(X_live)[0][1])
    p_lgbm = float(models["lgbm"].predict_proba(X_live)[0][1]) if "lgbm" in models else p_xgb
    p_rf   = float(models["rf"].predict_proba(X_live)[0][1])   if "rf"   in models else p_xgb

    up_prob = (p_xgb  * weights["xgb"]  +
               p_lgbm * weights["lgbm"] +
               p_rf   * weights["rf"]) * 100

    down_prob  = 100 - up_prob

    # STEP 8c — use optimised threshold, not hardcoded 0.50
    pred_int   = 1 if (up_prob / 100) >= threshold else 0
    confidence = up_prob if pred_int == 1 else down_prob
    tier_info  = confidence_tier(confidence)

    def safe_ret(col):
        if col in closes.columns and len(closes[col].dropna()) >= 2:
            s = closes[col].dropna()
            return float((s.iloc[-1] / s.iloc[-2] - 1) * 100)
        return 0.0

    # SHAP explanation
    explanation = {"available": False, "reasons": []}
    try:
        from explainer import explain_prediction
        explanation = explain_prediction(X_live, models)
    except Exception:
        pass

    result = {
        "prediction"   : "UP 🟢" if pred_int == 1 else "DOWN 🔴",
        "pred_int"     : pred_int,
        "up_prob"      : round(up_prob, 1),
        "down_prob"    : round(down_prob, 1),
        "confidence"   : round(confidence, 1),
        "tier"         : tier_info["tier"],
        "tier_label"   : tier_info["label"],
        "tier_emoji"   : tier_info["emoji"],
        "regime"       : regime_info["regime"],
        "regime_emoji" : regime_info.get("emoji", "🔵"),
        "regime_desc"  : regime_info.get("description", ""),
        "as_of_date"   : latest_date.strftime("%d %b %Y"),
        "sp500_ret"    : round(safe_ret("sp500"),  2),
        "nasdaq_ret"   : round(safe_ret("nasdaq"), 2),
        "nifty_ret"    : round(safe_ret("nifty"),  2),
        "crude_ret"    : round(safe_ret("crude"),  2),
        "usdinr_ret"   : round(safe_ret("usdinr"), 2),
        "vix"          : round(float(closes["vix_india"].dropna().iloc[-1]), 2)
                         if "vix_india" in closes.columns else None,
        "explanation"  : explanation,
        "weights_used" : weights,
        "threshold_used": threshold,
        "missing_features": len(missing_cols),
    }

    if verbose:
        print(f"\n  Data as of      : {result['as_of_date']}")
        print(f"  Regime          : {result['regime_emoji']} {result['regime']}")
        print(f"  Weights used    : XGB={weights['xgb']:.3f} "
              f"LGBM={weights['lgbm']:.3f} RF={weights['rf']:.3f}")
        print(f"  Threshold used  : {threshold:.2f} (optimised)")
        print(f"  Missing features: {len(missing_cols)} (filled with median)")
        print(f"  S&P500          : {result['sp500_ret']:+.2f}%")
        print(f"  Nasdaq          : {result['nasdaq_ret']:+.2f}%")
        print(f"  Nifty (prev)    : {result['nifty_ret']:+.2f}%")
        print(f"  Crude Oil       : {result['crude_ret']:+.2f}%")
        print(f"  USD/INR         : {result['usdinr_ret']:+.2f}%")
        if result["vix"]:
            print(f"  India VIX       : {result['vix']}")
        print()
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  PREDICTION    : {result['prediction']:<23}│")
        print(f"  │  CONFIDENCE    : {result['confidence']:>5.1f}%                   │")
        print(f"  │  SIGNAL TIER   : {result['tier_emoji']} {result['tier_label']:<20}│")
        print(f"  │  UP PROB       : {result['up_prob']:>5.1f}%                   │")
        print(f"  │  DOWN PROB     : {result['down_prob']:>5.1f}%                   │")
        print("  └─────────────────────────────────────────┘")

    return result


if __name__ == "__main__":
    predict_tomorrow_v2(verbose=True)