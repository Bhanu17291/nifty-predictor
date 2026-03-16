"""
LIVE PREDICT V2
- Fetches today's live data (all 7 markets)
- Detects market regime
- Runs ensemble prediction with regime-adjusted weights
- Confidence tier filter
- SHAP explanation
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
    "gift_nifty" : "^NSEI",
    "vix_india"  : "^INDIAVIX",
    "usdinr"     : "INR=X",
    "crude"      : "BZ=F",
}

# Confidence tiers
def confidence_tier(conf: float) -> dict:
    if conf >= 65:
        return {"tier": "STRONG",  "label": "Strong Signal",  "emoji": "🔥"}
    elif conf >= 57:
        return {"tier": "MODERATE","label": "Moderate Signal","emoji": "✅"}
    elif conf >= 52:
        return {"tier": "WEAK",    "label": "Weak Signal",    "emoji": "⚠️"}
    else:
        return {"tier": "UNCLEAR", "label": "No Clear Signal","emoji": "⚪"}


def load_models():
    models = {}
    for name in ["xgb", "lgbm", "rf"]:
        path = os.path.join(MODEL_DIR, f"{name}_model_v2.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def fetch_live(days_back: int = 60) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    closes = {}
    for name, sym in TICKERS.items():
        try:
            df = yf.download(sym,
                             start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False)
            if df.empty: continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes[name] = df["Close"]
        except:
            pass
    df = pd.DataFrame(closes)
    df.index = pd.to_datetime(df.index)
    return df.ffill(limit=2)


def build_live_features(closes: pd.DataFrame) -> pd.DataFrame:
    """Rebuild all features from closes — must match features_v2.py exactly."""
    df = pd.DataFrame(index=closes.index)

    for col in closes.columns:
        df[f"{col}_ret"] = closes[col].pct_change() * 100

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude", "usdinr"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_mom_3d"]  = df[r].rolling(3).sum()
        df[f"{market}_mom_5d"]  = df[r].rolling(5).sum()
        df[f"{market}_mom_10d"] = df[r].rolling(10).sum()

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_vol_5d"]  = df[r].rolling(5).std()
        df[f"{market}_vol_10d"] = df[r].rolling(10).std()

    if "sp500_ret" in df and "nasdaq_ret" in df:
        df["sp500_nasdaq_div"] = df["sp500_ret"] - df["nasdaq_ret"]
    if "sp500_ret" in df and "nasdaq_ret" in df and "nifty_ret" in df:
        df["us_avg_ret"]   = (df["sp500_ret"] + df["nasdaq_ret"]) / 2
        df["us_nifty_div"] = df["us_avg_ret"] - df["nifty_ret"]
    if "gift_nifty_ret" in df and "nifty_ret" in df:
        df["gift_nifty_basis"] = df["gift_nifty_ret"] - df["nifty_ret"]

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude", "usdinr"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_lag1"] = df[r].shift(1)
        df[f"{market}_lag2"] = df[r].shift(2)

    for market in ["sp500", "nasdaq", "nifty", "crude"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_up_3d"] = (df[r].rolling(3).sum() > 0).astype(int)
        df[f"{market}_up_5d"] = (df[r].rolling(5).sum() > 0).astype(int)

    if "vix_india_ret" in df:
        df["vix_level"]    = closes["vix_india"].ffill()
        df["vix_high"]     = (df["vix_level"] > 20).astype(int)
        df["vix_extreme"]  = (df["vix_level"] > 30).astype(int)
        df["vix_mom_3d"]   = df["vix_india_ret"].rolling(3).sum()
        df["vix_rising"]   = (df["vix_india_ret"].rolling(3).sum() > 0).astype(int)

    if "usdinr_ret" in df:
        df["usdinr_level"]    = closes["usdinr"].ffill()
        df["rupee_weak"]      = (df["usdinr_ret"] > 0).astype(int)
        df["rupee_very_weak"] = (df["usdinr_ret"] > 0.5).astype(int)

    if "crude_ret" in df:
        df["crude_spike"] = (df["crude_ret"].abs() > 2).astype(int)
        df["crude_up"]    = (df["crude_ret"] > 0).astype(int)

    df["day_of_week"]    = df.index.dayofweek
    df["month"]          = df.index.month
    df["is_monday"]      = (df.index.dayofweek == 0).astype(int)
    df["is_friday"]      = (df.index.dayofweek == 4).astype(int)
    df["is_month_end"]   = (df.index.day >= 25).astype(int)
    df["is_month_start"] = (df.index.day <= 5).astype(int)
    df["quarter"]        = df.index.quarter

    if "sp500_ret" in df and "nifty_ret" in df:
        df["sp500_nifty_corr_20d"] = df["sp500_ret"].rolling(20).corr(df["nifty_ret"])
    if "nasdaq_ret" in df and "nifty_ret" in df:
        df["nasdaq_nifty_corr_20d"] = df["nasdaq_ret"].rolling(20).corr(df["nifty_ret"])

    if "nifty" in closes.columns:
        nifty_price = closes["nifty"].ffill()
        ma20 = nifty_price.rolling(20).mean()
        ma50 = nifty_price.rolling(50).mean()
        df["nifty_dist_ma20"]  = ((nifty_price - ma20) / ma20 * 100)
        df["nifty_dist_ma50"]  = ((nifty_price - ma50) / ma50 * 100)
        df["nifty_above_ma20"] = (nifty_price > ma20).astype(int)
        df["nifty_above_ma50"] = (nifty_price > ma50).astype(int)

    if "sp500" in closes.columns:
        sp_price = closes["sp500"].ffill()
        ma50_sp  = sp_price.rolling(50).mean()
        df["sp500_dist_ma50"]  = ((sp_price - ma50_sp) / ma50_sp * 100)
        df["sp500_above_ma50"] = (sp_price > ma50_sp).astype(int)

    if "sp500_mom_5d" in df and "nifty_mom_5d" in df:
        df["cross_mom_diff"] = df["sp500_mom_5d"] - df["nifty_mom_5d"]

    return df.dropna()


def predict_tomorrow_v2(verbose: bool = True) -> dict:
    if verbose:
        print("=" * 55)
        print("  NIFTY PREDICTOR V2 — Live Prediction")
        print("=" * 55 + "\n")
        print("  Fetching live market data...")

    closes   = fetch_live(days_back=80)
    features = build_live_features(closes)

    if features.empty:
        raise ValueError("Not enough live data.")

    models = load_models()
    if not models:
        raise ValueError("No trained models found. Run ensemble_model.py first.")

    # Load feature columns from training data
    train_path   = os.path.join(DATA_DIR, "features_v2.csv")
    train_df     = pd.read_csv(train_path, index_col="Date", nrows=1)
    feature_cols = [c for c in train_df.columns if c != "target"]

    # Get latest row & align columns
    latest      = features.tail(1)
    latest_date = latest.index[0]

    X_live = pd.DataFrame(index=latest.index, columns=feature_cols)
    for col in feature_cols:
        X_live[col] = latest[col].values if col in latest.columns else 0.0
    X_live = X_live.astype(float)

    # Detect regime
    from regime_detector import detect_regime
    regime_info = detect_regime(closes.dropna(subset=["nifty"]))
    weights     = regime_info["weights"]

    # Ensemble predict with regime weights
    p_xgb  = models["xgb"].predict_proba(X_live)[0][1]
    p_lgbm = models["lgbm"].predict_proba(X_live)[0][1] if "lgbm" in models else p_xgb
    p_rf   = models["rf"].predict_proba(X_live)[0][1]   if "rf"   in models else p_xgb

    up_prob = (p_xgb  * weights["xgb"] +
               p_lgbm * weights["lgbm"] +
               p_rf   * weights["rf"]) * 100

    down_prob  = 100 - up_prob
    pred_int   = 1 if up_prob >= 50 else 0
    confidence = up_prob if pred_int == 1 else down_prob
    tier_info  = confidence_tier(confidence)

    # Market snapshot
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
    except:
        pass

    result = {
        "prediction"  : "UP 🟢" if pred_int == 1 else "DOWN 🔴",
        "pred_int"    : pred_int,
        "up_prob"     : round(up_prob, 1),
        "down_prob"   : round(down_prob, 1),
        "confidence"  : round(confidence, 1),
        "tier"        : tier_info["tier"],
        "tier_label"  : tier_info["label"],
        "tier_emoji"  : tier_info["emoji"],
        "regime"      : regime_info["regime"],
        "regime_emoji": regime_info["emoji"],
        "regime_desc" : regime_info["description"],
        "as_of_date"  : latest_date.strftime("%d %b %Y"),
        "sp500_ret"   : round(safe_ret("sp500"), 2),
        "nasdaq_ret"  : round(safe_ret("nasdaq"), 2),
        "nifty_ret"   : round(safe_ret("nifty"), 2),
        "crude_ret"   : round(safe_ret("crude"), 2),
        "usdinr_ret"  : round(safe_ret("usdinr"), 2),
        "vix"         : round(float(closes["vix_india"].dropna().iloc[-1]), 2)
                        if "vix_india" in closes.columns else None,
        "explanation" : explanation,
        "weights_used": weights,
    }

    if verbose:
        print(f"\n  Data as of      : {result['as_of_date']}")
        print(f"  Regime          : {result['regime_emoji']} {result['regime']}")
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