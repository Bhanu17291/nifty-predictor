"""
LIVE PREDICTION — Tier 1
- Fetches latest data for S&P500, Nasdaq, Nifty, GIFT Nifty
- Builds feature vector from most recent available data
- Runs through trained XGBoost model
- Outputs tomorrow's Nifty opening prediction
- Includes target price magnitude estimate
- Includes options strategy recommendation
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

MODEL_DIR  = "models"
DATA_DIR   = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")

TICKERS = {
    "sp500"      : "^GSPC",
    "nasdaq"     : "^IXIC",
    "nifty"      : "^NSEI",
    "gift_nifty" : "^NSEI",
}

def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def fetch_latest(days_back: int = 40) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    all_closes = {}
    for name, symbol in TICKERS.items():
        try:
            df = yf.download(symbol,
                             start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False)
            if df.empty:
                print(f"  No data for {name}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            all_closes[name] = df["Close"]
        except Exception as e:
            print(f"  Error fetching {name}: {e}")
    closes = pd.DataFrame(all_closes)
    closes.index = pd.to_datetime(closes.index)
    closes = closes.ffill(limit=1).dropna()
    return closes

def build_features(closes: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=closes.index)

    for col in closes.columns:
        df[f"{col}_ret"] = closes[col].pct_change() * 100

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_mom_3d"]  = df[r].rolling(3).sum()
        df[f"{market}_mom_5d"]  = df[r].rolling(5).sum()
        df[f"{market}_mom_10d"] = df[r].rolling(10).sum()

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
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

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_lag1"] = df[r].shift(1)
        df[f"{market}_lag2"] = df[r].shift(2)

    for market in ["sp500", "nasdaq"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_up_3d"] = (df[r].rolling(3).sum() > 0).astype(int)
        df[f"{market}_up_5d"] = (df[r].rolling(5).sum() > 0).astype(int)

    df = df.dropna()
    return df

def estimate_magnitude(closes: pd.DataFrame, pred_int: int) -> dict:
    """
    Simple magnitude estimate based on recent volatility and momentum.
    Uses the last 20 days of Nifty returns to estimate expected move size.
    """
    try:
        nifty_rets = closes["nifty"].pct_change().dropna() * 100
        if len(nifty_rets) < 5:
            return {"available": False}

        vol_5d     = float(nifty_rets.tail(5).std())
        mom_3d     = float(nifty_rets.tail(3).mean())
        sp_ret     = float((closes["sp500"].iloc[-1] / closes["sp500"].iloc[-2] - 1) * 100)
        nas_ret    = float((closes["nasdaq"].iloc[-1] / closes["nasdaq"].iloc[-2] - 1) * 100)
        us_avg     = (sp_ret + nas_ret) / 2

        base_move  = abs(us_avg) * 0.45 + vol_5d * 0.3
        direction  = 1 if pred_int == 1 else -1
        pred_ret   = round(direction * base_move, 2)
        std_err    = round(vol_5d * 0.6, 2)
        low_ret    = round(pred_ret - std_err, 2)
        high_ret   = round(pred_ret + std_err, 2)

        current_price = float(closes["nifty"].iloc[-1])
        pred_price    = round(current_price * (1 + pred_ret / 100))
        low_price     = round(current_price * (1 + low_ret  / 100))
        high_price    = round(current_price * (1 + high_ret / 100))

        return {
            "available"    : True,
            "predicted_ret": pred_ret,
            "range_low"    : low_ret,
            "range_high"   : high_ret,
            "current_price": current_price,
            "pred_price"   : pred_price,
            "low_price"    : low_price,
            "high_price"   : high_price,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}

def get_options_strategy(direction: str, confidence: float, vix_level: float = 15.0) -> dict:
    """
    Simple rule-based options strategy suggestion.
    """
    if confidence >= 65:
        conviction = "HIGH"
    elif confidence >= 55:
        conviction = "MODERATE"
    else:
        conviction = "WEAK"

    vix_cat = "HIGH" if vix_level > 18 else "LOW"

    STRATEGIES = {
        ("UP",   "HIGH",     "LOW") : ("Buy ATM Call",       "Strong bullish + low IV — directional call optimal."),
        ("UP",   "HIGH",     "HIGH"): ("Bull Call Spread",   "Strong bullish but high IV — spread reduces cost."),
        ("UP",   "MODERATE", "LOW") : ("ATM Call / Spread",  "Moderate bullish, low IV — single leg or spread."),
        ("UP",   "MODERATE", "HIGH"): ("Bull Put Spread",    "Moderate UP + high IV — sell premium instead."),
        ("UP",   "WEAK",     "LOW") : ("Sit Out",            "Weak signal + low IV — no clear edge."),
        ("UP",   "WEAK",     "HIGH"): ("Iron Condor",        "Weak signal + high IV — range-bound play."),
        ("DOWN", "HIGH",     "LOW") : ("Buy ATM Put",        "Strong bearish + low IV — directional put optimal."),
        ("DOWN", "HIGH",     "HIGH"): ("Bear Put Spread",    "Strong bearish + high IV — spread cuts cost."),
        ("DOWN", "MODERATE", "LOW") : ("ATM Put / Spread",  "Moderate bearish, low IV — single leg or spread."),
        ("DOWN", "MODERATE", "HIGH"): ("Bear Call Spread",  "Moderate DOWN + high IV — sell inflated calls."),
        ("DOWN", "WEAK",     "LOW") : ("Sit Out",           "Weak signal + low IV — no clear edge."),
        ("DOWN", "WEAK",     "HIGH"): ("Iron Condor",       "Weak signal + high IV — range-bound play."),
    }

    key      = (direction, conviction, vix_cat)
    strategy, reason = STRATEGIES.get(key, ("Insufficient data", "No matching strategy found."))

    return {
        "strategy"  : strategy,
        "reason"    : reason,
        "conviction": conviction,
        "vix_cat"   : vix_cat,
    }

def predict_tomorrow(verbose: bool = True) -> dict:
    """
    Main function — fetches live data, builds features, runs prediction.
    Returns a dict with prediction details including magnitude and options strategy.
    """
    if verbose:
        print("=" * 55)
        print("  NIFTY PREDICTOR — Live Prediction (Tier 1)")
        print("=" * 55 + "\n")
        print("  Fetching latest market data...")

    closes   = fetch_latest(days_back=40)
    features = build_features(closes)

    if features.empty:
        raise ValueError("Not enough data to build features. Try again later.")

    model = load_model()
    train_features_path = os.path.join(DATA_DIR, "features.csv")
    train_df     = pd.read_csv(train_features_path, index_col="date", nrows=1)
    feature_cols = [c for c in train_df.columns if c != "target"]

    latest_row  = features.tail(1)
    latest_date = latest_row.index[0]

    X_live = pd.DataFrame(index=latest_row.index, columns=feature_cols)
    for col in feature_cols:
        if col in latest_row.columns:
            X_live[col] = latest_row[col].values
        else:
            X_live[col] = 0.0
    X_live = X_live.astype(float)

    pred      = model.predict(X_live)[0]
    proba     = model.predict_proba(X_live)[0]
    up_prob   = proba[1] * 100
    down_prob = proba[0] * 100
    confidence = up_prob if pred == 1 else down_prob
    direction  = "UP" if pred == 1 else "DOWN"

    latest_closes = closes.tail(2)
    sp500_ret  = float((latest_closes["sp500"].iloc[-1]  / latest_closes["sp500"].iloc[-2]  - 1) * 100)
    nasdaq_ret = float((latest_closes["nasdaq"].iloc[-1] / latest_closes["nasdaq"].iloc[-2] - 1) * 100)
    nifty_ret  = float((latest_closes["nifty"].iloc[-1]  / latest_closes["nifty"].iloc[-2]  - 1) * 100)

    magnitude = estimate_magnitude(closes, int(pred))
    options   = get_options_strategy(direction, confidence)

    result = {
        "prediction"       : f"UP 🟢" if pred == 1 else "DOWN 🔴",
        "pred_int"         : int(pred),
        "up_prob"          : round(up_prob, 1),
        "down_prob"        : round(down_prob, 1),
        "confidence"       : round(confidence, 1),
        "as_of_date"       : latest_date.strftime("%d %b %Y"),
        "sp500_ret"        : round(sp500_ret, 2),
        "nasdaq_ret"       : round(nasdaq_ret, 2),
        "nifty_ret"        : round(nifty_ret, 2),
        "crude_ret"        : 0.0,
        "usdinr_ret"       : 0.0,
        "vix"              : None,
        "regime"           : "N/A",
        "regime_emoji"     : "",
        "regime_desc"      : "",
        "tier"             : "MODERATE",
        "tier_label"       : "Moderate Signal",
        "tier_emoji"       : "✅",
        "explanation"      : {"available": False, "reasons": []},
        "weights_used"     : {"xgb": 1.0, "lgbm": 0.0, "rf": 0.0},
        "magnitude"        : magnitude,
        "options_strategy" : options,
        "X_live"           : X_live,
    }

    if verbose:
        print(f"\n  Data as of      : {result['as_of_date']}")
        print(f"  S&P500          : {result['sp500_ret']:+.2f}%")
        print(f"  Nasdaq          : {result['nasdaq_ret']:+.2f}%")
        print(f"  Nifty (prev)    : {result['nifty_ret']:+.2f}%")
        print()
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  PREDICTION    : {result['prediction']:<23}│")
        print(f"  │  CONFIDENCE    : {result['confidence']:>5.1f}%                   │")
        print(f"  │  UP PROB       : {result['up_prob']:>5.1f}%                   │")
        print(f"  │  DOWN PROB     : {result['down_prob']:>5.1f}%                   │")
        if magnitude.get("available"):
            print(f"  │  TARGET RANGE  : {magnitude['low_price']:,.0f} – {magnitude['high_price']:,.0f}          │")
        print(f"  │  STRATEGY      : {options['strategy']:<23}│")
        print("  └─────────────────────────────────────────┘")
        print()
        print("=" * 55)

    return result


if __name__ == "__main__":
    result = predict_tomorrow(verbose=True)