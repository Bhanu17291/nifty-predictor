"""
LIVE PREDICTION
- Fetches latest data for S&P500, Nasdaq, Nifty, GIFT Nifty
- Builds feature vector from most recent available data
- Runs through trained XGBoost model
- Outputs tomorrow's Nifty opening prediction
"""

import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_DIR  = "models"
DATA_DIR   = "data"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")

TICKERS = {
    "sp500"      : "^GSPC",
    "nasdaq"     : "^IXIC",
    "nifty"      : "^NSEI",
    "gift_nifty" : "^NSEI",
}

# ── Load Model ────────────────────────────────────────────────────────────────
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

# ── Fetch Latest Data ─────────────────────────────────────────────────────────
def fetch_latest(days_back: int = 30) -> pd.DataFrame:
    """
    Fetch last N days of data for all markets.
    We fetch 30 days to have enough history for rolling features (10d window).
    """
    end   = datetime.today()
    start = end - timedelta(days=days_back)

    all_closes = {}

    for name, symbol in TICKERS.items():
        try:
            df = yf.download(symbol, start=start.strftime("%Y-%m-%d"),
                             end=end.strftime("%Y-%m-%d"),
                             progress=False)
            if df.empty:
                print(f"  ⚠️  No data for {name}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            all_closes[name] = df["Close"]
        except Exception as e:
            print(f"  ❌ Error fetching {name}: {e}")

    closes = pd.DataFrame(all_closes)
    closes.index = pd.to_datetime(closes.index)
    closes = closes.ffill(limit=1).dropna()
    return closes


# ── Build Features ────────────────────────────────────────────────────────────
def build_features(closes: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate all 40 features from the training pipeline
    on the latest available data.
    """
    df = pd.DataFrame(index=closes.index)

    # ── Returns
    for col in closes.columns:
        df[f"{col}_ret"] = closes[col].pct_change() * 100

    # ── Momentum
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_mom_3d"]  = df[r].rolling(3).sum()
        df[f"{market}_mom_5d"]  = df[r].rolling(5).sum()
        df[f"{market}_mom_10d"] = df[r].rolling(10).sum()

    # ── Volatility
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_vol_5d"]  = df[r].rolling(5).std()
        df[f"{market}_vol_10d"] = df[r].rolling(10).std()

    # ── Divergence
    if "sp500_ret" in df and "nasdaq_ret" in df:
        df["sp500_nasdaq_div"] = df["sp500_ret"] - df["nasdaq_ret"]
    if "sp500_ret" in df and "nasdaq_ret" in df and "nifty_ret" in df:
        df["us_avg_ret"]   = (df["sp500_ret"] + df["nasdaq_ret"]) / 2
        df["us_nifty_div"] = df["us_avg_ret"] - df["nifty_ret"]
    if "gift_nifty_ret" in df and "nifty_ret" in df:
        df["gift_nifty_basis"] = df["gift_nifty_ret"] - df["nifty_ret"]

    # ── Lags
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_lag1"] = df[r].shift(1)
        df[f"{market}_lag2"] = df[r].shift(2)

    # ── Trend flags
    for market in ["sp500", "nasdaq"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_up_3d"] = (df[r].rolling(3).sum() > 0).astype(int)
        df[f"{market}_up_5d"] = (df[r].rolling(5).sum() > 0).astype(int)

    df = df.dropna()
    return df


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_tomorrow(verbose: bool = True) -> dict:
    """
    Main function — fetches live data, builds features, runs prediction.
    Returns a dict with prediction details.
    """
    if verbose:
        print("=" * 55)
        print("  NIFTY PREDICTOR — Live Prediction")
        print("=" * 55 + "\n")
        print("  Fetching latest market data...")

    closes   = fetch_latest(days_back=40)
    features = build_features(closes)

    if features.empty:
        raise ValueError("Not enough data to build features. Try again later.")

    # Load model and get feature columns from training data
    model = load_model()
    train_features_path = os.path.join(DATA_DIR, "features.csv")
    train_df     = pd.read_csv(train_features_path, index_col="date", nrows=1)
    feature_cols = [c for c in train_df.columns if c != "target"]

    # Get latest row
    latest_row  = features.tail(1)
    latest_date = latest_row.index[0]

    # Align columns — fill missing with 0
    X_live = pd.DataFrame(index=latest_row.index, columns=feature_cols)
    for col in feature_cols:
        if col in latest_row.columns:
            X_live[col] = latest_row[col].values
        else:
            X_live[col] = 0.0
    X_live = X_live.astype(float)

    # Predict
    pred  = model.predict(X_live)[0]
    proba = model.predict_proba(X_live)[0]
    up_prob   = proba[1] * 100
    down_prob = proba[0] * 100
    confidence = up_prob if pred == 1 else down_prob

    # Latest market snapshot
    latest_closes = closes.tail(2)
    sp500_ret    = float((latest_closes["sp500"].iloc[-1]    / latest_closes["sp500"].iloc[-2]    - 1) * 100)
    nasdaq_ret   = float((latest_closes["nasdaq"].iloc[-1]   / latest_closes["nasdaq"].iloc[-2]   - 1) * 100)
    nifty_ret    = float((latest_closes["nifty"].iloc[-1]    / latest_closes["nifty"].iloc[-2]    - 1) * 100)

    result = {
        "prediction"  : "UP 🟢" if pred == 1 else "DOWN 🔴",
        "pred_int"    : int(pred),
        "up_prob"     : round(up_prob, 1),
        "down_prob"   : round(down_prob, 1),
        "confidence"  : round(confidence, 1),
        "as_of_date"  : latest_date.strftime("%d %b %Y"),
        "sp500_ret"   : round(sp500_ret, 2),
        "nasdaq_ret"  : round(nasdaq_ret, 2),
        "nifty_ret"   : round(nifty_ret, 2),
    }

    if verbose:
        print(f"\n  Data as of   : {result['as_of_date']}")
        print(f"  S&P500       : {result['sp500_ret']:+.2f}%")
        print(f"  Nasdaq       : {result['nasdaq_ret']:+.2f}%")
        print(f"  Nifty (prev) : {result['nifty_ret']:+.2f}%")
        print()
        print("  ┌─────────────────────────────────────────┐")
        print(f"  │  TOMORROW'S NIFTY OPENING: {result['prediction']:<14}│")
        print(f"  │  UP Probability   : {result['up_prob']:>5.1f}%           │")
        print(f"  │  DOWN Probability : {result['down_prob']:>5.1f}%           │")
        print(f"  │  Confidence       : {result['confidence']:>5.1f}%           │")
        print("  └─────────────────────────────────────────┘")
        print()
        print("=" * 55)

    return result


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = predict_tomorrow(verbose=True)