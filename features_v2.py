"""
FEATURES V2 — Expanded Feature Engineering
40 → 65+ features:
  + India VIX, USD/INR, Brent Crude
  + Calendar features (day, month, flags)
  + Rolling correlations
  + Moving average distances
  + Cross-market momentum
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
OUTPUT    = os.path.join(DATA_DIR, "features_v2.csv")
START     = "2019-01-01"
END       = "2024-12-31"

TICKERS = {
    "sp500"      : "^GSPC",
    "nasdaq"     : "^IXIC",
    "nifty"      : "^NSEI",
    "gift_nifty" : "^NSEI",
    "vix_india"  : "^INDIAVIX",
    "usdinr"     : "INR=X",
    "crude"      : "BZ=F",
}


# ── Fetch ─────────────────────────────────────────────────────────────────────
def fetch_all() -> pd.DataFrame:
    print("  Fetching all market data (this may take ~60s)...")
    closes = {}
    for name, sym in TICKERS.items():
        try:
            df = yf.download(sym, start=START, end=END, progress=False)
            if df.empty:
                print(f"  ⚠️  No data: {name}")
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            closes[name] = df["Close"]
            print(f"  ✅ {name}: {len(df)} rows")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    df = pd.DataFrame(closes)
    df.index = pd.to_datetime(df.index)
    df = df.ffill(limit=2).dropna(subset=["nifty"])
    return df


# ── Build features ────────────────────────────────────────────────────────────
def build_features(closes: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=closes.index)

    # ── 1. Returns
    for col in closes.columns:
        df[f"{col}_ret"] = closes[col].pct_change() * 100

    # ── 2. Momentum (3d, 5d, 10d)
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude", "usdinr"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_mom_3d"]  = df[r].rolling(3).sum()
        df[f"{market}_mom_5d"]  = df[r].rolling(5).sum()
        df[f"{market}_mom_10d"] = df[r].rolling(10).sum()

    # ── 3. Volatility
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_vol_5d"]  = df[r].rolling(5).std()
        df[f"{market}_vol_10d"] = df[r].rolling(10).std()

    # ── 4. Divergence
    if "sp500_ret" in df and "nasdaq_ret" in df:
        df["sp500_nasdaq_div"] = df["sp500_ret"] - df["nasdaq_ret"]
    if "sp500_ret" in df and "nasdaq_ret" in df and "nifty_ret" in df:
        df["us_avg_ret"]   = (df["sp500_ret"] + df["nasdaq_ret"]) / 2
        df["us_nifty_div"] = df["us_avg_ret"] - df["nifty_ret"]
    if "gift_nifty_ret" in df and "nifty_ret" in df:
        df["gift_nifty_basis"] = df["gift_nifty_ret"] - df["nifty_ret"]

    # ── 5. Lags
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude", "usdinr"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_lag1"] = df[r].shift(1)
        df[f"{market}_lag2"] = df[r].shift(2)

    # ── 6. Trend flags
    for market in ["sp500", "nasdaq", "nifty", "crude"]:
        r = f"{market}_ret"
        if r not in df: continue
        df[f"{market}_up_3d"] = (df[r].rolling(3).sum() > 0).astype(int)
        df[f"{market}_up_5d"] = (df[r].rolling(5).sum() > 0).astype(int)

    # ── 7. VIX features
    if "vix_india_ret" in df:
        df["vix_level"]    = closes["vix_india"].ffill()
        df["vix_high"]     = (df["vix_level"] > 20).astype(int)
        df["vix_extreme"]  = (df["vix_level"] > 30).astype(int)
        df["vix_mom_3d"]   = df["vix_india_ret"].rolling(3).sum()
        df["vix_rising"]   = (df["vix_india_ret"].rolling(3).sum() > 0).astype(int)

    # ── 8. USD/INR features
    if "usdinr_ret" in df:
        df["usdinr_level"]    = closes["usdinr"].ffill()
        df["rupee_weak"]      = (df["usdinr_ret"] > 0).astype(int)
        df["rupee_very_weak"] = (df["usdinr_ret"] > 0.5).astype(int)

    # ── 9. Crude oil features
    if "crude_ret" in df:
        df["crude_spike"]   = (df["crude_ret"].abs() > 2).astype(int)
        df["crude_up"]      = (df["crude_ret"] > 0).astype(int)

    # ── 10. Calendar features
    df["day_of_week"]   = df.index.dayofweek          # 0=Mon, 4=Fri
    df["month"]         = df.index.month
    df["is_monday"]     = (df.index.dayofweek == 0).astype(int)
    df["is_friday"]     = (df.index.dayofweek == 4).astype(int)
    df["is_month_end"]  = (df.index.day >= 25).astype(int)
    df["is_month_start"]= (df.index.day <= 5).astype(int)
    df["quarter"]       = df.index.quarter

    # ── 11. Rolling correlations (SP500 & Nasdaq vs Nifty)
    if "sp500_ret" in df and "nifty_ret" in df:
        df["sp500_nifty_corr_20d"] = (
            df["sp500_ret"].rolling(20)
            .corr(df["nifty_ret"])
        )
    if "nasdaq_ret" in df and "nifty_ret" in df:
        df["nasdaq_nifty_corr_20d"] = (
            df["nasdaq_ret"].rolling(20)
            .corr(df["nifty_ret"])
        )

    # ── 12. Moving average distance
    if "nifty" in closes.columns:
        nifty_price = closes["nifty"].ffill()
        ma20  = nifty_price.rolling(20).mean()
        ma50  = nifty_price.rolling(50).mean()
        df["nifty_dist_ma20"] = ((nifty_price - ma20) / ma20 * 100)
        df["nifty_dist_ma50"] = ((nifty_price - ma50) / ma50 * 100)
        df["nifty_above_ma20"] = (nifty_price > ma20).astype(int)
        df["nifty_above_ma50"] = (nifty_price > ma50).astype(int)

    if "sp500" in closes.columns:
        sp_price = closes["sp500"].ffill()
        ma50_sp  = sp_price.rolling(50).mean()
        df["sp500_dist_ma50"]   = ((sp_price - ma50_sp) / ma50_sp * 100)
        df["sp500_above_ma50"]  = (sp_price > ma50_sp).astype(int)

    # ── 13. Cross-market momentum diff
    if "sp500_mom_5d" in df and "nifty_mom_5d" in df:
        df["cross_mom_diff"] = df["sp500_mom_5d"] - df["nifty_mom_5d"]

    # ── 14. Target variable
    nifty_ret = closes["nifty"].pct_change()
    df["target"] = (nifty_ret.shift(-1) > 0).astype(int)

    df = df.dropna()
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Features V2")
    print("=" * 55 + "\n")

    closes  = fetch_all()
    df      = build_features(closes)

    # Save closes for regime detector
    closes.to_csv(os.path.join(DATA_DIR, "closes_v2.csv"))

    # Train/test split
    train = df[df.index.year <= 2022]
    test  = df[df.index.year >= 2023]

    df.to_csv(OUTPUT)
    train.to_csv(os.path.join(DATA_DIR, "train_v2.csv"))
    test.to_csv(os.path.join(DATA_DIR,  "test_v2.csv"))

    feat_cols = [c for c in df.columns if c != "target"]
    print()
    print("=" * 55)
    print(f"  ✅ Features V2 complete!")
    print(f"  Total features : {len(feat_cols)}")
    print(f"  Total rows     : {len(df)}")
    print(f"  Train rows     : {len(train)}")
    print(f"  Test rows      : {len(test)}")
    print("=" * 55)
    print()
    for c in feat_cols:
        print(f"    • {c}")
    print()
    print("  Next → run: python ensemble_model.py")


if __name__ == "__main__":
    main()