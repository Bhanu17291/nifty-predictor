"""
STEP 3 - FEATURE ENGINEERING
- Loads master_clean.csv
- Creates momentum, volatility, divergence features
- Saves feature-rich dataset ready for model training
"""

import pandas as pd
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = "data"
INPUT    = os.path.join(DATA_DIR, "master_clean.csv")
OUTPUT   = os.path.join(DATA_DIR, "features.csv")


# ── Load ─────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("  Loading master_clean.csv...")
    df = pd.read_csv(INPUT, index_col="date", parse_dates=True)
    print(f"  Shape: {df.shape}")
    return df


# ── Feature Group 1: Momentum ─────────────────────────────────────────────────
def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling returns over 3, 5, 10 day windows."""
    print("  Adding momentum features...")

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        ret_col = f"{market}_ret"
        if ret_col not in df.columns:
            continue
        # Cumulative rolling return over N days
        df[f"{market}_mom_3d"]  = df[ret_col].rolling(3).sum()
        df[f"{market}_mom_5d"]  = df[ret_col].rolling(5).sum()
        df[f"{market}_mom_10d"] = df[ret_col].rolling(10).sum()

    return df


# ── Feature Group 2: Volatility ───────────────────────────────────────────────
def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling standard deviation as a proxy for volatility."""
    print("  Adding volatility features...")

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        ret_col = f"{market}_ret"
        if ret_col not in df.columns:
            continue
        df[f"{market}_vol_5d"]  = df[ret_col].rolling(5).std()
        df[f"{market}_vol_10d"] = df[ret_col].rolling(10).std()

    return df


# ── Feature Group 3: Divergence / Cross-market signals ───────────────────────
def add_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Capture when markets diverge — very informative for opening prediction.
    e.g. SP500 up but Nasdaq down = mixed signal
    """
    print("  Adding divergence features...")

    # SP500 vs Nasdaq divergence
    if "sp500_ret" in df.columns and "nasdaq_ret" in df.columns:
        df["sp500_nasdaq_div"] = df["sp500_ret"] - df["nasdaq_ret"]

    # US average vs Nifty divergence (how much India diverges from US)
    if "sp500_ret" in df.columns and "nasdaq_ret" in df.columns and "nifty_ret" in df.columns:
        df["us_avg_ret"]      = (df["sp500_ret"] + df["nasdaq_ret"]) / 2
        df["us_nifty_div"]    = df["us_avg_ret"] - df["nifty_ret"]

    # GIFT Nifty vs Nifty divergence (basis / premium signal)
    if "gift_nifty_ret" in df.columns and "nifty_ret" in df.columns:
        df["gift_nifty_basis"] = df["gift_nifty_ret"] - df["nifty_ret"]

    return df


# ── Feature Group 4: Lag features ────────────────────────────────────────────
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previous day values as features.
    Model learns from recent history.
    """
    print("  Adding lag features...")

    for market in ["sp500", "nasdaq", "nifty", "gift_nifty"]:
        ret_col = f"{market}_ret"
        if ret_col not in df.columns:
            continue
        df[f"{market}_lag1"] = df[ret_col].shift(1)
        df[f"{market}_lag2"] = df[ret_col].shift(2)

    return df


# ── Feature Group 5: Trend direction flags ───────────────────────────────────
def add_trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary flags: was the market up or down over last N days?
    Helps XGBoost capture directional regime.
    """
    print("  Adding trend flag features...")

    for market in ["sp500", "nasdaq"]:
        ret_col = f"{market}_ret"
        if ret_col not in df.columns:
            continue
        df[f"{market}_up_3d"] = (df[ret_col].rolling(3).sum() > 0).astype(int)
        df[f"{market}_up_5d"] = (df[ret_col].rolling(5).sum() > 0).astype(int)

    return df


# ── Clean up ──────────────────────────────────────────────────────────────────
def finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN (from rolling windows) and reset."""
    before = len(df)
    df = df.dropna()
    after = len(df)
    print(f"\n  Dropped {before - after} rows due to rolling window NaNs")
    print(f"  Final shape: {df.shape}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Step 3: Feature Engineering")
    print("=" * 55 + "\n")

    df = load_data()
    df = add_momentum_features(df)
    df = add_volatility_features(df)
    df = add_divergence_features(df)
    df = add_lag_features(df)
    df = add_trend_flags(df)
    df = finalize(df)

    # Save
    df.to_csv(OUTPUT)
    print(f"\n  💾 Saved → {OUTPUT}")

    # Summary
    feature_cols = [c for c in df.columns if c != "target"]
    print()
    print("=" * 55)
    print(f"  ✅ Feature engineering complete!")
    print(f"  Total features: {len(feature_cols)}")
    print("=" * 55)
    print()
    print("  All features created:")
    for col in feature_cols:
        print(f"    • {col}")
    print()
    print("  Next step → run: python model.py")


if __name__ == "__main__":
    main()