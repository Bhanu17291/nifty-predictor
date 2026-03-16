"""
STEP 2 - DATA PREPROCESSING
- Loads raw CSVs
- Aligns US closing data to next India trading day
- Computes overnight returns (avoids data leakage)
- Handles missing values and holiday mismatches
- Saves clean dataset ready for feature engineering
"""

import pandas as pd
import numpy as np
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR  = "data"
INPUT     = os.path.join(DATA_DIR, "master_raw.csv")
OUTPUT    = os.path.join(DATA_DIR, "master_clean.csv")


# ── Load ─────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    print("  Loading master_raw.csv...")
    df = pd.read_csv(INPUT, index_col="date", parse_dates=True)
    print(f"  Raw shape: {df.shape}")
    return df


# ── Step 2a: Separate each market ────────────────────────────────────────────
def extract_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only closing prices for each market — that's what matters."""
    close_cols = {
        "sp500_close"      : "sp500",
        "nasdaq_close"     : "nasdaq",
        "nifty_close"      : "nifty",
        "gift_nifty_close" : "gift_nifty",
    }
    closes = df[[col for col in close_cols.keys() if col in df.columns]].copy()
    closes.columns = [close_cols[c] for c in closes.columns]
    print(f"  Extracted close prices: {closes.columns.tolist()}")
    return closes


# ── Step 2b: Handle missing values ───────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    US and India have different holidays.
    Forward fill max 1 day (e.g. US holiday but India trades).
    Drop rows where ALL markets are missing.
    """
    print(f"  Missing values before:\n{df.isnull().sum()}")

    # Forward fill max 1 day for holiday mismatches
    df = df.ffill(limit=1)

    # Drop rows where Nifty is missing (India didn't trade — useless for us)
    df = df.dropna(subset=["nifty"])

    print(f"\n  Missing values after:\n{df.isnull().sum()}")
    print(f"\n  Shape after cleaning: {df.shape}")
    return df


# ── Step 2c: Compute overnight returns ───────────────────────────────────────
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily returns for each market.
    
    IMPORTANT — Alignment logic:
    - US markets close AFTER India (at ~2:30 AM IST next day)
    - So today's S&P500/Nasdaq close = signal for TODAY's Nifty open
    - We shift US returns by 0 (same calendar day, different actual time)
    - Nifty return = what we want to predict (next day open behavior)
    """
    returns = pd.DataFrame(index=df.index)

    for col in df.columns:
        returns[f"{col}_ret"] = df[col].pct_change() * 100  # in percentage

    # Drop first row (NaN returns)
    returns = returns.dropna()

    print(f"\n  Returns computed for: {returns.columns.tolist()}")
    print(f"  Shape: {returns.shape}")
    return returns


# ── Step 2d: Create target variable ──────────────────────────────────────────
def create_target(df_returns: pd.DataFrame, df_close: pd.DataFrame) -> pd.DataFrame:
    """
    Target = direction of Nifty NEXT DAY opening move.
    
    Since we don't have free intraday 9:00-9:15 data,
    we use next day's open vs today's close as proxy.
    
    Label:
        1 = Nifty opens UP next day (bullish open)
        0 = Nifty opens DOWN or flat next day (bearish/flat open)
    """
    # Use nifty close as proxy for open (best free data available)
    nifty_close = df_close["nifty"].reindex(df_returns.index)

    # Next day return = shift by -1
    df_returns["target"] = (nifty_close.pct_change().shift(-1) > 0).astype(int)

    # Drop last row (no next day available)
    df_returns = df_returns.dropna(subset=["target"])

    up_days   = df_returns["target"].sum()
    down_days = len(df_returns) - up_days
    print(f"\n  Target distribution:")
    print(f"    UP   (1): {up_days}  ({up_days/len(df_returns)*100:.1f}%)")
    print(f"    DOWN (0): {down_days} ({down_days/len(df_returns)*100:.1f}%)")

    return df_returns


# ── Step 2e: Train / Test split ───────────────────────────────────────────────
def split_and_save(df: pd.DataFrame):
    """
    Walk-forward split (NO random shuffle — preserves time order):
        Train : 2019–2022
        Test  : 2023–2024
    """
    train = df[df.index.year <= 2022]
    test  = df[df.index.year >= 2023]

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path  = os.path.join(DATA_DIR, "test.csv")
    clean_path = OUTPUT

    df.to_csv(clean_path)
    train.to_csv(train_path)
    test.to_csv(test_path)

    print(f"\n  Train set : {train.shape[0]} rows → {train_path}")
    print(f"  Test set  : {test.shape[0]} rows  → {test_path}")
    print(f"  Full clean: {df.shape[0]} rows  → {clean_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Step 2: Preprocessing")
    print("=" * 55 + "\n")

    df_raw     = load_data()
    df_close   = extract_close_prices(df_raw)
    df_clean   = handle_missing(df_close)
    df_returns = compute_returns(df_clean)
    df_final   = create_target(df_returns, df_clean)

    split_and_save(df_final)

    print()
    print("=" * 55)
    print("  ✅ Preprocessing complete!")
    print(f"  Final dataset: {df_final.shape[0]} rows × {df_final.shape[1]} cols")
    print("=" * 55)
    print()
    print("  Columns in final dataset:")
    for col in df_final.columns:
        print(f"    • {col}")
    print()
    print("  Next step → run: python features.py")


if __name__ == "__main__":
    main()