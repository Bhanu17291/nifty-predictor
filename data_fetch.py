"""
STEP 1 - DATA FETCH
Fetches historical daily data for:
- S&P 500 (^GSPC)
- Nasdaq (^IXIC)
- Nifty50 (^NSEI)
- GIFT Nifty proxy (using Nifty Futures: NF=F or fallback to ^NSEI)
"""

import yfinance as yf
import pandas as pd
import os

# ── Config ──────────────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
DATA_DIR   = "data"

TICKERS = {
    "sp500"      : "^GSPC",
    "nasdaq"     : "^IXIC",
    "nifty"      : "^NSEI",
    "gift_nifty" : "^NSEI",   # Best free proxy; replace with actual GIFT Nifty data if available
}

# ── Helpers ──────────────────────────────────────────────────────────────────
def fetch_ticker(name: str, symbol: str) -> pd.DataFrame:
    """Download OHLCV data for a single ticker and return a clean DataFrame."""
    print(f"  Fetching {name} ({symbol})...")
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)

        if df.empty:
            print(f"  ⚠️  No data returned for {name}. Check the ticker symbol.")
            return pd.DataFrame()

        # Flatten multi-level columns if present (yfinance sometimes returns them)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Keep only OHLCV columns
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Rename columns to include ticker prefix for clarity after merging
        df.columns = [f"{name}_{col.lower()}" for col in df.columns]

        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

        print(f"  ✅ {name}: {len(df)} rows fetched ({df.index.min().date()} → {df.index.max().date()})")
        return df

    except Exception as e:
        print(f"  ❌ Error fetching {name}: {e}")
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, name: str):
    """Save a DataFrame to the data/ directory as CSV."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path)
    print(f"  💾 Saved → {path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Step 1: Data Fetch")
    print("=" * 55)
    print(f"  Period : {START_DATE} to {END_DATE}\n")

    all_data = {}

    for name, symbol in TICKERS.items():
        df = fetch_ticker(name, symbol)
        if not df.empty:
            save_csv(df, name)
            all_data[name] = df
        print()

    # ── Merge all into one master CSV ────────────────────────────────────────
    print("  Merging all data into master dataset...")
    master = pd.concat(all_data.values(), axis=1)

    # Only keep rows where ALL markets have data (inner join on dates)
    master.dropna(how="all", inplace=True)

    save_csv(master, "master_raw")

    print()
    print("=" * 55)
    print(f"  ✅ Done! Master dataset: {master.shape[0]} rows × {master.shape[1]} cols")
    print(f"  📁 All files saved in '{DATA_DIR}/' folder")
    print("=" * 55)
    print()
    print("  Next step → run: python data_preprocess.py")


if __name__ == "__main__":
    main()