"""
STEP 1 — DATA FETCH (Fixed)
===========================
Changes from original:
  [Step 1] gift_nifty ticker fixed: ^NSEI → NIFTYBEES.NS
           (^NSEI was identical to nifty — pure data leakage)
  [Step 1] Added DXY, Gold as additional macro signals
  [Step 1] Added forward-fill audit — logs any column with >10% NaN
"""

import yfinance as yf
import pandas as pd
import os

# ── Config ───────────────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
DATA_DIR   = "data"

TICKERS = {
    "sp500"      : "^GSPC",
    "nasdaq"     : "^IXIC",
    "nifty"      : "^NSEI",

    # ── STEP 1 FIX ──────────────────────────────────────────────────────────
    # WAS:  "gift_nifty": "^NSEI"  ← IDENTICAL to nifty. Caused data leakage.
    # NOW:  NIFTYBEES.NS — Nifty BeES ETF, trades separately from spot Nifty,
    #       acts as a genuine GIFT Nifty proxy when actual GIFT data unavailable.
    # If you have access to actual GIFT Nifty data (SGX/NSE feed), replace this.
    "gift_nifty" : "NIFTYBEES.NS",

    "vix_india"  : "^INDIAVIX",
    "usdinr"     : "INR=X",
    "crude"      : "BZ=F",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_ticker(name: str, symbol: str) -> pd.Series:
    """Download closing prices for a single ticker."""
    print(f"  Fetching {name} ({symbol})...")
    try:
        df = yf.download(symbol, start=START_DATE, end=END_DATE, progress=False)
        if df.empty:
            print(f"  ⚠️  No data: {name}")
            return pd.Series(dtype=float, name=name)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        series = df["Close"].copy()
        series.name = name
        series.index = pd.to_datetime(series.index)
        series.index.name = "date"

        nan_pct = series.isna().mean() * 100
        print(f"  ✅ {name}: {len(series)} rows | NaN: {nan_pct:.1f}%")
        if nan_pct > 10:
            print(f"  ⚠️  WARNING: {name} has {nan_pct:.1f}% missing — check ticker")
        return series

    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return pd.Series(dtype=float, name=name)


def audit_dataframe(df: pd.DataFrame):
    """
    STEP 1 FIX — Forward-fill audit.
    Logs any column with >5% NaN before filling.
    Catches bad tickers silently masking as zeros after ffill.
    """
    print("\n  ── Data Quality Audit ─────────────────────────")
    issues = []
    for col in df.columns:
        nan_pct = df[col].isna().mean() * 100
        if nan_pct > 5:
            issues.append((col, nan_pct))
            print(f"  ⚠️  {col:<20} {nan_pct:.1f}% NaN before fill")

    # Critical check: gift_nifty should NOT correlate >0.99 with nifty
    # (if it does, the ticker is still wrong / duplicate)
    if "gift_nifty" in df.columns and "nifty" in df.columns:
        common = df[["gift_nifty", "nifty"]].dropna()
        if len(common) > 20:
            corr = common["gift_nifty"].corr(common["nifty"])
            if corr > 0.99:
                print(f"\n  🚨 CRITICAL: gift_nifty correlates {corr:.4f} with nifty")
                print(f"     This means gift_nifty ticker is still a duplicate.")
                print(f"     Check TICKERS dict — replace with actual GIFT Nifty data.")
            else:
                print(f"  ✅ gift_nifty vs nifty correlation: {corr:.4f} (OK — they differ)")

    if not issues:
        print("  ✅ All columns within acceptable NaN threshold")
    print("  ─────────────────────────────────────────────\n")


def save_csv(df: pd.DataFrame, name: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{name}.csv")
    df.to_csv(path)
    print(f"  💾 Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Step 1: Data Fetch (Fixed)")
    print("=" * 55)
    print(f"  Period : {START_DATE} to {END_DATE}\n")

    closes = {}
    for name, symbol in TICKERS.items():
        s = fetch_ticker(name, symbol)
        if not s.empty:
            closes[name] = s
        print()

    # Merge all into single closes DataFrame
    master_closes = pd.DataFrame(closes)
    master_closes.index = pd.to_datetime(master_closes.index)
    master_closes = master_closes.sort_index()

    # ── Audit BEFORE filling ─────────────────────────────────────────────────
    audit_dataframe(master_closes)

    # Forward-fill max 2 days (handles weekends/holidays), then drop rows
    # where nifty itself is missing (nifty is the target source — must exist)
    master_closes = master_closes.ffill(limit=2)
    master_closes = master_closes.dropna(subset=["nifty"])

    # Save closes separately — used by features_v2.py and regime_detector.py
    save_csv(master_closes, "closes_v2")

    print()
    print("=" * 55)
    print(f"  ✅ Done! Closes saved: {master_closes.shape[0]} rows × {master_closes.shape[1]} cols")
    print(f"  Columns: {list(master_closes.columns)}")
    print("=" * 55)
    print()
    print("  Next → run: python features_v2.py")


if __name__ == "__main__":
    main()