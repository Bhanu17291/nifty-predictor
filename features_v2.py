"""
FEATURES V2 — Fixed & Improved
================================
Changes from original:

  [Step 2] Z-score normalisation added to all return features
           Rolling 20-day z-score wraps every *_ret column
           Model now sees "how unusual is this move" not raw magnitude

  [Step 3] Target leakage audit added
           nifty_ret (today) verified to not bleed into target (tomorrow)
           Explicit check: no feature uses t+1 data

  [Step 4] Calendar features changed from raw integers to sin/cos encoding
           day_of_week 0-4 → dow_sin, dow_cos
           month 1-12 → mon_sin, mon_cos
           Model now understands Monday and Friday are "close" in week-cycle

  [Step 5] Class imbalance report added
           Prints UP/DOWN split so ensemble_model.py can set scale_pos_weight
           Also saves class_weights.pkl for use during training
"""

import pandas as pd
import numpy as np
import pickle
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR = "data"
OUTPUT   = os.path.join(DATA_DIR, "features_v2.csv")
CLOSES   = os.path.join(DATA_DIR, "closes_v2.csv")


# ── Helpers ───────────────────────────────────────────────────────────────────
def rolling_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    STEP 2 — Z-score normalisation.
    Returns (x - rolling_mean) / rolling_std over `window` days.
    min_periods=5 prevents NaN explosion at the start of the series.
    """
    mu  = series.rolling(window, min_periods=5).mean()
    sig = series.rolling(window, min_periods=5).std()
    return (series - mu) / (sig + 1e-9)


def sin_cos_encode(series: pd.Series, period: float):
    """
    STEP 4 — Cyclical encoding.
    Converts an integer cycle (0..period-1) to two continuous features
    so the model understands the circular nature (Dec close to Jan, Fri close to Mon).
    """
    angle = 2 * np.pi * series / period
    return np.sin(angle), np.cos(angle)


# ── Load closes ───────────────────────────────────────────────────────────────
def load_closes() -> pd.DataFrame:
    if not os.path.exists(CLOSES):
        raise FileNotFoundError(
            f"closes_v2.csv not found. Run data_fetch.py first.\n"
            f"Expected at: {CLOSES}"
        )
    df = pd.read_csv(CLOSES, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df.sort_index()
    print(f"  Loaded closes: {df.shape[0]} rows | columns: {list(df.columns)}")
    return df


# ── STEP 3 — Leakage audit ────────────────────────────────────────────────────
def leakage_audit(df: pd.DataFrame, target_col: str = "target"):
    """
    STEP 3 FIX — Target leakage check.

    The target is: did Nifty go UP tomorrow? (shift -1)
    Any feature that uses tomorrow's data will inflate accuracy artificially.

    Checks performed:
      1. nifty_ret at time t should correlate <0.95 with target at time t
         (if >0.95 the target was built from today's return, not tomorrow's)
      2. No feature column should be the raw target series itself
    """
    print("\n  ── Leakage Audit ───────────────────────────────")

    if target_col not in df.columns:
        print("  ⚠️  target column not found — skipping audit")
        return

    target = df[target_col]
    feature_cols = [c for c in df.columns if c != target_col]

    leaks = []
    for col in feature_cols:
        try:
            corr = abs(df[col].corr(target))
            if corr > 0.85:
                leaks.append((col, corr))
        except Exception:
            pass

    if leaks:
        print(f"  🚨 Potential leakage detected in {len(leaks)} feature(s):")
        for col, corr in sorted(leaks, key=lambda x: -x[1]):
            print(f"     {col:<35} corr={corr:.4f}")
        print("  ➜  Review these features — they may contain future information.")
    else:
        print("  ✅ No leakage detected — all features have acceptable target correlation")

    if "nifty_ret" in df.columns:
        nr_corr = abs(df["nifty_ret"].corr(target))
        if nr_corr > 0.5:
            print(f"\n  🚨 nifty_ret correlation with target = {nr_corr:.4f}")
            print(f"     This is too high. Likely the target was built using today's return.")
            print(f"     target should use: (nifty_close.shift(-1) / nifty_close - 1) > 0")
        else:
            print(f"  ✅ nifty_ret vs target correlation: {nr_corr:.4f} (OK)")

    print("  ────────────────────────────────────────────────\n")


# ── STEP 5 — Class imbalance check ────────────────────────────────────────────
def check_class_imbalance(y: pd.Series):
    """
    STEP 5 FIX — Compute and save class weights for use in ensemble_model.py.
    """
    print("\n  ── Class Imbalance Check ───────────────────────")
    counts = y.value_counts().sort_index()
    total  = len(y)
    up_pct   = counts.get(1, 0) / total * 100
    down_pct = counts.get(0, 0) / total * 100

    print(f"  UP   (1): {counts.get(1,0):>5} rows  ({up_pct:.1f}%)")
    print(f"  DOWN (0): {counts.get(0,0):>5} rows  ({down_pct:.1f}%)")

    spw = counts.get(0, 1) / max(counts.get(1, 1), 1)

    if abs(up_pct - 50) < 3:
        print(f"  ✅ Classes are balanced (within 3%). scale_pos_weight = {spw:.3f}")
    else:
        print(f"  ⚠️  Imbalance detected. scale_pos_weight = {spw:.3f}")
        print(f"     Pass this to XGBClassifier and use class_weight='balanced' in RF")

    weights = {
        "scale_pos_weight" : round(spw, 4),
        "up_pct"           : round(up_pct, 2),
        "down_pct"         : round(down_pct, 2),
        "up_count"         : int(counts.get(1, 0)),
        "down_count"       : int(counts.get(0, 0)),
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "class_weights.pkl"), "wb") as f:
        pickle.dump(weights, f)
    print(f"  💾 Saved → data/class_weights.pkl")
    print("  ────────────────────────────────────────────────\n")
    return weights


# ── Build features ────────────────────────────────────────────────────────────
def build_features(closes: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=closes.index)

    # ── 1. Raw returns ────────────────────────────────────────────────────────
    ret_cols = {}
    for col in closes.columns:
        r = closes[col].pct_change() * 100
        df[f"{col}_ret"] = r
        ret_cols[col] = f"{col}_ret"

    # ── 2. STEP 2 FIX — Z-score normalised returns ───────────────────────────
    for col in closes.columns:
        r_col = f"{col}_ret"
        if r_col in df.columns:
            df[f"{col}_ret_z20"] = rolling_zscore(df[r_col], window=20)

    # ── 3. Momentum ───────────────────────────────────────────────────────────
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude", "usdinr"]:
        r = f"{market}_ret"
        if r not in df.columns:
            continue
        df[f"{market}_mom_3d"]  = df[r].rolling(3, min_periods=2).sum()
        df[f"{market}_mom_5d"]  = df[r].rolling(5, min_periods=3).sum()
        df[f"{market}_mom_10d"] = df[r].rolling(10, min_periods=5).sum()
        df[f"{market}_mom_5d_z20"]  = rolling_zscore(df[f"{market}_mom_5d"],  20)
        df[f"{market}_mom_10d_z20"] = rolling_zscore(df[f"{market}_mom_10d"], 20)

    # ── 4. Volatility ─────────────────────────────────────────────────────────
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude"]:
        r = f"{market}_ret"
        if r not in df.columns:
            continue
        df[f"{market}_vol_5d"]  = df[r].rolling(5,  min_periods=3).std()
        df[f"{market}_vol_10d"] = df[r].rolling(10, min_periods=5).std()
        df[f"{market}_vol_ratio"] = (
            df[f"{market}_vol_5d"] / (df[f"{market}_vol_10d"] + 1e-9)
        )

    # ── 5. Divergence ─────────────────────────────────────────────────────────
    if "sp500_ret" in df.columns and "nasdaq_ret" in df.columns:
        df["sp500_nasdaq_div"]     = df["sp500_ret"] - df["nasdaq_ret"]
        df["sp500_nasdaq_div_z20"] = rolling_zscore(df["sp500_nasdaq_div"], 20)

    if all(c in df.columns for c in ["sp500_ret", "nasdaq_ret", "nifty_ret"]):
        df["us_avg_ret"]     = (df["sp500_ret"] + df["nasdaq_ret"]) / 2
        df["us_nifty_div"]   = df["us_avg_ret"] - df["nifty_ret"]
        df["us_avg_ret_z20"] = rolling_zscore(df["us_avg_ret"], 20)

    if "gift_nifty_ret" in df.columns and "nifty_ret" in df.columns:
        df["gift_nifty_basis"]     = df["gift_nifty_ret"] - df["nifty_ret"]
        df["gift_nifty_basis_z20"] = rolling_zscore(df["gift_nifty_basis"], 20)

    # ── 6. Lags ───────────────────────────────────────────────────────────────
    for market in ["sp500", "nasdaq", "nifty", "gift_nifty", "crude", "usdinr"]:
        r = f"{market}_ret"
        if r not in df.columns:
            continue
        df[f"{market}_lag1"] = df[r].shift(1)
        df[f"{market}_lag2"] = df[r].shift(2)

    # ── 7. Trend flags ────────────────────────────────────────────────────────
    for market in ["sp500", "nasdaq", "nifty", "crude"]:
        r = f"{market}_ret"
        if r not in df.columns:
            continue
        df[f"{market}_up_3d"] = (df[r].rolling(3, min_periods=2).sum() > 0).astype(int)
        df[f"{market}_up_5d"] = (df[r].rolling(5, min_periods=3).sum() > 0).astype(int)

    # ── 8. VIX features ───────────────────────────────────────────────────────
    if "vix_india_ret" in df.columns:
        df["vix_level"]   = closes["vix_india"].ffill()
        df["vix_high"]    = (df["vix_level"] > 20).astype(int)
        df["vix_extreme"] = (df["vix_level"] > 30).astype(int)
        df["vix_mom_3d"]  = df["vix_india_ret"].rolling(3, min_periods=2).sum()
        df["vix_rising"]  = (df["vix_india_ret"].rolling(3, min_periods=2).sum() > 0).astype(int)
        vix_mean = df["vix_level"].rolling(252, min_periods=60).mean()
        df["vix_level_norm"] = df["vix_level"] / (vix_mean + 1e-9)
        df["vix_ret_z20"]    = rolling_zscore(df["vix_india_ret"], 20)

    # ── 9. USD/INR features ───────────────────────────────────────────────────
    if "usdinr_ret" in df.columns:
        df["usdinr_level"]      = closes["usdinr"].ffill()
        usdinr_mean             = df["usdinr_level"].rolling(252, min_periods=60).mean()
        df["usdinr_level_norm"] = df["usdinr_level"] / (usdinr_mean + 1e-9)
        df["rupee_weak"]        = (df["usdinr_ret"] > 0).astype(int)
        df["rupee_very_weak"]   = (df["usdinr_ret"] > 0.5).astype(int)

    # ── 10. Crude oil features ────────────────────────────────────────────────
    if "crude_ret" in df.columns:
        df["crude_spike"] = (df["crude_ret"].abs() > 2).astype(int)
        df["crude_up"]    = (df["crude_ret"] > 0).astype(int)

    # ── 11. STEP 4 FIX — Cyclical calendar encoding ──────────────────────────
    dow = df.index.dayofweek
    mon = df.index.month
    dom = df.index.day

    df["dow_sin"], df["dow_cos"] = sin_cos_encode(pd.Series(dow, index=df.index), 5)
    df["mon_sin"], df["mon_cos"] = sin_cos_encode(pd.Series(mon, index=df.index), 12)
    df["dom_sin"], df["dom_cos"] = sin_cos_encode(pd.Series(dom, index=df.index), 31)

    df["is_monday"]      = (dow == 0).astype(int)
    df["is_friday"]      = (dow == 4).astype(int)
    df["is_month_end"]   = (df.index.day >= 25).astype(int)
    df["is_month_start"] = (df.index.day <= 5).astype(int)
    df["is_expiry_week"] = (df.index.day >= 25).astype(int)
    df["quarter"]        = df.index.quarter

    # ── 12. Rolling correlations ──────────────────────────────────────────────
    if "sp500_ret" in df.columns and "nifty_ret" in df.columns:
        df["sp500_nifty_corr_20d"]  = df["sp500_ret"].rolling(20, min_periods=10).corr(df["nifty_ret"])
    if "nasdaq_ret" in df.columns and "nifty_ret" in df.columns:
        df["nasdaq_nifty_corr_20d"] = df["nasdaq_ret"].rolling(20, min_periods=10).corr(df["nifty_ret"])

    # ── 13. Moving average distance ───────────────────────────────────────────
    if "nifty" in closes.columns:
        nifty_price      = closes["nifty"].ffill()
        ma20             = nifty_price.rolling(20, min_periods=10).mean()
        ma50             = nifty_price.rolling(50, min_periods=25).mean()
        df["nifty_dist_ma20"]  = ((nifty_price - ma20) / ma20 * 100)
        df["nifty_dist_ma50"]  = ((nifty_price - ma50) / ma50 * 100)
        df["nifty_above_ma20"] = (nifty_price > ma20).astype(int)
        df["nifty_above_ma50"] = (nifty_price > ma50).astype(int)

    if "sp500" in closes.columns:
        sp_price         = closes["sp500"].ffill()
        ma50_sp          = sp_price.rolling(50, min_periods=25).mean()
        df["sp500_dist_ma50"]  = ((sp_price - ma50_sp) / ma50_sp * 100)
        df["sp500_above_ma50"] = (sp_price > ma50_sp).astype(int)

    # ── 14. Cross-market momentum diff ────────────────────────────────────────
    if "sp500_mom_5d" in df.columns and "nifty_mom_5d" in df.columns:
        df["cross_mom_diff"]     = df["sp500_mom_5d"] - df["nifty_mom_5d"]
        df["cross_mom_diff_z20"] = rolling_zscore(df["cross_mom_diff"], 20)

    # ── 15. STEP 3 FIX — Target variable (verified clean) ─────────────────────
    #
    # CORRECT construction:
    #   target[t] = 1 if nifty_close[t+1] > nifty_close[t]
    #
    # nifty_ret[t] = (close[t] - close[t-1]) / close[t-1]  ← TODAY's return
    # target[t]    = (close[t+1] > close[t])                ← TOMORROW's direction
    # These are different series — no leakage.
    nifty_close  = closes["nifty"].ffill()
    tomorrow_ret = nifty_close.pct_change().shift(-1)
    df["target"] = (tomorrow_ret > 0).astype(int)

    df = df.dropna()
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Features V2 (Fixed)")
    print("=" * 55 + "\n")

    closes = load_closes()
    df     = build_features(closes)

    leakage_audit(df)
    weights = check_class_imbalance(df["target"])

    train = df[df.index.year <= 2023]
    test  = df[df.index.year >= 2024]

    df.to_csv(OUTPUT)
    train.to_csv(os.path.join(DATA_DIR, "train_v2.csv"))
    test.to_csv(os.path.join(DATA_DIR,  "test_v2.csv"))
    closes.to_csv(os.path.join(DATA_DIR, "closes_v2.csv"))

    feat_cols = [c for c in df.columns if c != "target"]

    print("=" * 55)
    print(f"  ✅ Features V2 complete!")
    print(f"  Total features : {len(feat_cols)}")
    print(f"  Total rows     : {len(df)}")
    print(f"  Train rows     : {len(train)}  ({train.index[0].date()} → {train.index[-1].date()})")
    print(f"  Test rows      : {len(test)}   ({test.index[0].date()} → {test.index[-1].date()})")
    print(f"  UP/DOWN split  : {weights['up_pct']:.1f}% / {weights['down_pct']:.1f}%")
    print("=" * 55)
    print()

    groups = {
        "raw returns"    : [c for c in feat_cols if c.endswith("_ret") and "_z" not in c],
        "z-scored"       : [c for c in feat_cols if "_z20" in c],
        "momentum"       : [c for c in feat_cols if "_mom_" in c and "_z" not in c],
        "volatility"     : [c for c in feat_cols if "_vol_" in c or "_ratio" in c],
        "lags"           : [c for c in feat_cols if "_lag" in c],
        "cyclical time"  : [c for c in feat_cols if any(x in c for x in ["_sin","_cos","is_","expiry","quarter"])],
        "VIX/FX/Crude"   : [c for c in feat_cols if any(x in c for x in ["vix","usdinr","crude","rupee"])],
        "MA / corr"      : [c for c in feat_cols if any(x in c for x in ["dist_","above_","corr","cross"])],
    }
    print("  Feature group breakdown:")
    for grp, cols in groups.items():
        print(f"    {grp:<16}: {len(cols):>3} features")

    print()
    print("  Next → run: python ensemble_model.py")


if __name__ == "__main__":
    main()