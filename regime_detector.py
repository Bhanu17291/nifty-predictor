"""
REGIME DETECTOR
Detects current market regime: BULL / BEAR / FLAT
Adjusts ensemble weights per regime for better accuracy
"""

import pandas as pd
import numpy as np
import pickle
import os

DATA_DIR  = "data"
MODEL_DIR = "models"

# Regime-specific ensemble weights
REGIME_WEIGHTS = {
    "BULL" : {"xgb": 0.45, "lgbm": 0.40, "rf": 0.15},  # trust momentum more
    "BEAR" : {"xgb": 0.30, "lgbm": 0.35, "rf": 0.35},  # trust volatility more
    "FLAT" : {"xgb": 0.40, "lgbm": 0.40, "rf": 0.20},  # balanced
}

REGIME_COLORS = {
    "BULL": "#15803d",
    "BEAR": "#b91c1c",
    "FLAT": "#1e3a8a",
}

REGIME_EMOJI = {
    "BULL": "🟢",
    "BEAR": "🔴",
    "FLAT": "🔵",
}


def detect_regime(closes_df: pd.DataFrame,
                  row_idx: int = -1) -> dict:
    """
    Detect market regime from closing price data.

    Rules:
        BULL → Nifty above 50d MA AND 20d MA above 50d MA
        BEAR → Nifty below 50d MA AND 20d MA below 50d MA
        FLAT → everything else
    """
    nifty = closes_df["nifty"].ffill()

    ma20 = nifty.rolling(20).mean()
    ma50 = nifty.rolling(50).mean()

    price  = float(nifty.iloc[row_idx])
    m20    = float(ma20.iloc[row_idx])
    m50    = float(ma50.iloc[row_idx])

    above_ma50 = price > m50
    ma20_above = m20 > m50

    # VIX check if available
    vix_ok = True
    vix_val = None
    if "vix_india" in closes_df.columns:
        vix_series = closes_df["vix_india"].ffill()
        vix_val    = float(vix_series.iloc[row_idx])
        if vix_val > 25:
            vix_ok = False

    if above_ma50 and ma20_above and vix_ok:
        regime = "BULL"
    elif not above_ma50 and not ma20_above:
        regime = "BEAR"
    else:
        regime = "FLAT"

    return {
        "regime"      : regime,
        "emoji"       : REGIME_EMOJI[regime],
        "color"       : REGIME_COLORS[regime],
        "weights"     : REGIME_WEIGHTS[regime],
        "nifty_price" : round(price, 2),
        "ma20"        : round(m20, 2),
        "ma50"        : round(m50, 2),
        "above_ma50"  : above_ma50,
        "vix"         : round(vix_val, 2) if vix_val else None,
        "description" : _describe(regime, above_ma50, m20, m50, vix_val),
    }


def _describe(regime, above_ma50, m20, m50, vix):
    if regime == "BULL":
        return "Market trending UP — momentum features weighted higher"
    elif regime == "BEAR":
        return "Market trending DOWN — volatility features weighted higher"
    else:
        return "Market in consolidation — balanced model weights"


def detect_from_file(row_idx: int = -1) -> dict:
    """Load saved closes and detect regime."""
    path = os.path.join(DATA_DIR, "closes_v2.csv")
    if not os.path.exists(path):
        return {"regime": "FLAT", "emoji": "🔵",
                "color": "#1e3a8a",
                "weights": REGIME_WEIGHTS["FLAT"],
                "description": "No regime data — using balanced weights"}
    closes = pd.read_csv(path, index_col=0, parse_dates=True)
    return detect_regime(closes, row_idx)


def get_regime_history(closes_df: pd.DataFrame) -> pd.Series:
    """Return regime label for every row in closes_df."""
    nifty = closes_df["nifty"].ffill()
    ma20  = nifty.rolling(20).mean()
    ma50  = nifty.rolling(50).mean()

    regimes = []
    for i in range(len(closes_df)):
        try:
            r = detect_regime(closes_df, i)
            regimes.append(r["regime"])
        except:
            regimes.append("FLAT")

    return pd.Series(regimes, index=closes_df.index, name="regime")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Regime Detector")
    print("=" * 55 + "\n")

    result = detect_from_file()

    print(f"  Current Regime   : {result['emoji']} {result['regime']}")
    print(f"  Description      : {result['description']}")
    print(f"  Nifty Price      : {result.get('nifty_price', 'N/A')}")
    print(f"  20d MA           : {result.get('ma20', 'N/A')}")
    print(f"  50d MA           : {result.get('ma50', 'N/A')}")
    if result.get("vix"):
        print(f"  India VIX        : {result['vix']}")
    print(f"\n  Ensemble weights for {result['regime']} regime:")
    for model, w in result["weights"].items():
        print(f"    {model.upper():<8}: {w*100:.0f}%")

    print()
    print("=" * 55)
    print("  ✅ Regime detection complete!")
    print("=" * 55)
    print()
    print("  Next → run: python explainer.py")


if __name__ == "__main__":
    main()