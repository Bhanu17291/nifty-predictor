"""
EXPLAINER — SHAP-based prediction explanations
For every prediction, shows WHY the model predicted UP or DOWN
"""

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR  = "data"
MODEL_DIR = "models"

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("  ⚠️  SHAP not installed. Run: pip install shap")


def load_models():
    models = {}
    for name in ["xgb", "lgbm", "rf"]:
        path = os.path.join(MODEL_DIR, f"{name}_model_v2.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
    return models


def get_shap_values(model, X: pd.DataFrame, model_type: str = "xgb"):
    """Get SHAP values for XGBoost or LightGBM."""
    if not SHAP_AVAILABLE:
        return None
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        # For binary classification, shap_values is array or list
        if isinstance(shap_vals, list):
            return shap_vals[1]  # class 1 (UP)
        return shap_vals
    except Exception as e:
        print(f"  SHAP error ({model_type}): {e}")
        return None


def explain_prediction(X_row: pd.DataFrame,
                       models: dict,
                       top_n: int = 8) -> dict:
    """
    Generate top N SHAP-based reasons for a single prediction row.
    Returns dict with reasons list and chart path.
    """
    if not SHAP_AVAILABLE or not models:
        return {"available": False, "reasons": []}

    # Use XGBoost for explanation (most interpretable)
    model = models.get("xgb")
    if model is None:
        return {"available": False, "reasons": []}

    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_row)

        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]

        feat_names = X_row.columns.tolist()
        shap_df = pd.DataFrame({
            "feature"   : feat_names,
            "shap_value": sv,
            "abs_shap"  : np.abs(sv),
            "feat_value": X_row.iloc[0].values,
        }).sort_values("abs_shap", ascending=False).head(top_n)

        reasons = []
        for _, row in shap_df.iterrows():
            direction = "UP ▲" if row["shap_value"] > 0 else "DOWN ▼"
            strength  = abs(row["shap_value"])
            reasons.append({
                "feature"   : row["feature"],
                "direction" : direction,
                "strength"  : round(strength, 4),
                "shap"      : round(row["shap_value"], 4),
                "value"     : round(float(row["feat_value"]), 4),
            })

        return {"available": True, "reasons": reasons, "shap_df": shap_df}

    except Exception as e:
        return {"available": False, "reasons": [], "error": str(e)}


def plot_shap_waterfall(explanation: dict,
                        prediction: str = "UP",
                        save_path: str = None) -> str:
    """Generate a clean waterfall chart for SHAP values."""
    if not explanation.get("available"):
        return None

    shap_df  = explanation["shap_df"]
    save_path = save_path or os.path.join(MODEL_DIR, "shap_waterfall.png")

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    ax.set_facecolor("white")

    colors = ["#15803d" if v > 0 else "#b91c1c"
              for v in shap_df["shap_value"]]

    bars = ax.barh(
        range(len(shap_df)),
        shap_df["shap_value"].values[::-1],
        color=colors[::-1],
        edgecolor="none",
        height=0.6,
    )

    labels = [f"{f}  ({v:+.3f})" for f, v in
              zip(shap_df["feature"].values[::-1],
                  shap_df["feat_value"].values[::-1])]
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(labels, fontsize=8, color="#475569")

    ax.axvline(0, color="#e2e8f0", linewidth=1.5)
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=8, color="#64748b")
    ax.set_title(f"Why the model predicted {prediction} — Top {len(shap_df)} drivers",
                 fontsize=11, color="#0f172a", pad=12, fontweight="bold")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#e2e8f0")
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(colors="#94a3b8")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return save_path


def build_test_explanation():
    """Build explanation on latest test row for dashboard preview."""
    feat_path = os.path.join(DATA_DIR, "features_v2.csv")
    if not os.path.exists(feat_path):
        return None

    df   = pd.read_csv(feat_path, index_col="Date", parse_dates=True)
    test = df[df.index.year >= 2023]
    feat = [c for c in df.columns if c != "target"]

    X_latest = test[feat].tail(1)
    models   = load_models()

    if not models:
        return None

    explanation = explain_prediction(X_latest, models)
    pred = "UP" if models["xgb"].predict(X_latest)[0] == 1 else "DOWN"
    plot_shap_waterfall(explanation, pred)
    return explanation


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — SHAP Explainer")
    print("=" * 55 + "\n")

    if not SHAP_AVAILABLE:
        print("  ❌ SHAP not installed!")
        print("  Run: pip install shap")
        print("  Then re-run this script.")
        return

    explanation = build_test_explanation()

    if explanation and explanation.get("available"):
        print("  Top prediction drivers (latest test row):\n")
        for i, r in enumerate(explanation["reasons"], 1):
            arrow = "▲" if r["shap"] > 0 else "▼"
            color = "UP  " if r["shap"] > 0 else "DOWN"
            print(f"  {i:2}. {r['feature']:<30} {arrow} {color}  strength={r['strength']:.4f}")
        print(f"\n  💾 SHAP chart → models/shap_waterfall.png")
    else:
        print("  ⚠️  Could not generate SHAP explanation.")

    print()
    print("=" * 55)
    print("  ✅ Explainer ready!")
    print("=" * 55)
    print()
    print("  Next → run: python live_predict_v2.py")


if __name__ == "__main__":
    main()