"""
STEP 4 - MODEL TRAINING
- Loads features.csv
- Trains XGBoost classifier
- Evaluates on test set
- Plots feature importance
- Saves trained model
"""

import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from xgboost import XGBClassifier

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
INPUT      = os.path.join(DATA_DIR, "features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Load & Split ──────────────────────────────────────────────────────────────
def load_and_split():
    print("  Loading features.csv...")
    df = pd.read_csv(INPUT, index_col="date", parse_dates=True)

    # Walk-forward split — NO random shuffle
    train = df[df.index.year <= 2022]
    test  = df[df.index.year >= 2023]

    feature_cols = [c for c in df.columns if c != "target"]

    X_train = train[feature_cols]
    y_train = train["target"]
    X_test  = test[feature_cols]
    y_test  = test["target"]

    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"  Features: {len(feature_cols)}")
    return X_train, y_train, X_test, y_test, feature_cols


# ── Train ─────────────────────────────────────────────────────────────────────
def train_model(X_train, y_train):
    print("\n  Training XGBoost model...")

    model = XGBClassifier(
        n_estimators      = 200,
        max_depth         = 4,
        learning_rate     = 0.05,
        subsample         = 0.8,
        colsample_bytree  = 0.8,
        min_child_weight  = 5,
        gamma             = 0.1,
        random_state      = 42,
        eval_metric       = "logloss",
        verbosity         = 0,
    )

    model.fit(X_train, y_train)
    print("  ✅ Model trained!")
    return model


# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, X_train, y_train, X_test, y_test):
    print("\n  Evaluating model...")

    # Predictions
    train_preds = model.predict(X_train)
    test_preds  = model.predict(X_test)
    test_proba  = model.predict_proba(X_test)[:, 1]

    # Metrics
    train_acc = accuracy_score(y_train, train_preds)
    test_acc  = accuracy_score(y_test, test_preds)
    test_f1   = f1_score(y_test, test_preds)

    print()
    print("  ┌─────────────────────────────────────┐")
    print(f"  │  Train Accuracy : {train_acc*100:.1f}%              │")
    print(f"  │  Test Accuracy  : {test_acc*100:.1f}%              │")
    print(f"  │  Test F1 Score  : {test_f1:.3f}                │")
    print("  └─────────────────────────────────────┘")

    print("\n  Classification Report (Test Set):")
    print(classification_report(y_test, test_preds,
                                target_names=["DOWN", "UP"]))

    return test_preds, test_proba, test_acc


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, test_preds):
    cm = confusion_matrix(y_test, test_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["DOWN", "UP"],
                yticklabels=["DOWN", "UP"])
    plt.title("Confusion Matrix — Test Set", fontsize=14, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()
    print("  💾 Saved → models/confusion_matrix.png")


# ── Feature Importance Plot ───────────────────────────────────────────────────
def plot_feature_importance(model, feature_cols):
    importance = pd.Series(model.feature_importances_, index=feature_cols)
    importance = importance.sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    colors = ["#2196F3" if i < 5 else "#90CAF9" for i in range(len(importance))]
    importance.plot(kind="barh", color=colors[::-1])
    plt.title("Top 20 Feature Importances", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"), dpi=150)
    plt.close()
    print("  💾 Saved → models/feature_importance.png")

    print("\n  Top 10 Most Important Features:")
    for i, (feat, score) in enumerate(importance.head(10).items(), 1):
        bar = "█" * int(score * 300)
        print(f"    {i:2}. {feat:<25} {bar} {score:.4f}")


# ── Prediction curve ──────────────────────────────────────────────────────────
def plot_prediction_curve(y_test, test_proba):
    plt.figure(figsize=(14, 5))
    plt.plot(range(len(test_proba)), test_proba, color="#2196F3",
             alpha=0.7, linewidth=1, label="Predicted UP probability")
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Decision boundary (0.5)")
    plt.fill_between(range(len(test_proba)), test_proba, 0.5,
                     where=[p > 0.5 for p in test_proba],
                     alpha=0.2, color="green", label="Predicted UP")
    plt.fill_between(range(len(test_proba)), test_proba, 0.5,
                     where=[p <= 0.5 for p in test_proba],
                     alpha=0.2, color="red", label="Predicted DOWN")
    plt.title("Model Predictions on Test Set (2023–2024)", fontsize=14, fontweight="bold")
    plt.xlabel("Trading Days")
    plt.ylabel("P(Nifty UP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "prediction_curve.png"), dpi=150)
    plt.close()
    print("  💾 Saved → models/prediction_curve.png")


# ── Save Model ────────────────────────────────────────────────────────────────
def save_model(model):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"  💾 Model saved → {MODEL_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Step 4: Model Training")
    print("=" * 55 + "\n")

    X_train, y_train, X_test, y_test, feature_cols = load_and_split()
    model = train_model(X_train, y_train)
    test_preds, test_proba, test_acc = evaluate(model, X_train, y_train,
                                                X_test, y_test)

    print("\n  Generating plots...")
    plot_confusion_matrix(y_test, test_preds)
    plot_feature_importance(model, feature_cols)
    plot_prediction_curve(y_test, test_proba)

    save_model(model)

    print()
    print("=" * 55)
    print("  ✅ Model training complete!")
    print(f"  Test Accuracy: {test_acc*100:.1f}%")
    print("  3 plots saved in models/ folder")
    print("=" * 55)
    print()
    print("  Next step → run: python backtest.py")


if __name__ == "__main__":
    main()