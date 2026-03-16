"""
STEP 5 - BACKTESTING
- Loads trained model + features
- Simulates trading on test set predictions
- Compares strategy vs buy-and-hold benchmark
- Plots P&L curve
- Prints final performance report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# ── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = "data"
MODEL_DIR  = "models"
INPUT      = os.path.join(DATA_DIR, "features.csv")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.pkl")

INITIAL_CAPITAL = 100000  # ₹1,00,000 starting capital
TRANSACTION_COST = 0.001  # 0.1% per trade (brokerage + slippage)


# ── Load ─────────────────────────────────────────────────────────────────────
def load_data_and_model():
    print("  Loading features and model...")

    df = pd.read_csv(INPUT, index_col="date", parse_dates=True)
    test = df[df.index.year >= 2023].copy()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    feature_cols = [c for c in df.columns if c != "target"]
    X_test = test[feature_cols]
    y_test = test["target"]

    print(f"  Test period: {test.index.min().date()} → {test.index.max().date()}")
    print(f"  Test rows: {len(test)}")
    return model, X_test, y_test, test


# ── Run Backtest ──────────────────────────────────────────────────────────────
def run_backtest(model, X_test, y_test, test_df):
    print("\n  Running backtest simulation...")

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Actual daily returns from test set
    actual_returns = test_df["nifty_ret"].values / 100  # convert % to decimal

    # ── Strategy: buy when model predicts UP ─────────────────────────────────
    strategy_returns = []
    benchmark_returns = []

    capital_strategy  = INITIAL_CAPITAL
    capital_benchmark = INITIAL_CAPITAL

    strategy_curve  = [INITIAL_CAPITAL]
    benchmark_curve = [INITIAL_CAPITAL]

    wins = 0
    losses = 0
    total_trades = 0

    for i in range(len(preds)):
        actual_ret = actual_returns[i]
        benchmark_returns.append(actual_ret)

        if preds[i] == 1:  # Model says UP → BUY
            trade_ret = actual_ret - TRANSACTION_COST
            strategy_returns.append(trade_ret)
            capital_strategy *= (1 + trade_ret)
            total_trades += 1
            if actual_ret > 0:
                wins += 1
            else:
                losses += 1
        else:  # Model says DOWN → STAY OUT (cash)
            strategy_returns.append(0)
            capital_strategy = capital_strategy  # no change

        capital_benchmark *= (1 + actual_ret)
        strategy_curve.append(capital_strategy)
        benchmark_curve.append(capital_benchmark)

    return {
        "preds"            : preds,
        "proba"            : proba,
        "actual_returns"   : actual_returns,
        "strategy_returns" : strategy_returns,
        "benchmark_returns": benchmark_returns,
        "strategy_curve"   : strategy_curve,
        "benchmark_curve"  : benchmark_curve,
        "capital_strategy" : capital_strategy,
        "capital_benchmark": capital_benchmark,
        "wins"             : wins,
        "losses"           : losses,
        "total_trades"     : total_trades,
    }


# ── Compute Metrics ───────────────────────────────────────────────────────────
def compute_metrics(results):
    strat = np.array(results["strategy_returns"])
    bench = np.array(results["benchmark_returns"])

    # Remove zeros (days we didn't trade) for strategy stats
    traded = strat[strat != 0]

    def sharpe(returns):
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def max_drawdown(curve):
        curve = np.array(curve)
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / peak
        return dd.min() * 100

    metrics = {
        "total_return_strategy"  : (results["capital_strategy"] / INITIAL_CAPITAL - 1) * 100,
        "total_return_benchmark" : (results["capital_benchmark"] / INITIAL_CAPITAL - 1) * 100,
        "sharpe_strategy"        : sharpe(traded) if len(traded) > 0 else 0,
        "sharpe_benchmark"       : sharpe(bench),
        "max_dd_strategy"        : max_drawdown(results["strategy_curve"]),
        "max_dd_benchmark"       : max_drawdown(results["benchmark_curve"]),
        "win_rate"               : results["wins"] / results["total_trades"] * 100 if results["total_trades"] > 0 else 0,
        "total_trades"           : results["total_trades"],
    }
    return metrics


# ── Print Report ──────────────────────────────────────────────────────────────
def print_report(metrics, results):
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║         BACKTEST RESULTS (2023–2024)         ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Starting Capital    : ₹{INITIAL_CAPITAL:,.0f}          ║")
    print(f"  ║  Strategy Final      : ₹{results['capital_strategy']:,.0f}       ║")
    print(f"  ║  Benchmark Final     : ₹{results['capital_benchmark']:,.0f}       ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Strategy Return     : {metrics['total_return_strategy']:+.1f}%              ║")
    print(f"  ║  Benchmark Return    : {metrics['total_return_benchmark']:+.1f}%              ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Sharpe (Strategy)   : {metrics['sharpe_strategy']:.2f}                 ║")
    print(f"  ║  Sharpe (Benchmark)  : {metrics['sharpe_benchmark']:.2f}                 ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Max Drawdown (Str)  : {metrics['max_dd_strategy']:.1f}%             ║")
    print(f"  ║  Max Drawdown (Ben)  : {metrics['max_dd_benchmark']:.1f}%             ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  Win Rate            : {metrics['win_rate']:.1f}%               ║")
    print(f"  ║  Total Trades        : {metrics['total_trades']}                    ║")
    print("  ╚══════════════════════════════════════════════╝")


# ── Plot P&L Curve ────────────────────────────────────────────────────────────
def plot_pnl(results, metrics):
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Capital curve
    ax1 = axes[0]
    ax1.plot(results["strategy_curve"],
             color="#2196F3", linewidth=2, label="ML Strategy")
    ax1.plot(results["benchmark_curve"],
             color="#FF9800", linewidth=2, label="Buy & Hold Nifty")
    ax1.axhline(INITIAL_CAPITAL, color="gray", linestyle="--",
                linewidth=1, alpha=0.7, label="Starting Capital")
    ax1.set_title("Portfolio Value: ML Strategy vs Buy & Hold (2023–2024)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (₹)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))

    # Plot 2: Daily strategy returns
    ax2 = axes[1]
    strat_ret = np.array(results["strategy_returns"]) * 100
    colors = ["#4CAF50" if r > 0 else "#F44336" if r < 0 else "#9E9E9E"
              for r in strat_ret]
    ax2.bar(range(len(strat_ret)), strat_ret, color=colors, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Daily Strategy Returns (%)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("Return (%)")
    ax2.set_xlabel("Trading Days")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(MODEL_DIR, "backtest_pnl.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  💾 Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  NIFTY PREDICTOR — Step 5: Backtesting")
    print("=" * 55 + "\n")

    model, X_test, y_test, test_df = load_data_and_model()
    results = run_backtest(model, X_test, y_test, test_df)
    metrics = compute_metrics(results)

    print_report(metrics, results)

    print("\n  Generating P&L chart...")
    plot_pnl(results, metrics)

    print()
    print("=" * 55)
    print("  ✅ Backtest complete!")
    print("  Check models/backtest_pnl.png for the P&L chart")
    print("=" * 55)
    print()
    print("  Next step → run: python app.py  (Streamlit dashboard)")


if __name__ == "__main__":
    main()