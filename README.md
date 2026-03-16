# 📈 Nifty Intelligence

> ML-powered prediction of Nifty50 opening direction using global market signals

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)

---

## 🎯 What It Does

Predicts whether the **NSE Nifty50 will open UP or DOWN** the next trading day using an ensemble of machine learning models trained on global market data.

**Prediction window:** 9:00–9:15 AM IST (NSE opening session)

---

## 🧠 Model Architecture — Tier 3 Ensemble

```
Input Features (65+)
    ├── GIFT Nifty overnight return
    ├── S&P 500 close return
    ├── Nasdaq close return
    ├── India VIX level & momentum
    ├── USD/INR overnight move
    ├── Brent Crude Oil return
    ├── Rolling correlations (20d)
    ├── Moving average distances
    ├── Calendar features (day, month, week)
    └── Cross-market momentum divergence

Ensemble Model
    ├── XGBoost        (40% weight)
    ├── LightGBM       (40% weight)
    └── Random Forest  (20% weight)

Regime Detection
    ├── BULL  → Nifty above 50d MA, VIX < 15
    ├── BEAR  → Nifty below 50d MA, VIX > 20
    └── FLAT  → Consolidation phase

Output
    ├── Direction: UP / DOWN
    ├── Confidence: 50–100%
    ├── Signal Tier: Strong / Moderate / Weak / Unclear
    └── SHAP Explanation: Top drivers
```

---

## 📊 Performance (Backtest 2023–2024)

| Metric | Value |
|--------|-------|
| Test Accuracy | ~55% |
| Win Rate | ~60% |
| F1 Score | ~0.56 |
| Training Period | 2019–2022 |
| Test Period | 2023–2024 |

> Note: Even 52%+ sustained accuracy is considered a real signal in financial markets.

---

## 🖥️ Dashboard Features

| Page | Features |
|------|----------|
| 🏠 Live Prediction | Real-time prediction + Confidence gauge + Market tickers + AI Commentary + PDF Report |
| 🧪 What-If Simulator | 5 market sliders + 4 preset scenarios |
| 📊 Analytics | Performance charts + Model comparison + SHAP explainability |
| 📅 Heatmaps | Accuracy calendar + Monthly P&L + Live Nifty chart |
| 📋 History | Prediction log + Actual outcome tracking |
| ⚙️ Settings | Model info + One-click retrain |

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/nifty-predictor.git
cd nifty-predictor

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (first time only)
python data_fetch.py
python data_preprocess.py
python features_v2.py
python ensemble_model.py
python regime_detector.py
python explainer.py

# Launch dashboard
streamlit run app.py
```

---

## 📁 Project Structure

```
nifty-predictor/
├── app.py                  # Main Streamlit dashboard
├── data_fetch.py           # Fetch market data
├── data_preprocess.py      # Clean & align data
├── features.py             # Tier 1 features (40)
├── features_v2.py          # Tier 3 features (65+)
├── model.py                # Single XGBoost (Tier 1)
├── ensemble_model.py       # Ensemble training (Tier 3)
├── regime_detector.py      # Market regime detection
├── explainer.py            # SHAP explanations
├── live_predict.py         # Tier 1 live prediction
├── live_predict_v2.py      # Tier 3 live prediction
├── backtest.py             # Strategy backtesting
├── requirements.txt        # Dependencies
├── data/                   # CSV datasets
└── models/                 # Trained model files
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Always conduct independent research before making any investment decisions. Past model performance does not guarantee future results.

---

## 🛠️ Tech Stack

`Python` `Streamlit` `XGBoost` `LightGBM` `scikit-learn` `SHAP` `yfinance` `pandas` `matplotlib`

---

*Built as a quantitative finance learning project*