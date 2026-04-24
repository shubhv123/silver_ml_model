A comprehensive machine learning system for silver price prediction, featuring ensemble methods, deep learning, NLP sentiment analysis, Bayesian inference, and a live dashboard.

## 🚀 Features

- **Data Collection**: Pulls silver/gold prices, macro indicators (FRED), COT reports, ETF flows, and news sentiment
- **Feature Engineering**: Technical indicators, lag features, rolling statistics, cross-asset correlations
- **Multiple Models**: XGBoost, LightGBM, CatBoost, LSTM with attention, Monte Carlo Dropout, Bayesian regression
- **Ensemble Learning**: Stacking ensemble with SHAP explainability
- **NLP Sentiment**: FinBERT for financial news sentiment analysis
- **Risk Management**: Dynamic position sizing, circuit breaker, volatility scaling
- **Live Dashboard**: Plotly Dash web app with real-time predictions

## 📊 Performance (XGBoost - No Sentiment)

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.97 |
| Total Return (Test) | 129.7% |
| Directional Accuracy | 52.2% |
| Max Drawdown | -22.8% |

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/silver_ml_model.git
cd silver-price-prediction

# Create virtual environment
conda create -n silver_ml python=3.10 -y
conda activate silver_ml

# Install dependencies
pip install -r requirements.txt
```

## 🔑 API Keys Required

- FRED API (free): https://fred.stlouisfed.org/docs/api/api_key.html
- NewsAPI (optional, for real news): https://newsapi.org/register

- Add to .env file:

```bash
FRED_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
```

## 🧪 Run the Pipeline

```bash
# Collect all data
python src/data/run_data_pipeline.py

# Feature engineering
python src/features/build_features.py

# Train models
python src/models/train_xgboost.py
python src/models/train_lightgbm.py
python src/models/train_catboost.py

# Run backtest
python src/backtest/complete_model_backtest.py

# Launch dashboard
python src/dashboard/live_dashboard_fixed.py
```

## 🤝 Contributing

- We welcome contributions! Please check the issues tab and see the "Known Issues" section above. To contribute:
- Fork the repository
- Create a feature branch (git checkout -b feature/your-fix)
- Commit changes (git commit -m "Add fix for COT data")
- Push to branch (git push origin feature/your-fix)
- Open a Pull Request
