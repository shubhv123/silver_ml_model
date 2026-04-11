# Data Dictionary - Silver Price Prediction Project

## Price Data (from Yahoo Finance)

| Column | Description | Frequency |
|--------|-------------|-----------|
| open | Opening price | Daily |
| high | Highest price of the day | Daily |
| low | Lowest price of the day | Daily |
| close | Closing price | Daily |
| volume | Trading volume | Daily |
| symbol | Ticker symbol (SI=F or GC=F) | Daily |

## Macroeconomic Data (from FRED)

| Column | FRED Series | Description | Frequency |
|--------|-------------|-------------|-----------|
| treasury_10yr | DGS10 | 10-Year Treasury Constant Maturity Rate (%) | Daily |
| treasury_2yr | DGS2 | 2-Year Treasury Constant Maturity Rate (%) | Daily |
| treasury_30yr | DGS30 | 30-Year Treasury Constant Maturity Rate (%) | Daily |
| dollar_index | DTWEXBGS | Trade Weighted U.S. Dollar Index (Jan 2006=100) | Daily |
| cpi | CPIAUCSL | Consumer Price Index for All Urban Consumers | Monthly |
| fed_funds | FEDFUNDS | Effective Federal Funds Rate (%) | Daily |
| inflation_expectations | T10YIE | 10-Year Breakeven Inflation Rate (%) | Daily |
| industrial_production | INDPRO | Industrial Production Index | Monthly |
| unemployment | UNRATE | Unemployment Rate (%) | Monthly |
| gdp_growth | A191RL1Q225SBEA | Real Gross Domestic Product (Quarterly Growth %) | Quarterly |

## Derived Features

| Column | Formula | Description |
|--------|---------|-------------|
| cpi_mom | CPI pct change (1 month) | Monthly inflation rate |
| cpi_yoy | CPI pct change (12 months) | Annual inflation rate |
| yield_curve_10y2y | 10Y - 2Y | Yield curve slope (recession predictor) |
| yield_curve_30y10y | 30Y - 10Y | Long-end yield curve slope |
| real_rate_10yr | 10Y - inflation_expectations | Real interest rate |
| cpi_acceleration | CPI 12M change - previous 12M change | Inflation momentum |

## Market Features (from Yahoo Finance)

### Files: `vix_daily.csv`, `oil_daily.csv`, `copper_daily.csv`, `sp500_daily.csv`, `market_features_combined.csv`

| Feature | Symbol | Description | Typical Range |
|---------|--------|-------------|---------------|
| vix_close | ^VIX | CBOE Volatility Index (fear gauge) | 10-40 |
| oil_close | CL=F | Crude Oil futures price (USD/barrel) | $20-$120 |
| copper_close | HG=F | Copper futures price (USD/lb) | $2.00-$5.00 |
| sp500_close | ^GSPC | S&P 500 Index | 1000-5000 |

## COT Data (Commitment of Traders)

### File: `cot_silver_YYYYMMDD.csv`

| Column | Description | Interpretation |
|--------|-------------|----------------|
| date | Report date (Tuesday) | - |
| Open_Interest | Total open interest in silver futures | Higher = more market activity |
| Commercial_Long | Commercial traders long positions | Hedgers (producers/users) |
| Commercial_Short | Commercial traders short positions | Hedgers (producers/users) |
| Commercial_Net | Net commercial position (long - short) | Negative = commercial hedging |
| Commercial_Net_Pct | Commercial net as % of open interest | >20% is significant |
| Commercial_COT_Index | Commercial COT Index (0-100) | <20 = bullish, >80 = bearish |
| Noncommercial_Long | Large speculators (funds) long | Trend followers |
| Noncommercial_Short | Large speculators short | Trend followers |
| Noncommercial_Spread | Spreading positions | Market makers |
| Noncommercial_Net | Net large speculator position | Positive = bullish sentiment |
| Noncommercial_Net_Pct | Large spec net as % of open interest | Sentiment indicator |
| Noncommercial_COT_Index | Large spec COT Index (0-100) | <20 = oversold, >80 = overbought |
| Nonreportable_Long | Small traders long | Retail sentiment |
| Nonreportable_Short | Small traders short | Retail sentiment |

### COT Interpretation Guide

| Indicator | Bullish Signal | Bearish Signal |
|-----------|---------------|----------------|
| Commercial Net | Near historical lows | Near historical highs |
| Commercial COT Index | < 20 (commercial buying) | > 80 (commercial selling) |
| Noncommercial Net | Increasing (funds buying) | Decreasing (funds selling) |
| Noncommercial COT Index | > 80 (momentum) | < 20 (capitulation) |
EOF

## Feature Selection Summary

After analyzing all engineered features, we select the most predictive ones while:
- Removing features with >30% missing values
- Removing highly correlated features (>0.95 correlation)
- Keeping features with highest correlation to target

### Selected Features Count by Category
- Price Technical: ~25 features
- Cross-Asset: ~8 features
- Macroeconomic: ~12 features  
- Sentiment (COT/ETF): ~10 features
- Fundamental: ~6 features
- Lag/Rolling: ~15 features

**Total Selected Features**: ~75-80 features