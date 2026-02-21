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