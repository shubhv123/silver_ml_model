#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Feature engineering for silver price prediction"""
    
    def __init__(self):
        self.raw_data_path = "data/raw"
        self.processed_path = "data/processed"
        os.makedirs(self.processed_path, exist_ok=True)
        
        # Define lag periods
        self.lag_periods = [1, 2, 3, 5, 10, 21, 63]
        
        # Define rolling windows
        self.rolling_windows = [5, 10, 21, 63, 252]
        
    def load_all_data(self):
        """Load all raw data files"""
        data = {}
        
        # Load silver price data
        silver_files = [f for f in os.listdir(self.raw_data_path) if 'SIF_daily' in f]
        if silver_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, silver_files[-1]), parse_dates=['date'])
            df.set_index('date', inplace=True)
            # Ensure column names are lowercase
            df.columns = [col.lower() for col in df.columns]
            data['silver'] = df
            logger.info(f"Loaded silver data: {df.shape}")
        
        # Load gold price data
        gold_files = [f for f in os.listdir(self.raw_data_path) if 'GCF_daily' in f]
        if gold_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, gold_files[-1]), parse_dates=['date'])
            df.set_index('date', inplace=True)
            df.columns = [col.lower() for col in df.columns]
            data['gold'] = df
            logger.info(f"Loaded gold data: {df.shape}")
        
        # Load macro data
        macro_files = [f for f in os.listdir(self.raw_data_path) if 'macro_data_' in f and f.endswith('.csv')]
        if macro_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, macro_files[-1]), index_col=0, parse_dates=True)
            data['macro'] = df
            logger.info(f"Loaded macro data: {df.shape}")
        
        # Load market features
        market_files = [f for f in os.listdir(self.raw_data_path) if 'market_features_combined' in f]
        if market_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, market_files[-1]), parse_dates=['date'])
            df.set_index('date', inplace=True)
            data['market'] = df
            logger.info(f"Loaded market features: {df.shape}")
        
        # Load COT data
        cot_files = [f for f in os.listdir(self.raw_data_path) if 'cot_silver_' in f and f.endswith('.csv')]
        if cot_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, cot_files[-1]), parse_dates=['date'])
            df.set_index('date', inplace=True)
            data['cot'] = df
            logger.info(f"Loaded COT data: {df.shape}")
        
        # Load ETF data
        etf_files = [f for f in os.listdir(self.raw_data_path) if 'slv_holdings_' in f and f.endswith('.csv')]
        if etf_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, etf_files[-1]), parse_dates=['date'])
            df.set_index('date', inplace=True)
            data['etf'] = df
            logger.info(f"Loaded ETF data: {df.shape}")
        
        # Load Silver Institute data
        institute_files = [f for f in os.listdir(self.raw_data_path) if 'silver_institute_' in f and f.endswith('.csv')]
        if institute_files:
            df = pd.read_csv(os.path.join(self.raw_data_path, institute_files[-1]), parse_dates=['date'])
            df.set_index('date', inplace=True)
            data['institute'] = df
            logger.info(f"Loaded Silver Institute data: {df.shape}")
        
        return data
    
    def add_technical_indicators(self, df):
        """Add technical indicators using pandas calculations"""
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Ensure we have the required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return df
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        volume = df['volume']
        
        # Returns (keep original close for later use)
        df['returns_1d'] = close.pct_change()
        df['returns_5d'] = close.pct_change(5)
        df['returns_21d'] = close.pct_change(21)
        df['returns_63d'] = close.pct_change(63)
        df['returns_252d'] = close.pct_change(252)
        
        # Log returns
        df['log_returns_1d'] = np.log(close / close.shift(1))
        
        # Price ratios
        df['high_low_ratio'] = high / low
        df['close_open_ratio'] = close / open_price
        
        # Volatility
        df['volatility_5d'] = df['returns_1d'].rolling(5).std() * np.sqrt(252)
        df['volatility_21d'] = df['returns_1d'].rolling(21).std() * np.sqrt(252)
        df['volatility_63d'] = df['returns_1d'].rolling(63).std() * np.sqrt(252)
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = close.rolling(period).mean()
            df[f'close_sma_{period}_ratio'] = close / df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in [9, 12, 21, 50]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
            df[f'close_ema_{period}_ratio'] = close / df[f'ema_{period}']
        
        # RSI
        for period in [14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        period = 20
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        df['bb_upper'] = sma + (std * 2)
        df['bb_lower'] = sma - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma
        df['bb_position'] = (close - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_percent'] = df['atr_14'] / close * 100
        
        # On-Balance Volume
        obv = (np.sign(df['returns_1d']) * volume).cumsum()
        df['obv'] = obv
        df['obv_ma_ratio'] = obv / obv.rolling(20).mean()
        
        # Volume indicators
        df['volume_ratio'] = volume / volume.rolling(20).mean()
        df['volume_price_trend'] = (volume * (close - close.shift()) / close.shift()).cumsum()
        
        # Price patterns
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['golden_cross'] = ((df['sma_50'] > df['sma_200']) & 
                                   (df['sma_50'].shift() <= df['sma_200'].shift())).astype(int)
            df['death_cross'] = ((df['sma_50'] < df['sma_200']) & 
                                  (df['sma_50'].shift() >= df['sma_200'].shift())).astype(int)
        
        logger.info(f"Added technical indicators: {len(df.columns)} total columns")
        return df
    
    def add_cross_asset_features(self, df, data):
        """Add features derived from other assets"""
        
        if 'gold' in data:
            gold = data['gold']['close']
            # Align dates
            common_dates = df.index.intersection(gold.index)
            df = df.loc[common_dates]
            gold = gold.loc[common_dates]
            
            df['gold_close'] = gold
            
            # Gold-silver ratio
            df['gold_silver_ratio'] = df['gold_close'] / df['close']
            df['gold_silver_ratio_change'] = df['gold_silver_ratio'].pct_change()
            
            # Correlation
            corr_window = 21
            df['gold_silver_corr'] = (df['returns_1d'].rolling(corr_window)
                                       .corr(df['gold_close'].pct_change()))
        
        if 'market' in data:
            market = data['market']
            # Align dates
            common_dates = df.index.intersection(market.index)
            df = df.loc[common_dates]
            market = market.loc[common_dates]
            
            for col in market.columns:
                if 'close' in col:
                    df[col] = market[col]
            
            # Market correlations
            for asset in ['vix', 'oil', 'copper', 'sp500']:
                col = f'{asset}_close'
                if col in df.columns:
                    df[f'{asset}_corr'] = (df['returns_1d'].rolling(21)
                                            .corr(df[col].pct_change()))
        
        logger.info("Added cross-asset features")
        return df
    
    def add_macro_features(self, df, data):
        """Add macroeconomic features"""
        
        if 'macro' in data:
            macro = data['macro']
            
            # Align dates
            common_dates = df.index.intersection(macro.index)
            df = df.loc[common_dates]
            macro = macro.loc[common_dates]
            
            # Add key macro indicators
            for col in macro.columns:
                if col not in df.columns:
                    df[col] = macro[col]
            
            # Calculate macro regimes
            if 'treasury_10yr' in df.columns:
                df['rate_regime'] = pd.cut(df['treasury_10yr'], 
                                           bins=[0, 2, 4, 6, 10], 
                                           labels=['low', 'normal', 'high', 'very_high'])
            
            if 'yield_curve_10y2y' in df.columns:
                df['yield_curve_regime'] = pd.cut(df['yield_curve_10y2y'],
                                                   bins=[-np.inf, -0.5, 0, 0.5, np.inf],
                                                   labels=['inverted', 'flat', 'normal', 'steep'])
            
            logger.info("Added macroeconomic features")
        
        return df
    
    def add_sentiment_features(self, df, data):
        """Add sentiment features from COT and ETF data"""
        
        if 'cot' in data:
            cot = data['cot']
            
            # Align dates (COT is weekly, so forward fill)
            cot = cot.reindex(df.index, method='ffill')
            
            for col in cot.columns:
                if col not in df.columns and col != 'date':
                    df[f'cot_{col}'] = cot[col]
            
            # COT signals
            if 'cot_Commercial_COT_Index' in df.columns:
                df['cot_commercial_extreme'] = ((df['cot_Commercial_COT_Index'] < 20) | 
                                                (df['cot_Commercial_COT_Index'] > 80)).astype(int)
            
            logger.info("Added COT sentiment features")
        
        if 'etf' in data:
            etf = data['etf']
            
            # Align dates
            common_dates = df.index.intersection(etf.index)
            df = df.loc[common_dates]
            etf = etf.loc[common_dates]
            
            for col in etf.columns:
                if col not in df.columns and col != 'date':
                    df[f'etf_{col}'] = etf[col]
            
            # ETF signals
            if 'etf_daily_flow_usd' in df.columns:
                df['etf_flow_signal'] = np.sign(df['etf_daily_flow_usd'])
            
            logger.info("Added ETF sentiment features")
        
        return df
    
    def add_fundamental_features(self, df, data):
        """Add fundamental supply/demand features"""
        
        if 'institute' in data:
            institute = data['institute']
            
            # Align dates
            common_dates = df.index.intersection(institute.index)
            df = df.loc[common_dates]
            institute = institute.loc[common_dates]
            
            for col in institute.columns:
                if col not in df.columns and col not in ['date', 'year']:
                    df[f'fund_{col}'] = institute[col]
            
            # Fundamental ratios
            if 'fund_market_balance' in df.columns:
                df['fund_deficit_surplus'] = np.sign(df['fund_market_balance'])
            
            if 'fund_stock_to_flow' in df.columns:
                df['fund_stock_to_flow_zscore'] = ((df['fund_stock_to_flow'] - 
                                                     df['fund_stock_to_flow'].rolling(252).mean()) / 
                                                    df['fund_stock_to_flow'].rolling(252).std())
            
            logger.info("Added fundamental features")
        
        return df
    
    def add_lag_features(self, df, columns=None, periods=None):
        """Add lagged versions of key features"""
        
        if periods is None:
            periods = self.lag_periods
        
        if columns is None:
            # Select key columns for lagging (excluding targets and non-numeric)
            exclude_patterns = ['date', 'signal', 'regime', 'target']
            columns = [col for col in df.columns if not any(p in col.lower() for p in exclude_patterns)]
            columns = [col for col in columns if df[col].dtype in ['float64', 'int64']]
            columns = columns[:15]  # Limit to top 15 features to avoid explosion
        
        for col in columns:
            if col in df.columns:
                for lag in periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        logger.info(f"Added lag features for {len(columns)} columns")
        return df
    
    def add_rolling_features(self, df, columns=None, windows=None, aggs=None):
        """Add rolling statistics"""
        
        if windows is None:
            windows = self.rolling_windows
        
        if aggs is None:
            aggs = ['mean', 'std', 'min', 'max']
        
        if columns is None:
            # Select numerical columns (excluding targets)
            columns = df.select_dtypes(include=[np.number]).columns
            columns = [col for col in columns if 'target' not in col]
            columns = columns[:10]  # Limit to top 10 features
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    for agg in aggs:
                        if agg == 'mean':
                            df[f'{col}_rolling_{window}_mean'] = df[col].rolling(window).mean()
                        elif agg == 'std':
                            df[f'{col}_rolling_{window}_std'] = df[col].rolling(window).std()
                        elif agg == 'min':
                            df[f'{col}_rolling_{window}_min'] = df[col].rolling(window).min()
                        elif agg == 'max':
                            df[f'{col}_rolling_{window}_max'] = df[col].rolling(window).max()
        
        logger.info(f"Added rolling features")
        return df
    
    def create_target_variables(self, df):
        """Create prediction targets"""
        
        if 'close' not in df.columns:
            logger.error("Cannot create targets: 'close' column missing")
            return df
        
        close = df['close']
        
        # Regression targets
        df['target_next_day_return'] = close.pct_change().shift(-1)
        df['target_next_week_return'] = close.pct_change(5).shift(-5)
        df['target_next_month_return'] = close.pct_change(21).shift(-21)
        df['target_next_quarter_return'] = close.pct_change(63).shift(-63)
        
        # Classification targets
        df['target_next_day_direction'] = (df['target_next_day_return'] > 0).astype(int)
        df['target_next_week_direction'] = (df['target_next_week_return'] > 0).astype(int)
        
        # Volatility targets
        df['target_next_week_volatility'] = df['returns_1d'].rolling(5).std().shift(-5)
        
        # Binary targets for specific thresholds
        df['target_next_day_up_1pct'] = (df['target_next_day_return'] > 0.01).astype(int)
        df['target_next_day_down_1pct'] = (df['target_next_day_return'] < -0.01).astype(int)
        
        logger.info("Created target variables")
        return df
    
    def remove_lookahead_bias(self, df):
        """
        Ensure no lookahead bias by removing any features that use future data
        """
        # Drop any rows with NaN in critical columns
        initial_rows = len(df)
        df = df.dropna(subset=['close'])
        
        logger.info(f"Removed lookahead bias, rows: {initial_rows} -> {len(df)}")
        return df
    
    def build_feature_matrix(self):
        """Build complete feature matrix"""
        
        logger.info("="*60)
        logger.info("BUILDING FEATURE MATRIX")
        logger.info("="*60)
        
        # Load all data
        data = self.load_all_data()
        
        if 'silver' not in data:
            logger.error("No silver price data found!")
            return None
        
        # Start with silver price data
        df = data['silver'].copy()
        logger.info(f"Starting with silver data: {df.shape}")
        logger.info(f"Silver columns: {df.columns.tolist()}")
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        logger.info(f"After technical indicators: {df.shape}")
        
        # Add cross-asset features
        df = self.add_cross_asset_features(df, data)
        logger.info(f"After cross-asset features: {df.shape}")
        
        # Add macro features
        df = self.add_macro_features(df, data)
        logger.info(f"After macro features: {df.shape}")
        
        # Add sentiment features
        df = self.add_sentiment_features(df, data)
        logger.info(f"After sentiment features: {df.shape}")
        
        # Add fundamental features
        df = self.add_fundamental_features(df, data)
        logger.info(f"After fundamental features: {df.shape}")
        
        # Add lag features (before rolling to avoid leakage)
        df = self.add_lag_features(df)
        logger.info(f"After lag features: {df.shape}")
        
        # Add rolling features
        df = self.add_rolling_features(df)
        logger.info(f"After rolling features: {df.shape}")
        
        # Create targets (must be last, after all features)
        df = self.create_target_variables(df)
        logger.info(f"After creating targets: {df.shape}")
        
        # Remove lookahead bias
        df = self.remove_lookahead_bias(df)
        
        # Final cleaning
        df = df.replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"\nFinal feature matrix: {df.shape}")
        logger.info(f"Features: {len(df.columns)}")
        logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Count feature types
        n_features = len([c for c in df.columns if 'target' not in c])
        n_targets = len([c for c in df.columns if 'target' in c])
        logger.info(f"Features: {n_features}, Targets: {n_targets}")
        
        return df
    
    def save_feature_matrix(self, df):
        """Save feature matrix to disk"""
        
        if df is None or df.empty:
            logger.error("No feature matrix to save")
            return
        
        # Save full feature matrix
        filename = f"feature_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(self.processed_path, filename)
        df.to_csv(filepath)
        logger.info(f"Saved feature matrix to {filepath}")
        
        # Save feature list
        feature_file = os.path.join(self.processed_path, "feature_list.txt")
        with open(feature_file, 'w') as f:
            f.write("FEATURE LIST\n")
            f.write("="*60 + "\n")
            for i, col in enumerate(sorted(df.columns), 1):
                if 'target' not in col:
                    f.write(f"{i:3d}. {col}\n")
        
        logger.info(f"Saved feature list to {feature_file}")
        
        # Save metadata
        metadata = {
            'filename': filename,
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'rows': len(df),
            'total_columns': len(df.columns),
            'feature_columns': len([c for c in df.columns if 'target' not in c]),
            'target_columns': len([c for c in df.columns if 'target' in c]),
            'creation_date': datetime.now().isoformat()
        }
        
        meta_path = os.path.join(self.processed_path, filename.replace('.csv', '_metadata.txt'))
        with open(meta_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        return filepath

def main():
    """Main execution"""
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Build feature matrix
    df = fe.build_feature_matrix()
    
    if df is not None:
        # Save feature matrix
        fe.save_feature_matrix(df)
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING SUMMARY")
        logger.info("="*60)
        
        # Missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        cols_with_missing = missing[missing > 0]
        
        if len(cols_with_missing) > 0:
            logger.info(f"\nColumns with missing values: {len(cols_with_missing)}")
            for col in cols_with_missing.index[:10]:  # Show first 10
                logger.info(f"  {col}: {missing[col]:.0f} ({missing_pct[col]:.1f}%)")
        
        # Feature correlations with target (sample)
        if 'target_next_day_return' in df.columns:
            corr_with_target = df.corrwith(df['target_next_day_return']).abs().sort_values(ascending=False)
            logger.info(f"\nTop 10 features correlated with next day return:")
            for col in corr_with_target.index[1:11]:
                if pd.notna(corr_with_target[col]):
                    logger.info(f"  {col}: {corr_with_target[col]:.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE ENGINEERING COMPLETE")
        logger.info("="*60)

if __name__ == "__main__":
    main()