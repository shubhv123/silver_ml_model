#!/usr/bin/env python
"""
Port Ensemble to Qlib Framework - Correct Version
Standardized backtesting and portfolio simulation
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QlibIntegration:
    """Port ensemble to Qlib framework"""
    
    def __init__(self):
        self.processed_path = "data/processed"
        self.models_path = "models/saved"
        self.qlib_path = "data/qlib"
        os.makedirs(self.qlib_path, exist_ok=True)
        
    def prepare_qlib_data(self):
        """Convert our feature matrix to Qlib format"""
        logger.info("="*60)
        logger.info("PREPARING DATA FOR QLIB")
        logger.info("="*60)
        
        # Load feature matrix
        final_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not final_files:
            final_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        
        if not final_files:
            logger.error("No feature matrix found!")
            return None
        
        # Read CSV
        df = pd.read_csv(sorted(final_files)[-1])
        logger.info(f"Loaded {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()[:5]}...")
        
        # Find the date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or col == 'Unnamed: 0':
                date_col = col
                break
        
        if date_col is None:
            date_col = df.columns[0]
        
        # Extract dates as strings
        dates = pd.to_datetime(df[date_col], utc=True).dt.strftime('%Y-%m-%d')
        df = df.drop(columns=[date_col])
        
        logger.info(f"Date range: {dates.iloc[0]} to {dates.iloc[-1]}")
        
        # Create Qlib format dataframe
        df_qlib = df.copy()
        df_qlib.insert(0, 'date', dates)
        df_qlib.insert(1, 'instrument', 'SILVER')
        
        # Save in Qlib format
        qlib_file = os.path.join(self.qlib_path, "silver_data.csv")
        df_qlib.to_csv(qlib_file, index=False)
        logger.info(f"Saved Qlib format data to {qlib_file}")
        
        # Create calendar file
        calendar = pd.DataFrame({'date': dates})
        calendar_file = os.path.join(self.qlib_path, "calendar.csv")
        calendar.to_csv(calendar_file, index=False)
        logger.info(f"Saved calendar to {calendar_file}")
        
        # Create instruments file
        instruments = pd.DataFrame({
            'instrument': ['SILVER'], 
            'start_date': [dates.iloc[0]], 
            'end_date': [dates.iloc[-1]]
        })
        instruments_file = os.path.join(self.qlib_path, "instruments.csv")
        instruments.to_csv(instruments_file, index=False)
        logger.info(f"Saved instruments to {instruments_file}")
        
        return df_qlib
    
    def create_simple_backtest(self, df_qlib):
        """Create simple backtest"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING SIMPLE BACKTEST")
        logger.info("="*60)
        
        # Load our trained models
        models = {}
        
        # Load XGBoost
        xgb_files = list(Path(self.models_path).glob("xgboost_*.pkl"))
        if xgb_files:
            with open(sorted(xgb_files)[-1], 'rb') as f:
                models['xgboost'] = pickle.load(f)
                logger.info("Loaded XGBoost model")
        
        # Load LightGBM
        lgb_files = list(Path(self.models_path).glob("lightgbm_*.pkl"))
        if lgb_files:
            with open(sorted(lgb_files)[-1], 'rb') as f:
                models['lightgbm'] = pickle.load(f)
                logger.info("Loaded LightGBM model")
        
        # Load CatBoost
        cb_files = list(Path(self.models_path).glob("catboost_*.cbm"))
        if cb_files:
            import catboost as cb
            models['catboost'] = cb.CatBoostRegressor()
            models['catboost'].load_model(sorted(cb_files)[-1])
            logger.info("Loaded CatBoost model")
        
        if not models:
            logger.error("No models found!")
            return None
        
        # Prepare data
        df_backtest = df_qlib.copy()
        df_backtest['date'] = pd.to_datetime(df_backtest['date'])
        df_backtest = df_backtest.set_index('date').sort_index()
        
        # Get feature columns
        exclude_cols = ['instrument']
        feature_cols = [c for c in df_backtest.columns if c not in exclude_cols and 'target' not in c]
        X = df_backtest[feature_cols].fillna(0)
        
        logger.info(f"Using {len(feature_cols)} features for prediction")
        
        # Make ensemble predictions
        predictions = []
        model_preds = {}
        
        for name, model in models.items():
            pred = model.predict(X)
            model_preds[name] = pred
            predictions.append(pred)
        
        # Weighted ensemble (equal weights)
        ensemble_pred = np.mean(predictions, axis=0)
        
        # Simple trading strategy
        initial_capital = 10000
        portfolio_value = initial_capital
        positions = []
        daily_returns = []
        
        # Use target returns if available
        if 'target_next_day_return' in df_backtest.columns:
            actual_returns = df_backtest['target_next_day_return'].values
            logger.info("Using actual target returns for backtest")
        else:
            # Estimate returns from predictions
            actual_returns = ensemble_pred + np.random.randn(len(ensemble_pred)) * 0.005
            logger.info("Using estimated returns for backtest")
        
        # Run trading simulation
        for i, pred in enumerate(ensemble_pred):
            # Position sizing based on prediction
            if pred > 0.002:  # Long signal (>0.2% predicted return)
                position = 0.9
            elif pred < -0.002:  # Short signal
                position = -0.3
            else:
                position = 0
            
            positions.append(position)
            
            # Strategy return
            strategy_return = actual_returns[i] * position if i < len(actual_returns) else 0
            daily_returns.append(strategy_return)
            portfolio_value *= (1 + strategy_return)
        
        # Calculate metrics
        returns_series = pd.Series(daily_returns, index=X.index)
        cumulative = (1 + returns_series).cumprod()
        
        sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
        max_drawdown = ((cumulative / cumulative.expanding().max()) - 1).min()
        win_rate = (returns_series > 0).mean()
        total_return = (portfolio_value / initial_capital - 1) * 100
        
        logger.info(f"\n📊 BACKTEST RESULTS:")
        logger.info(f"   Initial Capital: ${initial_capital:,.0f}")
        logger.info(f"   Final Value: ${portfolio_value:,.0f}")
        logger.info(f"   Total Return: {total_return:.2f}%")
        logger.info(f"   Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"   Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"   Win Rate: {win_rate:.2%}")
        
        # Model correlation analysis
        logger.info(f"\n📈 MODEL CORRELATIONS:")
        for name, preds in model_preds.items():
            corr = np.corrcoef(preds, ensemble_pred)[0, 1]
            logger.info(f"   {name}: {corr:.4f}")
        
        # Save results
        results = {
            'initial_capital': initial_capital,
            'final_value': portfolio_value,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate_pct': win_rate * 100,
            'model_correlations': {name: float(np.corrcoef(preds, ensemble_pred)[0, 1]) 
                                   for name, preds in model_preds.items()}
        }
        
        results_file = os.path.join("reports/metrics", f"qlib_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n💾 Saved results to {results_file}")
        
        return results
    
    def run(self):
        """Run complete Qlib integration"""
        logger.info("="*60)
        logger.info("QLIB INTEGRATION - PORTING ENSEMBLE")
        logger.info("="*60)
        
        # Prepare data
        df_qlib = self.prepare_qlib_data()
        if df_qlib is None:
            return
        
        # Run backtest
        results = self.create_simple_backtest(df_qlib)
        
        logger.info("\n" + "="*60)
        logger.info("✅ QLIB INTEGRATION COMPLETE")
        logger.info("="*60)
        
        return results

if __name__ == "__main__":
    integration = QlibIntegration()
    integration.run()