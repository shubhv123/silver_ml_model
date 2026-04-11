#!/usr/bin/env python
"""
Walk-Forward Backtest - JSON Fixed Version
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

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WalkForwardBacktest:
    def __init__(self, target_col='target_next_day_return'):
        self.processed_path = "data/processed"
        self.models_path = "models/saved"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        self.target_col = target_col
        self.model = None
        
    def load_best_model(self):
        ensemble_files = list(Path(self.models_path).glob("ensemble_meta_*.pkl"))
        if ensemble_files:
            with open(sorted(ensemble_files)[-1], 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded ensemble model")
            return True
        xgb_files = list(Path(self.models_path).glob("xgboost_*.pkl"))
        if xgb_files:
            import xgboost as xgb
            with open(sorted(xgb_files)[-1], 'rb') as f:
                self.model = pickle.load(f)
            logger.info(f"Loaded XGBoost model")
            return True
        logger.error("No model found!")
        return False
    
    def load_data(self):
        final_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not final_files:
            final_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        if not final_files:
            logger.error("No feature matrix found!")
            return None
        df = pd.read_csv(sorted(final_files)[-1], index_col=0)
        df.index = pd.to_datetime(df.index, utc=True).tz_localize(None)
        logger.info(f"Loaded: {df.shape}")
        return df
    
    def run_backtest(self, df, initial_capital=10000):
        logger.info("\n" + "="*60)
        logger.info("WALK-FORWARD BACKTEST")
        logger.info("="*60)
        
        feature_cols = [c for c in df.columns if 'target' not in c]
        
        results = pd.DataFrame(index=df.index)
        results['actual_return'] = df[self.target_col]
        results['predicted_return'] = np.nan
        results['position'] = 0
        results['returns'] = 0
        results['portfolio_value'] = initial_capital
        
        train_window = 1000
        test_size = 63
        rebalance_days = 21
        
        logger.info(f"Training window: {train_window} days")
        logger.info(f"Test window: {test_size} days")
        
        for start_idx in range(0, len(df) - train_window - test_size, rebalance_days):
            train_end = start_idx + train_window
            test_start = train_end
            test_end = min(test_start + test_size, len(df))
            if test_end > len(df):
                break
            
            X_train = df.iloc[start_idx:train_end][feature_cols].fillna(0)
            y_train = df.iloc[start_idx:train_end][self.target_col]
            X_test = df.iloc[test_start:test_end][feature_cols].fillna(0)
            test_dates = df.index[test_start:test_end]
            
            try:
                predictions = self.model.predict(X_test)
            except:
                predictions = np.zeros(len(X_test))
            
            results.loc[test_dates, 'predicted_return'] = predictions
            
            for i, date in enumerate(test_dates):
                if i < len(test_dates) - 1:
                    actual_ret = results.loc[date, 'actual_return']
                    pred = predictions[i]
                    if pred > 0.002:
                        position = 0.9
                    elif pred < -0.002:
                        position = -0.3
                    else:
                        position = 0
                    results.loc[date, 'position'] = position
                    results.loc[date, 'returns'] = actual_ret * position
        
        portfolio_value = initial_capital
        portfolio_values = [initial_capital]
        for i in range(1, len(results)):
            if pd.notna(results.iloc[i]['returns']):
                portfolio_value *= (1 + results.iloc[i]['returns'])
            portfolio_values.append(portfolio_value)
        results['portfolio_value'] = portfolio_values
        
        metrics = self.calculate_metrics(results)
        return results, metrics
    
    def calculate_metrics(self, results):
        returns = results['returns'].dropna()
        portfolio = results['portfolio_value']
        
        if len(returns) == 0:
            return {}
        
        total_return = float((portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100)
        annual_return = total_return / (len(portfolio) / 252)
        volatility = float(returns.std() * np.sqrt(252) * 100)
        sharpe = float(annual_return / volatility) if volatility > 0 else 0
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = float(drawdown.min())
        win_rate = float((returns > 0).mean() * 100)
        
        gross_profits = float(returns[returns > 0].sum())
        gross_losses = float(abs(returns[returns < 0].sum()))
        profit_factor = float(gross_profits / gross_losses) if gross_losses > 0 else np.inf
        
        valid_idx = results['predicted_return'].notna() & results['actual_return'].notna()
        direction_acc = float(((results['predicted_return'][valid_idx] > 0) == 
                        (results['actual_return'][valid_idx] > 0)).mean() * 100) if valid_idx.sum() > 0 else 0
        
        return {
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'direction_accuracy_pct': direction_acc,
            'total_trades': int(len(returns)),
            'positive_trades': int((returns > 0).sum()),
            'negative_trades': int((returns < 0).sum())
        }
    
    def plot_results(self, results, metrics):
        logger.info("\n" + "="*60)
        logger.info("GENERATING PLOTS")
        logger.info("="*60)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        axes[0].plot(results.index, results['portfolio_value'], linewidth=1.5, color='darkgreen')
        axes[0].set_title(f'Portfolio Value: ${results["portfolio_value"].iloc[-1]:,.0f} (Return: {metrics["total_return_pct"]:.1f}%)', fontsize=14)
        axes[0].set_ylabel('Value ($)')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=10000, color='gray', linestyle='--', alpha=0.5)
        
        returns = results['returns'].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        axes[1].fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        axes[1].plot(drawdown.index, drawdown.values, linewidth=1, color='red')
        axes[1].set_title(f'Drawdown: {metrics["max_drawdown_pct"]:.1f}%', fontsize=14)
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].hist(returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=1)
        axes[2].axvline(x=returns.mean(), color='green', linestyle='--', linewidth=1, label=f'Mean: {returns.mean():.4f}')
        axes[2].set_title(f'Returns Distribution (Win Rate: {metrics["win_rate_pct"]:.1f}%)', fontsize=14)
        axes[2].set_xlabel('Daily Return')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f"backtest_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), dpi=150)
        plt.close()
        logger.info("Saved plots")
    
    def print_summary(self, metrics):
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"\n📈 RETURNS: {metrics['total_return_pct']:.2f}% (Annual: {metrics['annual_return_pct']:.2f}%)")
        logger.info(f"⚠️ RISK: Vol {metrics['volatility_pct']:.2f}%, DD {metrics['max_drawdown_pct']:.2f}%, Sharpe {metrics['sharpe_ratio']:.2f}")
        logger.info(f"🎯 WIN RATE: {metrics['win_rate_pct']:.1f}%, Direction Acc: {metrics['direction_accuracy_pct']:.1f}%")
        logger.info(f"📊 TRADES: {metrics['total_trades']} (Won: {metrics['positive_trades']}, Lost: {metrics['negative_trades']})")
    
    def save_results(self, results, metrics):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(os.path.join(self.results_path, f"backtest_metrics_{timestamp}.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {self.results_path}")
    
    def run(self):
        logger.info("="*60)
        logger.info("WALK-FORWARD BACKTEST")
        logger.info("="*60)
        if not self.load_best_model():
            return
        df = self.load_data()
        if df is None:
            return
        results, metrics = self.run_backtest(df)
        if metrics:
            self.print_summary(metrics)
            self.plot_results(results, metrics)
            self.save_results(results, metrics)
            logger.info("\n✅ BACKTEST COMPLETE")

if __name__ == "__main__":
    backtest = WalkForwardBacktest()
    backtest.run()