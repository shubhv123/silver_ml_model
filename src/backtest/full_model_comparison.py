#!/usr/bin/env python
"""
Complete Model Backtest - Retrain all models on same feature set
Compares: XGBoost (with/without sentiment), LightGBM, CatBoost, Ensemble
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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CompleteModelBacktest:
    """Retrain and backtest all models on same feature set"""
    
    def __init__(self):
        self.processed_path = "data/processed"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load feature matrix with sentiment"""
        
        # Load base feature matrix
        feature_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not feature_files:
            feature_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        
        df = pd.read_csv(sorted(feature_files)[-1], index_col=0, parse_dates=True)
        
        # Load sentiment
        sentiment_files = list(Path(self.processed_path).glob("finbert_daily_*.csv"))
        if sentiment_files:
            sentiment_df = pd.read_csv(sorted(sentiment_files)[-1], index_col=0, parse_dates=True)
            common_dates = df.index.intersection(sentiment_df.index)
            df = df.loc[common_dates]
            sentiment_df = sentiment_df.loc[common_dates]
            df['sentiment_net'] = sentiment_df['finbert_net']
            logger.info("Added sentiment feature")
        
        logger.info(f"Feature matrix: {df.shape}")
        return df
    
    def prepare_train_test(self, df, test_size=0.2):
        """Prepare train/test split with walk-forward"""
        
        # Create two versions: with and without sentiment
        feature_cols_no_sent = [c for c in df.columns if 'target' not in c and c != 'sentiment_net']
        feature_cols_with_sent = [c for c in df.columns if 'target' not in c]
        
        X_no_sent = df[feature_cols_no_sent].copy()
        X_with_sent = df[feature_cols_with_sent].copy()
        y = df['target_next_day_return'].copy()
        
        # Remove NaN
        valid_idx = ~y.isna()
        X_no_sent = X_no_sent[valid_idx]
        X_with_sent = X_with_sent[valid_idx]
        y = y[valid_idx]
        
        # Split
        split_idx = int(len(X_no_sent) * (1 - test_size))
        
        # Without sentiment
        X_train_no = X_no_sent.iloc[:split_idx]
        X_test_no = X_no_sent.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        test_dates = X_no_sent.index[split_idx:]
        
        # With sentiment
        X_train_with = X_with_sent.iloc[:split_idx]
        X_test_with = X_with_sent.iloc[split_idx:]
        
        # Handle missing values
        X_train_no = X_train_no.fillna(X_train_no.median())
        X_test_no = X_test_no.fillna(X_train_no.median())
        X_train_with = X_train_with.fillna(X_train_with.median())
        X_test_with = X_test_with.fillna(X_train_with.median())
        
        # Scale
        X_train_no_scaled = self.scaler.fit_transform(X_train_no)
        X_test_no_scaled = self.scaler.transform(X_test_no)
        X_train_with_scaled = self.scaler.fit_transform(X_train_with)
        X_test_with_scaled = self.scaler.transform(X_test_with)
        
        logger.info(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
        logger.info(f"Test dates: {test_dates[0].date()} to {test_dates[-1].date()}")
        
        return (X_train_no_scaled, X_test_no_scaled, 
                X_train_with_scaled, X_test_with_scaled, 
                y_train, y_test, test_dates)
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost"""
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM"""
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        return model
    
    def train_catboost(self, X_train, y_train):
        """Train CatBoost"""
        model = cb.CatBoostRegressor(
            iterations=200,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=False
        )
        model.fit(X_train, y_train)
        return model
    
    def train_ensemble(self, models, X_train, y_train):
        """Train ensemble meta-model"""
        # Get base model predictions
        predictions = []
        for model in models:
            pred = model.predict(X_train)
            predictions.append(pred.reshape(-1, 1))
        
        meta_features = np.hstack(predictions)
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_features, y_train)
        return meta_model
    
    def backtest_model(self, model, X_test, y_test, test_dates, model_name):
        """Run backtest for a single model"""
        
        predictions = model.predict(X_test)
        
        # Trading simulation
        signal_threshold = 0.002
        signals = np.where(predictions > signal_threshold, 1,
                          np.where(predictions < -signal_threshold, -1, 0))
        strategy_returns = signals * y_test.values
        
        portfolio = 10000 * (1 + strategy_returns).cumprod()
        total_return = (portfolio[-1] / 10000 - 1) * 100
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0
        
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        win_rate = (strategy_returns > 0).mean() * 100
        direction_acc = (np.sign(y_test) == np.sign(predictions)).mean()
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        
        return {
            'model': model_name,
            'rmse': rmse,
            'direction_acc': direction_acc,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': (signals != 0).sum(),
            'portfolio': portfolio,
            'predictions': predictions,
            'dates': test_dates
        }
    
    def run_backtest(self, X_train_no, X_test_no, X_train_with, X_test_with, y_train, y_test, test_dates):
        """Run complete backtest for all models"""
        
        results = []
        
        # 1. XGBoost without sentiment
        logger.info("\n1. Training XGBoost (No Sentiment)...")
        xgb_no = self.train_xgboost(X_train_no, y_train)
        results.append(self.backtest_model(xgb_no, X_test_no, y_test, test_dates, "XGBoost (No Sentiment)"))
        
        # 2. XGBoost with sentiment
        logger.info("2. Training XGBoost (With Sentiment)...")
        xgb_with = self.train_xgboost(X_train_with, y_train)
        results.append(self.backtest_model(xgb_with, X_test_with, y_test, test_dates, "XGBoost + Sentiment"))
        
        # 3. LightGBM with sentiment
        logger.info("3. Training LightGBM...")
        lgb_model = self.train_lightgbm(X_train_with, y_train)
        results.append(self.backtest_model(lgb_model, X_test_with, y_test, test_dates, "LightGBM"))
        
        # 4. CatBoost with sentiment
        logger.info("4. Training CatBoost...")
        cb_model = self.train_catboost(X_train_with, y_train)
        results.append(self.backtest_model(cb_model, X_test_with, y_test, test_dates, "CatBoost"))
        
        # 5. Ensemble (XGBoost + LightGBM + CatBoost)
        logger.info("5. Training Ensemble...")
        base_models = [xgb_with, lgb_model, cb_model]
        ensemble = self.train_ensemble(base_models, X_train_with, y_train)
        
        # Get ensemble predictions
        ensemble_pred = []
        for model in base_models:
            ensemble_pred.append(model.predict(X_test_with).reshape(-1, 1))
        ensemble_pred = np.hstack(ensemble_pred)
        ensemble_predictions = ensemble.predict(ensemble_pred)
        
        # Manual backtest for ensemble
        signals = np.where(ensemble_predictions > 0.002, 1,
                          np.where(ensemble_predictions < -0.002, -1, 0))
        strategy_returns = signals * y_test.values
        portfolio = 10000 * (1 + strategy_returns).cumprod()
        
        results.append({
            'model': 'Ensemble (XGB+LGB+CB)',
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_predictions)),
            'direction_acc': (np.sign(y_test) == np.sign(ensemble_predictions)).mean(),
            'total_return': (portfolio[-1] / 10000 - 1) * 100,
            'sharpe': strategy_returns.mean() / strategy_returns.std() * np.sqrt(252) if strategy_returns.std() > 0 else 0,
            'max_drawdown': ((portfolio / np.maximum.accumulate(portfolio)) - 1).min() * 100,
            'win_rate': (strategy_returns > 0).mean() * 100,
            'n_trades': (signals != 0).sum(),
            'portfolio': portfolio,
            'predictions': ensemble_predictions,
            'dates': test_dates
        })
        
        return results
    
    def plot_results(self, results):
        """Create comprehensive comparison plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        models = [r['model'] for r in results]
        
        # 1. Sharpe Ratio
        sharpe_values = [r['sharpe'] for r in results]
        colors = ['#2ecc71' if 'Sentiment' in m else '#3498db' for m in models]
        bars = axes[0, 0].bar(models, sharpe_values, color=colors, alpha=0.7)
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 0].axhline(y=1, color='green', linestyle='--', label='Target (1.0)')
        axes[0, 0].set_title('Sharpe Ratio (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        for bar, val in zip(bars, sharpe_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Total Return
        return_values = [r['total_return'] for r in results]
        bars2 = axes[0, 1].bar(models, return_values, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('Total Return (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        for bar, val in zip(bars2, return_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 3. Directional Accuracy
        dir_values = [r['direction_acc'] * 100 for r in results]
        bars3 = axes[0, 2].bar(models, dir_values, color=colors, alpha=0.7)
        axes[0, 2].axhline(y=50, color='red', linestyle='--', label='Random (50%)')
        axes[0, 2].set_title('Directional Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0, 2].set_ylabel('Accuracy (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        for bar, val in zip(bars3, dir_values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        # 4. Equity Curves
        for r in results:
            axes[1, 0].plot(r['portfolio'], label=r['model'], linewidth=1.5)
        axes[1, 0].axhline(y=10000, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('Equity Curves', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Trading Days')
        axes[1, 0].set_ylabel('Portfolio Value ($)')
        axes[1, 0].legend(loc='upper left', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. RMSE Comparison
        rmse_values = [r['rmse'] for r in results]
        bars4 = axes[1, 1].bar(models, rmse_values, color=colors, alpha=0.7)
        axes[1, 1].set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        for bar, val in zip(bars4, rmse_values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                           f'{val:.5f}', ha='center', va='bottom', fontsize=9)
        
        # 6. Win Rate
        win_values = [r['win_rate'] for r in results]
        bars5 = axes[1, 2].bar(models, win_values, color=colors, alpha=0.7)
        axes[1, 2].set_title('Win Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 2].set_ylabel('Win Rate (%)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        for bar, val in zip(bars5, win_values):
            axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_file = os.path.join(self.figures_path, f"complete_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved comparison plot: {plot_file}")
    
    def save_results(self, results):
        """Save backtest results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create DataFrame
        df_results = pd.DataFrame([{
            'Model': r['model'],
            'RMSE': r['rmse'],
            'Directional_Accuracy': f"{r['direction_acc']:.2%}",
            'Total_Return_%': r['total_return'],
            'Sharpe_Ratio': r['sharpe'],
            'Max_Drawdown_%': r['max_drawdown'],
            'Win_Rate_%': r['win_rate'],
            'Num_Trades': r['n_trades']
        } for r in results])
        
        # Sort by Sharpe Ratio
        df_results = df_results.sort_values('Sharpe_Ratio', ascending=False)
        
        csv_file = os.path.join(self.results_path, f"complete_backtest_{timestamp}.csv")
        df_results.to_csv(csv_file, index=False)
        logger.info(f"Saved results to {csv_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("COMPLETE MODEL BACKTEST RESULTS")
        print("="*80)
        print(df_results.to_string(index=False))
        
        print("\n" + "="*80)
        print("WINNER ANALYSIS")
        print("="*80)
        
        best_sharpe = df_results.iloc[0]
        best_return = df_results.loc[df_results['Total_Return_%'].idxmax()]
        best_dir = df_results.loc[df_results['Directional_Accuracy'].str.rstrip('%').astype(float).idxmax()]
        
        print(f"\n🏆 Best Sharpe Ratio: {best_sharpe['Model']} ({best_sharpe['Sharpe_Ratio']:.2f})")
        print(f"📈 Best Return: {best_return['Model']} ({best_return['Total_Return_%']:.2f}%)")
        print(f"🎯 Best Directional Accuracy: {best_dir['Model']} ({best_dir['Directional_Accuracy']})")
        
        # Sentiment impact
        xgb_no = next((r for r in results if r['model'] == 'XGBoost (No Sentiment)'), None)
        xgb_with = next((r for r in results if r['model'] == 'XGBoost + Sentiment'), None)
        
        if xgb_no and xgb_with:
            impr = (xgb_with['sharpe'] - xgb_no['sharpe']) / abs(xgb_no['sharpe']) * 100 if xgb_no['sharpe'] != 0 else 0
            print(f"\n📊 Sentiment Impact: {impr:+.1f}% change in Sharpe Ratio")
            if impr > 0:
                print(f"✅ Sentiment features IMPROVED performance!")
            else:
                print(f"⚠️ Sentiment needs more data")
        
        return df_results
    
    def run(self):
        """Run complete backtest"""
        
        logger.info("="*60)
        logger.info("COMPLETE MODEL BACKTEST")
        logger.info("Retraining all models on same feature set")
        logger.info("="*60)
        
        # Load data
        df = self.load_data()
        
        # Prepare train/test
        (X_train_no, X_test_no, X_train_with, X_test_with, 
         y_train, y_test, test_dates) = self.prepare_train_test(df)
        
        # Run backtest
        results = self.run_backtest(X_train_no, X_test_no, X_train_with, X_test_with, 
                                    y_train, y_test, test_dates)
        
        # Plot results
        self.plot_results(results)
        
        # Save results
        df_results = self.save_results(results)
        
        logger.info("\n✅ COMPLETE BACKTEST FINISHED")
        
        return results

if __name__ == "__main__":
    backtest = CompleteModelBacktest()
    results = backtest.run()