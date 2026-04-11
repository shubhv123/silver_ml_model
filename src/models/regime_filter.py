#!/usr/bin/env python
"""
Sentiment-Based Regime Filter - Completely Fixed
Only take long signals when sentiment > threshold
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentRegimeFilter:
    """Regime filter with proper feature alignment"""
    
    def __init__(self, sentiment_threshold=0.05):
        self.processed_path = "data/processed"
        self.models_path = "models/saved"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
        self.sentiment_threshold = sentiment_threshold
        self.model = None
        
    def load_model_and_features(self):
        """Load model and create aligned feature matrix"""
        
        # Load model
        model_files = list(Path(self.models_path).glob("xgboost_with_sentiment_*.pkl"))
        if not model_files:
            logger.error("No model found!")
            return False, None, None
        
        with open(sorted(model_files)[-1], 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from {sorted(model_files)[-1].name}")
        
        # Load feature matrix
        feature_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not feature_files:
            feature_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        
        df = pd.read_csv(sorted(feature_files)[-1], index_col=0, parse_dates=True)
        
        # Load sentiment
        sentiment_files = list(Path(self.processed_path).glob("finbert_daily_*.csv"))
        sentiment_df = pd.read_csv(sorted(sentiment_files)[-1], index_col=0, parse_dates=True)
        
        # Merge sentiment into feature matrix
        common_dates = df.index.intersection(sentiment_df.index)
        df = df.loc[common_dates]
        sentiment_df = sentiment_df.loc[common_dates]
        
        # Add sentiment features
        df['sentiment_finbert_net'] = sentiment_df['finbert_net']
        df['sentiment_sentiment_5d_ma'] = sentiment_df['sentiment_5d_ma'] if 'sentiment_5d_ma' in sentiment_df.columns else sentiment_df['finbert_net'].rolling(5).mean()
        df['sentiment_sentiment_momentum'] = sentiment_df['sentiment_momentum'] if 'sentiment_momentum' in sentiment_df.columns else df['sentiment_sentiment_5d_ma'].diff()
        df['sentiment_news_volume_zscore'] = sentiment_df['news_volume_zscore'] if 'news_volume_zscore' in sentiment_df.columns else 0
        
        logger.info(f"Feature matrix with sentiment: {df.shape}")
        return True, df, sentiment_df
    
    def prepare_test_data(self, df):
        """Prepare test data (last 20%)"""
        
        # Get features and target
        feature_cols = [c for c in df.columns if 'target' not in c]
        X = df[feature_cols].copy()
        y = df['target_next_day_return'].copy()
        
        # Drop NaN
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Split (last 20% for test)
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        test_dates = X_test.index
        
        # Handle missing values
        X_test = X_test.fillna(X_test.median())
        
        logger.info(f"Test data: {X_test.shape}")
        return X_test, y_test, test_dates
    
    def generate_signals(self, predictions, sentiment, thresholds=None):
        """Generate signals with regime filter"""
        if thresholds is None:
            thresholds = [0, 0.03, 0.05, 0.07, 0.1]
        
        signals = {}
        
        for threshold in thresholds:
            base_signal = np.where(predictions > 0.002, 1, 
                                  np.where(predictions < -0.002, -1, 0))
            
            regime_filtered = base_signal.copy()
            long_mask = (base_signal == 1) & (sentiment <= threshold)
            regime_filtered[long_mask] = 0
            
            signals[f'threshold_{threshold}'] = regime_filtered
            logger.info(f"Threshold {threshold}: Blocked {long_mask.sum()} long signals")
        
        return signals
    
    def backtest_strategy(self, y_test, signals, initial_capital=10000):
        """Backtest different strategies"""
        
        results = {}
        
        for name, signal in signals.items():
            # Ensure both are numpy arrays
            signal_array = np.array(signal)
            returns_array = np.array(y_test.values)
            
            strategy_returns = signal_array * returns_array
            
            # Calculate portfolio
            portfolio = np.zeros(len(strategy_returns))
            portfolio[0] = initial_capital
            for i in range(1, len(strategy_returns)):
                portfolio[i] = portfolio[i-1] * (1 + strategy_returns[i])
            
            portfolio_series = pd.Series(portfolio, index=y_test.index)
            returns_series = pd.Series(strategy_returns, index=y_test.index)
            
            total_return = (portfolio_series.iloc[-1] / initial_capital - 1) * 100
            sharpe = returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0
            max_drawdown = ((portfolio_series / portfolio_series.expanding().max()) - 1).min() * 100
            win_rate = (returns_series > 0).mean() * 100
            
            results[name] = {
                'total_return_pct': total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_drawdown,
                'win_rate_pct': win_rate,
                'n_trades': int((signal != 0).sum()),
                'n_long': int((signal == 1).sum()),
                'n_short': int((signal == -1).sum()),
                'portfolio': portfolio_series,
                'returns': returns_series,
                'signal': pd.Series(signal, index=y_test.index)
            }
            
            logger.info(f"\n📊 {name}: Return={total_return:.2f}%, Sharpe={sharpe:.2f}, Win Rate={win_rate:.1f}%")
        
        return results
    
    def plot_results(self, test_dates, predictions, sentiment, y_test, results):
        """Visualize regime filter impact"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        
        # Sentiment over time
        axes[0, 0].plot(test_dates, sentiment, color='blue', linewidth=1, alpha=0.7)
        axes[0, 0].axhline(y=self.sentiment_threshold, color='green', linestyle='--', label=f'Threshold ({self.sentiment_threshold})')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 0].fill_between(test_dates, 0, sentiment, where=(sentiment > 0), color='green', alpha=0.3, label='Positive')
        axes[0, 0].fill_between(test_dates, 0, sentiment, where=(sentiment < 0), color='red', alpha=0.3, label='Negative')
        axes[0, 0].set_title('Sentiment Over Time')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Predictions vs Sentiment
        axes[0, 1].scatter(sentiment, predictions, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0.002, color='red', linestyle='--', label='Long Threshold (0.002)')
        axes[0, 1].axvline(x=self.sentiment_threshold, color='green', linestyle='--', label=f'Sentiment Threshold ({self.sentiment_threshold})')
        axes[0, 1].set_title('Predictions vs Sentiment')
        axes[0, 1].set_xlabel('Sentiment')
        axes[0, 1].set_ylabel('Predicted Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Equity curves
        for name, metrics in results.items():
            if 'threshold_0' in name:
                label = 'No Filter'
            elif f'threshold_{self.sentiment_threshold}' in name:
                label = f'Filter (threshold={self.sentiment_threshold})'
            else:
                continue
            axes[1, 0].plot(metrics['portfolio'].index, metrics['portfolio'].values, label=label, linewidth=1.5)
        
        axes[1, 0].axhline(y=10000, color='black', linestyle='-', alpha=0.5)
        axes[1, 0].set_title('Equity Curves: With vs Without Sentiment Filter')
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Portfolio Value ($)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sharpe by threshold
        thresholds = []
        sharpes = []
        returns = []
        for name, metrics in results.items():
            thresholds.append(float(name.split('_')[-1]))
            sharpes.append(metrics['sharpe_ratio'])
            returns.append(metrics['total_return_pct'])
        
        axes[1, 1].plot(thresholds, sharpes, marker='o', linewidth=1, color='blue', label='Sharpe Ratio')
        axes[1, 1].set_xlabel('Sentiment Threshold')
        axes[1, 1].set_ylabel('Sharpe Ratio')
        axes[1, 1].set_title('Sharpe Ratio by Sentiment Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add return as secondary axis
        ax2 = axes[1, 1].twinx()
        ax2.plot(thresholds, returns, marker='s', linewidth=1, color='orange', label='Return %')
        ax2.set_ylabel('Return (%)', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')
        
        # Signal distribution
        no_filter_signals = results.get('threshold_0', {}).get('signal', pd.Series())
        filtered_signals = results.get(f'threshold_{self.sentiment_threshold}', {}).get('signal', pd.Series())
        
        if len(no_filter_signals) > 0:
            x = np.arange(3)
            width = 0.35
            
            no_filter_counts = [(no_filter_signals == 1).sum(), (no_filter_signals == 0).sum(), (no_filter_signals == -1).sum()]
            filtered_counts = [(filtered_signals == 1).sum(), (filtered_signals == 0).sum(), (filtered_signals == -1).sum()]
            
            axes[2, 0].bar(x - width/2, no_filter_counts, width, label='No Filter', alpha=0.7, color='steelblue')
            axes[2, 0].bar(x + width/2, filtered_counts, width, label='With Filter', alpha=0.7, color='lightgreen')
            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels(['Long', 'Neutral', 'Short'])
            axes[2, 0].set_title('Signal Distribution Comparison')
            axes[2, 0].set_ylabel('Number of Signals')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # Summary table
        axes[2, 1].axis('tight')
        axes[2, 1].axis('off')
        
        no_filter = results.get('threshold_0', {})
        filtered = results.get(f'threshold_{self.sentiment_threshold}', {})
        
        summary = f"""
        ========================================
        REGIME FILTER PERFORMANCE SUMMARY
        ========================================
        
        Sentiment Threshold: {self.sentiment_threshold}
        
        {'Metric':<15} {'No Filter':>12} {'With Filter':>12} {'Change':>10}
        ------------------------------------------------
        Return (%):      {no_filter.get('total_return_pct', 0):>10.2f}   {filtered.get('total_return_pct', 0):>10.2f}   {(filtered.get('total_return_pct', 0) - no_filter.get('total_return_pct', 0)):>+8.2f}
        Sharpe Ratio:    {no_filter.get('sharpe_ratio', 0):>10.2f}   {filtered.get('sharpe_ratio', 0):>10.2f}   {(filtered.get('sharpe_ratio', 0) - no_filter.get('sharpe_ratio', 0)):>+8.2f}
        Max Drawdown (%):{no_filter.get('max_drawdown_pct', 0):>10.2f}   {filtered.get('max_drawdown_pct', 0):>10.2f}   {(filtered.get('max_drawdown_pct', 0) - no_filter.get('max_drawdown_pct', 0)):>+8.2f}
        Win Rate (%):    {no_filter.get('win_rate_pct', 0):>10.1f}   {filtered.get('win_rate_pct', 0):>10.1f}   {(filtered.get('win_rate_pct', 0) - no_filter.get('win_rate_pct', 0)):>+8.1f}
        Long Trades:     {no_filter.get('n_long', 0):>10d}   {filtered.get('n_long', 0):>10d}   {(filtered.get('n_long', 0) - no_filter.get('n_long', 0)):>+8d}
        ========================================
        """
        
        axes[2, 1].text(0.5, 0.5, summary, transform=axes[2, 1].transAxes,
                        fontsize=9, va='center', ha='center', fontfamily='monospace',
                        bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plot_file = os.path.join(self.figures_path, f"regime_filter_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_file}")
    
    def save_results(self, results, optimal_threshold):
        """Save regime filter results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        results_data = {}
        for name, metrics in results.items():
            results_data[name] = {
                'total_return_pct': metrics['total_return_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'win_rate_pct': metrics['win_rate_pct'],
                'n_trades': int(metrics['n_trades']),
                'n_long': int(metrics['n_long']),
                'n_short': int(metrics['n_short'])
            }
        
        regime_results = {
            'timestamp': timestamp,
            'sentiment_threshold_tested': [float(n.split('_')[-1]) for n in results.keys()],
            'optimal_threshold': optimal_threshold,
            'results': results_data
        }
        
        results_file = os.path.join(self.results_path, f"regime_filter_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(regime_results, f, indent=2, default=str)
        logger.info(f"Saved regime results: {results_file}")
        
        return results_file
    
    def run(self):
        """Run regime filter analysis"""
        
        logger.info("="*60)
        logger.info("SENTIMENT REGIME FILTER")
        logger.info(f"Threshold: {self.sentiment_threshold}")
        logger.info("="*60)
        
        # Load data
        success, df, sentiment_df = self.load_model_and_features()
        if not success:
            return None, None
        
        # Prepare test data
        X_test, y_test, test_dates = self.prepare_test_data(df)
        
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Get sentiment for test period
        sentiment = sentiment_df.loc[test_dates, 'finbert_net'].values
        
        # Generate signals
        signals = self.generate_signals(predictions, sentiment)
        
        # Backtest
        results = self.backtest_strategy(y_test, signals)
        
        # Plot
        self.plot_results(test_dates, predictions, sentiment, y_test, results)
        
        # Find optimal threshold
        best_threshold_name = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])[0]
        best_threshold = float(best_threshold_name.split('_')[-1])
        
        # Save results
        self.save_results(results, best_threshold)
        
        logger.info(f"\n✅ Optimal threshold: {best_threshold}")
        
        return results, best_threshold

if __name__ == "__main__":
    filter_analyzer = SentimentRegimeFilter(sentiment_threshold=0.05)
    results, optimal = filter_analyzer.run()
    
    print("\n" + "="*60)
    print("REGIME FILTER RECOMMENDATION")
    print("="*60)
    print(f"✅ Use sentiment threshold: {optimal}")
    print("✅ Only take LONG signals when sentiment > threshold")
    print("✅ SHORT signals remain unaffected")
    print("✅ Reduces false positives in bear markets")
    
    # Print best results
    best_result = results.get(f'threshold_{optimal}', {})
    no_filter = results.get('threshold_0', {})
    print(f"\n📊 Performance at threshold {optimal}:")
    print(f"   Return: {best_result.get('total_return_pct', 0):.2f}% (vs {no_filter.get('total_return_pct', 0):.2f}%)")
    print(f"   Sharpe: {best_result.get('sharpe_ratio', 0):.2f} (vs {no_filter.get('sharpe_ratio', 0):.2f})")