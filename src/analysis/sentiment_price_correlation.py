#!/usr/bin/env python
"""
Correlate Sentiment Features with Silver Returns
Check if sentiment leads price by 1-3 days
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPriceCorrelation:
    """Analyze correlation between sentiment and silver returns"""
    
    def __init__(self):
        self.processed_path = "data/processed"
        self.raw_path = "data/raw"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
    def load_data(self):
        """Load sentiment and price data"""
        
        # Load sentiment data
        sentiment_files = list(Path(self.processed_path).glob("finbert_daily_*.csv"))
        if not sentiment_files:
            logger.error("No sentiment files found!")
            return None, None
        
        sentiment_df = pd.read_csv(sorted(sentiment_files)[-1], index_col=0, parse_dates=True)
        logger.info(f"Loaded sentiment data: {sentiment_df.shape}")
        
        # Load silver price data
        price_files = list(Path(self.raw_path).glob("SIF_daily.csv"))
        if not price_files:
            logger.error("No silver price data found!")
            return sentiment_df, None
        
        price_df = pd.read_csv(price_files[0], parse_dates=['date'])
        price_df.set_index('date', inplace=True)
        price_df.columns = [col.lower() for col in price_df.columns]
        logger.info(f"Loaded price data: {price_df.shape}")
        
        return sentiment_df, price_df
    
    def calculate_returns(self, price_df):
        """Calculate silver returns"""
        
        returns = pd.DataFrame(index=price_df.index)
        returns['silver_return'] = price_df['close'].pct_change()
        returns['silver_return_1d'] = returns['silver_return']
        returns['silver_return_2d'] = price_df['close'].pct_change(2)
        returns['silver_return_3d'] = price_df['close'].pct_change(3)
        returns['silver_return_5d'] = price_df['close'].pct_change(5)
        returns['silver_return_21d'] = price_df['close'].pct_change(21)
        
        # Log returns
        returns['silver_log_return'] = np.log(price_df['close'] / price_df['close'].shift(1))
        
        return returns
    
    def align_data(self, sentiment_df, returns_df):
        """Align sentiment and returns data"""
        
        # Find common dates
        common_dates = sentiment_df.index.intersection(returns_df.index)
        sentiment_aligned = sentiment_df.loc[common_dates]
        returns_aligned = returns_df.loc[common_dates]
        
        logger.info(f"Aligned data: {len(common_dates)} common days")
        logger.info(f"Date range: {common_dates.min()} to {common_dates.max()}")
        
        return sentiment_aligned, returns_aligned
    
    def calculate_lead_lag_correlations(self, sentiment_df, returns_df, max_lag=5):
        """Calculate lead-lag correlations"""
        
        logger.info("\n" + "="*60)
        logger.info("LEAD-LAG CORRELATION ANALYSIS")
        logger.info("="*60)
        
        sentiment_cols = ['finbert_net', 'sentiment_5d_ma', 'sentiment_momentum', 'news_volume_zscore']
        results = {}
        
        for sent_col in sentiment_cols:
            if sent_col not in sentiment_df.columns:
                continue
            
            correlations = {}
            for lag in range(-max_lag, max_lag + 1):
                if lag < 0:
                    # Sentiment leads price (negative lag = sentiment earlier)
                    shifted_sentiment = sentiment_df[sent_col].shift(-lag)
                    aligned_sentiment = shifted_sentiment.loc[returns_df.index]
                    aligned_returns = returns_df['silver_return']
                else:
                    # Price leads sentiment or concurrent
                    aligned_sentiment = sentiment_df[sent_col]
                    aligned_returns = returns_df['silver_return'].shift(lag)
                
                # Drop NaN
                mask = ~(aligned_sentiment.isna() | aligned_returns.isna())
                if mask.sum() > 10:
                    corr, p_value = pearsonr(aligned_sentiment[mask], aligned_returns[mask])
                    correlations[lag] = {'correlation': corr, 'p_value': p_value, 'samples': mask.sum()}
            
            results[sent_col] = correlations
            
            # Print results
            logger.info(f"\n📊 {sent_col.upper()}:")
            for lag in sorted(correlations.keys()):
                if lag < 0:
                    label = f"Sentiment leads by {-lag} day(s)"
                elif lag == 0:
                    label = "Same day"
                else:
                    label = f"Sentiment lags by {lag} day(s)"
                
                corr = correlations[lag]['correlation']
                p_val = correlations[lag]['p_value']
                sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.1 else ""
                logger.info(f"   {label:25s}: {corr:+.4f} {sig} (p={p_val:.4f})")
        
        return results
    
    def calculate_rolling_correlations(self, sentiment_df, returns_df, window=63):
        """Calculate rolling correlations over time"""
        
        logger.info("\n" + "="*60)
        logger.info("ROLLING CORRELATION ANALYSIS")
        logger.info(f"Window: {window} days")
        logger.info("="*60)
        
        rolling_corr = pd.DataFrame(index=sentiment_df.index)
        
        sentiment_cols = ['finbert_net', 'sentiment_5d_ma', 'sentiment_momentum']
        
        for sent_col in sentiment_cols:
            if sent_col in sentiment_df.columns:
                # Align data
                common_idx = sentiment_df.index.intersection(returns_df.index)
                sent_aligned = sentiment_df.loc[common_idx, sent_col]
                ret_aligned = returns_df.loc[common_idx, 'silver_return']
                
                # Calculate rolling correlation
                rolling_corr[f'{sent_col}_corr'] = sent_aligned.rolling(window).corr(ret_aligned)
                
                logger.info(f"   {sent_col}: mean={rolling_corr[f'{sent_col}_corr'].mean():.4f}, "
                          f"std={rolling_corr[f'{sent_col}_corr'].std():.4f}")
        
        return rolling_corr
    
    def test_causality(self, sentiment_df, returns_df, max_lag=3):
        """Simple Granger causality test (using correlation patterns)"""
        
        logger.info("\n" + "="*60)
        logger.info("CAUSALITY PATTERN ANALYSIS")
        logger.info("="*60)
        
        sentiment_col = 'finbert_net'
        if sentiment_col not in sentiment_df.columns:
            return {}
        
        results = {}
        
        # Test if sentiment predicts returns
        for lag in range(1, max_lag + 1):
            shifted_sentiment = sentiment_df[sentiment_col].shift(lag)
            aligned_sentiment = shifted_sentiment.loc[returns_df.index]
            aligned_returns = returns_df['silver_return']
            
            mask = ~(aligned_sentiment.isna() | aligned_returns.isna())
            if mask.sum() > 10:
                corr, p_value = pearsonr(aligned_sentiment[mask], aligned_returns[mask])
                results[f'Sentiment -> Return (lag {lag})'] = {'correlation': corr, 'p_value': p_value}
        
        # Test if returns predict sentiment (reverse)
        for lag in range(1, max_lag + 1):
            shifted_returns = returns_df['silver_return'].shift(lag)
            aligned_returns = shifted_returns.loc[sentiment_df.index]
            aligned_sentiment = sentiment_df[sentiment_col]
            
            mask = ~(aligned_returns.isna() | aligned_sentiment.isna())
            if mask.sum() > 10:
                corr, p_value = pearsonr(aligned_returns[mask], aligned_sentiment[mask])
                results[f'Return -> Sentiment (lag {lag})'] = {'correlation': corr, 'p_value': p_value}
        
        # Print results
        logger.info(f"\n📊 Sentiment: {sentiment_col}")
        for test, vals in results.items():
            sig = "***" if vals['p_value'] < 0.01 else "**" if vals['p_value'] < 0.05 else "*" if vals['p_value'] < 0.1 else ""
            logger.info(f"   {test}: {vals['correlation']:+.4f} {sig}")
        
        # Determine direction
        sent_to_return = np.mean([v['correlation'] for k, v in results.items() if 'Sentiment -> Return' in k]) if any('Sentiment -> Return' in k for k in results) else 0
        return_to_sent = np.mean([v['correlation'] for k, v in results.items() if 'Return -> Sentiment' in k]) if any('Return -> Sentiment' in k for k in results) else 0
        
        logger.info(f"\n�� SUMMARY:")
        logger.info(f"   Sentiment → Return (avg): {sent_to_return:+.4f}")
        logger.info(f"   Return → Sentiment (avg): {return_to_sent:+.4f}")
        
        if sent_to_return > return_to_sent:
            logger.info(f"   ✅ Sentiment LEADS price (stronger predictor)")
        elif return_to_sent > sent_to_return:
            logger.info(f"   ⚠️ Price LEADS sentiment (sentiment reacts to price)")
        else:
            logger.info(f"   🤝 Mutual relationship")
        
        return results
    
    def plot_correlations(self, lead_lag_results, rolling_corr, sentiment_df, returns_df):
        """Create correlation visualizations"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Lead-lag correlation heatmap
        sent_col = 'finbert_net'
        if sent_col in lead_lag_results:
            lags = sorted(lead_lag_results[sent_col].keys())
            corrs = [lead_lag_results[sent_col][lag]['correlation'] for lag in lags]
            
            colors = ['red' if c < 0 else 'green' for c in corrs]
            axes[0, 0].bar(range(len(lags)), corrs, color=colors, alpha=0.7)
            axes[0, 0].set_xticks(range(len(lags)))
            axes[0, 0].set_xticklabels([f"{lag}" for lag in lags])
            axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0, 0].set_title('Lead-Lag Correlation: Sentiment vs Returns', fontsize=12)
            axes[0, 0].set_xlabel('Lag (negative = sentiment leads)')
            axes[0, 0].set_ylabel('Correlation')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add best lag annotation
            best_lag = lags[np.argmax(np.abs(corrs))]
            axes[0, 0].annotate(f'Best: lag={best_lag}', xy=(lags.index(best_lag), corrs[np.argmax(np.abs(corrs))]),
                               xytext=(10, 10), textcoords='offset points', fontsize=9, fontweight='bold')
        
        # 2. Rolling correlation over time
        if rolling_corr is not None and len(rolling_corr) > 0:
            for col in rolling_corr.columns:
                axes[0, 1].plot(rolling_corr.index, rolling_corr[col], label=col.replace('_corr', ''), linewidth=1)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[0, 1].set_title('Rolling 63-Day Correlation', fontsize=12)
            axes[0, 1].set_xlabel('Date')
            axes[0, 1].set_ylabel('Correlation')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Sentiment vs Returns Scatter
        common_idx = sentiment_df.index.intersection(returns_df.index)
        if len(common_idx) > 0:
            sent_vals = sentiment_df.loc[common_idx, 'finbert_net']
            ret_vals = returns_df.loc[common_idx, 'silver_return']
            
            axes[0, 2].scatter(sent_vals, ret_vals, alpha=0.5, s=10)
            
            # Add regression line
            mask = ~(sent_vals.isna() | ret_vals.isna())
            if mask.sum() > 10:
                z = np.polyfit(sent_vals[mask], ret_vals[mask], 1)
                p = np.poly1d(z)
                axes[0, 2].plot(sent_vals[mask].sort_values(), p(sent_vals[mask].sort_values()), 
                              "r--", alpha=0.8, label=f'R² = {z[0]**2:.3f}')
            
            axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 2].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            axes[0, 2].set_title('Sentiment vs Next Day Returns', fontsize=12)
            axes[0, 2].set_xlabel('Net Sentiment')
            axes[0, 2].set_ylabel('Next Day Return')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Sentiment and Price Overlay
        if len(common_idx) > 0:
            ax4 = axes[1, 0]
            ax4_twin = ax4.twinx()
            
            # Normalize for visualization
            sent_norm = (sentiment_df.loc[common_idx, 'finbert_net'] - sentiment_df.loc[common_idx, 'finbert_net'].mean()) / sentiment_df.loc[common_idx, 'finbert_net'].std()
            
            ax4.plot(common_idx, sent_norm, color='blue', linewidth=1, alpha=0.7, label='Sentiment (norm)')
            ax4.set_ylabel('Sentiment (z-score)', color='blue')
            ax4.tick_params(axis='y', labelcolor='blue')
            
            ax4_twin.plot(common_idx, returns_df.loc[common_idx, 'silver_return'].cumsum(), 
                         color='orange', linewidth=1, alpha=0.7, label='Cumulative Returns')
            ax4_twin.set_ylabel('Cumulative Return', color='orange')
            ax4_twin.tick_params(axis='y', labelcolor='orange')
            
            ax4.set_title('Sentiment vs Cumulative Returns', fontsize=12)
            ax4.grid(True, alpha=0.3)
        
        # 5. Lagged Correlation by Day
        sent_col = 'finbert_net'
        if sent_col in lead_lag_results:
            lags = sorted(lead_lag_results[sent_col].keys())
            pos_lags = [l for l in lags if l >= 0]
            neg_lags = [l for l in lags if l < 0]
            
            pos_corrs = [lead_lag_results[sent_col][l]['correlation'] for l in pos_lags]
            neg_corrs = [lead_lag_results[sent_col][l]['correlation'] for l in neg_lags]
            
            axes[1, 1].bar(range(len(pos_lags)), pos_corrs, alpha=0.7, color='orange', label='Sentiment lags')
            axes[1, 1].bar(range(-len(neg_lags), 0), neg_corrs, alpha=0.7, color='blue', label='Sentiment leads')
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[1, 1].set_title('Correlation by Day (Sentiment Leads = Negative)', fontsize=12)
            axes[1, 1].set_xlabel('Days')
            axes[1, 1].set_ylabel('Correlation')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Feature Correlation Matrix
        feature_cols = ['finbert_net', 'silver_return', 'silver_return_2d', 'silver_return_3d']
        available_cols = [c for c in feature_cols if c in returns_df.columns or c in sentiment_df.columns]
        
        combined_df = pd.DataFrame(index=common_idx)
        for col in available_cols:
            if col in sentiment_df.columns:
                combined_df[col] = sentiment_df[col]
            elif col in returns_df.columns:
                combined_df[col] = returns_df[col]
        
        if len(combined_df.columns) > 1:
            corr_matrix = combined_df.corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0, 
                       ax=axes[1, 2], square=True)
            axes[1, 2].set_title('Feature Correlation Matrix', fontsize=12)
        
        plt.tight_layout()
        plot_file = os.path.join(self.figures_path, f"sentiment_price_correlation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation plot: {plot_file}")
    
    def save_results(self, lead_lag_results, rolling_corr, causality_results):
        """Save correlation analysis results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save lead-lag results
        lead_lag_data = {}
        for sent_col, lags in lead_lag_results.items():
            lead_lag_data[sent_col] = {str(lag): vals for lag, vals in lags.items()}
        
        results = {
            'timestamp': timestamp,
            'lead_lag_correlations': lead_lag_data,
            'causality_analysis': causality_results,
            'rolling_correlation_mean': {col: float(rolling_corr[col].mean()) for col in rolling_corr.columns} if rolling_corr is not None else {}
        }
        
        results_file = os.path.join(self.results_path, f"sentiment_correlation_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved correlation results: {results_file}")
        
        return results_file
    
    def run(self):
        """Run complete correlation analysis"""
        
        logger.info("="*60)
        logger.info("SENTIMENT-PRICE CORRELATION ANALYSIS")
        logger.info("="*60)
        
        # Load data
        sentiment_df, price_df = self.load_data()
        if sentiment_df is None or price_df is None:
            return
        
        # Calculate returns
        returns_df = self.calculate_returns(price_df)
        
        # Align data
        sentiment_aligned, returns_aligned = self.align_data(sentiment_df, returns_df)
        
        # Calculate lead-lag correlations
        lead_lag_results = self.calculate_lead_lag_correlations(sentiment_aligned, returns_aligned, max_lag=5)
        
        # Calculate rolling correlations
        rolling_corr = self.calculate_rolling_correlations(sentiment_aligned, returns_aligned, window=63)
        
        # Test causality patterns
        causality_results = self.test_causality(sentiment_aligned, returns_aligned, max_lag=3)
        
        # Create plots
        self.plot_correlations(lead_lag_results, rolling_corr, sentiment_aligned, returns_aligned)
        
        # Save results
        self.save_results(lead_lag_results, rolling_corr, causality_results)
        
        logger.info("\n" + "="*60)
        logger.info("✅ CORRELATION ANALYSIS COMPLETE")
        logger.info("="*60)
        
        return lead_lag_results, rolling_corr

if __name__ == "__main__":
    analyzer = SentimentPriceCorrelation()
    results, rolling = analyzer.run()
    
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)
    if 'finbert_net' in results:
        best_lag = max(results['finbert_net'].items(), key=lambda x: abs(x[1]['correlation']))
        print(f"Best lead-lag: lag={best_lag[0]} days, correlation={best_lag[1]['correlation']:.4f}")
        
        if best_lag[0] < 0:
            print(f"✅ Sentiment LEADS price by {abs(best_lag[0])} day(s)")
        elif best_lag[0] == 0:
            print(f"🤝 Sentiment and price move together")
        else:
            print(f"⚠️ Price LEADS sentiment by {best_lag[0]} day(s)")
