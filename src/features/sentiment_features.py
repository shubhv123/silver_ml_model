#!/usr/bin/env python
"""
Engineer Sentiment Features for ML Model
- Daily sentiment mean
- Rolling 5-day MA
- Sentiment momentum
- News volume (spikes matter)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentFeatureEngineer:
    """Engineer sentiment features for price prediction"""
    
    def __init__(self):
        self.processed_path = "data/processed"
        self.raw_path = "data/raw/news"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
    def load_sentiment_data(self):
        """Load FinBERT sentiment data"""
        
        # Load daily FinBERT sentiment
        finbert_files = list(Path(self.processed_path).glob("finbert_daily_*.csv"))
        
        if not finbert_files:
            logger.error("No FinBERT daily files found!")
            return None
        
        latest = sorted(finbert_files)[-1]
        df = pd.read_csv(latest, index_col=0, parse_dates=True)
        logger.info(f"Loaded FinBERT daily sentiment: {latest.name} ({len(df)} days)")
        
        return df
    
    def load_price_data(self):
        """Load silver price data for correlation analysis"""
        
        price_files = list(Path("data/raw").glob("SIF_daily.csv"))
        if price_files:
            df = pd.read_csv(price_files[0], parse_dates=['date'])
            df.set_index('date', inplace=True)
            df.columns = [col.lower() for col in df.columns]
            logger.info(f"Loaded silver price data: {df.shape}")
            return df
        return None
    
    def engineer_sentiment_features(self, daily_sentiment):
        """Engineer sentiment features from daily data"""
        
        logger.info("\n" + "="*60)
        logger.info("ENGINEERING SENTIMENT FEATURES")
        logger.info("="*60)
        
        features = pd.DataFrame(index=daily_sentiment.index)
        
        # 1. Daily sentiment mean (already have finbert_net)
        features['sentiment_daily'] = daily_sentiment['finbert_net']
        
        # 2. Rolling 5-day moving average
        features['sentiment_5d_ma'] = daily_sentiment['finbert_net'].rolling(5).mean()
        
        # 3. Rolling 21-day moving average (monthly)
        features['sentiment_21d_ma'] = daily_sentiment['finbert_net'].rolling(21).mean()
        
        # 4. Sentiment momentum (5d MA - 21d MA)
        features['sentiment_momentum'] = features['sentiment_5d_ma'] - features['sentiment_21d_ma']
        
        # 5. Sentiment acceleration (momentum change)
        features['sentiment_acceleration'] = features['sentiment_momentum'].diff()
        
        # 6. News volume (number of articles per day)
        features['news_volume'] = daily_sentiment.get('finbert_positive_mean', 0) * 100  # Approximate
        
        # If we have actual news count, use it
        if 'finbert_positive_mean' in daily_sentiment.columns:
            # Estimate volume from sentiment scores (higher volume = more news)
            features['news_volume'] = (daily_sentiment['finbert_positive_mean'] + 
                                       daily_sentiment['finbert_neutral_mean'] + 
                                       daily_sentiment['finbert_negative_mean']) * 100
        
        # 7. News volume z-score (spikes detection)
        features['news_volume_zscore'] = (
            (features['news_volume'] - features['news_volume'].rolling(21).mean()) / 
            features['news_volume'].rolling(21).std()
        )
        
        # 8. Sentiment extremes (overbought/oversold)
        features['sentiment_extreme'] = 0
        features.loc[features['sentiment_daily'] > 0.3, 'sentiment_extreme'] = 1  # Very positive
        features.loc[features['sentiment_daily'] < -0.3, 'sentiment_extreme'] = -1  # Very negative
        
        # 9. Sentiment volatility (std of last 5 days)
        features['sentiment_volatility'] = daily_sentiment['finbert_net'].rolling(5).std()
        
        # 10. Positive vs negative ratio
        if 'finbert_positive_mean' in daily_sentiment.columns and 'finbert_negative_mean' in daily_sentiment.columns:
            features['pos_neg_ratio'] = (
                daily_sentiment['finbert_positive_mean'] / 
                (daily_sentiment['finbert_negative_mean'] + 0.001)
            )
            features['pos_neg_ratio_log'] = np.log1p(features['pos_neg_ratio'])
        
        # 11. Confidence-weighted sentiment
        if 'finbert_confidence_mean' in daily_sentiment.columns:
            features['sentiment_weighted'] = (
                daily_sentiment['finbert_net'] * daily_sentiment['finbert_confidence_mean']
            )
        
        # 12. Sentiment regime (bullish/bearish/neutral)
        features['sentiment_regime'] = 'neutral'
        features.loc[features['sentiment_5d_ma'] > 0.1, 'sentiment_regime'] = 'bullish'
        features.loc[features['sentiment_5d_ma'] < -0.1, 'sentiment_regime'] = 'bearish'
        
        # 13. Rolling quantiles
        features['sentiment_5d_max'] = daily_sentiment['finbert_net'].rolling(5).max()
        features['sentiment_5d_min'] = daily_sentiment['finbert_net'].rolling(5).min()
        features['sentiment_5d_range'] = features['sentiment_5d_max'] - features['sentiment_5d_min']
        
        # 14. Week-over-week change
        features['sentiment_wow'] = daily_sentiment['finbert_net'].diff(5)
        
        # 15. Day of week effects
        features['day_of_week'] = daily_sentiment.index.dayofweek
        features['is_monday'] = (features['day_of_week'] == 0).astype(int)
        features['is_friday'] = (features['day_of_week'] == 4).astype(int)
        
        # Log all engineered features
        logger.info(f"\n📊 ENGINEERED FEATURES:")
        logger.info(f"   Total features: {len(features.columns)}")
        logger.info(f"   Date range: {features.index.min()} to {features.index.max()}")
        
        for col in features.columns:
            non_null = features[col].notna().sum()
            logger.info(f"   {col}: {non_null} non-null values")
        
        return features
    
    def align_with_price_data(self, sentiment_features, price_df):
        """Align sentiment features with price data"""
        
        if price_df is None:
            return sentiment_features
        
        # Align dates
        common_dates = sentiment_features.index.intersection(price_df.index)
        aligned_sentiment = sentiment_features.loc[common_dates]
        aligned_price = price_df.loc[common_dates]
        
        logger.info(f"\n📈 ALIGNED WITH PRICE DATA:")
        logger.info(f"   Common dates: {len(common_dates)}")
        logger.info(f"   Date range: {common_dates.min()} to {common_dates.max()}")
        
        # Calculate correlations with next day returns
        if 'close' in aligned_price.columns:
            returns = aligned_price['close'].pct_change().shift(-1)
            correlations = {}
            
            for col in aligned_sentiment.columns:
                if aligned_sentiment[col].dtype in ['float64', 'int64']:
                    corr = aligned_sentiment[col].corr(returns)
                    if not pd.isna(corr):
                        correlations[col] = corr
            
            logger.info(f"\n📊 CORRELATION WITH NEXT DAY RETURNS:")
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            for col, corr in sorted_corr[:10]:
                logger.info(f"   {col}: {corr:+.4f}")
        
        return aligned_sentiment
    
    def create_lag_features(self, sentiment_features, lags=[1, 2, 3, 5, 10, 21]):
        """Create lagged versions of sentiment features"""
        
        logger.info("\n" + "="*60)
        logger.info("CREATING LAG FEATURES")
        logger.info("="*60)
        
        lagged = sentiment_features.copy()
        
        # Select key features for lagging
        key_features = ['sentiment_daily', 'sentiment_momentum', 'news_volume_zscore', 
                       'sentiment_volatility', 'pos_neg_ratio_log']
        
        for feature in key_features:
            if feature in sentiment_features.columns:
                for lag in lags:
                    lagged[f'{feature}_lag_{lag}'] = sentiment_features[feature].shift(lag)
                    logger.info(f"   Created {feature}_lag_{lag}")
        
        return lagged
    
    def save_sentiment_features(self, features):
        """Save engineered sentiment features"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save features
        feature_file = os.path.join(self.processed_path, f"sentiment_features_{timestamp}.csv")
        features.to_csv(feature_file)
        logger.info(f"\n💾 Saved sentiment features: {feature_file}")
        
        # Save feature list
        feature_list_file = os.path.join(self.processed_path, f"sentiment_feature_list_{timestamp}.txt")
        with open(feature_list_file, 'w') as f:
            f.write("SENTIMENT FEATURES FOR ML MODEL\n")
            f.write("="*60 + "\n\n")
            for col in features.columns:
                f.write(f"• {col}\n")
        logger.info(f"Saved feature list: {feature_list_file}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_features': len(features.columns),
            'date_range': f"{features.index.min()} to {features.index.max()}",
            'feature_names': list(features.columns),
            'feature_stats': {
                col: {
                    'mean': float(features[col].mean()) if features[col].dtype in ['float64', 'int64'] else None,
                    'std': float(features[col].std()) if features[col].dtype in ['float64', 'int64'] else None,
                    'min': float(features[col].min()) if features[col].dtype in ['float64', 'int64'] else None,
                    'max': float(features[col].max()) if features[col].dtype in ['float64', 'int64'] else None,
                }
                for col in features.columns[:20]  # First 20 features
            }
        }
        
        meta_file = os.path.join(self.results_path, f"sentiment_features_metadata_{timestamp}.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {meta_file}")
        
        return feature_file
    
    def plot_sentiment_features(self, features, price_df=None):
        """Visualize engineered sentiment features"""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Sentiment vs Price
        if price_df is not None and 'close' in price_df.columns:
            common_idx = features.index.intersection(price_df.index)
            if len(common_idx) > 0:
                ax1 = axes[0, 0]
                ax1_twin = ax1.twinx()
                
                ax1.plot(common_idx, features.loc[common_idx, 'sentiment_daily'], 
                        color='blue', linewidth=1, alpha=0.7, label='Sentiment')
                ax1.set_ylabel('Sentiment', color='blue')
                ax1.tick_params(axis='y', labelcolor='blue')
                
                ax1_twin.plot(common_idx, price_df.loc[common_idx, 'close'], 
                             color='orange', linewidth=1, alpha=0.7, label='Silver Price')
                ax1_twin.set_ylabel('Silver Price ($)', color='orange')
                ax1_twin.tick_params(axis='y', labelcolor='orange')
                
                axes[0, 0].set_title('Sentiment vs Silver Price', fontsize=12)
                axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Sentiment momentum
        axes[0, 1].plot(features.index, features['sentiment_momentum'], 
                       color='purple', linewidth=1)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('Sentiment Momentum (5d - 21d MA)', fontsize=12)
        axes[0, 1].set_ylabel('Momentum')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. News volume spikes
        axes[1, 0].fill_between(features.index, 0, features['news_volume'], 
                               alpha=0.5, color='green')
        axes[1, 0].plot(features.index, features['news_volume_zscore'] * features['news_volume'].std() + 
                       features['news_volume'].mean(), color='red', linewidth=0.5, alpha=0.7)
        axes[1, 0].set_title('News Volume (Green) & Z-Score Spikes (Red)', fontsize=12)
        axes[1, 0].set_ylabel('Volume')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Sentiment volatility
        axes[1, 1].plot(features.index, features['sentiment_volatility'], 
                       color='orange', linewidth=1)
        axes[1, 1].set_title('Sentiment Volatility (5-day)', fontsize=12)
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Positive/Negative ratio
        if 'pos_neg_ratio' in features.columns:
            axes[2, 0].plot(features.index, features['pos_neg_ratio'], 
                           color='brown', linewidth=1)
            axes[2, 0].axhline(y=1, color='black', linestyle='--', alpha=0.5)
            axes[2, 0].set_title('Positive/Negative News Ratio', fontsize=12)
            axes[2, 0].set_ylabel('Ratio')
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Sentiment regime distribution
        if 'sentiment_regime' in features.columns:
            regime_counts = features['sentiment_regime'].value_counts()
            colors = ['green', 'gray', 'red']
            axes[2, 1].bar(regime_counts.index, regime_counts.values, color=colors, alpha=0.7)
            axes[2, 1].set_title('Sentiment Regime Distribution', fontsize=12)
            axes[2, 1].set_ylabel('Days')
            axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.figures_path, f"sentiment_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sentiment features plot: {plot_file}")
    
    def run(self):
        """Run complete sentiment feature engineering"""
        
        logger.info("="*60)
        logger.info("SENTIMENT FEATURE ENGINEERING")
        logger.info("="*60)
        
        # Load data
        daily_sentiment = self.load_sentiment_data()
        if daily_sentiment is None:
            return None
        
        price_df = self.load_price_data()
        
        # Engineer features
        features = self.engineer_sentiment_features(daily_sentiment)
        
        # Align with price data
        features = self.align_with_price_data(features, price_df)
        
        # Create lag features
        features = self.create_lag_features(features)
        
        # Save features
        feature_file = self.save_sentiment_features(features)
        
        # Plot
        self.plot_sentiment_features(features, price_df)
        
        logger.info("\n" + "="*60)
        logger.info("✅ SENTIMENT FEATURE ENGINEERING COMPLETE")
        logger.info("="*60)
        logger.info(f"Final feature count: {len(features.columns)}")
        logger.info(f"Date range: {features.index.min()} to {features.index.max()}")
        
        return features

if __name__ == "__main__":
    engineer = SentimentFeatureEngineer()
    features = engineer.run()
    
    if features is not None:
        print("\n" + "="*60)
        print("SENTIMENT FEATURES SUMMARY")
        print("="*60)
        print(f"Features created: {len(features.columns)}")
        print(f"Days with data: {len(features)}")
        print("\nFeature list:")
        for col in features.columns[:15]:
            print(f"  • {col}")
        if len(features.columns) > 15:
            print(f"  ... and {len(features.columns) - 15} more")
