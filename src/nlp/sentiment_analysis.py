#!/usr/bin/env python
"""
Sentiment Analysis on News Headlines
Using VADER (Valence Aware Dictionary and sEntiment Reasoner)
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

# VADER sentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Text processing
import re
from textblob import TextBlob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Apply sentiment analysis to news headlines"""
    
    def __init__(self):
        self.raw_path = "data/raw/news"
        self.processed_path = "data/processed"
        self.results_path = "reports/metrics"
        
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        # Initialize VADER
        self.vader = SentimentIntensityAnalyzer()
        
        # Add financial domain specific words
        self.add_financial_lexicon()
        
    def add_financial_lexicon(self):
        """Add finance-specific words to VADER lexicon"""
        
        financial_words = {
            # Bullish words
            'surge': 2.5, 'soar': 2.8, 'rally': 2.3, 'breakout': 2.0,
            'bullish': 2.5, 'upside': 1.5, 'gain': 1.8, 'profit': 1.5,
            'positive': 2.0, 'strong': 1.5, 'growth': 1.3, 'opportunity': 1.2,
            'upgrade': 1.8, 'outperform': 2.0, 'buy': 1.5, 'accumulate': 1.3,
            
            # Bearish words
            'plunge': -2.5, 'crash': -3.0, 'selloff': -2.5, 'downturn': -2.0,
            'bearish': -2.5, 'downside': -1.8, 'loss': -1.5, 'risk': -1.2,
            'negative': -2.0, 'weak': -1.5, 'decline': -1.8, 'downgrade': -1.8,
            'underperform': -2.0, 'sell': -1.5, 'reduce': -1.3, 'warning': -1.5,
            
            # Silver/Gold specific
            'silver': 0.5, 'gold': 0.5, 'precious': 0.8, 'metal': 0.3,
            'mining': 0.2, 'etf': 0.3, 'inventory': 0.5, 'supply': 0.3,
            'demand': 0.5, 'shortage': 1.0, 'deficit': 1.2, 'surplus': -1.0,
            
            # Macro
            'fed': -0.5, 'inflation': -0.8, 'rate_hike': -1.0, 'dovish': 0.8,
            'hawkish': -0.8, 'stimulus': 1.0, 'recession': -1.5, 'recovery': 1.0,
        }
        
        self.vader.lexicon.update(financial_words)
        logger.info(f"Added {len(financial_words)} financial terms to VADER lexicon")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text.lower()
    
    def get_vader_sentiment(self, text):
        """Get VADER sentiment scores"""
        
        if not text or len(text) < 5:
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        scores = self.vader.polarity_scores(text)
        return scores
    
    def get_textblob_sentiment(self, text):
        """Get TextBlob sentiment as alternative"""
        
        if not text or len(text) < 5:
            return {'polarity': 0, 'subjectivity': 0}
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def classify_sentiment(self, compound_score):
        """Classify sentiment into categories"""
        
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_article(self, row):
        """Analyze single article"""
        
        # Combine title and summary for analysis
        text = f"{row.get('title', '')} {row.get('summary', '')} {row.get('description', '')}"
        text = self.clean_text(text)
        
        # VADER scores
        vader_scores = self.get_vader_sentiment(text)
        
        # TextBlob scores
        blob_scores = self.get_textblob_sentiment(text)
        
        return {
            'vader_compound': vader_scores['compound'],
            'vader_positive': vader_scores['pos'],
            'vader_negative': vader_scores['neg'],
            'vader_neutral': vader_scores['neu'],
            'vader_sentiment': self.classify_sentiment(vader_scores['compound']),
            'textblob_polarity': blob_scores['polarity'],
            'textblob_subjectivity': blob_scores['subjectivity']
        }
    
    def load_news_data(self):
        """Load the latest news data"""
        
        news_files = list(Path(self.raw_path).glob("news_headlines_*.csv"))
        
        if not news_files:
            logger.error("No news files found!")
            return None
        
        latest = sorted(news_files)[-1]
        df = pd.read_csv(latest, parse_dates=['published_at'])
        logger.info(f"Loaded news data: {latest.name} ({len(df)} articles)")
        
        return df
    
    def apply_sentiment_analysis(self, df):
        """Apply sentiment analysis to all articles"""
        
        logger.info("\n" + "="*60)
        logger.info("APPLYING SENTIMENT ANALYSIS")
        logger.info("="*60)
        
        # Apply sentiment to each article
        sentiment_results = []
        
        for idx, row in df.iterrows():
            sentiment = self.analyze_article(row)
            sentiment_results.append(sentiment)
            
            if (idx + 1) % 500 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} articles")
        
        # Add sentiment columns
        sentiment_df = pd.DataFrame(sentiment_results)
        df = pd.concat([df, sentiment_df], axis=1)
        
        # Summary statistics
        logger.info(f"\n📊 SENTIMENT SUMMARY:")
        logger.info(f"   Positive articles: {(df['vader_sentiment'] == 'positive').sum()}")
        logger.info(f"   Neutral articles: {(df['vader_sentiment'] == 'neutral').sum()}")
        logger.info(f"   Negative articles: {(df['vader_sentiment'] == 'negative').sum()}")
        logger.info(f"   Avg Compound Score: {df['vader_compound'].mean():.4f}")
        logger.info(f"   Avg Polarity: {df['textblob_polarity'].mean():.4f}")
        
        return df
    
    def aggregate_daily_sentiment(self, df):
        """Aggregate sentiment to daily level"""
        
        logger.info("\n" + "="*60)
        logger.info("AGGREGATING DAILY SENTIMENT")
        logger.info("="*60)
        
        # Group by date
        df['date'] = df['published_at'].dt.date
        
        daily_sentiment = df.groupby('date').agg({
            'vader_compound': ['mean', 'std', 'count'],
            'vader_positive': 'mean',
            'vader_negative': 'mean',
            'vader_sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral',
            'textblob_polarity': 'mean'
        }).round(4)
        
        # Flatten column names
        daily_sentiment.columns = [
            'sentiment_mean', 'sentiment_std', 'news_count',
            'sentiment_positive', 'sentiment_negative',
            'sentiment_mode', 'polarity_mean'
        ]
        
        # Create rolling averages
        daily_sentiment['sentiment_5d_ma'] = daily_sentiment['sentiment_mean'].rolling(5).mean()
        daily_sentiment['sentiment_21d_ma'] = daily_sentiment['sentiment_mean'].rolling(21).mean()
        daily_sentiment['sentiment_momentum'] = daily_sentiment['sentiment_5d_ma'] - daily_sentiment['sentiment_21d_ma']
        
        # Volume spikes (news attention)
        daily_sentiment['news_volume_zscore'] = (
            (daily_sentiment['news_count'] - daily_sentiment['news_count'].rolling(21).mean()) / 
            daily_sentiment['news_count'].rolling(21).std()
        )
        
        logger.info(f"Daily sentiment data: {len(daily_sentiment)} days")
        logger.info(f"Date range: {daily_sentiment.index.min()} to {daily_sentiment.index.max()}")
        
        return daily_sentiment
    
    def save_sentiment_data(self, df, daily_df):
        """Save sentiment analysis results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save article-level sentiment
        article_file = os.path.join(self.processed_path, f"sentiment_articles_{timestamp}.csv")
        df.to_csv(article_file, index=False)
        logger.info(f"Saved article sentiment: {article_file}")
        
        # Save daily aggregated sentiment
        daily_file = os.path.join(self.processed_path, f"sentiment_daily_{timestamp}.csv")
        daily_df.to_csv(daily_file)
        logger.info(f"Saved daily sentiment: {daily_file}")
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'total_articles': len(df),
            'daily_days': len(daily_df),
            'date_range': f"{daily_df.index.min()} to {daily_df.index.max()}",
            'avg_daily_sentiment': float(daily_df['sentiment_mean'].mean()),
            'avg_news_per_day': float(daily_df['news_count'].mean())
        }
        
        meta_file = os.path.join(self.results_path, f"sentiment_metadata_{timestamp}.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {meta_file}")
        
        return article_file, daily_file
    
    def plot_sentiment(self, daily_df):
        """Create sentiment visualization"""
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Daily sentiment
        axes[0].plot(daily_df.index, daily_df['sentiment_mean'], label='Daily Sentiment', linewidth=1, alpha=0.7)
        axes[0].plot(daily_df.index, daily_df['sentiment_5d_ma'], label='5-day MA', linewidth=2)
        axes[0].plot(daily_df.index, daily_df['sentiment_21d_ma'], label='21-day MA', linewidth=2)
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0].axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Positive threshold')
        axes[0].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5, label='Negative threshold')
        axes[0].set_title('News Sentiment Over Time', fontsize=14)
        axes[0].set_ylabel('VADER Compound Score')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # News volume
        axes[1].bar(daily_df.index, daily_df['news_count'], alpha=0.7, color='steelblue')
        axes[1].set_title('Daily News Volume', fontsize=14)
        axes[1].set_ylabel('Number of Articles')
        axes[1].grid(True, alpha=0.3)
        
        # Sentiment distribution
        axes[2].hist(daily_df['sentiment_mean'], bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[2].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[2].axvline(x=daily_df['sentiment_mean'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {daily_df["sentiment_mean"].mean():.4f}')
        axes[2].set_title('Distribution of Daily Sentiment Scores', fontsize=14)
        axes[2].set_xlabel('Sentiment Score')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        figures_path = "reports/figures"
        os.makedirs(figures_path, exist_ok=True)
        plot_file = os.path.join(figures_path, f"sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Sentiment plot saved: {plot_file}")
    
    def run(self):
        """Run complete sentiment analysis"""
        
        logger.info("="*60)
        logger.info("SENTIMENT ANALYSIS PIPELINE")
        logger.info("="*60)
        
        # Load news data
        df = self.load_news_data()
        if df is None:
            return None
        
        # Apply sentiment analysis
        df = self.apply_sentiment_analysis(df)
        
        # Aggregate to daily
        daily_df = self.aggregate_daily_sentiment(df)
        
        # Save results
        article_file, daily_file = self.save_sentiment_data(df, daily_df)
        
        # Create visualization
        self.plot_sentiment(daily_df)
        
        # Print sample
        logger.info("\n" + "="*60)
        logger.info("SAMPLE SENTIMENT RESULTS")
        logger.info("="*60)
        sample = df[['title', 'vader_sentiment', 'vader_compound']].head(10)
        for i, row in sample.iterrows():
            logger.info(f"{row['vader_sentiment'].upper()}: {row['vader_compound']:.3f} - {row['title'][:60]}...")
        
        logger.info("\n" + "="*60)
        logger.info("✅ SENTIMENT ANALYSIS COMPLETE")
        logger.info("="*60)
        
        return df, daily_df

if __name__ == "__main__":
    # Install required packages
    # pip install vaderSentiment textblob
    
    analyzer = SentimentAnalyzer()
    df, daily_df = analyzer.run()
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    if daily_df is not None:
        print(f"Daily sentiment data: {len(daily_df)} days")
        print(f"Average sentiment: {daily_df['sentiment_mean'].mean():.4f}")
        print(f"Sentiment std dev: {daily_df['sentiment_std'].mean():.4f}")
        print(f"Average news/day: {daily_df['news_count'].mean():.1f}")
