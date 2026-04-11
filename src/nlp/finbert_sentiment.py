#!/usr/bin/env python
"""
FinBERT Sentiment Analysis - Working Version
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from scipy.special import softmax
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinBERTAnalyzer:
    def __init__(self):
        self.raw_path = "data/raw/news"
        self.processed_path = "data/processed"
        self.results_path = "reports/metrics"
        
        os.makedirs(self.processed_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        self.model_name = "ProsusAI/finbert"
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        logger.info(f"Loading FinBERT model on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("FinBERT loaded successfully")
        self.labels = ['negative', 'neutral', 'positive']
        self.batch_size = 16
        
    def clean_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        import re
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = ' '.join(text.split())
        if len(text) > 1000:
            text = text[:1000]
        return text
    
    def get_finbert_sentiment_batch(self, texts):
        if not texts:
            return []
        
        inputs = self.tokenizer(texts, return_tensors="pt", truncation=True, 
                                padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = softmax(logits.cpu().numpy(), axis=1)
        
        results = []
        for probs in probabilities:
            sentiment_idx = np.argmax(probs)
            results.append({
                'finbert_sentiment': self.labels[sentiment_idx],
                'finbert_positive': float(probs[2]),
                'finbert_neutral': float(probs[1]),
                'finbert_negative': float(probs[0]),
                'finbert_confidence': float(np.max(probs)),
                'finbert_net': float(probs[2] - probs[0])  # positive - negative
            })
        return results
    
    def load_news_data(self):
        news_files = list(Path(self.raw_path).glob("news_headlines_*.csv"))
        if not news_files:
            logger.error("No news files found!")
            return None
        
        latest = sorted(news_files)[-1]
        df = pd.read_csv(latest, parse_dates=['published_at'])
        logger.info(f"Loaded news data: {latest.name} ({len(df)} articles)")
        return df
    
    def apply_finbert_analysis(self, df):
        logger.info("\n" + "="*60)
        logger.info("APPLYING FINBERT SENTIMENT ANALYSIS")
        logger.info("="*60)
        
        texts = []
        for idx, row in df.iterrows():
            text = f"{row.get('title', '')} {row.get('summary', '')} {row.get('description', '')}"
            text = self.clean_text(text)
            texts.append(text)
        
        all_results = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="FinBERT Processing"):
            batch_texts = texts[i:i + self.batch_size]
            batch_results = self.get_finbert_sentiment_batch(batch_texts)
            all_results.extend(batch_results)
        
        results_df = pd.DataFrame(all_results)
        df = pd.concat([df, results_df], axis=1)
        
        logger.info(f"\n📊 FINBERT SENTIMENT SUMMARY:")
        logger.info(f"   Positive: {(df['finbert_sentiment'] == 'positive').sum()} articles")
        logger.info(f"   Neutral: {(df['finbert_sentiment'] == 'neutral').sum()} articles")
        logger.info(f"   Negative: {(df['finbert_sentiment'] == 'negative').sum()} articles")
        logger.info(f"   Avg Confidence: {df['finbert_confidence'].mean():.4f}")
        logger.info(f"   Avg Net Sentiment: {df['finbert_net'].mean():.4f}")
        
        return df
    
    def aggregate_daily_sentiment(self, df):
        logger.info("\n" + "="*60)
        logger.info("AGGREGATING DAILY FINBERT SENTIMENT")
        logger.info("="*60)
        
        df['date'] = df['published_at'].dt.date
        
        daily_sentiment = df.groupby('date').agg({
            'finbert_positive': ['mean', 'std'],
            'finbert_neutral': ['mean', 'std'],
            'finbert_negative': ['mean', 'std'],
            'finbert_net': ['mean', 'std'],
            'finbert_confidence': 'mean',
            'finbert_sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral',
        }).round(4)
        
        daily_sentiment.columns = [
            'finbert_positive_mean', 'finbert_positive_std',
            'finbert_neutral_mean', 'finbert_neutral_std',
            'finbert_negative_mean', 'finbert_negative_std',
            'finbert_net', 'finbert_net_std',
            'finbert_confidence_mean', 'finbert_sentiment_mode'
        ]
        
        # Create rolling averages
        daily_sentiment['finbert_net_5d_ma'] = daily_sentiment['finbert_net'].rolling(5).mean()
        daily_sentiment['finbert_net_21d_ma'] = daily_sentiment['finbert_net'].rolling(21).mean()
        daily_sentiment['finbert_momentum'] = daily_sentiment['finbert_net_5d_ma'] - daily_sentiment['finbert_net_21d_ma']
        
        logger.info(f"Daily FinBERT data: {len(daily_sentiment)} days")
        logger.info(f"Date range: {daily_sentiment.index.min()} to {daily_sentiment.index.max()}")
        
        return daily_sentiment
    
    def save_finbert_results(self, df, daily_df):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        article_file = os.path.join(self.processed_path, f"finbert_articles_{timestamp}.csv")
        df.to_csv(article_file, index=False)
        logger.info(f"Saved FinBERT article sentiment: {article_file}")
        
        daily_file = os.path.join(self.processed_path, f"finbert_daily_{timestamp}.csv")
        daily_df.to_csv(daily_file)
        logger.info(f"Saved FinBERT daily sentiment: {daily_file}")
        
        metadata = {
            'timestamp': timestamp,
            'model': self.model_name,
            'total_articles': len(df),
            'daily_days': len(daily_df),
            'positive_articles': int((df['finbert_sentiment'] == 'positive').sum()),
            'neutral_articles': int((df['finbert_sentiment'] == 'neutral').sum()),
            'negative_articles': int((df['finbert_sentiment'] == 'negative').sum()),
            'avg_net_sentiment': float(daily_df['finbert_net'].mean()),
            'avg_confidence': float(df['finbert_confidence'].mean())
        }
        
        meta_file = os.path.join(self.results_path, f"finbert_metadata_{timestamp}.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata: {meta_file}")
        
        return article_file, daily_file
    
    def run(self):
        logger.info("="*60)
        logger.info("FINBERT SENTIMENT ANALYSIS")
        logger.info("="*60)
        
        df = self.load_news_data()
        if df is None:
            return None, None
        
        df = self.apply_finbert_analysis(df)
        daily_df = self.aggregate_daily_sentiment(df)
        self.save_finbert_results(df, daily_df)
        
        logger.info("\n" + "="*60)
        logger.info("SAMPLE FINBERT RESULTS")
        logger.info("="*60)
        sample = df[['title', 'finbert_sentiment', 'finbert_net', 'finbert_confidence']].head(10)
        for i, row in sample.iterrows():
            logger.info(f"{row['finbert_sentiment'].upper()}: net={row['finbert_net']:.3f}, "
                       f"conf={row['finbert_confidence']:.3f} - {row['title'][:50]}...")
        
        logger.info("\n" + "="*60)
        logger.info("✅ FINBERT ANALYSIS COMPLETE")
        logger.info("="*60)
        
        return df, daily_df

if __name__ == "__main__":
    analyzer = FinBERTAnalyzer()
    df, daily_df = analyzer.run()
    
    if daily_df is not None:
        print("\n" + "="*60)
        print("FINBERT SENTIMENT SUMMARY")
        print("="*60)
        print(f"Daily sentiment data: {len(daily_df)} days")
        print(f"Average net sentiment: {daily_df['finbert_net'].mean():.4f}")
        print(f"Average confidence: {daily_df['finbert_confidence_mean'].mean():.4f}")
        print(f"Positive days: {(daily_df['finbert_net'] > 0).sum()}")
        print(f"Negative days: {(daily_df['finbert_net'] < 0).sum()}")