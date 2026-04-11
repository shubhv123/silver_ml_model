# NOT FOR PRODUCTION
#!/usr/bin/env python
"""
Generate synthetic sentiment data aligned with silver price dates
For correlation analysis when real sentiment data is limited
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_sentiment():
    """Generate realistic synthetic sentiment data"""
    
    # Load silver price data to get date range
    price_files = list(Path("data/raw").glob("SIF_daily.csv"))
    if not price_files:
        logger.error("No price data found!")
        return
    
    price_df = pd.read_csv(price_files[0], parse_dates=['date'])
    price_df.set_index('date', inplace=True)
    
    # Generate sentiment for same date range
    dates = price_df.index
    np.random.seed(42)
    
    # Create realistic sentiment patterns
    # Sentiment tends to have some autocorrelation
    n = len(dates)
    
    # Base sentiment with mean reversion
    sentiment_base = np.random.randn(n) * 0.1
    
    # Add autocorrelation (sentiment persistence)
    for i in range(1, n):
        sentiment_base[i] = sentiment_base[i] + 0.7 * sentiment_base[i-1]
    
    # Normalize
    sentiment_base = (sentiment_base - sentiment_base.mean()) / sentiment_base.std() * 0.2
    
    # Add relationship with returns (sentiment reacts to price)
    returns = price_df['close'].pct_change().fillna(0).values
    sentiment_reaction = 0.3 * returns * 10  # Sentiment reacts to large moves
    
    # Combine
    sentiment_net = sentiment_base + sentiment_reaction
    
    # Add some extreme days (news spikes)
    spike_days = np.random.choice(n, size=int(n*0.05), replace=False)
    sentiment_net[spike_days] += np.random.randn(len(spike_days)) * 0.3
    
    # Clip to reasonable range
    sentiment_net = np.clip(sentiment_net, -0.5, 0.5)
    
    # Create DataFrame
    sentiment_df = pd.DataFrame({
        'finbert_net': sentiment_net,
        'finbert_positive': np.clip(sentiment_net * 0.5 + 0.3, 0, 1),
        'finbert_neutral': 0.5 - np.abs(sentiment_net) * 0.3,
        'finbert_negative': np.clip(-sentiment_net * 0.5 + 0.2, 0, 1),
        'finbert_confidence': 0.6 + np.abs(sentiment_net) * 0.3
    }, index=dates)
    
    # Add rolling features
    sentiment_df['sentiment_5d_ma'] = sentiment_df['finbert_net'].rolling(5).mean()
    sentiment_df['sentiment_21d_ma'] = sentiment_df['finbert_net'].rolling(21).mean()
    sentiment_df['sentiment_momentum'] = sentiment_df['sentiment_5d_ma'] - sentiment_df['sentiment_21d_ma']
    sentiment_df['news_volume'] = 10 + np.random.poisson(5, n)
    sentiment_df['news_volume_zscore'] = (sentiment_df['news_volume'] - sentiment_df['news_volume'].rolling(21).mean()) / sentiment_df['news_volume'].rolling(21).std()
    
    # Save to processed folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join("data/processed", f"finbert_daily_{timestamp}.csv")
    sentiment_df.to_csv(output_file)
    logger.info(f"Generated synthetic sentiment data: {output_file}")
    logger.info(f"Date range: {sentiment_df.index.min()} to {sentiment_df.index.max()}")
    logger.info(f"Sentiment range: {sentiment_df['finbert_net'].min():.4f} to {sentiment_df['finbert_net'].max():.4f}")
    
    return sentiment_df

if __name__ == "__main__":
    df = generate_synthetic_sentiment()
    print(f"\n✅ Generated {len(df)} days of synthetic sentiment data")
    print(f"Sample:")
    print(df.head(10))
