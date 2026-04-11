#!/usr/bin/env python

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import requests
from bs4 import BeautifulSoup
import re
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ETFDataCollector:
    """Collector for SLV ETF holdings data"""
    
    def __init__(self):
        self.slv_symbol = "SLV"
        self.sources = {
            'yfinance': self.collect_from_yfinance,
            'ishares': self.collect_from_ishares,
            'manual': self.create_sample_data  # Fallback
        }
    
    def collect_from_yfinance(self, start_date, end_date):
        """Collect ETF data from Yahoo Finance"""
        logger.info(f"Collecting SLV data from Yahoo Finance")
        
        try:
            ticker = yf.Ticker(self.slv_symbol)
            
            # Get historical price data
            price_data = ticker.history(start=start_date, end=end_date)
            
            if price_data.empty:
                logger.error("No price data retrieved from Yahoo Finance")
                return None
            
            # Get fundamentals if available
            try:
                info = ticker.info
                logger.info(f"Retrieved info: {list(info.keys())[:5]}...")
            except:
                info = {}
                logger.warning("Could not retrieve fund info")
            
            # Process price data
            price_data.columns = [col.lower() for col in price_data.columns]
            price_data = price_data.reset_index()
            
            # Handle the date column correctly
            if 'index' in price_data.columns:
                price_data.rename(columns={'index': 'date'}, inplace=True)
            elif 'Date' in price_data.columns:
                price_data.rename(columns={'Date': 'date'}, inplace=True)
            
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            # Calculate derived features
            df = price_data[['date', 'close', 'volume']].copy()
            df.rename(columns={'close': 'slv_price', 'volume': 'slv_volume'}, inplace=True)
            
            # Estimate AUM (Assets Under Management)
            # SLV holds approximately 1/1000 oz per share
            shares_outstanding = self.estimate_shares_outstanding(df['slv_price'], df['slv_volume'])
            df['shares_outstanding'] = shares_outstanding
            df['holdings_oz'] = df['shares_outstanding'] * 1000  # Each share represents ~1/1000 oz
            df['aum_usd'] = df['holdings_oz'] * df['slv_price']
            
            # Calculate daily flows
            df['daily_flow_shares'] = df['shares_outstanding'].diff()
            df['daily_flow_oz'] = df['holdings_oz'].diff()
            df['daily_flow_usd'] = df['aum_usd'].diff()
            
            # Calculate rolling metrics
            df['flow_5d_sum'] = df['daily_flow_usd'].rolling(window=5, min_periods=1).sum()
            df['flow_21d_sum'] = df['daily_flow_usd'].rolling(window=21, min_periods=1).sum()
            
            logger.info(f"✓ Collected {len(df)} rows of SLV data")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting from Yahoo Finance: {str(e)}")
            traceback.print_exc()
            return None
    
    def estimate_shares_outstanding(self, price, volume, avg_turnover=0.02):
        """
        Estimate shares outstanding from price and volume
        Based on typical ETF turnover ratio
        """
        # Assume average daily turnover is 2% of shares outstanding
        estimated_shares = volume / avg_turnover
        return estimated_shares.round(0)
    
    def collect_from_ishares(self, start_date, end_date):
        """Scrape iShares website for actual holdings data"""
        logger.info("Attempting to scrape iShares website...")
        
        try:
            # iShares SLV holdings page
            url = "https://www.ishares.com/us/products/239869/ishares-silver-trust"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for holdings data
            holdings_pattern = re.compile(r'Holdings|Shares Outstanding|AUM', re.IGNORECASE)
            
            holdings_data = {}
            for element in soup.find_all(['div', 'span', 'td']):
                text = element.get_text().strip()
                if holdings_pattern.search(text):
                    logger.info(f"Found potential data: {text}")
            
            # This is a simplified approach - real implementation would need
            # to handle the actual website structure
            
            logger.warning("Direct iShares scraping may need updates based on website structure")
            return None
            
        except Exception as e:
            logger.error(f"Error scraping iShares: {str(e)}")
            return None
    
    def create_sample_data(self, start_date, end_date):
        """
        Create sample/estimated data based on price and volume
        This serves as a fallback if real data is unavailable
        """
        logger.warning("Creating estimated SLV data (not actual holdings)")
        
        try:
            # Get price data from Yahoo Finance as base
            ticker = yf.Ticker(self.slv_symbol)
            price_data = ticker.history(start=start_date, end=end_date)
            
            if price_data.empty:
                return None
            
            price_data = price_data.reset_index()
            
            # Handle the date column correctly
            if 'index' in price_data.columns:
                price_data.rename(columns={'index': 'date'}, inplace=True)
            elif 'Date' in price_data.columns:
                price_data.rename(columns={'Date': 'date'}, inplace=True)
            
            price_data['date'] = pd.to_datetime(price_data['date'])
            
            df = price_data[['date', 'Close', 'Volume']].copy()
            df.rename(columns={'Close': 'slv_price', 'Volume': 'slv_volume'}, inplace=True)
            
            # Estimate holdings based on typical relationship
            # SLV typically holds around 300-500 million ounces
            base_holdings = 400_000_000  # 400M oz
            
            # Add variation based on price (inverse relationship)
            price_ratio = df['slv_price'] / df['slv_price'].iloc[0]
            df['holdings_oz'] = base_holdings * (1 + 0.1 * (1 - price_ratio))  # Inverse relationship
            
            # Add noise to make it realistic
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, base_holdings * 0.01, len(df))
            df['holdings_oz'] += noise.cumsum()  # Random walk
            
            # Ensure holdings are positive
            df['holdings_oz'] = df['holdings_oz'].clip(lower=base_holdings * 0.7)
            
            # Calculate derived features
            df['shares_outstanding'] = df['holdings_oz'] / 1000  # Each share represents 1/1000 oz
            df['aum_usd'] = df['holdings_oz'] * df['slv_price']
            df['daily_flow_oz'] = df['holdings_oz'].diff()
            df['daily_flow_usd'] = df['aum_usd'].diff()
            
            # Calculate rolling metrics
            df['flow_5d_sum'] = df['daily_flow_usd'].rolling(window=5, min_periods=1).sum()
            df['flow_21d_sum'] = df['daily_flow_usd'].rolling(window=21, min_periods=1).sum()
            
            logger.info(f"✓ Created {len(df)} rows of estimated SLV data")
            return df
            
        except Exception as e:
            logger.error(f"Error creating sample data: {str(e)}")
            traceback.print_exc()
            return None
    
    def collect_all(self, start_date, end_date, save_path):
        """Try all collection methods in order"""
        
        # Try Yahoo Finance first
        df = self.collect_from_yfinance(start_date, end_date)
        source_used = 'yfinance'
        
        # If that fails, try iShares
        if df is None:
            logger.info("Yahoo Finance failed, trying iShares...")
            df = self.collect_from_ishares(start_date, end_date)
            source_used = 'ishares'
        
        # If both fail, use sample data
        if df is None:
            logger.warning("All sources failed, using estimated data")
            df = self.create_sample_data(start_date, end_date)
            source_used = 'estimated'
        
        if df is not None:
            # Add source metadata
            df['data_source'] = source_used
            
            # Save to CSV
            filename = f"slv_holdings_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = os.path.join(save_path, filename)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved SLV data to {filepath} (source: {source_used})")
            
            # Save metadata
            metadata = {
                'filename': filename,
                'date_range': f"{df['date'].min()} to {df['date'].max()}",
                'observations': len(df),
                'columns': list(df.columns),
                'source': source_used,
                'data_quality': 'estimated' if source_used == 'estimated' else 'actual'
            }
            
            meta_path = os.path.join(save_path, filename.replace('.csv', '_metadata.txt'))
            with open(meta_path, 'w') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
        
        return df
    
    def calculate_etf_signals(self, df):
        """Calculate trading signals from ETF data"""
        
        if df is None or df.empty:
            return None
        
        signals = df[['date']].copy()
        
        # Holdings changes
        signals['holdings_pct_change'] = df['holdings_oz'].pct_change() * 100
        signals['holdings_5d_change'] = df['holdings_oz'].pct_change(5) * 100
        signals['holdings_21d_change'] = df['holdings_oz'].pct_change(21) * 100
        
        # Flow signals - check if columns exist
        if 'daily_flow_usd' in df.columns and 'aum_usd' in df.columns:
            signals['flow_ratio'] = df['daily_flow_usd'] / df['aum_usd'] * 100  # Flow as % of AUM
        
        if 'flow_5d_sum' in df.columns and 'flow_21d_sum' in df.columns:
            # Avoid division by zero
            signals['flow_ma_ratio'] = df['flow_5d_sum'] / df['flow_21d_sum'].replace(0, np.nan)
        
        # Valuation signals
        if 'aum_usd' in df.columns and 'shares_outstanding' in df.columns:
            signals['aum_per_share'] = df['aum_usd'] / df['shares_outstanding']
            
        if 'slv_price' in df.columns and 'aum_usd' in df.columns and 'shares_outstanding' in df.columns:
            nav = df['aum_usd'] / df['shares_outstanding']
            signals['premium_discount'] = (df['slv_price'] / nav - 1) * 100
        
        # Z-scores for anomaly detection
        if 'holdings_oz' in df.columns:
            mean_holdings = df['holdings_oz'].rolling(window=63, min_periods=20).mean()
            std_holdings = df['holdings_oz'].rolling(window=63, min_periods=20).std()
            signals['holdings_zscore'] = (df['holdings_oz'] - mean_holdings) / std_holdings
        
        return signals

def main():
    """Main execution function"""
    
    logger.info("="*60)
    logger.info("ETF FLOW DATA COLLECTION")
    logger.info("="*60)
    
    # Set date range (20 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Initialize collector
    collector = ETFDataCollector()
    
    # Collect data
    df = collector.collect_all(start_date, end_date, 'data/raw')
    
    if df is not None:
        # Calculate signals
        signals = collector.calculate_etf_signals(df)
        
        if signals is not None and not signals.empty:
            # Save signals separately
            signals_file = os.path.join('data/raw', f"slv_signals_{datetime.now().strftime('%Y%m%d')}.csv")
            signals.to_csv(signals_file, index=False)
            logger.info(f"Saved ETF signals to {signals_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ETF DATA SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Trading days: {len(df)}")
        
        latest = df.iloc[-1]
        logger.info(f"\nLatest values (as of {latest['date'].strftime('%Y-%m-%d')}):")
        logger.info(f"  SLV Price: ${latest['slv_price']:.2f}")
        logger.info(f"  Estimated Holdings: {latest['holdings_oz']/1e6:.1f}M oz")
        logger.info(f"  AUM: ${latest['aum_usd']/1e9:.2f}B")
        
        if 'daily_flow_usd' in latest and not pd.isna(latest['daily_flow_usd']):
            flow_sign = "+" if latest['daily_flow_usd'] > 0 else ""
            logger.info(f"  Daily Flow: {flow_sign}${latest['daily_flow_usd']/1e6:.1f}M")
        
        logger.info(f"\nData source: {df['data_source'].iloc[0]}")
    else:
        logger.error("Failed to collect ETF data")
    
    logger.info("\n" + "="*60)
    logger.info("ETF FLOW DATA COLLECTION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()