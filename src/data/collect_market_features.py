#!/usr/bin/env python

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketFeaturesCollector:
    """Collector for market features from Yahoo Finance"""
    
    def __init__(self):
        self.symbols = {
            'vix': '^VIX',
            'oil': 'CL=F',
            'copper': 'HG=F',
            'sp500': '^GSPC'
        }
        
        self.feature_names = {
            '^VIX': 'vix',
            'CL=F': 'oil',
            'HG=F': 'copper',
            '^GSPC': 'sp500'
        }
    
    def collect_symbol(self, symbol, name, start_date, end_date):
        """Collect data for a single symbol"""
        logger.info(f"Collecting {name} ({symbol}) from {start_date} to {end_date}")
        
        try:
            # Download data
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            if data.empty:
                logger.error(f"No data retrieved for {name}")
                return None
            
            logger.info(f"Raw data shape for {name}: {data.shape}")
            logger.info(f"Raw data columns: {data.columns.tolist()}")
            
            # Basic cleaning
            data.columns = [col.lower() for col in data.columns]
            
            # Keep only relevant columns
            keep_cols = ['open', 'high', 'low', 'close', 'volume']
            available_cols = [col for col in keep_cols if col in data.columns]
            data = data[available_cols]
            
            # Add symbol and name
            data['symbol'] = symbol
            data['name'] = name
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # The index column might be named 'index' or 'Date' depending on version
            if 'index' in data.columns:
                data.rename(columns={'index': 'date'}, inplace=True)
            elif 'Date' in data.columns:
                data.rename(columns={'Date': 'date'}, inplace=True)
            
            # Ensure date is datetime
            data['date'] = pd.to_datetime(data['date'])
            
            logger.info(f"✓ Collected {len(data)} rows for {name}")
            logger.info(f"  Date range: {data['date'].min()} to {data['date'].max()}")
            
            return data
            
        except Exception as e:
            logger.error(f"Error collecting {name}: {str(e)}")
            logger.error("Full traceback:")
            traceback.print_exc()
            return None
    
    def collect_all(self, start_date, end_date, save_path):
        """Collect all market features"""
        all_data = {}
        
        for name, symbol in self.symbols.items():
            data = self.collect_symbol(symbol, name, start_date, end_date)
            if data is not None:
                all_data[name] = data
                
                # Save individual file
                filename = f"{name}_daily.csv"
                filepath = os.path.join(save_path, filename)
                data.to_csv(filepath, index=False)
                logger.info(f"  Saved to {filename}")
            
            # Add small delay to avoid rate limiting
            time.sleep(1)
        
        return all_data
    
    def save_combined(self, all_data, save_path):
        """Save combined dataset"""
        
        if not all_data:
            logger.error("No data to combine")
            return None
        
        combined_dfs = []
        
        for name, df in all_data.items():
            # Prepare for combined dataset
            df_pivot = df[['date', 'close']].copy()
            df_pivot.rename(columns={'close': f'{name}_close'}, inplace=True)
            combined_dfs.append(df_pivot)
        
        # Create combined dataset with all closes
        if combined_dfs:
            from functools import reduce
            combined = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), combined_dfs)
            combined = combined.sort_values('date')
            
            combined_file = os.path.join(save_path, "market_features_combined.csv")
            combined.to_csv(combined_file, index=False)
            logger.info(f"Saved combined market features to {combined_file}")
            logger.info(f"  Rows: {len(combined)}, Columns: {len(combined.columns)}")
            
            return combined
        
        return None

def test_yahoo_finance_market():
    """Test if yfinance can fetch market indices"""
    logger.info("Testing market data connection...")
    try:
        # Test with a simple market index
        test_ticker = yf.Ticker("^GSPC")
        test_data = test_ticker.history(period="5d")
        logger.info(f"Test successful. Got {len(test_data)} rows for S&P 500")
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def main():
    """Main execution function"""
    
    # First test the connection
    if not test_yahoo_finance_market():
        logger.error("Market data connection test failed. Please check your internet connection.")
        return
    
    logger.info("="*60)
    logger.info("MARKET FEATURES COLLECTION")
    logger.info("="*60)
    
    # Set date range (20 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Initialize collector
    collector = MarketFeaturesCollector()
    
    # Collect data
    all_data = collector.collect_all(start_date, end_date, 'data/raw')
    
    # Save combined data
    if all_data:
        combined = collector.save_combined(all_data, 'data/raw')
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("COLLECTION SUMMARY")
        logger.info("="*60)
        
        for name, df in all_data.items():
            logger.info(f"\n{name.upper()}:")
            logger.info(f"  Rows: {len(df)}")
            logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            logger.info(f"  Close price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")
            
            # Check for zero volume days
            if 'volume' in df.columns:
                zero_volume = (df['volume'] == 0).sum()
                if zero_volume > 0:
                    logger.info(f"  Zero volume days: {zero_volume} (normal for holidays)")
        
        if combined is not None:
            logger.info(f"\nCombined dataset: {len(combined)} rows, {len(combined.columns)} columns")
    else:
        logger.error("No data collected successfully")
        
    logger.info("\n" + "="*60)
    logger.info("MARKET FEATURES COLLECTION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()