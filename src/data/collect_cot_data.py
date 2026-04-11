#!/usr/bin/env python

import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_price_data(symbol, start_date, end_date, save_path):
    """
    Collect OHLCV data from Yahoo Finance
    """
    logger.info(f"Collecting data for {symbol} from {start_date} to {end_date}")
    
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            logger.error(f"No data retrieved for {symbol}")
            return None
        
        # Basic cleaning
        data.columns = [col.lower() for col in data.columns]
        data = data[['open', 'high', 'low', 'close', 'volume']]
        
        # Add symbol column
        data['symbol'] = symbol
        
        # Reset index to make date a column
        data = data.reset_index()
        data.rename(columns={'index': 'date'}, inplace=True)
        
        # Ensure date is datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Save to CSV
        filename = f"{symbol.replace('=', '')}_daily.csv"
        filepath = os.path.join(save_path, filename)
        data.to_csv(filepath, index=False)
        
        logger.info(f"✓ Saved {len(data)} rows to {filepath}")
        logger.info(f"  Date range: {data['date'].min()} to {data['date'].max()}")
        logger.info(f"  Close price - Min: ${data['close'].min():.2f}, Max: ${data['close'].max():.2f}")
        
        return data
        
    except Exception as e:
        logger.error(f"Error collecting data for {symbol}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def validate_data(data, symbol):
    """
    Perform basic data validation
    """
    issues = []
    
    if data is None:
        issues.append("No data to validate")
        return issues
    
    # Check for missing values
    missing = data.isnull().sum()
    if missing.sum() > 0:
        issues.append(f"Missing values: {missing[missing>0].to_dict()}")
    
    # Check for duplicate dates
    if 'date' in data.columns and data['date'].duplicated().any():
        issues.append("Duplicate dates found")
    
    # Check for monotonic dates
    if 'date' in data.columns and not data['date'].is_monotonic_increasing:
        issues.append("Dates are not in order")
    
    # Check for zero volume (trading holidays)
    if 'volume' in data.columns:
        zero_volume_days = (data['volume'] == 0).sum()
        if zero_volume_days > 0:
            issues.append(f"Found {zero_volume_days} days with zero volume")
    
    # Check for extreme price movements
    if 'close' in data.columns:
        price_change = data['close'].pct_change().abs()
        extreme_moves = (price_change > 0.20).sum()
        if extreme_moves > 0:
            issues.append(f"Found {extreme_moves} days with >20% price change")
    
    return issues

def main():
    """Main execution function"""
    
    # Set date range (20 years)
    end_date = "2024-01-01"
    start_date = "2004-01-01"
    raw_data_path = "data/raw"
    
    # Create data directory if it doesn't exist
    os.makedirs(raw_data_path, exist_ok=True)
    
    all_data = []
    
    # Collect silver data
    logger.info("="*50)
    logger.info("COLLECTING SILVER PRICE DATA")
    logger.info("="*50)
    
    silver_data = collect_price_data("SI=F", start_date, end_date, raw_data_path)
    
    if silver_data is not None:
        issues = validate_data(silver_data, "Silver")
        if issues:
            logger.warning("Data validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ Silver data validation passed")
        all_data.append(silver_data)
    
    # Collect gold data
    logger.info("\n" + "="*50)
    logger.info("COLLECTING GOLD PRICE DATA")
    logger.info("="*50)
    
    gold_data = collect_price_data("GC=F", start_date, end_date, raw_data_path)
    
    if gold_data is not None:
        issues = validate_data(gold_data, "Gold")
        if issues:
            logger.warning("Data validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("✓ Gold data validation passed")
        all_data.append(gold_data)
    
    # Create a combined file for convenience
    if len(all_data) > 0:
        combined = pd.concat(all_data, ignore_index=True)
        combined.to_csv(os.path.join(raw_data_path, "precious_metals_combined.csv"), index=False)
        logger.info(f"\nSaved combined data: {len(combined)} total rows")
    
    logger.info("\n" + "="*50)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("="*50)

if __name__ == "__main__":
    main()