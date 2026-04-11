#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dotenv import load_dotenv
from fredapi import Fred
import requests_cache
from retry import retry
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Setup requests cache (to avoid hitting API limits)
requests_cache.install_cache('fred_cache', expire_after=3600)  # Cache for 1 hour

class MacroDataCollector:
    """Collector for macroeconomic data from FRED"""
    
    def __init__(self, api_key=None):
        """Initialize FRED connection"""
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        if not self.api_key:
            raise ValueError("FRED API key not found. Set FRED_API_KEY in .env file")
        
        self.fred = Fred(api_key=self.api_key)
        
        # Define FRED series IDs
        self.series_dict = {
            'treasury_10yr': 'DGS10',      # 10-Year Treasury Yield
            'dollar_index': 'DTWEXBGS',     # Trade Weighted U.S. Dollar Index
            'cpi': 'CPIAUCSL',              # Consumer Price Index
            'fed_funds': 'FEDFUNDS',         # Federal Funds Rate
            'treasury_2yr': 'DGS2',          # 2-Year Treasury (for spread)
            'treasury_30yr': 'DGS30',        # 30-Year Treasury
            'inflation_expectations': 'T10YIE',  # 10-Year Inflation Expectations
            'industrial_production': 'INDPRO',   # Industrial Production
            'unemployment': 'UNRATE',         # Unemployment Rate
            'gdp_growth': 'A191RL1Q225SBEA',  # Real GDP Growth (quarterly)
        }
        
        # Define frequency and transformation for each series
        self.series_info = {
            'treasury_10yr': {'freq': 'D', 'transform': None, 'description': '10-Year Treasury Yield (%)'},
            'treasury_2yr': {'freq': 'D', 'transform': None, 'description': '2-Year Treasury Yield (%)'},
            'treasury_30yr': {'freq': 'D', 'transform': None, 'description': '30-Year Treasury Yield (%)'},
            'dollar_index': {'freq': 'D', 'transform': None, 'description': 'Dollar Index (Jan 2006=100)'},
            'cpi': {'freq': 'M', 'transform': 'pct_change', 'description': 'CPI (Index 1982-84=100)'},
            'fed_funds': {'freq': 'D', 'transform': None, 'description': 'Federal Funds Rate (%)'},
            'inflation_expectations': {'freq': 'D', 'transform': None, 'description': '10-Year Inflation Expectations (%)'},
            'industrial_production': {'freq': 'M', 'transform': 'pct_change', 'description': 'Industrial Production Index'},
            'unemployment': {'freq': 'M', 'transform': None, 'description': 'Unemployment Rate (%)'},
            'gdp_growth': {'freq': 'Q', 'transform': None, 'description': 'Real GDP Growth (QoQ %)'},
        }
    
    @retry(tries=3, delay=2, backoff=2)
    def get_series(self, series_id, start_date, end_date):
        """
        Get data for a single series with retry logic
        """
        try:
            logger.info(f"Fetching {series_id} from {start_date} to {end_date}")
            data = self.fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
            
            # Add small delay to avoid rate limiting
            time.sleep(0.5)
            
            return data
        except Exception as e:
            logger.error(f"Error fetching {series_id}: {str(e)}")
            raise
    
    def collect_all_series(self, start_date, end_date):
        """
        Collect all macro series
        """
        all_data = {}
        
        for name, series_id in self.series_dict.items():
            try:
                # Get data
                data = self.get_series(series_id, start_date, end_date)
                
                # Convert to DataFrame
                df = pd.DataFrame(data)
                df.columns = [name]
                
                # Apply transformations if specified
                info = self.series_info.get(name, {})
                if info.get('transform') == 'pct_change':
                    df[name + '_mom'] = df[name].pct_change() * 100  # MoM percentage change
                    df[name + '_yoy'] = df[name].pct_change(12) * 100  # YoY percentage change
                
                all_data[name] = df
                
                logger.info(f"✓ Collected {name}: {len(df)} observations from {df.index.min()} to {df.index.max()}")
                
            except Exception as e:
                logger.error(f"✗ Failed to collect {name}: {str(e)}")
                all_data[name] = None
        
        return all_data
    
    def combine_series(self, all_data):
        """
        Combine all series into a single DataFrame
        """
        # Start with None, then join all available data
        combined = None
        
        for name, df in all_data.items():
            if df is not None and not df.empty:
                if combined is None:
                    combined = df
                else:
                    combined = combined.join(df, how='outer')
        
        # Sort by date
        if combined is not None:
            combined = combined.sort_index()
            
            # Forward fill for business days (carry forward last known value)
            # But don't fill too far into the future
            combined = combined.fillna(method='ffill', limit=5)
        
        return combined
    
    def calculate_derived_features(self, df):
        """
        Calculate derived macro features
        """
        if df is None or df.empty:
            return df
        
        # Yield curve spreads
        if all(col in df.columns for col in ['treasury_10yr', 'treasury_2yr']):
            df['yield_curve_10y2y'] = df['treasury_10yr'] - df['treasury_2yr']
            logger.info("✓ Calculated yield curve spread (10Y-2Y)")
        
        if all(col in df.columns for col in ['treasury_30yr', 'treasury_10yr']):
            df['yield_curve_30y10y'] = df['treasury_30yr'] - df['treasury_10yr']
            logger.info("✓ Calculated yield curve spread (30Y-10Y)")
        
        # Real rates (nominal yield - inflation expectations)
        if all(col in df.columns for col in ['treasury_10yr', 'inflation_expectations']):
            df['real_rate_10yr'] = df['treasury_10yr'] - df['inflation_expectations']
            logger.info("✓ Calculated real 10-year rate")
        
        # Inflation momentum
        if 'cpi' in df.columns:
            df['cpi_acceleration'] = df['cpi'].pct_change(12) - df['cpi'].pct_change(12).shift(12)
            logger.info("✓ Calculated CPI acceleration")
        
        return df
    
    def save_data(self, df, filename):
        """
        Save data to CSV with metadata
        """
        save_path = os.path.join('data', 'raw', filename)
        df.to_csv(save_path)
        logger.info(f"Saved data to {save_path}")
        
        # Save metadata
        metadata = {
            'filename': filename,
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'observations': len(df),
            'columns': list(df.columns),
            'series_info': self.series_info
        }
        
        meta_path = os.path.join('data', 'raw', filename.replace('.csv', '_metadata.txt'))
        with open(meta_path, 'w') as f:
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        
        logger.info(f"Saved metadata to {meta_path}")
        
        return save_path

def main():
    """Main execution function"""
    
    logger.info("="*60)
    logger.info("MACROECONOMIC DATA COLLECTION")
    logger.info("="*60)
    
    # Set date range (20 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=20*365)).strftime('%Y-%m-%d')
    
    logger.info(f"Date range: {start_date} to {end_date}")
    
    # Initialize collector
    try:
        collector = MacroDataCollector()
    except ValueError as e:
        logger.error(str(e))
        logger.info("\nPlease set your FRED API key in .env file:")
        logger.info("echo 'FRED_API_KEY=your_key_here' >> .env")
        sys.exit(1)
    
    # Collect all series
    logger.info("\n" + "-"*40)
    logger.info("Collecting FRED series...")
    logger.info("-"*40)
    
    all_data = collector.collect_all_series(start_date, end_date)
    
    # Combine into single DataFrame
    logger.info("\n" + "-"*40)
    logger.info("Combining series...")
    logger.info("-"*40)
    
    combined = collector.combine_series(all_data)
    
    if combined is None or combined.empty:
        logger.error("No data collected successfully")
        sys.exit(1)
    
    # Calculate derived features
    logger.info("\n" + "-"*40)
    logger.info("Calculating derived features...")
    logger.info("-"*40)
    
    combined = collector.calculate_derived_features(combined)
    
    # Save data
    logger.info("\n" + "-"*40)
    logger.info("Saving data...")
    logger.info("-"*40)
    
    filename = f"macro_data_{datetime.now().strftime('%Y%m%d')}.csv"
    collector.save_data(combined, filename)
    
    # Print summary statistics
    logger.info("\n" + "="*60)
    logger.info("COLLECTION SUMMARY")
    logger.info("="*60)
    
    logger.info(f"Total observations: {len(combined)}")
    logger.info(f"Date range: {combined.index.min()} to {combined.index.max()}")
    logger.info(f"Features collected: {len(combined.columns)}")
    
    # Show latest values
    logger.info("\nLatest values (as of {}):".format(combined.index.max().strftime('%Y-%m-%d')))
    latest = combined.iloc[-1]
    for col in combined.columns[:10]:  # Show first 10 columns
        if pd.notna(latest[col]):
            logger.info(f"  {col}: {latest[col]:.4f}")
    
    # Check for missing data
    missing = combined.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"\nMissing values found: {missing[missing>0].to_dict()}")
    
    logger.info("\n" + "="*60)
    logger.info("MACRO DATA COLLECTION COMPLETE")
    logger.info("="*60)

if __name__ == "__main__":
    main()