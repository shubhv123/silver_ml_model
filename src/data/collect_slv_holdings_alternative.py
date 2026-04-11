#!/usr/bin/env python

import os
import sys
import pandas as pd
import requests
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

def download_ishares_holdings():
    """Download SLV holdings from iShares public CSV"""
    
    # iShares provides daily holdings CSVs
    urls = [
        "https://www.ishares.com/us/products/239869/ishares-silver-trust/1467271812596.ajax?fileType=csv&fileName=SLV_holdings&dataType=fund",
        "https://www.ishares.com/us/products/239869/ishares-silver-trust/1467271812596.ajax?fileType=csv&fileName=SLV_daily_holdings&dataType=fund"
    ]
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }
    
    for url in urls:
        try:
            logger.info(f"Trying URL: {url}")
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                # Save the CSV
                filename = f"slv_holdings_official_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = os.path.join('data/raw', filename)
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                logger.info(f"✓ Downloaded official SLV holdings to {filepath}")
                
                # Try to parse it
                try:
                    df = pd.read_csv(filepath)
                    logger.info(f"  Rows: {len(df)}, Columns: {df.columns.tolist()}")
                    return df
                except:
                    logger.warning("  Could not parse CSV, but file saved")
                    return None
                    
        except Exception as e:
            logger.warning(f"Failed with URL {url}: {str(e)}")
    
    return None

if __name__ == "__main__":
    download_ishares_holdings()