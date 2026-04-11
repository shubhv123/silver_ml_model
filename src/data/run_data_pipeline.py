#!/usr/bin/env python

import os
import sys
import subprocess
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_script(script_path, description):
    """Run a Python script and log output"""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        # Only log stdout if it contains useful information
        if result.stdout and len(result.stdout.strip()) > 0:
            # Filter out common warnings we don't care about
            filtered_stdout = '\n'.join([
                line for line in result.stdout.split('\n')
                if 'FutureWarning' not in line 
                and 'DeprecationWarning' not in line
                and 'UserWarning' not in line
            ])
            if filtered_stdout.strip():
                logger.info(filtered_stdout)
        
        if result.stderr:
            # Filter stderr too
            filtered_stderr = '\n'.join([
                line for line in result.stderr.split('\n')
                if 'FutureWarning' not in line 
                and 'DeprecationWarning' not in line
                and 'UserWarning' not in line
            ])
            if filtered_stderr.strip():
                logger.warning(f"Stderr: {filtered_stderr}")
                
        logger.info(f"✓ Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed: {description}")
        logger.error(f"Error: {e}")
        if e.stdout:
            logger.error(f"Stdout: {e.stdout}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def main():
    """Run the complete data pipeline"""
    
    start_time = datetime.now()
    logger.info(f"Starting data pipeline at {start_time}")
    
    # Step 1: Collect price data (silver & gold)
    price_success = run_script(
        'src/data/collect_price_data.py',
        'Collecting Silver/Gold Price Data'
    )
    
    # Step 2: Collect macro data (FRED)
    macro_success = run_script(
        'src/data/collect_macro_data.py',
        'Collecting Macroeconomic Data from FRED'
    )
    
    # Step 3: Collect market features (VIX, Oil, Copper, S&P 500)
    market_success = run_script(
        'src/data/collect_market_features.py',
        'Collecting Market Features (VIX, Oil, Copper, S&P 500)'
    )
    
    # Step 4: Collect COT data
    cot_success = run_script(
        'src/data/collect_cot_data.py',
        'Collecting COT (Commitment of Traders) Data'
    )
    
    # Step 5: Collect ETF flow data
    etf_success = run_script(
        'src/data/collect_etf_data.py',
        'Collecting SLV ETF Flow Data'
    )
    
    # Step 6: Collect Silver Institute supply/demand data
    institute_success = run_script(
        'src/data/collect_silver_institute_data.py',
        'Collecting World Silver Institute Supply/Demand Data'
    )
    
    # Step 7: Validate all data
    validate_success = run_script(
        'src/data/validate_data.py',
        'Validating All Data'
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("\n" + "="*60)
    logger.info("DATA PIPELINE COMPLETE")
    logger.info("="*60)
    logger.info(f"Started: {start_time}")
    logger.info(f"Ended: {end_time}")
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Price Data: {'✓' if price_success else '✗'}")
    logger.info(f"Macro Data: {'✓' if macro_success else '✗'}")
    logger.info(f"Market Features: {'✓' if market_success else '✗'}")
    logger.info(f"COT Data: {'✓' if cot_success else '✗'}")
    logger.info(f"ETF Flow Data: {'✓' if etf_success else '✗'}")
    logger.info(f"Silver Institute Data: {'✓' if institute_success else '✗'}")
    logger.info(f"Validation: {'✓' if validate_success else '✗'}")
    
    # Show output files
    logger.info("\nOutput files in data/raw:")
    if os.path.exists('data/raw'):
        raw_files = os.listdir('data/raw')
        csv_files = [f for f in sorted(raw_files) if f.endswith('.csv')]
        for f in csv_files:
            size = os.path.getsize(os.path.join('data/raw', f))
            logger.info(f"  - {f} ({size:,} bytes)")

if __name__ == "__main__":
    main()