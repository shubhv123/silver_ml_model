import pandas as pd
import os
from pathlib import Path

print("="*60)
print("CHECKING DATE RANGES OF ALL DATASETS")
print("="*60)

raw_path = "data/raw"

# Dictionary to store date ranges (normalized to UTC naive)
date_ranges = {}

def normalize_to_utc_naive(df):
    """Convert index to UTC then remove timezone for comparison"""
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        # Convert to UTC and then make naive
        df.index = df.index.tz_convert('UTC').tz_localize(None)
    return df

# Check silver data
silver_files = [f for f in os.listdir(raw_path) if 'SIF_daily' in f]
if silver_files:
    df = pd.read_csv(os.path.join(raw_path, silver_files[0]), parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = normalize_to_utc_naive(df)
    date_ranges['silver'] = (df.index.min(), df.index.max(), len(df))
    print(f"Silver: {df.index.min()} to {df.index.max()} ({len(df)} rows)")

# Check gold data
gold_files = [f for f in os.listdir(raw_path) if 'GCF_daily' in f]
if gold_files:
    df = pd.read_csv(os.path.join(raw_path, gold_files[0]), parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = normalize_to_utc_naive(df)
    date_ranges['gold'] = (df.index.min(), df.index.max(), len(df))
    print(f"Gold: {df.index.min()} to {df.index.max()} ({len(df)} rows)")

# Check macro data
macro_files = [f for f in os.listdir(raw_path) if 'macro_data_' in f and f.endswith('.csv')]
if macro_files:
    df = pd.read_csv(os.path.join(raw_path, macro_files[-1]), index_col=0, parse_dates=True)
    df = normalize_to_utc_naive(df)
    date_ranges['macro'] = (df.index.min(), df.index.max(), len(df))
    print(f"Macro: {df.index.min()} to {df.index.max()} ({len(df)} rows)")

# Check market features
market_files = [f for f in os.listdir(raw_path) if 'market_features_combined' in f]
if market_files:
    df = pd.read_csv(os.path.join(raw_path, market_files[-1]), parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = normalize_to_utc_naive(df)
    date_ranges['market'] = (df.index.min(), df.index.max(), len(df))
    print(f"Market: {df.index.min()} to {df.index.max()} ({len(df)} rows)")

# Check ETF data
etf_files = [f for f in os.listdir(raw_path) if 'slv_holdings_' in f and f.endswith('.csv')]
if etf_files:
    df = pd.read_csv(os.path.join(raw_path, etf_files[-1]), parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = normalize_to_utc_naive(df)
    date_ranges['etf'] = (df.index.min(), df.index.max(), len(df))
    print(f"ETF: {df.index.min()} to {df.index.max()} ({len(df)} rows)")

# Check Silver Institute data
institute_files = [f for f in os.listdir(raw_path) if 'silver_institute_' in f and f.endswith('.csv')]
if institute_files:
    df = pd.read_csv(os.path.join(raw_path, institute_files[-1]), parse_dates=['date'])
    df.set_index('date', inplace=True)
    df = normalize_to_utc_naive(df)
    date_ranges['institute'] = (df.index.min(), df.index.max(), len(df))
    print(f"Institute: {df.index.min()} to {df.index.max()} ({len(df)} rows)")

# Find common date range
print("\n" + "="*60)
print("COMMON DATE RANGE ANALYSIS")
print("="*60)

if date_ranges:
    # Get the latest start date and earliest end date
    latest_start = max([r[0] for r in date_ranges.values()])
    earliest_end = min([r[1] for r in date_ranges.values()])
    
    print(f"\nLatest start date across all datasets: {latest_start}")
    print(f"Earliest end date across all datasets: {earliest_end}")
    
    if latest_start <= earliest_end:
        print(f"\n✓ Common date range: {latest_start} to {earliest_end}")
        print(f"  Total days: {(earliest_end - latest_start).days + 1}")
        
        # Show which datasets have full coverage
        print("\nDataset coverage within common range:")
        for name, (start, end, count) in date_ranges.items():
            if start <= latest_start and end >= earliest_end:
                print(f"  ✓ {name}: full coverage")
            else:
                print(f"  ✗ {name}: partial coverage (starts {start}, ends {end})")
    else:
        print(f"\n✗ No common date range!")
        print(f"  Latest start ({latest_start}) is after earliest end ({earliest_end})")
        
        # Find which dataset is causing the issue
        print("\nIssues by dataset:")
        for name, (start, end, count) in date_ranges.items():
            if start > earliest_end:
                print(f"  → {name} starts too late: {start}")
            if end < latest_start:
                print(f"  → {name} ends too early: {end}")
                
    # Check silver's date range specifically
    if 'silver' in date_ranges:
        silver_start, silver_end, _ = date_ranges['silver']
        print(f"\nSilver date range: {silver_start} to {silver_end}")
        print(f"Silver is the limiting factor: {silver_end} is the earliest end date")
        
        # Show which datasets extend beyond silver
        for name, (start, end, count) in date_ranges.items():
            if name != 'silver':
                if end > silver_end:
                    print(f"  {name} extends { (end - silver_end).days } days beyond silver")
                if start < silver_start:
                    print(f"  {name} starts { (silver_start - start).days } days before silver")
