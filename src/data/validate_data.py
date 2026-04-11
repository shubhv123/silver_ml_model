#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Validate collected data quality"""
    
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'files': {},
            'overall_quality': 'PASS'
        }
    
    def load_data(self, filepath):
        """Load CSV data"""
        try:
            # Skip metadata files
            if filepath.endswith('_metadata.txt'):
                return None
                
            df = pd.read_csv(filepath)
            
            # Try to parse date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            elif df.index.name == 'date':
                df.index = pd.to_datetime(df.index)
            else:
                # Try to parse index as date if it looks like dates
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    pass  # Keep as is
            
            logger.info(f"Loaded {filepath}: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            if not filepath.endswith('_metadata.txt'):
                logger.error(f"Error loading {filepath}: {str(e)}")
            return None
    
    def validate_price_data(self, df, filename):
        """Validate price data (silver/gold)"""
        issues = []
        stats = {}
        
        # Basic checks
        if df is None or df.empty:
            issues.append("EMPTY: No data")
            return issues, stats
        
        # Check date range
        if isinstance(df.index, pd.DatetimeIndex):
            date_range = (df.index.max() - df.index.min()).days
            stats['date_range_days'] = date_range
            
            if date_range < 365 * 10:  # Less than 10 years
                issues.append(f"INSUFFICIENT_HISTORY: Only {date_range} days of data")
            
            # Check for missing dates
            date_min = df.index.min()
            date_max = df.index.max()
            expected_dates = pd.date_range(start=date_min, end=date_max, freq='D')
            missing_dates = len(expected_dates) - len(df)
            missing_pct = (missing_dates / len(expected_dates)) * 100 if len(expected_dates) > 0 else 0
            stats['missing_dates'] = missing_dates
            stats['missing_pct'] = missing_pct
            
            if missing_pct > 10:
                issues.append(f"MISSING_DATA: {missing_pct:.1f}% dates missing")
        else:
            issues.append("INVALID_INDEX: Index is not datetime")
            stats['date_range_days'] = 0
        
        # Check required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            issues.append(f"MISSING_COLUMNS: {missing_cols}")
        
        # Check for null values
        cols_to_check = [col for col in required if col in df.columns]
        if cols_to_check:
            null_counts = df[cols_to_check].isnull().sum()
            if null_counts.sum() > 0:
                issues.append(f"NULL_VALUES: {null_counts[null_counts>0].to_dict()}")
        
        # Check price validity
        if 'close' in df.columns:
            # No negative prices
            if (df['close'] < 0).any():
                issues.append("NEGATIVE_PRICES")
            
            # No zero prices (except possibly)
            zero_days = (df['close'] == 0).sum()
            if zero_days > 0:
                issues.append(f"ZERO_PRICES: {zero_days} days")
            
            # Check for extreme moves
            pct_change = df['close'].pct_change().abs()
            extreme = (pct_change > 0.20).sum()  # 20% daily move
            if extreme > 0:
                issues.append(f"EXTREME_MOVES: {extreme} days with >20% change")
            
            stats['avg_close'] = float(df['close'].mean())
            stats['min_close'] = float(df['close'].min())
            stats['max_close'] = float(df['close'].max())
        
        return issues, stats
    
    def validate_macro_data(self, df, filename):

        issues = []
        stats = {}
    
        if df is None or df.empty:
            issues.append("EMPTY: No data")
            return issues, stats
    
        # Check date range
        if not isinstance(df.index, pd.DatetimeIndex):
            issues.append("INVALID_INDEX: Index is not datetime")
            stats['date_range_days'] = 0
        else:
            date_range = (df.index.max() - df.index.min()).days
            stats['date_range_days'] = date_range
            
            # Check data freshness
        last_date = df.index.max()
        
        # Fix timezone handling
        current_time = pd.Timestamp.now()
        
        # If last_date is timezone-aware, make it naive for comparison
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        
        days_since_update = (current_time - last_date).days
        stats['days_since_update'] = days_since_update
        
        # Only warn if it's been more than 30 days AND the data is recent (after 2020)
        # This avoids warnings for historical data
        if days_since_update > 30 and last_date.year > 2020:
            issues.append(f"STALE_DATA: Last update was {days_since_update} days ago")
        else:
            stats['date_range_days'] = 0
            stats['days_since_update'] = 999
    
    # Check for stale data (forward-filled too much)
        for col in df.columns[:10]:  # Check first 10 columns
            if col in df.columns and len(df) > 1:
                repeated = (df[col] == df[col].shift(1)).sum()
                repeated_pct = (repeated / len(df)) * 100
                if repeated_pct > 50:  # More than 50% repeated values
                    issues.append(f"STALE_DATA: {col} has {repeated_pct:.1f}% repeated values")
    
        return issues, stats
    
    def generate_report(self):
        """Generate validation report"""
        # Create reports directory if it doesn't exist
        os.makedirs('reports/metrics', exist_ok=True)
        
        # Save report
        report_path = os.path.join('reports', 'metrics', f'data_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        with open(report_path, 'w') as f:
            json.dump(self.report, f, indent=2, default=str)
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("DATA VALIDATION REPORT")
        logger.info("="*60)
        
        total_issues = 0
        for filename, info in self.report['files'].items():
            status = "✓" if not info.get('issues') else "✗"
            logger.info(f"\n{status} {filename}")
            logger.info(f"  Rows: {info.get('rows', 'N/A')}")
            logger.info(f"  Columns: {info.get('columns', 'N/A')}")
            
            issues = info.get('issues', [])
            if issues:
                total_issues += len(issues)
                logger.warning(f"  Issues: {len(issues)}")
                for issue in issues[:5]:  # Show first 5 issues
                    logger.warning(f"    - {issue}")
            
            if info.get('stats'):
                logger.info(f"  Stats: {info['stats']}")
        
        logger.info("\n" + "="*60)
        logger.info(f"Overall Quality: {self.report['overall_quality']}")
        logger.info(f"Total Issues: {total_issues}")
        logger.info("="*60)
        
        return report_path
    
    def validate_all(self):
        """Validate all data files"""
        
        # Get all CSV files in raw data directory
        if not os.path.exists('data/raw'):
            logger.error("data/raw directory not found")
            return None
            
        all_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
        
        for filename in all_files:
            filepath = os.path.join('data/raw', filename)
            df = self.load_data(filepath)
            
            if df is None:
                continue
            
            # Determine file type and validate accordingly
            if 'SI_F' in filename or 'GC_F' in filename or 'precious_metals' in filename:
                issues, stats = self.validate_price_data(df, filename)
            elif filename.startswith('macro_data_'):
                issues, stats = self.validate_macro_data(df, filename)
            else:
                issues, stats = [], {}  # Unknown file type
            
            self.report['files'][filename] = {
                'rows': len(df) if df is not None else 0,
                'columns': len(df.columns) if df is not None else 0,
                'issues': issues,
                'stats': stats
            }
        
        # Determine overall quality
        all_issues = []
        for f in self.report['files'].values():
            all_issues.extend(f.get('issues', []))
        
        if len(all_issues) == 0:
            self.report['overall_quality'] = 'EXCELLENT'
        elif len(all_issues) < 5:
            self.report['overall_quality'] = 'GOOD'
        elif len(all_issues) < 10:
            self.report['overall_quality'] = 'FAIR'
        else:
            self.report['overall_quality'] = 'POOR'
        
        return self.generate_report()

def main():
    """Main execution"""
    logger.info("Starting data validation...")
    
    validator = DataValidator()
    report_path = validator.validate_all()
    
    if report_path:
        logger.info(f"\nFull report saved to: {report_path}")

if __name__ == "__main__":
    main()