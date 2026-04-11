
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    """Analyze and select features for modeling"""
    
    def __init__(self):
        self.processed_path = "data/processed"
        self.reports_path = "reports/figures"
        os.makedirs(self.reports_path, exist_ok=True)
        
    def load_feature_matrix(self):
        """Load the latest feature matrix"""
        feature_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        
        if not feature_files:
            feature_files = list(Path(self.processed_path).glob("feature_matrix_*.csv"))
        
        if not feature_files:
            logger.error("No feature matrix found in data/processed/")
            return None
        
        latest_file = sorted(feature_files)[-1]
        logger.info(f"Loading feature matrix: {latest_file.name}")
        
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded: {df.shape}")
        
        return df
    
    def analyze_missing_values(self, df):
        """Analyze missing values in features"""
        logger.info("\n" + "="*50)
        logger.info("MISSING VALUES ANALYSIS")
        logger.info("="*50)
        
        # Calculate missing percentages
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        missing_df = pd.DataFrame({
            'Missing_Count': missing,
            'Missing_Percent': missing_pct
        }).sort_values('Missing_Percent', ascending=False)
        
        # Filter to features only (not targets)
        feature_cols = [col for col in missing_df.index if 'target' not in col]
        missing_df = missing_df.loc[feature_cols]
        
        # Summary statistics
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info(f"Features with any missing: {(missing_df['Missing_Count'] > 0).sum()}")
        logger.info(f"Features with >20% missing: {(missing_df['Missing_Percent'] > 20).sum()}")
        logger.info(f"Features with >50% missing: {(missing_df['Missing_Percent'] > 50).sum()}")
        
        # Plot missing values
        plt.figure(figsize=(12, 6))
        top_missing = missing_df.head(30)
        plt.barh(range(len(top_missing)), top_missing['Missing_Percent'])
        plt.yticks(range(len(top_missing)), top_missing.index)
        plt.xlabel('Missing (%)')
        plt.title('Top 30 Features by Missing Percentage')
        plt.tight_layout()
        
        plot_file = os.path.join(self.reports_path, f"missing_values_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved missing values plot to {plot_file}")
        
        return missing_df
    
    def analyze_correlations(self, df):
        """Analyze feature correlations with target"""
        logger.info("\n" + "="*50)
        logger.info("CORRELATION ANALYSIS")
        logger.info("="*50)
        
        # Find target columns
        target_cols = [col for col in df.columns if 'target' in col]
        logger.info(f"Target variables found: {target_cols}")
        
        # Choose primary target for analysis
        primary_target = 'target_next_day_return'
        if primary_target not in df.columns:
            primary_target = target_cols[0] if target_cols else None
        
        if primary_target is None:
            logger.error("No target variable found")
            return None
        
        # Calculate correlations
        feature_cols = [col for col in df.columns if 'target' not in col]
        
        # Calculate correlations safely
        correlations = []
        for col in feature_cols:
            try:
                corr = df[col].corr(df[primary_target])
                if pd.notna(corr):
                    correlations.append((col, corr))
            except:
                continue
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        logger.info(f"\nTop 10 positive correlations with {primary_target}:")
        pos_corr = [(col, corr) for col, corr in correlations if corr > 0][:10]
        for col, corr in pos_corr:
            logger.info(f"  {col}: {corr:.4f}")
        
        logger.info(f"\nTop 10 negative correlations with {primary_target}:")
        neg_corr = [(col, corr) for col, corr in correlations if corr < 0][-10:]
        for col, corr in reversed(neg_corr):
            logger.info(f"  {col}: {corr:.4f}")
        
        # Plot top correlations
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        top_positive = correlations[:15]
        axes[0].barh(range(len(top_positive)), [c[1] for c in top_positive], color='green')
        axes[0].set_yticks(range(len(top_positive)))
        axes[0].set_yticklabels([c[0] for c in top_positive])
        axes[0].set_xlabel('Correlation')
        axes[0].set_title(f'Top 15 Positive Correlations with {primary_target}')
        axes[0].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        top_negative = correlations[-15:]
        axes[1].barh(range(len(top_negative)), [c[1] for c in top_negative], color='red')
        axes[1].set_yticks(range(len(top_negative)))
        axes[1].set_yticklabels([c[0] for c in top_negative])
        axes[1].set_xlabel('Correlation')
        axes[1].set_title(f'Top 15 Negative Correlations with {primary_target}')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        plot_file = os.path.join(self.reports_path, f"correlations_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlations plot to {plot_file}")
        
        return correlations
    
    def analyze_feature_groups(self, df):
        """Analyze different groups of features"""
        logger.info("\n" + "="*50)
        logger.info("FEATURE GROUP ANALYSIS")
        logger.info("="*50)
        
        # Define feature groups
        feature_groups = {
            'price_technical': ['sma', 'ema', 'rsi', 'macd', 'bb', 'atr', 'returns', 'volatility', 'high_low', 'close_open'],
            'cross_asset': ['gold', 'vix', 'oil', 'copper', 'sp500'],
            'macro': ['treasury', 'dollar', 'cpi', 'fed_funds', 'unemployment', 'gdp'],
            'sentiment': ['cot_', 'etf_'],
            'fundamental': ['fund_'],
            'lag': ['_lag_'],
            'rolling': ['_rolling_']
        }
        
        feature_cols = [col for col in df.columns if 'target' not in col]
        
        group_counts = {}
        for group, patterns in feature_groups.items():
            count = 0
            for col in feature_cols:
                if any(pattern in col.lower() for pattern in patterns):
                    count += 1
            group_counts[group] = count
        
        logger.info("Feature group counts:")
        for group, count in group_counts.items():
            logger.info(f"  {group}: {count} features")
        
        # Pie chart
        plt.figure(figsize=(10, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(group_counts)))
        plt.pie(group_counts.values(), labels=group_counts.keys(), autopct='%1.1f%%', colors=colors)
        plt.title('Feature Distribution by Group')
        
        plot_file = os.path.join(self.reports_path, f"feature_groups_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature groups plot to {plot_file}")
        
        return group_counts
    
    def detect_multicollinearity(self, df, threshold=0.95):
        """Detect highly correlated features"""
        logger.info("\n" + "="*50)
        logger.info("MULTICOLLINEARITY ANALYSIS")
        logger.info("="*50)
        
        feature_cols = [col for col in df.columns if 'target' not in col]
        
        # Take a subset for correlation matrix (to avoid memory issues)
        if len(feature_cols) > 50:
            # Select features with highest variance
            variances = df[feature_cols].var().sort_values(ascending=False)
            top_features = variances.head(50).index.tolist()
            corr_matrix = df[top_features].corr()
        else:
            corr_matrix = df[feature_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > threshold:
                    high_corr_pairs.append((corr_matrix.columns[i], 
                                           corr_matrix.columns[j], 
                                           corr_matrix.iloc[i, j]))
        
        if high_corr_pairs:
            logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs (>{threshold}):")
            for feat1, feat2, corr in high_corr_pairs[:10]:  # Show first 10
                logger.info(f"  {feat1} - {feat2}: {corr:.3f}")
        else:
            logger.info(f"No highly correlated feature pairs found above {threshold}")
        
        # Correlation heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix (Top 50 Features)')
        plt.tight_layout()
        
        plot_file = os.path.join(self.reports_path, f"correlation_matrix_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation matrix to {plot_file}")
        
        return high_corr_pairs
    
    def select_features(self, df, missing_threshold=0.3, corr_threshold=0.95):
        """Select features based on analysis"""
        logger.info("\n" + "="*50)
        logger.info("FEATURE SELECTION")
        logger.info("="*50)
        
        feature_cols = [col for col in df.columns if 'target' not in col]
        initial_count = len(feature_cols)
        
        # 1. Remove features with too many missing values
        missing_pct = df[feature_cols].isnull().mean()
        cols_to_keep = missing_pct[missing_pct < missing_threshold].index.tolist()
        logger.info(f"After missing threshold ({missing_threshold}): {len(cols_to_keep)} features (removed {initial_count - len(cols_to_keep)})")
        
        # 2. Remove features with zero variance
        if cols_to_keep:
            variances = df[cols_to_keep].var()
            cols_to_keep = variances[variances > 0].index.tolist()
            logger.info(f"After removing zero variance: {len(cols_to_keep)} features")
        
        # 3. Handle multicollinearity (remove highly correlated features)
        if len(cols_to_keep) > 1:
            # Use correlation matrix on remaining features
            corr_matrix = df[cols_to_keep].corr().abs()
            
            # Select features to drop
            to_drop = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > corr_threshold:
                        # Drop the one with lower variance (less informative)
                        if df[corr_matrix.columns[i]].var() < df[corr_matrix.columns[j]].var():
                            to_drop.add(corr_matrix.columns[i])
                        else:
                            to_drop.add(corr_matrix.columns[j])
            
            cols_to_keep = [col for col in cols_to_keep if col not in to_drop]
            logger.info(f"After removing highly correlated (> {corr_threshold}): {len(cols_to_keep)} features (removed {len(to_drop)})")
        
        # Save selected features
        selected_features = {
            'initial_count': initial_count,
            'selected_count': len(cols_to_keep),
            'features': cols_to_keep,
            'missing_threshold': missing_threshold,
            'corr_threshold': corr_threshold
        }
        
        # Save to file
        feature_file = os.path.join(self.processed_path, f"selected_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(feature_file, 'w') as f:
            json.dump(selected_features, f, indent=2)
        
        logger.info(f"\nSelected {len(cols_to_keep)} features out of {initial_count}")
        logger.info(f"Saved selected features to {feature_file}")
        
        return cols_to_keep
    
    def create_feature_summary_report(self, df):
        """Create comprehensive feature summary report"""
        logger.info("\n" + "="*50)
        logger.info("FEATURE SUMMARY REPORT")
        logger.info("="*50)
        
        feature_cols = [col for col in df.columns if 'target' not in col]
        
        # Basic statistics
        summary = {
            'total_features': len(feature_cols),
            'total_targets': len([col for col in df.columns if 'target' in col]),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'total_rows': len(df),
            'feature_stats': {}
        }
        
        # Statistics for each feature (limit to first 20 for readability)
        for col in feature_cols[:20]:
            try:
                summary['feature_stats'][col] = {
                    'dtype': str(df[col].dtype),
                    'missing': int(df[col].isnull().sum()),
                    'missing_pct': float(df[col].isnull().mean() * 100),
                    'mean': float(df[col].mean()) if df[col].dtype in ['float64', 'int64'] else None,
                    'std': float(df[col].std()) if df[col].dtype in ['float64', 'int64'] else None,
                    'min': float(df[col].min()) if df[col].dtype in ['float64', 'int64'] else None,
                    'max': float(df[col].max()) if df[col].dtype in ['float64', 'int64'] else None
                }
            except:
                continue
        
        # Save report
        report_file = os.path.join(self.reports_path, f"feature_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Saved feature summary to {report_file}")
        
        # Print summary
        logger.info(f"\nDataset Summary:")
        logger.info(f"  Rows: {summary['total_rows']}")
        logger.info(f"  Features: {summary['total_features']}")
        logger.info(f"  Targets: {summary['total_targets']}")
        logger.info(f"  Date Range: {summary['date_range']}")
        
        return summary
    
    def run_full_analysis(self):
        """Run complete feature analysis pipeline"""
        
        logger.info("="*60)
        logger.info("FEATURE ANALYSIS PIPELINE")
        logger.info("="*60)
        
        # Load data
        df = self.load_feature_matrix()
        if df is None:
            return
        
        # Run analyses
        missing_df = self.analyze_missing_values(df)
        correlations = self.analyze_correlations(df)
        group_counts = self.analyze_feature_groups(df)
        high_corr = self.detect_multicollinearity(df)
        
        # Select features
        selected_features = self.select_features(df)
        
        # Create summary report
        summary = self.create_feature_summary_report(df)
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Reports saved to: {self.reports_path}")
        logger.info(f"Selected features saved to: {self.processed_path}")
        
        return {
            'missing_analysis': missing_df,
            'correlations': correlations,
            'group_counts': group_counts,
            'high_correlations': high_corr,
            'selected_features': selected_features,
            'summary': summary
        }

if __name__ == "__main__":
    analyzer = FeatureAnalyzer()
    results = analyzer.run_full_analysis()
