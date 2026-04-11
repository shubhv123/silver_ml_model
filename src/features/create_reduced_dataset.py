#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_reduced_dataset():
    """Create dataset with selected features"""
    
    processed_path = "data/processed"
    
    # Load feature matrix
    feature_files = list(Path(processed_path).glob("feature_matrix_*.csv"))
    if not feature_files:
        logger.error("No feature matrix found")
        return
    
    latest_matrix = sorted(feature_files)[-1]
    df = pd.read_csv(latest_matrix, index_col=0, parse_dates=True)
    logger.info(f"Loaded feature matrix: {df.shape}")
    
    # Load selected features
    selected_files = list(Path(processed_path).glob("selected_features_*.json"))
    if selected_files:
        latest_selected = sorted(selected_files)[-1]
        with open(latest_selected, 'r') as f:
            selected_data = json.load(f)
        selected_features = selected_data['features']
        logger.info(f"Loaded {len(selected_features)} selected features from {latest_selected.name}")
    else:
        # If no selection file, use simple heuristic
        logger.warning("No feature selection file found, using heuristic")
        feature_cols = [col for col in df.columns if 'target' not in col]
        
        # Remove features with too many missing values
        missing_pct = df[feature_cols].isnull().mean()
        selected_features = missing_pct[missing_pct < 0.3].index.tolist()
        logger.info(f"Selected {len(selected_features)} features based on missing threshold")
    
    # Create reduced dataset
    target_cols = [col for col in df.columns if 'target' in col]
    all_cols = selected_features + target_cols
    
    df_reduced = df[all_cols].copy()
    
    # Handle remaining missing values
    # For features, fill with median
    for col in selected_features:
        if df_reduced[col].isnull().any():
            median_val = df_reduced[col].median()
            df_reduced[col] = df_reduced[col].fillna(median_val)
            logger.info(f"Filled missing in {col} with median: {median_val:.4f}")
    
    # Save reduced dataset
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    reduced_file = os.path.join(processed_path, f"feature_matrix_reduced_{timestamp}.csv")
    df_reduced.to_csv(reduced_file)
    logger.info(f"Saved reduced feature matrix to {reduced_file}")
    logger.info(f"Shape: {df_reduced.shape}")
    logger.info(f"Features: {len(selected_features)}, Targets: {len(target_cols)}")
    
    # Save feature list
    feature_file = os.path.join(processed_path, f"features_final_{timestamp}.txt")
    with open(feature_file, 'w') as f:
        for feat in sorted(selected_features):
            f.write(f"{feat}\n")
    logger.info(f"Saved final feature list to {feature_file}")
    
    return df_reduced

if __name__ == "__main__":
    create_reduced_dataset()