import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Load the latest feature matrix
processed_path = "data/processed"
feature_files = list(Path(processed_path).glob("feature_matrix_*.csv"))
latest_file = sorted(feature_files)[-1]

print(f"Loading {latest_file.name}")
df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
print(f"Original shape: {df.shape}")
print(f"Original columns: {df.columns.tolist()}")

# Remove any non-numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df_numeric = df[numeric_cols].copy()
print(f"Numeric shape: {df_numeric.shape}")
print(f"Dropped {len(df.columns) - len(numeric_cols)} non-numeric columns")

# Save cleaned version
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
cleaned_file = os.path.join(processed_path, f"feature_matrix_cleaned_{timestamp}.csv")
df_numeric.to_csv(cleaned_file)
print(f"Saved cleaned feature matrix to {cleaned_file}")

# Calculate correlations
if 'target_next_day_return' in df_numeric.columns:
    # Get correlations with target
    correlations = []
    for col in df_numeric.columns:
        if col != 'target_next_day_return':
            try:
                corr = df_numeric[col].corr(df_numeric['target_next_day_return'])
                if pd.notna(corr):
                    correlations.append((col, abs(corr)))
            except:
                continue
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 features correlated with next day return:")
    for col, corr in correlations[:10]:
        print(f"  {col}: {corr:.4f}")
    
    # Also show top positive and negative
    pos_corr = [(col, corr) for col, corr in correlations if corr > 0]
    neg_corr = [(col, -corr) for col, corr in correlations if corr < 0]
    
    print("\nTop 5 positive correlations:")
    for col, corr in sorted(pos_corr, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {col}: {corr:.4f}")
    
    print("\nTop 5 negative correlations:")
    for col, corr in sorted(neg_corr, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {col}: {-corr:.4f}")
else:
    print("Target column 'target_next_day_return' not found")
    print("Available target columns:", [c for c in df_numeric.columns if 'target' in c])

print("\nFeature matrix is ready for modeling!")