import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*60)
print("VIF ANALYSIS FOR MULTICOLLINEARITY DETECTION")
print("="*60)

# Load the reduced feature matrix
processed_path = "data/processed"
reduced_files = list(Path(processed_path).glob("feature_matrix_reduced_*.csv"))

if not reduced_files:
    logger.error("No reduced feature matrix found")
    exit(1)

latest = sorted(reduced_files)[-1]
df = pd.read_csv(latest, index_col=0, parse_dates=True)
print(f"\n📊 Loaded: {latest.name}")
print(f"   Shape: {df.shape}")

# Get feature columns (exclude targets)
feature_cols = [c for c in df.columns if 'target' not in c]
print(f"   Features to analyze: {len(feature_cols)}")

# Drop rows with NaN for VIF calculation
df_clean = df[feature_cols].dropna()
print(f"   Rows after dropping NaN: {len(df_clean)}")

if len(df_clean) < 100:
    logger.warning("Too few rows after dropping NaN. Using alternative approach...")
    # Fill NaN with median instead of dropping
    df_clean = df[feature_cols].fillna(df[feature_cols].median())
    print(f"   Filled NaN with median: {len(df_clean)} rows")

# Add constant for VIF
X = add_constant(df_clean)
print(f"\n🔄 Calculating VIF for {len(X.columns)} variables (including constant)...")

# Calculate VIF for each feature
vif_data = []
for i in range(1, len(X.columns)):  # Start from 1 to skip constant
    try:
        vif = variance_inflation_factor(X.values, i)
        vif_data.append({
            'feature': X.columns[i],
            'vif': vif
        })
    except Exception as e:
        print(f"   Error on {X.columns[i]}: {str(e)[:50]}")
        vif_data.append({
            'feature': X.columns[i],
            'vif': np.nan
        })

# Create DataFrame
vif_df = pd.DataFrame(vif_data)
vif_df = vif_df.sort_values('vif', ascending=False)

# Summary statistics
print("\n📈 VIF SUMMARY:")
print(f"   Mean VIF: {vif_df['vif'].mean():.2f}")
print(f"   Median VIF: {vif_df['vif'].median():.2f}")
print(f"   Max VIF: {vif_df['vif'].max():.2f}")
print(f"   Min VIF: {vif_df['vif'].min():.2f}")

# Identify features with high VIF (>10)
high_vif = vif_df[vif_df['vif'] > 10]
print(f"\n⚠️  Features with high multicollinearity (VIF > 10): {len(high_vif)}")
if len(high_vif) > 0:
    for _, row in high_vif.head(10).iterrows():
        print(f"   • {row['feature']}: {row['vif']:.2f}")
    if len(high_vif) > 10:
        print(f"   ... and {len(high_vif) - 10} more")

# Identify features with moderate VIF (5-10)
moderate_vif = vif_df[(vif_df['vif'] >= 5) & (vif_df['vif'] <= 10)]
print(f"\n📊 Features with moderate VIF (5-10): {len(moderate_vif)}")

# Identify features with low VIF (<5)
low_vif = vif_df[vif_df['vif'] < 5]
print(f"\n✅ Features with low VIF (<5): {len(low_vif)}")

# Save full VIF results
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
vif_file = os.path.join(processed_path, f"vif_analysis_{timestamp}.csv")
vif_df.to_csv(vif_file, index=False)
print(f"\n💾 Saved full VIF results to: {vif_file}")

# Create final feature set excluding high VIF features
features_to_keep = low_vif['feature'].tolist() + moderate_vif['feature'].tolist()
print(f"\n🔍 Final feature selection:")
print(f"   Original features: {len(feature_cols)}")
print(f"   Keeping (VIF < 10): {len(features_to_keep)}")
print(f"   Removing (VIF > 10): {len(high_vif)}")

# Update selected features file
selected_features = {
    'original_count': len(feature_cols),
    'vif_analysis_count': len(features_to_keep),
    'removed_by_vif': len(high_vif),
    'features': features_to_keep,
    'vif_summary': {
        'mean_vif': float(vif_df['vif'].mean()),
        'max_vif': float(vif_df['vif'].max()),
        'min_vif': float(vif_df['vif'].min())
    }
}

selected_file = os.path.join(processed_path, f"selected_features_vif_{timestamp}.json")
with open(selected_file, 'w') as f:
    json.dump(selected_features, f, indent=2)
print(f"💾 Saved updated feature selection to: {selected_file}")

# Create final feature matrix with VIF-selected features
target_cols = [c for c in df.columns if 'target' in c]
final_df = df[features_to_keep + target_cols].copy()

final_file = os.path.join(processed_path, f"feature_matrix_final_{timestamp}.csv")
final_df.to_csv(final_file)
print(f"💾 Saved final feature matrix to: {final_file}")
print(f"   Final shape: {final_df.shape}")

print("\n" + "="*60)
print("✅ VIF ANALYSIS COMPLETE")
print("="*60)
