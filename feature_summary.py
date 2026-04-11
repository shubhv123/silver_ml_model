import pandas as pd
import json
from pathlib import Path
import os

print("="*60)
print("FEATURE ENGINEERING SUMMARY - WEEK 3 COMPLETE")
print("="*60)

# Load the reduced feature matrix
processed_path = "data/processed"
reduced_files = list(Path(processed_path).glob("feature_matrix_reduced_*.csv"))

if reduced_files:
    latest = sorted(reduced_files)[-1]
    df = pd.read_csv(latest, index_col=0, parse_dates=True)
    
    print(f"\n✅ Reduced Feature Matrix: {latest.name}")
    print(f"   Shape: {df.shape}")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")
    
    feature_cols = [c for c in df.columns if 'target' not in c]
    target_cols = [c for c in df.columns if 'target' in c]
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Targets: {len(target_cols)}")
    print(f"   Total columns: {len(df.columns)}")
    
    # Load selected features list
    selected_files = list(Path(processed_path).glob("selected_features_*.json"))
    if selected_files:
        latest_sel = sorted(selected_files)[-1]
        with open(latest_sel, 'r') as f:
            sel_data = json.load(f)
        print(f"\n✅ Feature Selection Summary:")
        print(f"   Initial features: {sel_data.get('initial_count', 'N/A')}")
        print(f"   Selected features: {sel_data.get('selected_count', 'N/A')}")
        print(f"   Reduction: {(1 - sel_data['selected_count']/sel_data['initial_count'])*100:.1f}%")
    
    # Show sample of selected features
    print(f"\n📊 Sample of Selected Features:")
    for i, feat in enumerate(sorted(feature_cols)[:10]):
        print(f"   {i+1}. {feat}")
    print("   ...")
    
    # Show targets
    print(f"\n🎯 Target Variables Available:")
    for target in sorted(target_cols):
        print(f"   • {target}")
        
    # Quick correlation check
    if 'target_next_day_return' in df.columns:
        print(f"\n📈 Quick Correlation Check:")
        correlations = []
        for col in feature_cols[:20]:  # Check first 20 features
            try:
                corr = df[col].corr(df['target_next_day_return'])
                if pd.notna(corr) and abs(corr) > 0.01:
                    correlations.append((col, corr))
            except:
                continue
        
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        for col, corr in correlations[:5]:
            print(f"   {col}: {corr:+.4f}")
else:
    print("No reduced feature matrix found. Run create_reduced_dataset.py first")

print("\n" + "="*60)
print("✅ WEEK 3 COMPLETE! Ready for Week 4 - Model Training")
print("="*60)
