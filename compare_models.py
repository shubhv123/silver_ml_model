import pandas as pd
import json
import os
from pathlib import Path

print("="*60)
print("MODEL COMPARISON - SILVER PRICE PREDICTION")
print("="*60)

results_path = "reports/metrics"

# Collect all results
model_results = []

for model_name in ['xgboost', 'lightgbm', 'catboost']:
    result_files = list(Path(results_path).glob(f"{model_name}_results_*.json"))
    
    if result_files:
        latest = sorted(result_files)[-1]
        with open(latest, 'r') as f:
            results = json.load(f)
        
        metrics = results['metrics']
        model_results.append({
            'Model': model_name.upper(),
            'Test RMSE': metrics['test_rmse'],
            'Test MAE': metrics['test_mae'],
            'Test R²': metrics['test_r2'],
            'Directional Accuracy': f"{metrics['test_directional_accuracy']:.2%}",
            'Best Params': results.get('best_params', {})
        })

# Create comparison dataframe
df_comparison = pd.DataFrame(model_results)
df_comparison = df_comparison.sort_values('Test RMSE')

print("\n📊 MODEL PERFORMANCE COMPARISON:")
print(df_comparison.to_string(index=False))

# Find best model
best_model = df_comparison.iloc[0]['Model']
best_accuracy = df_comparison.iloc[0]['Directional Accuracy']
best_rmse = df_comparison.iloc[0]['Test RMSE']

print("\n" + "="*60)
print(f"🏆 BEST MODEL: {best_model}")
print(f"   Test RMSE: {best_rmse:.6f}")
print(f"   Directional Accuracy: {best_accuracy}")
print("="*60)

# Save comparison
comparison_file = os.path.join(results_path, "model_comparison.csv")
df_comparison.to_csv(comparison_file, index=False)
print(f"\n💾 Comparison saved to: {comparison_file}")

# Load feature importances for best model
if best_model.lower() == 'xgboost':
    imp_files = list(Path(results_path).glob("xgboost_importance_*.csv"))
elif best_model.lower() == 'lightgbm':
    imp_files = list(Path(results_path).glob("lightgbm_importance_*.csv"))
else:
    imp_files = list(Path(results_path).glob("catboost_importance_*.csv"))

if imp_files:
    latest_imp = sorted(imp_files)[-1]
    importance_df = pd.read_csv(latest_imp)
    print(f"\n📈 TOP 10 FEATURES FROM {best_model}:")
    for i, row in importance_df.head(10).iterrows():
        print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
