#!/usr/bin/env python
"""
Ensemble Model & SHAP Analysis
Combines XGBoost, LightGBM, and CatBoost into a stacking ensemble
Provides SHAP explanations for model predictions
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import pickle
import json
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# ML libraries
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import matplotlib.pyplot as plt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnsembleModel:
    """Stacking ensemble of XGBoost, LightGBM, and CatBoost"""
    
    def __init__(self, target_col='target_next_day_return'):
        self.processed_path = "data/processed"
        self.models_path = "models/saved"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.figures_path, exist_ok=True)
        
        self.target_col = target_col
        self.base_models = {}
        self.meta_model = None
        
    def load_data(self):
        """Load the final feature matrix"""
        final_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        
        if not final_files:
            final_files = list(Path(self.processed_path).glob("feature_matrix_reduced_*.csv"))
        
        if not final_files:
            logger.error("No feature matrix found!")
            return None
        
        latest = sorted(final_files)[-1]
        logger.info(f"Loading: {latest.name}")
        df = pd.read_csv(latest, index_col=0, parse_dates=True)
        logger.info(f"Shape: {df.shape}")
        
        return df
    
    def load_trained_models(self):
        """Load previously trained models"""
        
        # Load XGBoost
        xgb_files = list(Path(self.models_path).glob("xgboost_*.pkl"))
        if xgb_files:
            latest_xgb = sorted(xgb_files)[-1]
            with open(latest_xgb, 'rb') as f:
                self.base_models['xgboost'] = pickle.load(f)
            logger.info(f"Loaded XGBoost from {latest_xgb.name}")
        
        # Load LightGBM
        lgb_files = list(Path(self.models_path).glob("lightgbm_*.pkl"))
        if lgb_files:
            latest_lgb = sorted(lgb_files)[-1]
            with open(latest_lgb, 'rb') as f:
                self.base_models['lightgbm'] = pickle.load(f)
            logger.info(f"Loaded LightGBM from {latest_lgb.name}")
        
        # Load CatBoost
        cb_files = list(Path(self.models_path).glob("catboost_*.cbm"))
        if cb_files:
            latest_cb = sorted(cb_files)[-1]
            self.base_models['catboost'] = cb.CatBoostRegressor()
            self.base_models['catboost'].load_model(latest_cb)
            logger.info(f"Loaded CatBoost from {latest_cb.name}")
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data with walk-forward split"""
        
        feature_cols = [c for c in df.columns if 'target' not in c]
        X = df[feature_cols].copy()
        y = df[self.target_col].copy()
        
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        n = len(X)
        split_idx = int(n * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        return X_train, X_test, y_train, y_test
    
    def create_meta_features(self, X):
        """Create meta-features from base model predictions"""
        
        meta_features = []
        model_names = []
        
        for name, model in self.base_models.items():
            preds = model.predict(X)
            meta_features.append(preds)
            model_names.append(f"{name}_pred")
        
        meta_features = np.column_stack(meta_features)
        return meta_features, model_names
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train meta-model on base model predictions"""
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING STACKING ENSEMBLE")
        logger.info("="*50)
        
        # Get base model predictions
        X_train_meta, meta_names = self.create_meta_features(X_train)
        X_test_meta, _ = self.create_meta_features(X_test)
        
        logger.info(f"Meta-features shape: {X_train_meta.shape}")
        
        # Train meta-model (Ridge regression is simple and effective)
        self.meta_model = Ridge(alpha=1.0, random_state=42)
        self.meta_model.fit(X_train_meta, y_train)
        
        logger.info(f"Meta-model coefficients:")
        for name, coef in zip(meta_names, self.meta_model.coef_):
            logger.info(f"  {name}: {coef:.4f}")
        
        # Make ensemble predictions
        y_pred_train = self.meta_model.predict(X_train_meta)
        y_pred_test = self.meta_model.predict(X_test_meta)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        return metrics
    
    def calculate_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """Calculate regression metrics"""
        
        train_direction = (np.sign(y_train) == np.sign(y_pred_train)).mean()
        test_direction = (np.sign(y_test) == np.sign(y_pred_test)).mean()
        
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_directional_accuracy': train_direction,
            'test_directional_accuracy': test_direction
        }
        
        return metrics
    
    def shap_analysis(self, X_test, model_name='xgboost'):
        """Perform SHAP analysis on the best model"""
        
        logger.info("\n" + "="*50)
        logger.info(f"SHAP ANALYSIS - {model_name.upper()}")
        logger.info("="*50)
        
        if model_name not in self.base_models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.base_models[model_name]
        
        # Use a sample for SHAP (it can be slow on full dataset)
        X_sample = X_test.sample(min(500, len(X_test)))
        
        logger.info(f"Computing SHAP values for {len(X_sample)} samples...")
        
        # Create SHAP explainer based on model type
        if model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'lightgbm':
            explainer = shap.TreeExplainer(model)
        else:  # catboost
            explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.tight_layout()
        
        shap_file = os.path.join(self.figures_path, f"shap_summary_{model_name}_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(shap_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP summary to {shap_file}")
        
        # Bar plot of feature importance
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        
        shap_bar_file = os.path.join(self.figures_path, f"shap_bar_{model_name}_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(shap_bar_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP bar plot to {shap_bar_file}")
        
        # Get top features by SHAP importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_features_idx = np.argsort(feature_importance)[-10:][::-1]
        
        logger.info(f"\nTop 10 features by SHAP importance:")
        for i, idx in enumerate(top_features_idx):
            logger.info(f"  {i+1}. {X_sample.columns[idx]}: {feature_importance[idx]:.4f}")
        
        return shap_values
    
    def compare_all_models(self, X_train, X_test, y_train, y_test):
        """Compare all models including ensemble"""
        
        logger.info("\n" + "="*60)
        logger.info("FINAL MODEL COMPARISON")
        logger.info("="*60)
        
        results = []
        
        # Individual models
        for name, model in self.base_models.items():
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            direction = (np.sign(y_test) == np.sign(y_pred)).mean()
            results.append({
                'Model': name.upper(),
                'Test RMSE': rmse,
                'Directional Accuracy': f"{direction:.2%}"
            })
        
        # Ensemble
        X_test_meta, _ = self.create_meta_features(X_test)
        y_pred_ensemble = self.meta_model.predict(X_test_meta)
        rmse_ensemble = np.sqrt(mean_squared_error(y_test, y_pred_ensemble))
        direction_ensemble = (np.sign(y_test) == np.sign(y_pred_ensemble)).mean()
        results.append({
            'Model': 'ENSEMBLE',
            'Test RMSE': rmse_ensemble,
            'Directional Accuracy': f"{direction_ensemble:.2%}"
        })
        
        # Create comparison dataframe
        df_compare = pd.DataFrame(results).sort_values('Test RMSE')
        
        logger.info("\n" + df_compare.to_string(index=False))
        
        # Save comparison
        compare_file = os.path.join(self.results_path, f"final_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df_compare.to_csv(compare_file, index=False)
        logger.info(f"\nSaved comparison to {compare_file}")
        
        return df_compare
    
    def save_ensemble(self):
        """Save ensemble model"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        ensemble_file = os.path.join(self.models_path, f"ensemble_meta_{timestamp}.pkl")
        with open(ensemble_file, 'wb') as f:
            pickle.dump(self.meta_model, f)
        logger.info(f"Ensemble model saved to {ensemble_file}")
        
        return timestamp
    
    def run(self):
        """Run the complete ensemble and SHAP pipeline"""
        
        logger.info("="*60)
        logger.info("ENSEMBLE MODEL & SHAP ANALYSIS")
        logger.info("="*60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Load trained models
        self.load_trained_models()
        
        if not self.base_models:
            logger.error("No base models found!")
            return
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train ensemble
        metrics = self.train_ensemble(X_train, y_train, X_test, y_test)
        
        logger.info("\n" + "="*50)
        logger.info("ENSEMBLE RESULTS")
        logger.info("="*50)
        logger.info(f"Test RMSE: {metrics['test_rmse']:.6f}")
        logger.info(f"Test MAE: {metrics['test_mae']:.6f}")
        logger.info(f"Test R²: {metrics['test_r2']:.4f}")
        logger.info(f"Directional Accuracy: {metrics['test_directional_accuracy']:.2%}")
        
        # Compare all models
        comparison = self.compare_all_models(X_train, X_test, y_train, y_test)
        
        # Find best model for SHAP analysis
        best_model = comparison.iloc[0]['Model'].lower()
        if best_model == 'ensemble':
            best_model = 'xgboost'  # Use XGBoost for SHAP if ensemble wins
        
        # Perform SHAP analysis on best model
        self.shap_analysis(X_test, best_model)
        
        # Save ensemble
        self.save_ensemble()
        
        logger.info("\n" + "="*60)
        logger.info("✅ ENSEMBLE & SHAP COMPLETE")
        logger.info("="*60)
        
        return metrics

if __name__ == "__main__":
    ensemble = EnsembleModel(target_col='target_next_day_return')
    ensemble.run()
