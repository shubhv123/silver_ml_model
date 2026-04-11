#!/usr/bin/env python
"""
Joint Hyperparameter Optimization for All 3 Base Models Simultaneously
Optimizes XGBoost, LightGBM, and CatBoost together for ensemble performance
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JointOptimizer:
    """Optimize all 3 models simultaneously for ensemble performance"""
    
    def __init__(self, target_col='target_next_day_return'):
        self.processed_path = "data/processed"
        self.target_col = target_col
        
    def load_data(self):
        final_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not final_files:
            final_files = list(Path(self.processed_path).glob("feature_matrix_reduced_*.csv"))
        latest = sorted(final_files)[-1]
        df = pd.read_csv(latest, index_col=0, parse_dates=True)
        feature_cols = [c for c in df.columns if 'target' not in c]
        X = df[feature_cols].fillna(0)
        y = df[self.target_col].dropna()
        X = X.loc[y.index]
        return X, y
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Optimize all 3 models together"""
        
        # XGBoost params
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample', 0.6, 1.0),
        }
        
        # LightGBM params
        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 300),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 8),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
        }
        
        # CatBoost params
        cb_params = {
            'iterations': trial.suggest_int('cb_iterations', 100, 300),
            'depth': trial.suggest_int('cb_depth', 3, 6),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.2, log=True),
        }
        
        # Ensemble weights
        weight_xgb = trial.suggest_float('weight_xgb', 0, 1)
        weight_lgb = trial.suggest_float('weight_lgb', 0, 1)
        weight_cb = trial.suggest_float('weight_cb', 0, 1)
        
        # Normalize weights
        total = weight_xgb + weight_lgb + weight_cb
        w_xgb, w_lgb, w_cb = weight_xgb/total, weight_lgb/total, weight_cb/total
        
        # Train models
        model_xgb = xgb.XGBRegressor(**xgb_params, random_state=42, verbose=0)
        model_lgb = lgb.LGBMRegressor(**lgb_params, random_state=42, verbose=-1)
        model_cb = cb.CatBoostRegressor(**cb_params, random_seed=42, verbose=False)
        
        model_xgb.fit(X_train, y_train)
        model_lgb.fit(X_train, y_train)
        model_cb.fit(X_train, y_train)
        
        # Ensemble predictions
        pred_xgb = model_xgb.predict(X_val)
        pred_lgb = model_lgb.predict(X_val)
        pred_cb = model_cb.predict(X_val)
        pred_ensemble = w_xgb * pred_xgb + w_lgb * pred_lgb + w_cb * pred_cb
        
        rmse = np.sqrt(mean_squared_error(y_val, pred_ensemble))
        return rmse
    
    def run(self, n_trials=50):
        logger.info("="*60)
        logger.info("JOINT OPTIMIZATION - ALL 3 MODELS SIMULTANEOUSLY")
        logger.info("="*60)
        
        X, y = self.load_data()
        logger.info(f"Data shape: {X.shape}")
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
        # Optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        logger.info(f"\nBest RMSE: {study.best_value:.6f}")
        logger.info(f"Best params: {study.best_params}")
        
        # Save results
        results_file = f"reports/metrics/joint_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'best_rmse': study.best_value,
                'best_params': study.best_params
            }, f, indent=2)
        
        logger.info(f"Saved to {results_file}")
        return study.best_params

if __name__ == "__main__":
    optimizer = JointOptimizer()
    optimizer.run(n_trials=50)
