#!/usr/bin/env python
"""
LightGBM Model Training with Walk-Forward Validation - FIXED
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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import optuna

# Suppress lightgbm warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LightGBMTrainer:
    """Train LightGBM model with walk-forward validation"""
    
    def __init__(self, target_col='target_next_day_return'):
        self.processed_path = "data/processed"
        self.models_path = "models/saved"
        self.results_path = "reports/metrics"
        
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        
        self.target_col = target_col
        self.best_params = None
        self.model = None
        self.feature_importance = None
        
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
        logger.info(f"Train dates: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"Test dates: {X_test.index[0]} to {X_test.index[-1]}")
        
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        return X_train, X_test, y_train, y_test
    
    def objective(self, trial, X_train, y_train):
        """Optuna objective function"""
        
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'random_state': 42,
            'verbose': -1
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(10, verbose=False)]
            )
            
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            cv_scores.append(rmse)
        
        return np.mean(cv_scores)
    
    def train(self, X_train, y_train, X_test, y_test, n_trials=20):
        """Train model with hyperparameter optimization"""
        
        logger.info("\n" + "="*50)
        logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
        logger.info("="*50)
        
        study = optuna.create_study(direction='minimize')
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        self.best_params = study.best_params
        logger.info(f"\nBest parameters: {self.best_params}")
        logger.info(f"Best CV RMSE: {study.best_value:.6f}")
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING FINAL MODEL")
        logger.info("="*50)
        
        self.model = lgb.LGBMRegressor(**self.best_params, random_state=42, verbose=-1)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='rmse',
            callbacks=[lgb.early_stopping(10, verbose=False)]
        )
        
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        metrics = self.calculate_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
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
    
    def save_model(self, metrics):
        """Save model and results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_file = os.path.join(self.models_path, f"lightgbm_{timestamp}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"\n💾 Model saved: {model_file}")
        
        results = {
            'timestamp': timestamp,
            'target': self.target_col,
            'best_params': self.best_params,
            'metrics': metrics,
            'feature_importance': self.feature_importance.head(20).to_dict('records'),
            'n_features': len(self.feature_importance)
        }
        
        results_file = os.path.join(self.results_path, f"lightgbm_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"📊 Results saved: {results_file}")
        
        imp_file = os.path.join(self.results_path, f"lightgbm_importance_{timestamp}.csv")
        self.feature_importance.to_csv(imp_file, index=False)
        logger.info(f"📈 Feature importance saved: {imp_file}")
        
        return timestamp
    
    def print_summary(self, metrics):
        """Print model summary"""
        
        logger.info("\n" + "="*60)
        logger.info("LIGHTGBM MODEL RESULTS")
        logger.info("="*60)
        
        logger.info(f"\n📊 Test Metrics:")
        logger.info(f"   RMSE: {metrics['test_rmse']:.6f}")
        logger.info(f"   MAE: {metrics['test_mae']:.6f}")
        logger.info(f"   R²: {metrics['test_r2']:.4f}")
        logger.info(f"   Directional Accuracy: {metrics['test_directional_accuracy']:.2%}")
        
        logger.info(f"\n📈 Train Metrics:")
        logger.info(f"   RMSE: {metrics['train_rmse']:.6f}")
        logger.info(f"   MAE: {metrics['train_mae']:.6f}")
        logger.info(f"   R²: {metrics['train_r2']:.4f}")
        logger.info(f"   Directional Accuracy: {metrics['train_directional_accuracy']:.2%}")
        
        logger.info(f"\n🔝 Top 10 Most Important Features:")
        for i, row in self.feature_importance.head(10).iterrows():
            logger.info(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
    
    def run(self):
        """Run the complete training pipeline"""
        
        logger.info("="*60)
        logger.info("LIGHTGBM MODEL TRAINING")
        logger.info("="*60)
        
        df = self.load_data()
        if df is None:
            return
        
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        metrics = self.train(X_train, y_train, X_test, y_test, n_trials=20)
        
        self.print_summary(metrics)
        
        timestamp = self.save_model(metrics)
        
        logger.info("\n" + "="*60)
        logger.info("✅ LIGHTGBM TRAINING COMPLETE")
        logger.info("="*60)
        
        return metrics

if __name__ == "__main__":
    trainer = LightGBMTrainer(target_col='target_next_day_return')
    trainer.run()