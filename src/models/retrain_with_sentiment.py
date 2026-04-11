#!/usr/bin/env python
"""
Merge Sentiment Features into Master Feature Matrix
Retrain XGBoost Ensemble with Sentiment
Check SHAP to see sentiment impact - Fixed Version
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import logging
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Try to import shap with fallback
try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    SHAP_AVAILABLE = False
    logging.warning(f"SHAP not available: {e}")

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainWithSentiment:
    """Merge sentiment features and retrain XGBoost"""
    
    def __init__(self, target_col='target_next_day_return'):
        self.processed_path = "data/processed"
        self.models_path = "models/saved"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
        self.target_col = target_col
        self.scaler = StandardScaler()
        
    def load_feature_matrix(self):
        """Load existing feature matrix"""
        feature_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not feature_files:
            feature_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        
        if not feature_files:
            logger.error("No feature matrix found!")
            return None
        
        latest = sorted(feature_files)[-1]
        df = pd.read_csv(latest, index_col=0, parse_dates=True)
        logger.info(f"Loaded feature matrix: {latest.name} ({df.shape})")
        return df
    
    def load_sentiment_features(self):
        """Load engineered sentiment features"""
        sentiment_files = list(Path(self.processed_path).glob("finbert_daily_*.csv"))
        
        if not sentiment_files:
            logger.warning("No sentiment files found!")
            return None
        
        latest = sorted(sentiment_files)[-1]
        df = pd.read_csv(latest, index_col=0, parse_dates=True)
        logger.info(f"Loaded sentiment features: {latest.name} ({df.shape})")
        return df
    
    def merge_features(self, feature_df, sentiment_df):
        """Merge sentiment features with existing features"""
        
        logger.info("\n" + "="*60)
        logger.info("MERGING SENTIMENT FEATURES")
        logger.info("="*60)
        
        # Align dates
        common_dates = feature_df.index.intersection(sentiment_df.index)
        feature_aligned = feature_df.loc[common_dates]
        sentiment_aligned = sentiment_df.loc[common_dates]
        
        logger.info(f"Common dates: {len(common_dates)}")
        
        # Select sentiment features to add
        sentiment_cols = ['finbert_net', 'sentiment_5d_ma', 'sentiment_momentum', 'news_volume_zscore']
        available_cols = [col for col in sentiment_cols if col in sentiment_aligned.columns]
        
        # Merge
        merged_df = feature_aligned.copy()
        for col in available_cols:
            merged_df[f'sentiment_{col}'] = sentiment_aligned[col]
        
        logger.info(f"Added {len(available_cols)} sentiment features")
        logger.info(f"New feature matrix shape: {merged_df.shape}")
        
        return merged_df
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data for training"""
        
        # Separate features and target
        feature_cols = [c for c in df.columns if 'target' not in c]
        X = df[feature_cols].copy()
        y = df[self.target_col].copy()
        
        # Drop NaN targets
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Time-based split
        n = len(X)
        split_idx = int(n * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        X_train = pd.DataFrame(X_train_scaled, index=X_train.index, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_scaled, index=X_test.index, columns=X_test.columns)
        
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Train dates: {X_train.index[0]} to {X_train.index[-1]}")
        logger.info(f"Test dates: {X_test.index[0]} to {X_test.index[-1]}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost with sentiment features"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING XGBOOST WITH SENTIMENT")
        logger.info("="*60)
        
        # Parameters (optimized for sentiment)
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'rmse'
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_directional_accuracy': (np.sign(y_train) == np.sign(y_pred_train)).mean(),
            'test_directional_accuracy': (np.sign(y_test) == np.sign(y_pred_test)).mean()
        }
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info(f"\n📊 RESULTS WITH SENTIMENT:")
        logger.info(f"   Test RMSE: {metrics['test_rmse']:.6f}")
        logger.info(f"   Test Directional Accuracy: {metrics['test_directional_accuracy']:.2%}")
        
        return model, metrics, importance
    
    def shap_analysis_fallback(self, model, X_test, feature_cols):
        """Alternative SHAP analysis using feature importance"""
        
        logger.info("\n" + "="*60)
        logger.info("FEATURE IMPORTANCE ANALYSIS (SHAP Alternative)")
        logger.info("="*60)
        
        # Get feature importance from model
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Find sentiment features in top 20
        logger.info(f"\n📊 Sentiment Features in Top 20:")
        sentiment_found = False
        for i, row in importance.head(20).iterrows():
            if 'sentiment' in row['feature']:
                logger.info(f"   {row['feature']}: importance={row['importance']:.4f}")
                sentiment_found = True
        
        if not sentiment_found:
            logger.info("   No sentiment features in top 20 yet")
            logger.info("   Need more real sentiment data or better feature engineering")
        
        # Create feature importance plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_features = importance.head(20)
        colors = ['green' if 'sentiment' in f else 'steelblue' for f in top_features['feature']]
        
        ax.barh(range(len(top_features)), top_features['importance'].values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values)
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 20 Feature Importance (Green = Sentiment Features)')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        imp_file = os.path.join(self.figures_path, f"feature_importance_with_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(imp_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved feature importance plot: {imp_file}")
        
        # Return importance as DataFrame
        return importance
    
    def compare_without_sentiment(self, X_train, X_test, y_train, y_test):
        """Train model without sentiment for comparison"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING BASELINE (NO SENTIMENT)")
        logger.info("="*60)
        
        # Remove sentiment features
        non_sentiment_cols = [c for c in X_train.columns if 'sentiment' not in c]
        X_train_no_sent = X_train[non_sentiment_cols]
        X_test_no_sent = X_test[non_sentiment_cols]
        
        # Train model
        params = {
            'n_estimators': 300,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train_no_sent, y_train)
        
        y_pred = model.predict(X_test_no_sent)
        
        metrics = {
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_directional_accuracy': (np.sign(y_test) == np.sign(y_pred)).mean()
        }
        
        logger.info(f"\n📊 BASELINE (NO SENTIMENT):")
        logger.info(f"   Test RMSE: {metrics['test_rmse']:.6f}")
        logger.info(f"   Test Directional Accuracy: {metrics['test_directional_accuracy']:.2%}")
        
        return metrics
    
    def save_results(self, model, metrics, importance, comparison):
        """Save all results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = os.path.join(self.models_path, f"xgboost_with_sentiment_{timestamp}.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Saved model: {model_file}")
        
        # Save metrics
        results = {
            'timestamp': timestamp,
            'model': 'XGBoost_with_Sentiment',
            'metrics': metrics,
            'comparison_without_sentiment': comparison,
            'feature_importance': importance.head(20).to_dict('records')
        }
        
        results_file = os.path.join(self.results_path, f"sentiment_model_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Saved results: {results_file}")
        
        # Save feature importance CSV
        imp_file = os.path.join(self.results_path, f"feature_importance_with_sentiment_{timestamp}.csv")
        importance.to_csv(imp_file, index=False)
        logger.info(f"Saved feature importance: {imp_file}")
        
        return timestamp
    
    def plot_sentiment_impact(self, importance):
        """Visualize sentiment feature impact"""
        
        # Get sentiment features only
        sentiment_imp = importance[importance['feature'].str.contains('sentiment', case=False)]
        
        if len(sentiment_imp) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.barh(sentiment_imp['feature'], sentiment_imp['importance'], color='green', alpha=0.7)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Sentiment Feature Importance in XGBoost Model')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_file = os.path.join(self.figures_path, f"sentiment_impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved sentiment impact plot: {plot_file}")
        else:
            logger.info("No sentiment features found in importance")
    
    def run(self):
        """Run complete retraining pipeline"""
        
        logger.info("="*60)
        logger.info("RETRAINING XGBOOST WITH SENTIMENT FEATURES")
        logger.info("="*60)
        
        # Load data
        feature_df = self.load_feature_matrix()
        if feature_df is None:
            return
        
        sentiment_df = self.load_sentiment_features()
        if sentiment_df is None:
            return
        
        # Merge features
        merged_df = self.merge_features(feature_df, sentiment_df)
        
        # Prepare data
        X_train, X_test, y_train, y_test, feature_cols = self.prepare_data(merged_df)
        
        # Train with sentiment
        model, metrics, importance = self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Feature importance analysis (instead of SHAP)
        importance_df = self.shap_analysis_fallback(model, X_test, feature_cols)
        
        # Compare without sentiment
        comparison = self.compare_without_sentiment(X_train, X_test, y_train, y_test)
        
        # Visualize sentiment impact
        self.plot_sentiment_impact(importance_df)
        
        # Save results
        self.save_results(model, metrics, importance_df, comparison)
        
        # Print improvement summary
        logger.info("\n" + "="*60)
        logger.info("SENTIMENT IMPROVEMENT SUMMARY")
        logger.info("="*60)
        
        improvement_rmse = ((comparison['test_rmse'] - metrics['test_rmse']) / comparison['test_rmse']) * 100
        improvement_dir = metrics['test_directional_accuracy'] - comparison['test_directional_accuracy']
        
        logger.info(f"\n📈 RMSE Improvement: {improvement_rmse:+.2f}%")
        logger.info(f"📈 Directional Accuracy Improvement: {improvement_dir:+.2%}")
        
        if improvement_rmse > 0:
            logger.info(f"\n✅ Sentiment features improved model performance!")
        else:
            logger.info(f"\n⚠️ Sentiment features did not improve performance yet")
            logger.info(f"   Need more real sentiment data or feature engineering")
        
        # Find top sentiment feature
        sentiment_features = importance_df[importance_df['feature'].str.contains('sentiment')]
        if len(sentiment_features) > 0:
            top_sentiment = sentiment_features.iloc[0]
            logger.info(f"\n🔝 Top sentiment feature: {top_sentiment['feature']} (importance: {top_sentiment['importance']:.4f})")
        else:
            logger.info(f"\n⚠️ No sentiment features in top importance")
        
        logger.info("\n" + "="*60)
        logger.info("✅ RETRAINING COMPLETE")
        logger.info("="*60)
        
        return model, metrics

if __name__ == "__main__":
    trainer = RetrainWithSentiment()
    model, metrics = trainer.run()
    
    print("\n" + "="*60)
    print("FINAL RESULTS WITH SENTIMENT")
    print("="*60)
    print(f"Test RMSE: {metrics['test_rmse']:.6f}")
    print(f"Test Directional Accuracy: {metrics['test_directional_accuracy']:.2%}")
    
    # Calculate improvement over random
    random_accuracy = 0.5
    improvement = (metrics['test_directional_accuracy'] - random_accuracy) * 100
    print(f"Improvement over random: {improvement:+.2f}%")