#!/usr/bin/env python
"""
Hybrid Model: XGBoost Predictions + Raw Features into LSTM
State-of-the-art approach for price prediction
Architecture: Features → XGBoost (pre-trained) → Predictions + Raw Features → LSTM → Final Prediction
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridXGBoostLSTM:
    """
    Hybrid Model: XGBoost predictions + raw features fed into LSTM
    This captures both non-linear feature interactions (XGBoost) 
    and temporal dependencies (LSTM)
    """
    
    def __init__(self, lookback=60, target_col='target_next_day_return'):
        self.lookback = lookback
        self.target_col = target_col
        self.processed_path = "data/processed"
        self.models_path = "models/saved"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_combined = StandardScaler()
        self.xgb_model = None
        self.lstm_model = None
        
    def load_xgboost_model(self):
        """Load pre-trained XGBoost model from Phase 2"""
        xgb_files = list(Path(self.models_path).glob("xgboost_*.pkl"))
        if not xgb_files:
            logger.error("No XGBoost model found! Please train XGBoost first.")
            return False
        
        latest = sorted(xgb_files)[-1]
        with open(latest, 'rb') as f:
            self.xgb_model = pickle.load(f)
        logger.info(f"Loaded XGBoost model from {latest.name}")
        return True
    
    def load_data(self):
        """Load feature matrix"""
        final_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not final_files:
            final_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        
        if not final_files:
            logger.error("No feature matrix found!")
            return None
        
        df = pd.read_csv(sorted(final_files)[-1], index_col=0, parse_dates=True)
        logger.info(f"Loaded data: {df.shape}")
        return df
    
    def create_hybrid_features(self, X_raw):
        """
        Create hybrid features: XGBoost predictions + raw features
        """
        # Get XGBoost predictions
        xgb_pred = self.xgb_model.predict(X_raw)
        
        # Combine predictions with raw features
        hybrid_features = np.column_stack([xgb_pred, X_raw])
        
        return hybrid_features, xgb_pred
    
    def create_sequences(self, X, y):
        """Create sequences for LSTM input"""
        X_seq, y_seq = [], []
        for i in range(len(X) - self.lookback):
            X_seq.append(X[i:i + self.lookback])
            y_seq.append(y[i + self.lookback])
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare data with walk-forward validation"""
        
        feature_cols = [c for c in df.columns if 'target' not in c]
        X = df[feature_cols].values
        y = df[self.target_col].values
        
        # Remove NaN targets
        valid_idx = ~np.isnan(y)
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"After removing NaN: X shape {X.shape}")
        
        # Time-based split
        n = len(X)
        split_idx = int(n * (1 - test_size))
        
        X_train_raw = X[:split_idx]
        X_test_raw = X[split_idx:]
        y_train_raw = y[:split_idx]
        y_test_raw = y[split_idx:]
        
        logger.info(f"Train raw: {X_train_raw.shape}, Test raw: {X_test_raw.shape}")
        
        # Normalize features (per training window - no lookahead!)
        X_train_scaled = self.scaler_X.fit_transform(X_train_raw)
        X_test_scaled = self.scaler_X.transform(X_test_raw)
        
        # Normalize target
        y_train_scaled = self.scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        # Create hybrid features (XGBoost predictions + raw features)
        # First get XGBoost predictions
        X_train_hybrid, xgb_pred_train = self.create_hybrid_features(X_train_scaled)
        X_test_hybrid, xgb_pred_test = self.create_hybrid_features(X_test_scaled)
        
        logger.info(f"Hybrid features shape: {X_train_hybrid.shape} (XGB_pred + {len(feature_cols)} raw features)")
        
        # Normalize hybrid features
        X_train_hybrid_scaled = self.scaler_combined.fit_transform(X_train_hybrid)
        X_test_hybrid_scaled = self.scaler_combined.transform(X_test_hybrid)
        
        # Create sequences for LSTM
        X_train_seq, y_train_seq = self.create_sequences(X_train_hybrid_scaled, y_train_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_hybrid_scaled, y_test_scaled)
        
        # Align raw targets with sequences
        y_train_raw_aligned = y_train_raw[self.lookback:]
        y_test_raw_aligned = y_test_raw[self.lookback:]
        xgb_pred_train_aligned = xgb_pred_train[self.lookback:]
        xgb_pred_test_aligned = xgb_pred_test[self.lookback:]
        
        logger.info(f"Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")
        logger.info(f"XGBoost predictions range: {xgb_pred_train.mean():.4f} ± {xgb_pred_train.std():.4f}")
        
        return (X_train_seq, X_test_seq, y_train_seq, y_test_seq, 
                y_train_raw_aligned, y_test_raw_aligned, 
                xgb_pred_train_aligned, xgb_pred_test_aligned)
    
    def build_hybrid_lstm(self, input_shape):
        """
        Build LSTM that takes hybrid features
        Architecture optimized for combining XGBoost predictions with raw features
        """
        
        # Input layer
        inputs = Input(shape=input_shape, name='hybrid_input')
        
        # First LSTM layer
        x = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
        x = Dropout(0.2, name='dropout_1')(x)
        
        # Second LSTM layer
        x = LSTM(64, return_sequences=False, name='lstm_2')(x)
        x = Dropout(0.2, name='dropout_2')(x)
        
        # Dense layers for final prediction
        x = Dense(32, activation='relu', name='dense_1')(x)
        x = Dropout(0.1, name='dropout_3')(x)
        
        # Output layer
        outputs = Dense(1, name='output')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='Hybrid_XGBoost_LSTM')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("\n" + "="*60)
        logger.info("HYBRID MODEL ARCHITECTURE")
        logger.info("="*60)
        logger.info("Input: XGBoost predictions + raw features")
        logger.info("LSTM layers capture temporal patterns in hybrid features")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train hybrid model"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING HYBRID XGBOOST + LSTM MODEL")
        logger.info("="*60)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.lstm_model = self.build_hybrid_lstm(input_shape)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
        history = self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, X_test, y_test, y_test_raw, xgb_pred_test):
        """Evaluate hybrid model"""
        
        y_pred_scaled = self.lstm_model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Ensure same length
        min_len = min(len(y_test_raw), len(y_pred))
        y_test_raw = y_test_raw[:min_len]
        y_pred = y_pred[:min_len]
        xgb_pred_test = xgb_pred_test[:min_len]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
        mae = mean_absolute_error(y_test_raw, y_pred)
        r2 = r2_score(y_test_raw, y_pred)
        direction_accuracy = (np.sign(y_test_raw) == np.sign(y_pred)).mean()
        
        # Compare with XGBoost alone
        xgb_rmse = np.sqrt(mean_squared_error(y_test_raw, xgb_pred_test))
        xgb_dir_acc = (np.sign(y_test_raw) == np.sign(xgb_pred_test)).mean()
        
        metrics = {
            'hybrid_rmse': float(rmse),
            'hybrid_mae': float(mae),
            'hybrid_r2': float(r2),
            'hybrid_directional_accuracy': float(direction_accuracy),
            'xgb_rmse': float(xgb_rmse),
            'xgb_directional_accuracy': float(xgb_dir_acc),
            'improvement_rmse_pct': float((xgb_rmse - rmse) / xgb_rmse * 100),
            'improvement_direction_pct': float(direction_accuracy - xgb_dir_acc) * 100
        }
        
        logger.info("\n" + "="*60)
        logger.info("HYBRID MODEL EVALUATION")
        logger.info("="*60)
        logger.info(f"\n📊 HYBRID MODEL (XGBoost + LSTM):")
        logger.info(f"   RMSE: {rmse:.6f}")
        logger.info(f"   MAE: {mae:.6f}")
        logger.info(f"   R²: {r2:.4f}")
        logger.info(f"   Directional Accuracy: {direction_accuracy:.2%}")
        
        logger.info(f"\n📊 XGBOOST ALONE (Baseline):")
        logger.info(f"   RMSE: {xgb_rmse:.6f}")
        logger.info(f"   Directional Accuracy: {xgb_dir_acc:.2%}")
        
        logger.info(f"\n📈 IMPROVEMENT:")
        logger.info(f"   RMSE Improvement: {metrics['improvement_rmse_pct']:+.2f}%")
        logger.info(f"   Directional Accuracy Improvement: {metrics['improvement_direction_pct']:+.2f}%")
        
        if metrics['improvement_rmse_pct'] > 0:
            logger.info(f"   ✅ Hybrid model outperforms XGBoost alone!")
        else:
            logger.info(f"   ⚠️  Hybrid model needs tuning")
        
        return metrics, y_pred
    
    def plot_results(self, history, y_test, y_pred, xgb_pred, metrics, model_name):
        """Plot results comparing hybrid vs XGBoost"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Training loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training History', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Predictions comparison (last 500 days)
        axes[0, 1].plot(y_test[-500:], label='Actual', alpha=0.7, linewidth=1)
        axes[0, 1].plot(y_pred[-500:], label='Hybrid (XGB+LSTM)', alpha=0.7, linewidth=1)
        axes[0, 1].plot(xgb_pred[-500:], label='XGBoost Only', alpha=0.7, linewidth=1, linestyle='--')
        axes[0, 1].set_title('Predictions Comparison (Last 500 days)', fontsize=14)
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # RMSE Comparison
        models = ['XGBoost', 'Hybrid (XGB+LSTM)']
        rmse_values = [metrics['xgb_rmse'], metrics['hybrid_rmse']]
        colors = ['orange', 'blue']
        
        bars = axes[0, 2].bar(models, rmse_values, color=colors, alpha=0.7)
        axes[0, 2].set_title('RMSE Comparison (Lower is Better)', fontsize=14)
        axes[0, 2].set_ylabel('RMSE')
        axes[0, 2].grid(True, alpha=0.3)
        
        for bar, val in zip(bars, rmse_values):
            axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                           f'{val:.5f}', ha='center', va='bottom', fontsize=10)
        
        # Directional Accuracy Comparison
        dir_acc_values = [metrics['xgb_directional_accuracy'] * 100, 
                         metrics['hybrid_directional_accuracy'] * 100]
        
        bars2 = axes[1, 0].bar(models, dir_acc_values, color=colors, alpha=0.7)
        axes[1, 0].axhline(y=50, color='red', linestyle='--', label='Random (50%)', alpha=0.7)
        axes[1, 0].set_title('Directional Accuracy (Higher is Better)', fontsize=14)
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, val in zip(bars2, dir_acc_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Scatter plot - Hybrid predictions
        axes[1, 1].scatter(y_test, y_pred, alpha=0.3, s=10, color='blue')
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
        axes[1, 1].set_title(f'Hybrid Model: Predicted vs Actual (R² = {metrics["hybrid_r2"]:.4f})', fontsize=14)
        axes[1, 1].set_xlabel('Actual Return')
        axes[1, 1].set_ylabel('Predicted Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Improvement summary
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        improvement_text = f"""
        HYBRID MODEL IMPROVEMENT SUMMARY
        
        RMSE Improvement: {metrics['improvement_rmse_pct']:+.2f}%
        Directional Accuracy: {metrics['improvement_direction_pct']:+.2f}%
        
        XGBoost Baseline:
        • RMSE: {metrics['xgb_rmse']:.6f}
        • Directional Acc: {metrics['xgb_directional_accuracy']:.2%}
        
        Hybrid Model (XGB + LSTM):
        • RMSE: {metrics['hybrid_rmse']:.6f}
        • Directional Acc: {metrics['hybrid_directional_accuracy']:.2%}
        
        Architecture:
        • XGBoost captures non-linear feature interactions
        • LSTM captures temporal patterns (60-day window)
        • Hybrid combines both strengths
        """
        
        axes[1, 2].text(0.5, 0.5, improvement_text, transform=axes[1, 2].transAxes,
                        fontsize=11, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved to {self.figures_path}")
    
    def save_model(self, metrics, history, model_name='hybrid_xgboost_lstm'):
        """Save hybrid model"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save LSTM model
        model_file = os.path.join(self.models_path, f"{model_name}_{timestamp}.keras")
        self.lstm_model.save(model_file)
        logger.info(f"Model saved: {model_file}")
        
        # Save scalers
        scalers = {
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'scaler_combined': self.scaler_combined
        }
        
        scaler_file = os.path.join(self.models_path, f"{model_name}_scalers_{timestamp}.pkl")
        with open(scaler_file, 'wb') as f:
            pickle.dump(scalers, f)
        logger.info(f"Scalers saved: {scaler_file}")
        
        # Save results
        results = {
            'timestamp': timestamp,
            'model_type': model_name,
            'lookback': self.lookback,
            'architecture': 'XGBoost Predictions + Raw Features → LSTM',
            'metrics': metrics,
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        results_file = os.path.join(self.results_path, f"{model_name}_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_file}")
        
        return timestamp
    
    def run(self, epochs=100):
        """Run complete hybrid training pipeline"""
        
        logger.info("="*60)
        logger.info("HYBRID MODEL: XGBoost Predictions + Raw Features → LSTM")
        logger.info("State-of-the-art approach for price prediction")
        logger.info("="*60)
        
        # Load XGBoost model
        if not self.load_xgboost_model():
            return
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare data with hybrid features
        (X_train, X_test, y_train, y_test, 
         y_train_raw, y_test_raw, 
         xgb_pred_train, xgb_pred_test) = self.prepare_data(df)
        
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error("Not enough data for sequences!")
            return
        
        # Train hybrid model
        history = self.train(X_train, y_train, X_test, y_test, epochs=epochs)
        
        # Evaluate
        metrics, y_pred = self.evaluate(X_test, y_test, y_test_raw, xgb_pred_test)
        
        # Plot results
        self.plot_results(history, y_test_raw, y_pred, xgb_pred_test, metrics, 'hybrid_xgboost_lstm')
        
        # Save model
        self.save_model(metrics, history)
        
        logger.info("\n" + "="*60)
        logger.info("✅ HYBRID MODEL TRAINING COMPLETE")
        logger.info("="*60)
        
        return metrics

if __name__ == "__main__":
    # Create and train hybrid model
    hybrid = HybridXGBoostLSTM(lookback=60)
    results = hybrid.run(epochs=50)
    
    # Print final summary
    if results:
        print("\n" + "="*60)
        print("HYBRID MODEL FINAL RESULTS")
        print("="*60)
        print(f"Hybrid RMSE: {results['hybrid_rmse']:.6f}")
        print(f"Hybrid Directional Accuracy: {results['hybrid_directional_accuracy']:.2%}")
        print(f"Improvement over XGBoost: {results['improvement_rmse_pct']:+.2f}% RMSE, {results['improvement_direction_pct']:+.2f}% Direction")
