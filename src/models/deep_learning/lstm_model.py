#!/usr/bin/env python
"""
LSTM Model using Keras Functional API
Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Output
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMFunctionalAPI:
    """LSTM model using Keras Functional API with specified architecture"""
    
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
        self.model = None
        
    def load_data(self):
        """Load the feature matrix"""
        final_files = list(Path(self.processed_path).glob("feature_matrix_final_*.csv"))
        if not final_files:
            final_files = list(Path(self.processed_path).glob("feature_matrix_cleaned_*.csv"))
        
        if not final_files:
            logger.error("No feature matrix found!")
            return None
        
        df = pd.read_csv(sorted(final_files)[-1], index_col=0, parse_dates=True)
        logger.info(f"Loaded data: {df.shape}")
        return df
    
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
        
        # Normalize per training window (no lookahead bias!)
        X_train_scaled = self.scaler_X.fit_transform(X_train_raw)
        X_test_scaled = self.scaler_X.transform(X_test_raw)
        
        y_train_scaled = self.scaler_y.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test_raw.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)
        
        # Align raw targets with sequences
        y_train_raw_aligned = y_train_raw[self.lookback:]
        y_test_raw_aligned = y_test_raw[self.lookback:]
        
        logger.info(f"Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")
        
        return X_train_seq, X_test_seq, y_train_seq, y_test_seq, y_train_raw_aligned, y_test_raw_aligned
    
    def build_model(self, input_shape, use_bidirectional=False):
        """
        Build LSTM model using Keras Functional API
        Architecture: Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Output
        """
        
        # Input layer
        inputs = Input(shape=input_shape, name='input_layer')
        
        # LSTM(128) with return_sequences=True for stacking
        if use_bidirectional:
            x = Bidirectional(LSTM(128, return_sequences=True, name='bilstm_1'))(inputs)
        else:
            x = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
        
        # Dropout(0.2)
        x = Dropout(0.2, name='dropout_1')(x)
        
        # LSTM(64)
        if use_bidirectional:
            x = Bidirectional(LSTM(64, return_sequences=False, name='bilstm_2'))(x)
        else:
            x = LSTM(64, return_sequences=False, name='lstm_2')(x)
        
        # Dropout(0.2)
        x = Dropout(0.2, name='dropout_2')(x)
        
        # Dense(32)
        x = Dense(32, activation='relu', name='dense_1')(x)
        
        # Dropout for regularization
        x = Dropout(0.1, name='dropout_3')(x)
        
        # Output layer
        outputs = Dense(1, name='output_layer')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_Functional_API')
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("\n" + "="*60)
        logger.info("MODEL ARCHITECTURE (Functional API)")
        logger.info("="*60)
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, use_bidirectional=False):
        """Train LSTM model"""
        
        logger.info("\n" + "="*60)
        logger.info("BUILDING LSTM MODEL WITH FUNCTIONAL API")
        logger.info("="*60)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model(input_shape, use_bidirectional)
        
        model_name = "bidirectional_lstm_functional" if use_bidirectional else "lstm_functional"
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING LSTM MODEL")
        logger.info("="*60)
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history, model_name
    
    def evaluate(self, X_test, y_test, y_test_raw):
        """Evaluate model"""
        
        y_pred_scaled = self.model.predict(X_test)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).flatten()
        
        # Ensure same length
        min_len = min(len(y_test_raw), len(y_pred))
        y_test_raw = y_test_raw[:min_len]
        y_pred = y_pred[:min_len]
        
        rmse = np.sqrt(mean_squared_error(y_test_raw, y_pred))
        mae = mean_absolute_error(y_test_raw, y_pred)
        r2 = r2_score(y_test_raw, y_pred)
        direction_accuracy = (np.sign(y_test_raw) == np.sign(y_pred)).mean()
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'directional_accuracy': float(direction_accuracy)
        }
        
        logger.info("\n" + "="*60)
        logger.info("LSTM EVALUATION")
        logger.info("="*60)
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Directional Accuracy: {direction_accuracy:.2%}")
        
        return metrics, y_pred
    
    def plot_results(self, history, y_test, y_pred, metrics, model_name):
        """Plot results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title(f'{model_name.upper()} - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Predictions
        axes[0, 1].plot(y_test[-500:], label='Actual', alpha=0.7)
        axes[0, 1].plot(y_pred[-500:], label='Predicted', alpha=0.7)
        axes[0, 1].set_title('Predictions vs Actual (Last 500 days)')
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter
        axes[1, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        axes[1, 0].set_title(f'Predicted vs Actual (R² = {metrics["r2"]:.4f})')
        axes[1, 0].set_xlabel('Actual')
        axes[1, 0].set_ylabel('Predicted')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].axvline(x=0, color='red', linestyle='--')
        axes[1, 1].set_title(f'Residuals (MAE = {metrics["mae"]:.6f})')
        axes[1, 1].set_xlabel('Residual')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f"{model_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), dpi=150)
        plt.close()
        logger.info(f"Plot saved to {self.figures_path}")
    
    def save_results(self, metrics, history, model_name):
        """Save results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_file = os.path.join(self.models_path, f"{model_name}_{timestamp}.keras")
        self.model.save(model_file)
        logger.info(f"Model saved: {model_file}")
        
        results = {
            'timestamp': timestamp,
            'model_type': model_name,
            'lookback': self.lookback,
            'architecture': 'Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Output',
            'functional_api': True,
            'metrics': metrics,
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        results_file = os.path.join(self.results_path, f"{model_name}_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_file}")
    
    def run(self, use_bidirectional=False, epochs=50):
        """Run training pipeline"""
        
        logger.info("="*60)
        logger.info("LSTM MODEL - KERAS FUNCTIONAL API")
        logger.info(f"Lookback window: {self.lookback} days")
        logger.info(f"Architecture: Input → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → Dense(32) → Output")
        logger.info(f"Type: {'Bidirectional' if use_bidirectional else 'Standard'}")
        logger.info("="*60)
        
        df = self.load_data()
        if df is None:
            return
        
        X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = self.prepare_data(df)
        
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error("Not enough data for sequences!")
            return
        
        history, model_name = self.train(X_train, y_train, X_test, y_test, epochs, use_bidirectional=use_bidirectional)
        
        metrics, y_pred = self.evaluate(X_test, y_test, y_test_raw)
        
        self.plot_results(history, y_test_raw, y_pred, metrics, model_name)
        
        self.save_results(metrics, history, model_name)
        
        logger.info("\n" + "="*60)
        logger.info("✅ LSTM MODEL (FUNCTIONAL API) COMPLETE")
        logger.info("="*60)
        
        return metrics

if __name__ == "__main__":
    # Train standard LSTM with Functional API
    lstm = LSTMFunctionalAPI(lookback=60)
    lstm.run(use_bidirectional=False, epochs=50)
    
    # Train Bidirectional LSTM with Functional API
    lstm_bi = LSTMFunctionalAPI(lookback=60)
    lstm_bi.run(use_bidirectional=True, epochs=50)