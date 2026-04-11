#!/usr/bin/env python
"""
LSTM with Self-Attention Layer - Fixed
Self-attention helps model focus on the most relevant timesteps
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
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, 
    Flatten, GlobalAveragePooling1D, LayerNormalization, Add
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LSTMWithAttention:
    """LSTM Model with Self-Attention Layer"""
    
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
        
        # Normalize per training window
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
    
    def attention_layer(self, inputs):
        """
        Custom self-attention layer
        Computes attention weights for each timestep
        """
        # Compute attention scores using a dense layer
        attention_scores = Dense(1, activation='tanh', name='attention_dense')(inputs)
        attention_weights = Flatten(name='attention_flatten')(attention_scores)
        attention_weights = tf.nn.softmax(attention_weights, name='attention_softmax')
        
        # Reshape to (batch_size, timesteps, 1)
        attention_weights = tf.expand_dims(attention_weights, axis=-1)
        
        # Apply attention weights
        weighted_output = inputs * attention_weights
        context_vector = tf.reduce_sum(weighted_output, axis=1, name='attention_context')
        
        return context_vector, attention_weights
    
    def build_attention_model(self, input_shape):
        """Build LSTM model with self-attention"""
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        # First LSTM layer
        x = LSTM(128, return_sequences=True, name='lstm_1')(inputs)
        x = Dropout(0.2, name='dropout_1')(x)
        
        # Second LSTM layer
        x = LSTM(64, return_sequences=True, name='lstm_2')(x)
        x = Dropout(0.2, name='dropout_2')(x)
        
        # Self-attention layer
        context_vector, attention_weights = self.attention_layer(x)
        
        # Dense layers
        x = Dense(32, activation='relu', name='dense_1')(context_vector)
        x = Dropout(0.1, name='dropout_3')(x)
        
        # Output
        outputs = Dense(1, name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_Attention')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("\n" + "="*60)
        logger.info("LSTM WITH SELF-ATTENTION ARCHITECTURE")
        logger.info("="*60)
        model.summary()
        
        return model
    
    def build_multihead_attention(self, input_shape, num_heads=4):
        """Build LSTM with Multi-Head Attention"""
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        # LSTM layers with unique names
        x = LSTM(128, return_sequences=True, name='lstm_1_multi')(inputs)
        x = Dropout(0.2, name='dropout_1_multi')(x)
        
        x = LSTM(64, return_sequences=True, name='lstm_2_multi')(x)
        x = Dropout(0.2, name='dropout_2_multi')(x)
        
        # Multi-Head Self-Attention
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=64,
            dropout=0.1,
            name='multihead_attention'
        )(x, x)
        
        # Add & Normalize
        x = Add(name='attention_add')([x, attention_output])
        x = LayerNormalization(epsilon=1e-6, name='attention_norm')(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D(name='global_pool')(x)
        
        # Dense layers
        x = Dense(32, activation='relu', name='dense_1_multi')(x)
        x = Dropout(0.1, name='dropout_3_multi')(x)
        
        outputs = Dense(1, name='output_multi')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='LSTM_MultiHead_Attention')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("\n" + "="*60)
        logger.info(f"LSTM WITH MULTI-HEAD ATTENTION ({num_heads} heads)")
        logger.info("="*60)
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, use_multihead=False, num_heads=4):
        """Train LSTM with attention"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING LSTM WITH ATTENTION")
        logger.info("="*60)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if use_multihead:
            self.model = self.build_multihead_attention(input_shape, num_heads)
            model_name = "lstm_multihead_attention"
        else:
            self.model = self.build_attention_model(input_shape)
            model_name = "lstm_attention"
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
        ]
        
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
        
        y_pred_scaled = self.model.predict(X_test, verbose=0)
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
        logger.info("LSTM WITH ATTENTION - EVALUATION")
        logger.info("="*60)
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.4f}")
        logger.info(f"Directional Accuracy: {direction_accuracy:.2%}")
        
        return metrics, y_pred
    
    def plot_results(self, history, y_test, y_pred, metrics, model_name):
        """Plot results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Training loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title(f'{model_name.upper()} - Training History', fontsize=14)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Predictions vs Actual
        axes[0, 1].plot(y_test[-500:], label='Actual', alpha=0.7, linewidth=1)
        axes[0, 1].plot(y_pred[-500:], label='Predicted', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('Predictions vs Actual (Last 500 days)', fontsize=14)
        axes[0, 1].set_xlabel('Days')
        axes[0, 1].set_ylabel('Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot
        axes[0, 2].scatter(y_test, y_pred, alpha=0.3, s=10)
        axes[0, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
        axes[0, 2].set_title(f'Predicted vs Actual (R² = {metrics["r2"]:.4f})', fontsize=14)
        axes[0, 2].set_xlabel('Actual Return')
        axes[0, 2].set_ylabel('Predicted Return')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_test - y_pred
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='red', linestyle='--')
        axes[1, 0].set_title(f'Residuals (MAE = {metrics["mae"]:.6f})', fontsize=14)
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance metrics
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        
        metrics_text = f"""
        MODEL PERFORMANCE SUMMARY
        
        RMSE: {metrics['rmse']:.6f}
        MAE: {metrics['mae']:.6f}
        R²: {metrics['r2']:.4f}
        Directional Accuracy: {metrics['directional_accuracy']:.2%}
        
        ATTENTION MECHANISM:
        • Focuses on most relevant timesteps
        • Learns which historical days matter
        • Improves temporal pattern recognition
        """
        
        axes[1, 1].text(0.5, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.5))
        
        # Model architecture summary
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        arch_text = f"""
        MODEL ARCHITECTURE
        
        Input: {self.lookback} days × features
        ↓
        LSTM(128) + Dropout(0.2)
        ↓
        LSTM(64) + Dropout(0.2)
        ↓
        SELF-ATTENTION LAYER
        • Computes importance weights
        • Weighted sum of timesteps
        • Focuses on relevant patterns
        ↓
        Dense(32) + Dropout(0.1)
        ↓
        Output: Next day return
        
        Attention helps model focus on
        the most important historical days!
        """
        
        axes[1, 2].text(0.5, 0.5, arch_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle="round", facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Plot saved to {self.figures_path}")
    
    def save_model(self, metrics, history, model_name):
        """Save model and results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = os.path.join(self.models_path, f"{model_name}_{timestamp}.keras")
        self.model.save(model_file)
        logger.info(f"Model saved: {model_file}")
        
        # Save results
        results = {
            'timestamp': timestamp,
            'model_type': model_name,
            'lookback': self.lookback,
            'architecture': 'LSTM + Self-Attention Layer',
            'attention_mechanism': 'Self-attention focuses on most relevant timesteps',
            'metrics': metrics,
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        results_file = os.path.join(self.results_path, f"{model_name}_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_file}")
        
        return timestamp
    
    def run(self, use_multihead=False, num_heads=4, epochs=50):
        """Run complete training pipeline"""
        
        logger.info("="*60)
        logger.info("LSTM WITH SELF-ATTENTION LAYER")
        logger.info("Attention helps model focus on most relevant timesteps")
        logger.info("="*60)
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare data
        X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = self.prepare_data(df)
        
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error("Not enough data for sequences!")
            return
        
        # Train model
        history, model_name = self.train(X_train, y_train, X_test, y_test, epochs, use_multihead, num_heads)
        
        # Evaluate
        metrics, y_pred = self.evaluate(X_test, y_test, y_test_raw)
        
        # Plot results
        self.plot_results(history, y_test_raw, y_pred, metrics, model_name)
        
        # Save model
        self.save_model(metrics, history, model_name)
        
        logger.info("\n" + "="*60)
        logger.info("✅ LSTM WITH ATTENTION COMPLETE")
        logger.info("="*60)
        
        return metrics

if __name__ == "__main__":
    # Train standard LSTM with self-attention
    logger.info("Training LSTM with Self-Attention...")
    attention_lstm = LSTMWithAttention(lookback=60)
    results_attention = attention_lstm.run(use_multihead=False, epochs=50)
    
    # Train LSTM with Multi-Head Attention (Transformer-like)
    logger.info("\n" + "="*60)
    logger.info("Training LSTM with Multi-Head Attention (4 heads)...")
    logger.info("="*60)
    multihead_lstm = LSTMWithAttention(lookback=60)
    results_multihead = multihead_lstm.run(use_multihead=True, num_heads=4, epochs=50)
    
    # Compare results
    print("\n" + "="*60)
    print("ATTENTION MODELS COMPARISON")
    print("="*60)
    if results_attention:
        print(f"Self-Attention LSTM - RMSE: {results_attention['rmse']:.6f}, Dir Acc: {results_attention['directional_accuracy']:.2%}")
    if results_multihead:
        print(f"Multi-Head Attention (4 heads) - RMSE: {results_multihead['rmse']:.6f}, Dir Acc: {results_multihead['directional_accuracy']:.2%}")
