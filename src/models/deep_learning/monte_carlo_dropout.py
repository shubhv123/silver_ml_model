#!/usr/bin/env python
"""
Monte Carlo Dropout for Uncertainty Estimation
Outputs: predicted price ± confidence interval
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
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonteCarloDropoutLSTM:
    """
    LSTM with Monte Carlo Dropout for Uncertainty Estimation
    Performs multiple stochastic forward passes to estimate prediction intervals
    """
    
    def __init__(self, lookback=60, target_col='target_next_day_return', mc_samples=100):
        self.lookback = lookback
        self.target_col = target_col
        self.mc_samples = mc_samples  # Number of Monte Carlo samples
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
    
    def build_model_with_dropout(self, input_shape):
        """
        Build LSTM model with dropout layers for Monte Carlo sampling
        Dropout remains active during inference for uncertainty estimation
        """
        
        inputs = Input(shape=input_shape, name='input_layer')
        
        # LSTM layers with dropout (keep them active for MC sampling)
        x = LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2, name='lstm_1')(inputs)
        x = Dropout(0.2, name='dropout_1')(x)
        
        x = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2, name='lstm_2')(x)
        x = Dropout(0.2, name='dropout_2')(x)
        
        # Dense layers with dropout
        x = Dense(32, activation='relu', name='dense_1')(x)
        x = Dropout(0.1, name='dropout_3')(x)
        
        outputs = Dense(1, name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='MC_Dropout_LSTM')
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("\n" + "="*60)
        logger.info("MONTE CARLO DROPOUT LSTM ARCHITECTURE")
        logger.info("="*60)
        logger.info("Dropout layers remain ACTIVE during inference")
        logger.info("Multiple stochastic forward passes = uncertainty estimates")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the model"""
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING MONTE CARLO DROPOUT LSTM")
        logger.info("="*60)
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        self.model = self.build_model_with_dropout(input_shape)
        
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
        
        return history
    
    def predict_with_uncertainty(self, X_test, confidence_level=0.95):
        """
        Perform Monte Carlo Dropout sampling
        Returns: mean prediction, lower bound, upper bound, standard deviation
        """
        logger.info(f"\nPerforming Monte Carlo Dropout with {self.mc_samples} samples...")
        
        # Collect predictions from multiple stochastic forward passes
        predictions = []
        
        for i in range(self.mc_samples):
            # Forward pass with dropout active
            pred = self.model.predict(X_test, verbose=0)
            predictions.append(pred.flatten())
            
            if (i + 1) % 20 == 0:
                logger.info(f"  Sample {i + 1}/{self.mc_samples}")
        
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Calculate confidence intervals
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        # Calculate prediction interval width (uncertainty)
        uncertainty_width = upper_bound - lower_bound
        relative_uncertainty = (uncertainty_width / np.abs(mean_pred)) * 100
        
        logger.info(f"\n📊 UNCERTAINTY ESTIMATION:")
        logger.info(f"   Mean prediction range: {mean_pred.min():.4f} to {mean_pred.max():.4f}")
        logger.info(f"   Avg uncertainty width: {uncertainty_width.mean():.4f}")
        logger.info(f"   Avg relative uncertainty: {relative_uncertainty.mean():.1f}%")
        
        return mean_pred, lower_bound, upper_bound, std_pred, predictions
    
    def evaluate_with_uncertainty(self, y_test, mean_pred, lower_bound, upper_bound):
        """Evaluate model with uncertainty metrics"""
        
        # Standard metrics
        rmse = np.sqrt(mean_squared_error(y_test, mean_pred))
        mae = mean_absolute_error(y_test, mean_pred)
        r2 = r2_score(y_test, mean_pred)
        direction_accuracy = (np.sign(y_test) == np.sign(mean_pred)).mean()
        
        # Uncertainty metrics
        # Prediction interval coverage (how often actual falls within interval)
        within_interval = ((y_test >= lower_bound) & (y_test <= upper_bound)).mean()
        
        # Mean prediction interval width
        avg_interval_width = (upper_bound - lower_bound).mean()
        
        # Pinball loss for quantile evaluation
        def pinball_loss(y_true, y_pred, quantile):
            residual = y_true - y_pred
            return np.mean(np.maximum(quantile * residual, (quantile - 1) * residual))
        
        pinball_95 = pinball_loss(y_test, upper_bound, 0.975) + pinball_loss(y_test, lower_bound, 0.025)
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'directional_accuracy': float(direction_accuracy),
            'prediction_interval_coverage': float(within_interval),
            'avg_interval_width': float(avg_interval_width),
            'pinball_loss': float(pinball_95)
        }
        
        logger.info("\n" + "="*60)
        logger.info("MONTE CARLO DROPOUT - EVALUATION")
        logger.info("="*60)
        logger.info(f"\n📈 POINT PREDICTION METRICS:")
        logger.info(f"   RMSE: {rmse:.6f}")
        logger.info(f"   MAE: {mae:.6f}")
        logger.info(f"   R²: {r2:.4f}")
        logger.info(f"   Directional Accuracy: {direction_accuracy:.2%}")
        
        logger.info(f"\n📊 UNCERTAINTY METRICS:")
        logger.info(f"   Prediction Interval Coverage: {within_interval:.2%} (target: 95%)")
        logger.info(f"   Avg Interval Width: {avg_interval_width:.6f}")
        logger.info(f"   Pinball Loss: {pinball_95:.6f}")
        
        if within_interval >= 0.90:
            logger.info(f"   ✅ Good calibration - intervals capture actual values")
        else:
            logger.info(f"   ⚠️ Under-confident - intervals too narrow")
        
        return metrics
    
    def plot_uncertainty(self, y_test, mean_pred, lower_bound, upper_bound, predictions, metrics):
        """Plot predictions with uncertainty bands"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Predictions with uncertainty bands
        n_plot = min(500, len(y_test))
        indices = range(n_plot)
        
        axes[0, 0].fill_between(indices, lower_bound[:n_plot], upper_bound[:n_plot], 
                                alpha=0.3, color='blue', label='95% Confidence Interval')
        axes[0, 0].plot(indices, y_test[:n_plot], label='Actual', color='green', linewidth=1, alpha=0.7)
        axes[0, 0].plot(indices, mean_pred[:n_plot], label='Predicted', color='red', linewidth=1, alpha=0.7)
        axes[0, 0].set_title('Predictions with Uncertainty Bands (95% CI)', fontsize=14)
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Uncertainty distribution
        uncertainty_widths = upper_bound - lower_bound
        axes[0, 1].hist(uncertainty_widths, bins=50, edgecolor='black', alpha=0.7, color='purple')
        axes[0, 1].axvline(x=uncertainty_widths.mean(), color='red', linestyle='--', 
                          label=f'Mean: {uncertainty_widths.mean():.4f}')
        axes[0, 1].set_title('Uncertainty Width Distribution', fontsize=14)
        axes[0, 1].set_xlabel('Uncertainty Width')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Standard deviation of predictions
        std_pred = np.std(predictions, axis=0)
        axes[0, 2].plot(std_pred[:n_plot], linewidth=1, color='orange')
        axes[0, 2].set_title('Prediction Standard Deviation (Uncertainty)', fontsize=14)
        axes[0, 2].set_xlabel('Days')
        axes[0, 2].set_ylabel('Std Dev')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Prediction interval coverage
        within_interval = ((y_test >= lower_bound) & (y_test <= upper_bound)).astype(int)
        axes[1, 0].plot(within_interval[:n_plot], 'g.', alpha=0.5, markersize=2)
        axes[1, 0].axhline(y=0.95, color='red', linestyle='--', label='Target 95%')
        axes[1, 0].set_title(f'Prediction Interval Coverage: {metrics["prediction_interval_coverage"]:.2%}', fontsize=14)
        axes[1, 0].set_xlabel('Days')
        axes[1, 0].set_ylabel('Within Interval (1=Yes, 0=No)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Scatter with error bars
        n_sample = min(200, len(y_test))
        sample_idx = np.random.choice(len(y_test), n_sample, replace=False)
        
        axes[1, 1].errorbar(y_test[sample_idx], mean_pred[sample_idx],
                           yerr=2 * std_pred[sample_idx], fmt='o', alpha=0.5, 
                           capsize=2, markersize=3, color='blue')
        axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=1)
        axes[1, 1].set_title('Predictions with Error Bars (2σ)', fontsize=14)
        axes[1, 1].set_xlabel('Actual Return')
        axes[1, 1].set_ylabel('Predicted Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Summary metrics
        axes[1, 2].axis('tight')
        axes[1, 2].axis('off')
        
        summary_text = f"""
        MONTE CARLO DROPOUT SUMMARY
        
        Monte Carlo Samples: {self.mc_samples}
        
        Point Prediction Metrics:
        • RMSE: {metrics['rmse']:.6f}
        • MAE: {metrics['mae']:.6f}
        • R²: {metrics['r2']:.4f}
        • Directional Acc: {metrics['directional_accuracy']:.2%}
        
        Uncertainty Metrics:
        • Interval Coverage: {metrics['prediction_interval_coverage']:.2%}
        • Avg Interval Width: {metrics['avg_interval_width']:.6f}
        • Pinball Loss: {metrics['pinball_loss']:.6f}
        
        Output: Prediction ± Uncertainty Band
        Example: {mean_pred[0]:.4f} ± {2*std_pred[0]:.4f}
        """
        
        axes[1, 2].text(0.5, 0.5, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=10, verticalalignment='center', horizontalalignment='center',
                        fontfamily='monospace', bbox=dict(boxstyle="round", facecolor='lightgray', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_path, f"mc_dropout_uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Uncertainty plot saved to {self.figures_path}")
    
    def save_model_with_uncertainty(self, metrics, history):
        """Save model and uncertainty results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_file = os.path.join(self.models_path, f"mc_dropout_lstm_{timestamp}.keras")
        self.model.save(model_file)
        logger.info(f"Model saved: {model_file}")
        
        # Save results
        results = {
            'timestamp': timestamp,
            'model_type': 'Monte_Carlo_Dropout_LSTM',
            'lookback': self.lookback,
            'mc_samples': self.mc_samples,
            'architecture': 'LSTM with MC Dropout for uncertainty',
            'metrics': metrics,
            'final_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1])
        }
        
        results_file = os.path.join(self.results_path, f"mc_dropout_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {results_file}")
        
        return timestamp
    
    def run(self, epochs=50, mc_samples=100):
        """Run complete training with uncertainty estimation"""
        
        self.mc_samples = mc_samples
        
        logger.info("="*60)
        logger.info("MONTE CARLO DROPOUT LSTM")
        logger.info(f"Monte Carlo Samples: {self.mc_samples}")
        logger.info("Output: Prediction ± Confidence Interval")
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
        history = self.train(X_train, y_train, X_test, y_test, epochs)
        
        # Predict with uncertainty
        mean_pred, lower_bound, upper_bound, std_pred, all_predictions = self.predict_with_uncertainty(X_test)
        
        # Inverse transform predictions
        mean_pred_original = self.scaler_y.inverse_transform(mean_pred.reshape(-1, 1)).flatten()
        lower_bound_original = self.scaler_y.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
        upper_bound_original = self.scaler_y.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
        
        # Evaluate
        metrics = self.evaluate_with_uncertainty(y_test_raw, mean_pred_original, 
                                                  lower_bound_original, upper_bound_original)
        
        # Plot results
        self.plot_uncertainty(y_test_raw, mean_pred_original, lower_bound_original, 
                              upper_bound_original, all_predictions, metrics)
        
        # Save model
        self.save_model_with_uncertainty(metrics, history)
        
        # Print sample output
        logger.info("\n" + "="*60)
        logger.info("SAMPLE PREDICTIONS WITH UNCERTAINTY")
        logger.info("="*60)
        for i in range(min(5, len(mean_pred_original))):
            logger.info(f"Day {i+1}: {mean_pred_original[i]:.6f} ± {2*std_pred[i]:.6f} "
                       f"(95% CI: [{lower_bound_original[i]:.6f}, {upper_bound_original[i]:.6f}])")
        
        logger.info("\n" + "="*60)
        logger.info("✅ MONTE CARLO DROPOUT COMPLETE")
        logger.info("="*60)
        
        return metrics

if __name__ == "__main__":
    # Create and train Monte Carlo Dropout model
    mc_lstm = MonteCarloDropoutLSTM(lookback=60, mc_samples=100)
    results = mc_lstm.run(epochs=50, mc_samples=100)
    
    print("\n" + "="*60)
    print("MONTE CARLO DROPOUT FINAL RESULTS")
    print("="*60)
    if results:
        print(f"RMSE: {results['rmse']:.6f}")
        print(f"Directional Accuracy: {results['directional_accuracy']:.2%}")
        print(f"Prediction Interval Coverage: {results['prediction_interval_coverage']:.2%}")
        print(f"Average Uncertainty Width: {results['avg_interval_width']:.6f}")
