#!/usr/bin/env python
"""
Position Sizing Logic with Risk Management - Fixed
Position Size = Signal Strength × Confidence Score
Includes Max Drawdown Circuit Breaker
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PositionSizingEngine:
    """Dynamic Position Sizing with Risk Management"""
    
    def __init__(self, max_position_pct=0.95, max_drawdown_pct=-0.15, 
                 circuit_breaker_days=5, volatility_scaling=True):
        self.max_position_pct = max_position_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.circuit_breaker_days = circuit_breaker_days
        self.volatility_scaling = volatility_scaling
        
        self.circuit_breaker_active = False
        self.circuit_breaker_end_date = None
        self.peak_portfolio = None
        self.circuit_breaker_triggered = False
        
    def calculate_signal_strength(self, predictions):
        """Convert predictions to signal strength (0 to 1)"""
        strength = np.abs(predictions)
        max_expected_return = 0.03
        strength = np.clip(strength / max_expected_return, 0, 1)
        return strength
    
    def calculate_confidence_score(self, predictions, uncertainties=None, model_agreement=None):
        """Calculate confidence score"""
        confidence = np.ones_like(predictions)
        
        if uncertainties is not None:
            uncertainty_factor = 1 - np.clip(uncertainties / 0.02, 0, 0.5)
            confidence *= uncertainty_factor
        
        if model_agreement is not None:
            confidence *= model_agreement
        
        confidence = np.clip(confidence, 0.2, 0.95)
        return confidence
    
    def calculate_volatility_scaling(self, returns, window=21):
        """Scale position based on volatility regime"""
        if not self.volatility_scaling or len(returns) < window:
            return 1.0
        
        volatility = returns.rolling(window).std() * np.sqrt(252)
        current_vol = volatility.iloc[-1] if len(volatility) > 0 else 0.2
        target_vol = 0.20
        scale_factor = target_vol / max(current_vol, 0.1)
        scale_factor = np.clip(scale_factor, 0.3, 1.5)
        
        return scale_factor
    
    def check_circuit_breaker(self, portfolio_value, initial_capital=10000):
        """Check if max drawdown breached"""
        if self.peak_portfolio is None:
            self.peak_portfolio = portfolio_value
        
        if portfolio_value > self.peak_portfolio:
            self.peak_portfolio = portfolio_value
        
        current_drawdown = (portfolio_value - self.peak_portfolio) / self.peak_portfolio
        
        if current_drawdown <= self.max_drawdown_pct and not self.circuit_breaker_active:
            self.circuit_breaker_active = True
            self.circuit_breaker_triggered = True
            self.circuit_breaker_end_date = datetime.now() + timedelta(days=self.circuit_breaker_days)
            logger.warning(f"🚨 CIRCUIT BREAKER ACTIVATED! Drawdown: {current_drawdown*100:.1f}%")
            return True
        
        if self.circuit_breaker_active and datetime.now() >= self.circuit_breaker_end_date:
            self.circuit_breaker_active = False
            self.peak_portfolio = portfolio_value
            logger.info(f"✅ Circuit breaker deactivated. New peak: ${portfolio_value:,.0f}")
        
        return self.circuit_breaker_active
    
    def calculate_position_size(self, predictions, confidence_scores=None, 
                                uncertainties=None, model_agreement=None,
                                returns_history=None, portfolio_value=None):
        """Calculate position size = Signal Strength × Confidence Score"""
        
        signal_strength = self.calculate_signal_strength(predictions)
        
        if confidence_scores is None:
            confidence_scores = self.calculate_confidence_score(
                predictions, uncertainties, model_agreement
            )
        
        base_position = signal_strength * confidence_scores
        direction = np.sign(predictions)
        position_size = base_position * direction
        
        if returns_history is not None and len(returns_history) > 20:
            vol_scale = self.calculate_volatility_scaling(returns_history)
            position_size = position_size * vol_scale
        
        position_size = np.clip(position_size, -self.max_position_pct, self.max_position_pct)
        
        if self.circuit_breaker_active:
            position_size = np.zeros_like(position_size)
        
        return position_size
    
    def calculate_position_for_signal(self, prediction, confidence, 
                                       returns_history=None, portfolio_value=None):
        """Calculate position size for a single signal"""
        predictions = np.array([prediction])
        confidence_scores = np.array([confidence])
        
        position = self.calculate_position_size(
            predictions, confidence_scores, 
            returns_history=returns_history,
            portfolio_value=portfolio_value
        )
        
        return position[0]


class BacktestWithPositionSizing:
    """Backtest with dynamic position sizing"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.position_sizer = PositionSizingEngine(
            max_position_pct=0.95,
            max_drawdown_pct=-0.15,
            circuit_breaker_days=5,
            volatility_scaling=True
        )
        
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
    def run_backtest(self, predictions, returns, confidence_scores=None, 
                     uncertainties=None, model_agreement=None):
        """Run backtest with dynamic position sizing"""
        
        n = len(predictions)
        positions = np.zeros(n)
        portfolio = np.zeros(n)
        portfolio[0] = self.initial_capital
        
        rolling_returns = []
        
        logger.info("="*60)
        logger.info("RUNNING BACKTEST WITH POSITION SIZING")
        logger.info("="*60)
        
        for i in range(1, n):
            if confidence_scores is not None:
                confidence = confidence_scores[i]
            else:
                confidence = 0.7
            
            returns_history = pd.Series(returns[:i]) if i > 20 else None
            
            position = self.position_sizer.calculate_position_for_signal(
                predictions[i], confidence, 
                returns_history=returns_history,
                portfolio_value=portfolio[i-1]
            )
            
            positions[i] = position
            
            if self.position_sizer.circuit_breaker_active:
                positions[i] = 0
            
            portfolio_return = position * returns[i]
            portfolio[i] = portfolio[i-1] * (1 + portfolio_return)
            rolling_returns.append(portfolio_return)
            
            self.position_sizer.check_circuit_breaker(portfolio[i], self.initial_capital)
        
        return {
            'positions': positions,
            'portfolio': portfolio,
            'returns': pd.Series(portfolio).pct_change().fillna(0).values,
            'circuit_breaker_triggered': self.position_sizer.circuit_breaker_triggered
        }
    
    def calculate_metrics(self, results):
        """Calculate performance metrics"""
        
        portfolio = results['portfolio']
        returns = results['returns']
        
        total_return = (portfolio[-1] / self.initial_capital - 1) * 100
        annual_return = total_return / (len(portfolio) / 252)
        volatility = returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return / volatility) if volatility > 0 else 0
        
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        win_rate = (returns > 0).mean() * 100
        avg_position = np.abs(results['positions']).mean() * 100
        
        metrics = {
            'total_return_pct': total_return,
            'annual_return_pct': annual_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown,
            'win_rate_pct': win_rate,
            'avg_position_size_pct': avg_position,
            'max_position_size_pct': np.abs(results['positions']).max() * 100,
            'circuit_breaker_triggered': results['circuit_breaker_triggered']
        }
        
        return metrics
    
    def plot_results(self, results, predictions, returns):
        """Plot backtest results"""
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Portfolio equity curve
        axes[0, 0].plot(results['portfolio'], linewidth=1.5, color='darkgreen')
        axes[0, 0].axhline(y=self.initial_capital, color='black', linestyle='-', alpha=0.5)
        axes[0, 0].set_title('Portfolio Equity Curve with Position Sizing', fontsize=12)
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Position sizes over time
        axes[0, 1].fill_between(range(len(results['positions'])), 0, results['positions'] * 100, 
                               where=(results['positions'] > 0), color='green', alpha=0.5, label='Long')
        axes[0, 1].fill_between(range(len(results['positions'])), 0, results['positions'] * 100, 
                               where=(results['positions'] < 0), color='red', alpha=0.5, label='Short')
        axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0, 1].set_title('Position Sizes (% of Capital)', fontsize=12)
        axes[0, 1].set_ylabel('Position Size (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Drawdown
        running_max = np.maximum.accumulate(results['portfolio'])
        drawdown = (results['portfolio'] - running_max) / running_max * 100
        axes[1, 0].fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.5)
        axes[1, 0].plot(drawdown, linewidth=1, color='darkred')
        axes[1, 0].axhline(y=-15, color='orange', linestyle='--', label='Circuit Breaker Threshold (-15%)')
        axes[1, 0].set_title('Drawdown with Circuit Breaker', fontsize=12)
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Signal strength vs actual returns
        axes[1, 1].scatter(np.abs(predictions), returns, alpha=0.3, s=10)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('Signal Strength vs Actual Returns', fontsize=12)
        axes[1, 1].set_xlabel('Signal Strength')
        axes[1, 1].set_ylabel('Actual Return')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 5. Position size distribution
        axes[2, 0].hist(results['positions'] * 100, bins=50, edgecolor='black', alpha=0.7)
        axes[2, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[2, 0].set_title('Position Size Distribution', fontsize=12)
        axes[2, 0].set_xlabel('Position Size (%)')
        axes[2, 0].set_ylabel('Frequency')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Circuit breaker indicator
        axes[2, 1].plot(results['portfolio'], linewidth=1, color='darkgreen', label='Portfolio')
        axes[2, 1].set_title('Circuit Breaker Protection', fontsize=12)
        axes[2, 1].set_ylabel('Portfolio Value ($)')
        axes[2, 1].grid(True, alpha=0.3)
        
        if results['circuit_breaker_triggered']:
            axes[2, 1].axvspan(0, len(results['portfolio']), alpha=0.2, color='red', label='Circuit Breaker Active')
        
        axes[2, 1].legend()
        
        plt.tight_layout()
        plot_file = os.path.join(self.figures_path, f"position_sizing_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved plot: {plot_file}")
    
    def run(self, predictions, returns, confidence_scores=None):
        """Run complete backtest"""
        
        results = self.run_backtest(predictions, returns, confidence_scores)
        metrics = self.calculate_metrics(results)
        self.plot_results(results, predictions, returns)
        
        print("\n" + "="*60)
        print("POSITION SIZING BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return: {metrics['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        print(f"Win Rate: {metrics['win_rate_pct']:.1f}%")
        print(f"Avg Position Size: {metrics['avg_position_size_pct']:.1f}%")
        print(f"Max Position Size: {metrics['max_position_size_pct']:.1f}%")
        print(f"Circuit Breaker: {'Triggered' if metrics['circuit_breaker_triggered'] else 'Not Triggered'}")
        
        return results, metrics

if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n = 1000
    
    predictions = np.random.randn(n) * 0.01
    returns = predictions + np.random.randn(n) * 0.015
    confidence_scores = np.random.uniform(0.5, 0.95, n)
    
    backtest = BacktestWithPositionSizing(initial_capital=10000)
    results, metrics = backtest.run(predictions, returns, confidence_scores)
    
    print("\n" + "="*60)
    print("POSITION SIZING FORMULA")
    print("="*60)
    print("Position Size = Signal Strength × Confidence Score")
    print(f"  - Signal Strength = |prediction| / 0.03 (capped at 1)")
    print(f"  - Max Position: {backtest.position_sizer.max_position_pct*100:.0f}% of capital")
    print(f"  - Circuit Breaker: {backtest.position_sizer.max_drawdown_pct*100:.0f}% drawdown threshold")
    print(f"  - Volatility Scaling: {'Enabled' if backtest.position_sizer.volatility_scaling else 'Disabled'}")