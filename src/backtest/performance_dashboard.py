#!/usr/bin/env python
"""
Performance Dashboard for Best Model (XGBoost No Sentiment)
Fixed version with proper date handling
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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceDashboard:
    """Generate comprehensive performance metrics and visualizations"""
    
    def __init__(self):
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        
    def load_best_model_results(self):
        """Load the best model (XGBoost No Sentiment) results from backtest"""
        
        # Find the latest backtest results
        result_files = list(Path(self.results_path).glob("complete_backtest_*.csv"))
        if not result_files:
            logger.error("No backtest results found!")
            return None
        
        df_results = pd.read_csv(sorted(result_files)[-1])
        
        # Get XGBoost No Sentiment results
        best_model = df_results[df_results['Model'] == 'XGBoost (No Sentiment)']
        if best_model.empty:
            logger.error("Best model not found in results")
            return None
        
        # Load the actual backtest portfolio data
        from src.backtest.full_model_comparison import CompleteModelBacktest  # type: ignore
        
        backtest = CompleteModelBacktest()
        df = backtest.load_data()
        (X_train_no, X_test_no, X_train_with, X_test_with, 
         y_train, y_test, test_dates) = backtest.prepare_train_test(df)
        
        # Train and predict with XGBoost No Sentiment
        import xgboost as xgb
        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train_no, y_train)
        predictions = model.predict(X_test_no)
        
        # Generate signals and returns
        signal_threshold = 0.002
        signals = np.where(predictions > signal_threshold, 1,
                          np.where(predictions < -signal_threshold, -1, 0))
        strategy_returns = signals * y_test.values
        
        portfolio = 10000 * (1 + strategy_returns).cumprod()
        
        # Ensure dates are datetime
        dates = pd.to_datetime(test_dates, utc=True)
        
        return {
            'returns': strategy_returns,
            'portfolio': portfolio,
            'dates': dates,
            'predictions': predictions,
            'signals': signals,
            'metrics': best_model.iloc[0].to_dict()
        }
    
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.02, periods_per_year=252):
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate / periods_per_year
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        return sharpe
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.02, periods_per_year=252):
        """Calculate Sortino ratio (uses downside deviation only)"""
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation if downside_deviation > 0 else 0
        return sortino
    
    def calculate_calmar_ratio(self, returns, max_drawdown):
        """Calculate Calmar ratio (return / max drawdown)"""
        annual_return = returns.mean() * 252
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        return calmar
    
    def calculate_max_drawdown(self, portfolio):
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - running_max) / running_max
        max_dd = drawdown.min()
        return max_dd, drawdown
    
    def calculate_rolling_metrics(self, returns, window=63):
        """Calculate rolling metrics"""
        returns_series = pd.Series(returns)
        rolling_sharpe = returns_series.rolling(window).apply(
            lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
        )
        rolling_volatility = returns_series.rolling(window).std() * np.sqrt(252)
        rolling_return = returns_series.rolling(window).mean() * 252
        return rolling_sharpe, rolling_volatility, rolling_return
    
    def create_monthly_returns_heatmap(self, returns, dates):
        """Create monthly returns heatmap"""
        # Create DataFrame with returns
        df = pd.DataFrame({'return': returns}, index=dates)
        
        # Resample to monthly
        monthly_returns = df.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table
        monthly_pivot = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'return': monthly_returns['return'].values * 100
        }).pivot(index='year', columns='month', values='return')
        
        # Rename months
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_pivot.columns = month_names
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Define color map
        cmap = sns.diverging_palette(10, 130, center='light', as_cmap=True)
        
        # Create heatmap
        im = ax.imshow(monthly_pivot.values, cmap=cmap, aspect='auto', vmin=-10, vmax=10)
        
        # Add text annotations
        for i in range(len(monthly_pivot.index)):
            for j in range(len(monthly_pivot.columns)):
                value = monthly_pivot.values[i, j]
                if not np.isnan(value):
                    color = 'white' if abs(value) > 5 else 'black'
                    ax.text(j, i, f'{value:.1f}%', ha='center', va='center', color=color, fontsize=9)
        
        ax.set_xticks(range(len(monthly_pivot.columns)))
        ax.set_xticklabels(monthly_pivot.columns)
        ax.set_yticks(range(len(monthly_pivot.index)))
        ax.set_yticklabels(monthly_pivot.index)
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)
        ax.set_title('Monthly Returns Heatmap (%)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Monthly Return (%)')
        
        plt.tight_layout()
        heatmap_file = os.path.join(self.figures_path, f"monthly_returns_heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved monthly returns heatmap: {heatmap_file}")
        
        return monthly_pivot
    
    def plot_equity_curve(self, portfolio, returns, drawdown, dates):
        """Plot equity curve with drawdown"""
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Equity Curve
        axes[0].plot(dates, portfolio, linewidth=1.5, color='darkgreen')
        axes[0].fill_between(dates, 10000, portfolio, where=(portfolio >= 10000), 
                             color='green', alpha=0.3)
        axes[0].fill_between(dates, portfolio, 10000, where=(portfolio < 10000), 
                             color='red', alpha=0.3)
        axes[0].axhline(y=10000, color='black', linestyle='-', alpha=0.5)
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True, alpha=0.3)
        
        # 2. Drawdown
        axes[1].fill_between(dates, 0, drawdown * 100, color='red', alpha=0.5)
        axes[1].plot(dates, drawdown * 100, linewidth=1, color='darkred')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1].set_title(f'Drawdown (Max: {drawdown.min() * 100:.1f}%)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Daily Returns Distribution
        axes[2].hist(returns, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[2].axvline(x=0, color='red', linestyle='--', linewidth=1.5)
        axes[2].axvline(x=returns.mean(), color='green', linestyle='--', linewidth=1.5,
                       label=f'Mean: {returns.mean():.4f}')
        axes[2].set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Daily Return')
        axes[2].set_ylabel('Frequency')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        equity_file = os.path.join(self.figures_path, f"equity_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(equity_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved equity curve: {equity_file}")
    
    def plot_rolling_performance(self, returns, dates):
        """Plot rolling performance metrics"""
        
        # Calculate rolling metrics
        rolling_sharpe, rolling_vol, rolling_return = self.calculate_rolling_metrics(returns, window=63)
        
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # 1. Rolling Sharpe Ratio
        axes[0].plot(dates[63:], rolling_sharpe[63:], linewidth=1.5, color='blue')
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[0].axhline(y=1, color='green', linestyle='--', alpha=0.7, label='Good (1.0)')
        axes[0].fill_between(dates[63:], 0, rolling_sharpe[63:], 
                             where=(rolling_sharpe[63:] > 0), color='green', alpha=0.3)
        axes[0].fill_between(dates[63:], 0, rolling_sharpe[63:], 
                             where=(rolling_sharpe[63:] < 0), color='red', alpha=0.3)
        axes[0].set_title('Rolling 63-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Rolling Volatility
        axes[1].plot(dates[63:], rolling_vol[63:] * 100, linewidth=1.5, color='orange')
        axes[1].set_title('Rolling 63-Day Volatility (Annualized)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Volatility (%)')
        axes[1].grid(True, alpha=0.3)
        
        # 3. Rolling Returns
        axes[2].plot(dates[63:], rolling_return[63:] * 100, linewidth=1.5, color='purple')
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2].fill_between(dates[63:], 0, rolling_return[63:] * 100,
                             where=(rolling_return[63:] > 0), color='green', alpha=0.3)
        axes[2].fill_between(dates[63:], 0, rolling_return[63:] * 100,
                             where=(rolling_return[63:] < 0), color='red', alpha=0.3)
        axes[2].set_title('Rolling 63-Day Annualized Return', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Return (%)')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        rolling_file = os.path.join(self.figures_path, f"rolling_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(rolling_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved rolling performance plot: {rolling_file}")
    
    def create_performance_summary_table(self, metrics, returns, portfolio):
        """Create performance summary table"""
        
        # Calculate additional metrics
        total_return = metrics.get('Total_Return_%', 0)
        sharpe = metrics.get('Sharpe_Ratio', 0)
        max_dd = metrics.get('Max_Drawdown_%', 0)
        win_rate = metrics.get('Win_Rate_%', 0)
        
        # Calculate Sortino
        sortino = self.calculate_sortino_ratio(pd.Series(returns))
        
        # Calculate Calmar
        calmar = self.calculate_calmar_ratio(pd.Series(returns), max_dd / 100)
        
        # Calculate other metrics
        annual_return = returns.mean() * 252 * 100
        volatility = returns.std() * np.sqrt(252) * 100
        avg_trade = returns.mean() * 100
        best_trade = returns.max() * 100
        worst_trade = returns.min() * 100
        
        # Create summary
        summary = {
            'Total Return': f"{total_return:.2f}%",
            'Annual Return': f"{annual_return:.2f}%",
            'Volatility': f"{volatility:.2f}%",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Sortino Ratio': f"{sortino:.2f}",
            'Calmar Ratio': f"{calmar:.2f}",
            'Max Drawdown': f"{max_dd:.2f}%",
            'Win Rate': f"{win_rate:.1f}%",
            'Average Trade': f"{avg_trade:.3f}%",
            'Best Trade': f"{best_trade:.3f}%",
            'Worst Trade': f"{worst_trade:.3f}%",
            'Final Portfolio': f"${portfolio[-1]:,.0f}"
        }
        
        # Save to JSON
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(self.results_path, f"performance_summary_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved performance summary: {summary_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY - XGBoost (No Sentiment)")
        print("="*60)
        for key, value in summary.items():
            print(f"{key:20s}: {value}")
        
        return summary
    
    def create_returns_distribution_plot(self, returns, dates):
        """Create returns distribution with statistics"""
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Histogram with KDE
        axes[0].hist(returns, bins=50, edgecolor='black', alpha=0.7, density=True, color='steelblue')
        kde_x = np.linspace(returns.min(), returns.max(), 100)
        kde = stats.gaussian_kde(returns)
        axes[0].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
        axes[0].axvline(x=0, color='black', linestyle='--', alpha=0.7)
        axes[0].axvline(x=returns.mean(), color='green', linestyle='--', 
                       label=f'Mean: {returns.mean():.4f}')
        axes[0].set_title('Returns Distribution with KDE', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Daily Return')
        axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        stats.probplot(returns, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        dist_file = os.path.join(self.figures_path, f"returns_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(dist_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved returns distribution plot: {dist_file}")
    
    def run(self):
        """Generate complete performance dashboard"""
        
        logger.info("="*60)
        logger.info("GENERATING PERFORMANCE DASHBOARD")
        logger.info("Best Model: XGBoost (No Sentiment)")
        logger.info("="*60)
        
        # Load results
        data = self.load_best_model_results()
        if data is None:
            logger.error("Could not load model results")
            return
        
        returns = pd.Series(data['returns'])
        portfolio = data['portfolio']
        dates = data['dates']
        
        # Calculate metrics
        max_dd, drawdown = self.calculate_max_drawdown(portfolio)
        
        # Create plots
        self.plot_equity_curve(portfolio, returns, drawdown, dates)
        self.plot_rolling_performance(returns, dates)
        self.create_returns_distribution_plot(returns, dates)
        
        # Monthly returns heatmap
        monthly_pivot = self.create_monthly_returns_heatmap(returns, dates)
        
        # Create summary table
        summary = self.create_performance_summary_table(data['metrics'], returns, portfolio)
        
        # Print additional insights
        print("\n" + "="*60)
        print("PERFORMANCE INSIGHTS")
        print("="*60)
        
        # Best month
        df_returns = pd.DataFrame({'return': returns}, index=dates)
        monthly_returns = df_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
        best_month = monthly_returns['return'].idxmax()
        worst_month = monthly_returns['return'].idxmin()
        
        print(f"\n📅 Best Month: {best_month.strftime('%Y-%m')} ({monthly_returns['return'].max()*100:.2f}%)")
        print(f"📅 Worst Month: {worst_month.strftime('%Y-%m')} ({monthly_returns['return'].min()*100:.2f}%)")
        
        # Positive months ratio
        positive_months = (monthly_returns['return'] > 0).mean()
        print(f"📈 Positive Months: {positive_months:.1%}")
        
        # Consecutive wins/losses
        wins = (returns > 0).astype(int)
        max_win_streak = wins.groupby((wins != wins.shift()).cumsum()).cumsum().max()
        max_loss_streak = ((returns < 0).astype(int)).groupby(((returns < 0) != (returns < 0).shift()).cumsum()).cumsum().max()
        
        print(f"🏆 Max Win Streak: {max_win_streak} days")
        print(f"⚠️ Max Loss Streak: {max_loss_streak} days")
        
        # Recovery time
        drawdown_series = pd.Series(drawdown, index=dates)
        below_10 = drawdown_series[drawdown_series < -0.1]
        if len(below_10) > 0:
            recovery_time = (drawdown_series.index[-1] - below_10.index[0]).days
            print(f"🔄 Time to recover from 10% drawdown: {recovery_time} days")
        
        print("\n" + "="*60)
        logger.info("✅ PERFORMANCE DASHBOARD COMPLETE")
        
        return summary

if __name__ == "__main__":
    dashboard = PerformanceDashboard()
    dashboard.run()