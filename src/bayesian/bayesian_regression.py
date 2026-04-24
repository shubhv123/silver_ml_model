#!/usr/bin/env python
"""
Bayesian Regression for Silver Returns - Working Version
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import json
import warnings
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BayesianSilverRegression:
    """Bayesian Regression for Silver Returns"""
    
    def __init__(self):
        self.processed_path = "data/processed"
        self.raw_path = "data/raw"
        self.results_path = "reports/metrics"
        self.figures_path = "reports/figures"
        self.models_path = "models/saved"
        
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.figures_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        
        self.trace = None
        self.model = None
        
    def load_data(self):
        """Load silver price and macro data"""
        
        price_files = list(Path(self.raw_path).glob("SIF_daily.csv"))
        if not price_files:
            logger.error("No silver price data found!")
            return None, None
        
        price_df = pd.read_csv(price_files[0], parse_dates=['date'])
        price_df.set_index('date', inplace=True)
        price_df.columns = [col.lower() for col in price_df.columns]
        
        # Load or create macro features
        macro_df = self.generate_synthetic_macro(price_df.index)
        
        common_dates = price_df.index.intersection(macro_df.index)
        price_df = price_df.loc[common_dates]
        macro_df = macro_df.loc[common_dates]
        
        returns = price_df['close'].pct_change().dropna()
        macro_aligned = macro_df.loc[returns.index]
        
        logger.info(f"Data loaded: {len(returns)} observations")
        logger.info(f"Date range: {returns.index[0]} to {returns.index[-1]}")
        
        return returns, macro_aligned
    
    def generate_synthetic_macro(self, dates):
        """Generate synthetic macro features"""
        np.random.seed(42)
        n = len(dates)
        
        macro_df = pd.DataFrame(index=dates)
        macro_df['treasury_10yr'] = 2 + np.random.randn(n) * 0.5 + np.sin(np.linspace(0, 20, n)) * 0.5
        macro_df['dollar_index'] = 100 + np.random.randn(n) * 5
        macro_df['cpi'] = 2 + np.random.randn(n) * 0.3
        macro_df['fed_funds'] = 1.5 + np.random.randn(n) * 0.5
        
        logger.info("Generated synthetic macro features")
        return macro_df
    
    def build_bayesian_model(self, returns, macro_df):
        """Build Bayesian regression model"""
        
        macro_scaled = (macro_df - macro_df.mean()) / macro_df.std()
        X = macro_scaled.values
        y = returns.values
        
        with pm.Model() as model:
            # Informative priors
            beta_0 = pm.Normal('beta_0', mu=0, sigma=0.01)
            beta_treasury = pm.Normal('beta_treasury', mu=-0.05, sigma=0.02)
            beta_dollar = pm.Normal('beta_dollar', mu=-0.03, sigma=0.015)
            beta_cpi = pm.Normal('beta_cpi', mu=0.04, sigma=0.02)
            beta_fed = pm.Normal('beta_fed', mu=-0.02, sigma=0.015)
            
            # Linear model
            mu = beta_0 + beta_treasury * X[:, 0] + beta_dollar * X[:, 1] + beta_cpi * X[:, 2] + beta_fed * X[:, 3]
            
            # Noise
            sigma = pm.HalfNormal('sigma', sigma=0.02)
            
            # Likelihood
            likelihood = pm.Normal('y', mu=mu, sigma=sigma, observed=y)
            
        self.model = model
        logger.info("Bayesian model built")
        return model
    
    def sample_posterior(self, model, draws=2000, tune=1000, chains=4):
        """Sample from posterior"""
        
        logger.info(f"\nSampling posterior with {chains} chains...")
        
        with model:
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                return_inferencedata=True,
                progressbar=True
            )
            
        self.trace = trace
        logger.info(f"Sampling complete! Samples: {trace.posterior.dims['draw'] * trace.posterior.dims['chain']}")
        
        return trace
    
    def analyze_posterior(self, trace):
        """Analyze posterior"""
        
        logger.info("\n" + "="*60)
        logger.info("POSTERIOR ANALYSIS")
        logger.info("="*60)
        
        summary = az.summary(trace, hdi_prob=0.94)
        logger.info("\nPosterior Summary (94% HDI):")
        logger.info(summary.to_string())
        
        logger.info("\n📊 PROBABILITY OF DIRECTIONAL EFFECTS:")
        for var in ['beta_treasury', 'beta_dollar', 'beta_cpi', 'beta_fed']:
            if var in trace.posterior.data_vars:
                prob_positive = (trace.posterior[var] > 0).mean().item()
                prob_negative = (trace.posterior[var] < 0).mean().item()
                logger.info(f"  {var}: Positive: {prob_positive:.1%}, Negative: {prob_negative:.1%}")
        
        return summary
    
    def plot_posterior_distributions(self, trace):
        """Plot posterior distributions"""
        
        # Trace plots
        az.plot_trace(trace, figsize=(15, 10))
        plt.suptitle('Trace Plots - MCMC Convergence', fontsize=14, y=1.02)
        plt.tight_layout()
        trace_file = os.path.join(self.figures_path, f"bayesian_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(trace_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved trace plots: {trace_file}")
        
        # Posterior distributions
        az.plot_posterior(trace, hdi_prob=0.94, figsize=(15, 10))
        plt.suptitle('Posterior Distributions with 94% HDI', fontsize=14, y=1.02)
        plt.tight_layout()
        posterior_file = os.path.join(self.figures_path, f"posterior_distributions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(posterior_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved posterior distributions: {posterior_file}")
    
    def predict_with_uncertainty(self, trace, macro_df):
        """Generate predictions with uncertainty"""
        
        macro_scaled = (macro_df - macro_df.mean()) / macro_df.std()
        X_new = macro_scaled.values[-100:]
        
        posterior_samples = trace.posterior.stack(samples=("chain", "draw"))
        
        beta_0 = posterior_samples['beta_0'].values
        beta_treasury = posterior_samples['beta_treasury'].values
        beta_dollar = posterior_samples['beta_dollar'].values
        beta_cpi = posterior_samples['beta_cpi'].values
        beta_fed = posterior_samples['beta_fed'].values
        
        predictions = []
        for i in range(min(1000, len(beta_0))):
            pred = (beta_0[i] + beta_treasury[i] * X_new[:, 0] +
                   beta_dollar[i] * X_new[:, 1] + beta_cpi[i] * X_new[:, 2] +
                   beta_fed[i] * X_new[:, 3])
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'lower_94': np.percentile(predictions, 3, axis=0),
            'upper_94': np.percentile(predictions, 97, axis=0),
            'samples': predictions,
            'dates': macro_df.index[-100:]
        }
    
    def plot_predictions(self, predictions, returns):
        """Plot predictions with uncertainty"""
        
        dates = predictions['dates']
        returns_aligned = returns.loc[dates]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Predictions with uncertainty
        axes[0, 0].fill_between(dates, predictions['lower_94'], predictions['upper_94'], 
                                alpha=0.2, color='blue', label='94% HDI')
        axes[0, 0].plot(dates, predictions['mean'], 'b-', linewidth=1.5, label='Mean')
        axes[0, 0].plot(dates, returns_aligned, 'r-', linewidth=1, alpha=0.7, label='Actual')
        axes[0, 0].set_title('Bayesian Predictions with 94% Uncertainty')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Latest prediction distribution
        latest_pred = predictions['samples'][:, -1]
        axes[0, 1].hist(latest_pred, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].axvline(x=latest_pred.mean(), color='red', linestyle='--', 
                          label=f'Mean: {latest_pred.mean():.4f}')
        axes[0, 1].axvline(x=np.percentile(latest_pred, 2.5), color='green', 
                          linestyle='--', alpha=0.7, label='2.5%')
        axes[0, 1].axvline(x=np.percentile(latest_pred, 97.5), color='green', 
                          linestyle='--', alpha=0.7, label='97.5%')
        axes[0, 1].set_title(f'Latest Prediction Distribution\n{dates[-1].date()}')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Uncertainty over time
        axes[1, 0].plot(dates, predictions['std'], 'purple', linewidth=1.5)
        axes[1, 0].set_title('Prediction Uncertainty (Standard Deviation)')
        axes[1, 0].set_ylabel('Uncertainty (σ)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Parameter summary
        param_names = ['beta_treasury', 'beta_dollar', 'beta_cpi', 'beta_fed']
        param_means = []
        param_hdi = []
        
        for param in param_names:
            samples = self.trace.posterior[param].values.flatten()
            param_means.append(samples.mean())
            param_hdi.append(np.percentile(samples, [3, 97]))
        
        y_pos = np.arange(len(param_names))
        axes[1, 1].barh(y_pos, param_means, 
                       xerr=[np.array(param_means)-np.array([h[0] for h in param_hdi]),
                             np.array([h[1] for h in param_hdi])-np.array(param_means)],
                       capsize=5, color='coral', alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(param_names)
        axes[1, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].set_title('Parameter Estimates with 94% HDI')
        axes[1, 1].set_xlabel('Coefficient')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        pred_file = os.path.join(self.figures_path, f"bayesian_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(pred_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved predictions plot: {pred_file}")
    
    def save_results(self, summary, predictions):
        """Save results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        summary_dict = {
            'timestamp': timestamp,
            'model': 'Bayesian Regression',
            'posterior_summary': summary.to_dict(),
            'prediction_stats': {
                'mean_prediction': float(predictions['mean'].mean()),
                'prediction_std': float(predictions['std'].mean())
            }
        }
        
        summary_file = os.path.join(self.results_path, f"bayesian_results_{timestamp}.json")
        with open(summary_file, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)
        logger.info(f"Saved results: {summary_file}")
        
        trace_file = os.path.join(self.models_path, f"bayesian_trace_{timestamp}.nc")
        az.to_netcdf(self.trace, trace_file)
        logger.info(f"Saved trace: {trace_file}")
        
        return summary_file
    
    def run(self):
        """Run Bayesian analysis"""
        
        logger.info("="*60)
        logger.info("BAYESIAN REGRESSION FOR SILVER RETURNS")
        logger.info("="*60)
        
        returns, macro_df = self.load_data()
        if returns is None:
            return
        
        model = self.build_bayesian_model(returns, macro_df)
        trace = self.sample_posterior(model)
        
        summary = self.analyze_posterior(trace)
        self.plot_posterior_distributions(trace)
        
        predictions = self.predict_with_uncertainty(trace, macro_df)
        self.plot_predictions(predictions, returns)
        
        self.save_results(summary, predictions)
        
        print("\n" + "="*60)
        print("BAYESIAN RESULTS")
        print("="*60)
        print(f"Latest prediction: {predictions['mean'][-1]:.4f}")
        print(f"94% HDI: [{predictions['lower_94'][-1]:.4f}, {predictions['upper_94'][-1]:.4f}]")
        print(f"Probability of positive return: {(predictions['samples'][:, -1] > 0).mean():.1%}")
        
        return trace, predictions

if __name__ == "__main__":
    bayesian = BayesianSilverRegression()
    trace, predictions = bayesian.run()
    
    print("\n" + "="*60)
    print("CONVERGENCE CHECK")
    print("="*60)
    rhat = az.rhat(trace)
    rhat_max = float(rhat.to_array().max())
    print(f"Max R-hat: {rhat_max:.4f}")
    if rhat_max < 1.01:
        print("✅ Model converged successfully!")
    else:
        print("⚠️ Model may need more sampling")