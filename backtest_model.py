#!/usr/bin/env python3
"""
Comprehensive Backtesting for Numerai Ensemble Model
===================================================

This script loads the trained ensemble model and performs detailed backtesting
on validation data with comprehensive metrics and visualizations.

Usage:
    python backtest_model.py [options]

Options:
    --model-path PATH       Path to the saved model (default: models/best_ensemble_model.pkl)
    --data-path PATH        Path to validation data (default: data/validation.parquet)
    --output-dir PATH       Output directory for results (default: backtest_results/)
    --era-analysis          Perform detailed era-by-era analysis
    --save-predictions      Save detailed predictions to CSV
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from config import *
from data_manager import NumeraiDataManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NumeraiBacktester:
    def __init__(self, model_path: str, data_path: str, output_dir: str = "backtest_results"):
        self.model_path = model_path
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.data = None
        self.predictions = None
        self.results = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup data manager
        self.data_manager = NumeraiDataManager()
        
    def load_model(self):
        """Load the trained ensemble model."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info("✅ Model loaded successfully")
            logger.info(f"Model type: {type(self.model).__name__}")
            
            if hasattr(self.model, 'models'):
                logger.info(f"Ensemble models: {list(self.model.models.keys())}")
                if hasattr(self.model, 'weights'):
                    logger.info(f"Model weights: {self.model.weights}")
                    
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
            
    def load_data(self):
        """Load validation data."""
        logger.info(f"Loading validation data from {self.data_path}")
        
        try:
            if self.data_path.endswith('.parquet'):
                self.data = pd.read_parquet(self.data_path)
            else:
                self.data = pd.read_csv(self.data_path)
                
            logger.info(f"✅ Data loaded: {self.data.shape}")
            logger.info(f"Columns: {list(self.data.columns)}")
            
            # Check for required columns
            feature_cols = [c for c in self.data.columns if c.startswith('feature_')]
            logger.info(f"Found {len(feature_cols)} features")
            
            if 'target' in self.data.columns:
                logger.info("✅ Target column found")
            else:
                logger.warning("⚠️ No target column found - backtesting will be limited")
                
            if 'era' in self.data.columns:
                eras = self.data['era'].nunique()
                logger.info(f"✅ Found {eras} unique eras")
            else:
                logger.warning("⚠️ No era column found - era analysis will be limited")
                
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            raise
            
    def make_predictions(self):
        """Generate predictions using the loaded model."""
        logger.info("Generating predictions...")
        
        try:
            # Get feature columns
            feature_cols = [c for c in self.data.columns if c.startswith('feature_')]
            
            if not feature_cols:
                raise ValueError("No feature columns found in data")
                
            X = self.data[feature_cols].values
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Make predictions
            self.predictions = self.model.predict(X)
            
            logger.info(f"✅ Generated {len(self.predictions)} predictions")
            logger.info(f"Prediction range: [{self.predictions.min():.4f}, {self.predictions.max():.4f}]")
            logger.info(f"Prediction mean: {self.predictions.mean():.4f}")
            logger.info(f"Prediction std: {self.predictions.std():.4f}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate predictions: {e}")
            raise
            
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        logger.info("Calculating performance metrics...")
        
        metrics = {}
        
        if 'target' not in self.data.columns:
            logger.warning("No target column - skipping target-based metrics")
            return metrics
            
        targets = self.data['target'].values
        
        try:
            # Basic correlation metrics
            correlation = np.corrcoef(self.predictions, targets)[0, 1]
            metrics['correlation'] = correlation
            
            # Rank correlation (Spearman)
            from scipy.stats import spearmanr
            rank_corr, _ = spearmanr(self.predictions, targets)
            metrics['rank_correlation'] = rank_corr
            
            # Mean metrics
            metrics['mae'] = np.mean(np.abs(self.predictions - targets))
            metrics['mse'] = np.mean((self.predictions - targets) ** 2)
            metrics['rmse'] = np.sqrt(metrics['mse'])
            
            # Sharpe ratio (using prediction differences as returns)
            returns = np.diff(self.predictions)
            if len(returns) > 1 and np.std(returns) > 0:
                metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
            else:
                metrics['sharpe_ratio'] = 0.0
                
            # Max drawdown
            cumulative = np.cumsum(self.predictions - self.predictions.mean())
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            metrics['max_drawdown'] = np.min(drawdown)
            
            # Feature neutral correlation (if multiple features)
            feature_cols = [c for c in self.data.columns if c.startswith('feature_')]
            if len(feature_cols) > 1:
                try:
                    neutralized_preds = self.data_manager.neutralize_predictions(
                        self.predictions, 
                        self.data[feature_cols].values
                    )
                    metrics['feature_neutral_correlation'] = np.corrcoef(neutralized_preds, targets)[0, 1]
                except:
                    metrics['feature_neutral_correlation'] = np.nan
            else:
                metrics['feature_neutral_correlation'] = correlation
                
            # Era-wise correlation if era data available
            if 'era' in self.data.columns:
                era_correlations = []
                for era in self.data['era'].unique():
                    era_mask = self.data['era'] == era
                    if era_mask.sum() > 1:
                        era_preds = self.predictions[era_mask]
                        era_targets = targets[era_mask]
                        era_corr = np.corrcoef(era_preds, era_targets)[0, 1]
                        if not np.isnan(era_corr):
                            era_correlations.append(era_corr)
                            
                if era_correlations:
                    metrics['mean_era_correlation'] = np.mean(era_correlations)
                    metrics['std_era_correlation'] = np.std(era_correlations)
                    metrics['sharpe_era'] = metrics['mean_era_correlation'] / metrics['std_era_correlation'] if metrics['std_era_correlation'] > 0 else 0
                else:
                    metrics['mean_era_correlation'] = np.nan
                    metrics['std_era_correlation'] = np.nan
                    metrics['sharpe_era'] = np.nan
                    
            logger.info("✅ Metrics calculated successfully")
            
        except Exception as e:
            logger.error(f"❌ Error calculating metrics: {e}")
            
        return metrics
        
    def era_analysis(self) -> pd.DataFrame:
        """Perform detailed era-by-era analysis."""
        logger.info("Performing era analysis...")
        
        if 'era' not in self.data.columns:
            logger.warning("No era column found - skipping era analysis")
            return pd.DataFrame()
            
        if 'target' not in self.data.columns:
            logger.warning("No target column found - skipping era analysis")
            return pd.DataFrame()
            
        era_results = []
        
        for era in sorted(self.data['era'].unique()):
            era_mask = self.data['era'] == era
            era_data = self.data[era_mask]
            era_preds = self.predictions[era_mask]
            era_targets = era_data['target'].values
            
            if len(era_preds) > 1:
                era_result = {
                    'era': era,
                    'count': len(era_preds),
                    'correlation': np.corrcoef(era_preds, era_targets)[0, 1] if len(era_preds) > 1 else np.nan,
                    'mae': np.mean(np.abs(era_preds - era_targets)),
                    'prediction_mean': np.mean(era_preds),
                    'prediction_std': np.std(era_preds),
                    'target_mean': np.mean(era_targets),
                    'target_std': np.std(era_targets),
                }
                
                # Rank correlation
                try:
                    from scipy.stats import spearmanr
                    rank_corr, _ = spearmanr(era_preds, era_targets)
                    era_result['rank_correlation'] = rank_corr
                except:
                    era_result['rank_correlation'] = np.nan
                    
                era_results.append(era_result)
                
        era_df = pd.DataFrame(era_results)
        logger.info(f"✅ Era analysis completed for {len(era_df)} eras")
        
        return era_df
        
    def create_visualizations(self, era_df: Optional[pd.DataFrame] = None):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        # Create figure directory
        fig_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)
        
        # 1. Prediction vs Target scatter plot
        if 'target' in self.data.columns:
            self._plot_prediction_scatter(fig_dir)
            
        # 2. Prediction distribution
        self._plot_prediction_distribution(fig_dir)
        
        # 3. Era analysis plots
        if era_df is not None and not era_df.empty:
            self._plot_era_analysis(era_df, fig_dir)
            
        # 4. Cumulative performance
        if 'target' in self.data.columns:
            self._plot_cumulative_performance(fig_dir)
            
        # 5. Model ensemble weights (if available)
        self._plot_ensemble_weights(fig_dir)
        
        logger.info("✅ Visualizations created")
        
    def _plot_prediction_scatter(self, fig_dir: str):
        """Create prediction vs target scatter plot."""
        try:
            plt.figure(figsize=(10, 8))
            
            targets = self.data['target'].values
            
            # Main scatter plot
            plt.scatter(targets, self.predictions, alpha=0.6, s=20)
            
            # Add correlation line
            correlation = np.corrcoef(self.predictions, targets)[0, 1]
            z = np.polyfit(targets, self.predictions, 1)
            p = np.poly1d(z)
            plt.plot(targets, p(targets), "r--", alpha=0.8, linewidth=2, 
                    label=f'Correlation: {correlation:.4f}')
            
            # Perfect correlation line
            min_val = min(targets.min(), self.predictions.min())
            max_val = max(targets.max(), self.predictions.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, 
                    label='Perfect Correlation')
            
            plt.xlabel('True Target')
            plt.ylabel('Predicted Target')
            plt.title('Prediction vs Target Scatter Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add text box with metrics
            textstr = f'Correlation: {correlation:.4f}\nMAE: {np.mean(np.abs(self.predictions - targets)):.4f}\nRMSE: {np.sqrt(np.mean((self.predictions - targets) ** 2)):.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create prediction scatter plot: {e}")
            
    def _plot_prediction_distribution(self, fig_dir: str):
        """Plot prediction distribution."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Histogram
            ax1.hist(self.predictions, bins=50, alpha=0.7, density=True, label='Predictions')
            if 'target' in self.data.columns:
                ax1.hist(self.data['target'], bins=50, alpha=0.7, density=True, label='Targets')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Density')
            ax1.set_title('Distribution Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Box plot
            box_data = [self.predictions]
            labels = ['Predictions']
            if 'target' in self.data.columns:
                box_data.append(self.data['target'].values)
                labels.append('Targets')
                
            ax2.boxplot(box_data, labels=labels)
            ax2.set_title('Distribution Box Plots')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'prediction_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create distribution plot: {e}")
            
    def _plot_era_analysis(self, era_df: pd.DataFrame, fig_dir: str):
        """Create era analysis plots."""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
            
            # Era correlation over time
            ax1.plot(range(len(era_df)), era_df['correlation'], marker='o', alpha=0.7)
            ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax1.axhline(y=era_df['correlation'].mean(), color='g', linestyle='--', alpha=0.7, 
                       label=f'Mean: {era_df["correlation"].mean():.4f}')
            ax1.set_title('Era Correlation Over Time')
            ax1.set_xlabel('Era Index')
            ax1.set_ylabel('Correlation')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Correlation distribution
            ax2.hist(era_df['correlation'].dropna(), bins=20, alpha=0.7, density=True)
            ax2.axvline(x=era_df['correlation'].mean(), color='r', linestyle='--', 
                       label=f'Mean: {era_df["correlation"].mean():.4f}')
            ax2.set_title('Era Correlation Distribution')
            ax2.set_xlabel('Correlation')
            ax2.set_ylabel('Density')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Cumulative correlation
            cumulative_corr = era_df['correlation'].fillna(0).cumsum() / (np.arange(len(era_df)) + 1)
            ax3.plot(range(len(era_df)), cumulative_corr, marker='o', alpha=0.7)
            ax3.set_title('Cumulative Mean Correlation')
            ax3.set_xlabel('Era Index')
            ax3.set_ylabel('Cumulative Mean Correlation')
            ax3.grid(True, alpha=0.3)
            
            # Era correlation vs count
            ax4.scatter(era_df['count'], era_df['correlation'], alpha=0.7)
            ax4.set_title('Correlation vs Era Size')
            ax4.set_xlabel('Era Sample Count')
            ax4.set_ylabel('Correlation')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'era_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create era analysis plots: {e}")
            
    def _plot_cumulative_performance(self, fig_dir: str):
        """Plot cumulative performance."""
        try:
            plt.figure(figsize=(12, 8))
            
            # Calculate cumulative returns (using predictions as proxy)
            returns = np.diff(self.predictions)
            cumulative_returns = np.cumsum(returns)
            
            plt.plot(range(len(cumulative_returns)), cumulative_returns, linewidth=2)
            plt.title('Cumulative Performance')
            plt.xlabel('Time')
            plt.ylabel('Cumulative Returns')
            plt.grid(True, alpha=0.3)
            
            # Add drawdown
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            plt.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red', label='Drawdown')
            
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, 'cumulative_performance.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create cumulative performance plot: {e}")
            
    def _plot_ensemble_weights(self, fig_dir: str):
        """Plot ensemble model weights."""
        try:
            if hasattr(self.model, 'weights') and self.model.weights:
                plt.figure(figsize=(10, 6))
                
                models = list(self.model.weights.keys())
                weights = list(self.model.weights.values())
                
                # Convert numpy values to float if needed
                weights = [float(w) for w in weights]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
                bars = plt.bar(models, weights, color=colors)
                
                plt.title('Ensemble Model Weights')
                plt.ylabel('Weight')
                plt.xticks(rotation=45)
                
                # Add value labels on bars
                for bar, weight in zip(bars, weights):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{weight:.3f}', ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(os.path.join(fig_dir, 'ensemble_weights.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.warning(f"Could not create ensemble weights plot: {e}")
            
    def save_results(self, metrics: Dict[str, float], era_df: Optional[pd.DataFrame] = None):
        """Save all results to files."""
        logger.info("Saving results...")
        
        # Save predictions
        results_df = self.data.copy()
        results_df['predictions'] = self.predictions
        
        if 'target' in results_df.columns:
            results_df['residual'] = results_df['predictions'] - results_df['target']
            
        pred_path = os.path.join(self.output_dir, 'predictions.csv')
        results_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to: {pred_path}")
        
        # Save metrics
        metrics_path = os.path.join(self.output_dir, 'metrics.json')
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_metrics[k] = float(v)
            elif isinstance(v, np.ndarray):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = v
                
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        logger.info(f"Metrics saved to: {metrics_path}")
        
        # Save era analysis
        if era_df is not None and not era_df.empty:
            era_path = os.path.join(self.output_dir, 'era_analysis.csv')
            era_df.to_csv(era_path, index=False)
            logger.info(f"Era analysis saved to: {era_path}")
            
    def print_summary(self, metrics: Dict[str, float]):
        """Print summary of results."""
        print("\n" + "="*60)
        print("BACKTESTING SUMMARY")
        print("="*60)
        
        print(f"Model: {self.model_path}")
        print(f"Data: {self.data_path}")
        print(f"Samples: {len(self.predictions)}")
        
        if metrics:
            print(f"\nPerformance Metrics:")
            print(f"  Correlation: {metrics.get('correlation', 'N/A'):.4f}")
            print(f"  Rank Correlation: {metrics.get('rank_correlation', 'N/A'):.4f}")
            print(f"  Feature Neutral Correlation: {metrics.get('feature_neutral_correlation', 'N/A'):.4f}")
            print(f"  MAE: {metrics.get('mae', 'N/A'):.4f}")
            print(f"  RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.4f}")
            
            if 'mean_era_correlation' in metrics:
                print(f"\nEra Analysis:")
                print(f"  Mean Era Correlation: {metrics.get('mean_era_correlation', 'N/A'):.4f}")
                print(f"  Era Correlation Std: {metrics.get('std_era_correlation', 'N/A'):.4f}")
                print(f"  Era Sharpe: {metrics.get('sharpe_era', 'N/A'):.4f}")
                
        print(f"\nResults saved to: {self.output_dir}")
        print("="*60)
        
    def run_backtest(self, era_analysis: bool = True, save_predictions: bool = True):
        """Run complete backtesting pipeline."""
        logger.info("Starting comprehensive backtesting...")
        
        # Load model and data
        self.load_model()
        self.load_data()
        
        # Generate predictions
        self.make_predictions()
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Era analysis
        era_df = None
        if era_analysis:
            era_df = self.era_analysis()
            
        # Create visualizations
        self.create_visualizations(era_df)
        
        # Save results
        if save_predictions:
            self.save_results(metrics, era_df)
            
        # Print summary
        self.print_summary(metrics)
        
        logger.info("✅ Backtesting completed successfully!")
        
        return metrics, era_df

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Backtesting for Numerai Ensemble Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--model-path", default="models/best_ensemble_model.pkl",
        help="Path to the saved model"
    )
    
    parser.add_argument(
        "--data-path", default="data/validation.parquet",
        help="Path to validation data"
    )
    
    parser.add_argument(
        "--output-dir", default="backtest_results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--era-analysis", action="store_true",
        help="Perform detailed era-by-era analysis"
    )
    
    parser.add_argument(
        "--save-predictions", action="store_true", default=True,
        help="Save detailed predictions to CSV"
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
        
    if not os.path.exists(args.data_path):
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)
    
    # Create backtester and run
    backtester = NumeraiBacktester(args.model_path, args.data_path, args.output_dir)
    metrics, era_df = backtester.run_backtest(
        era_analysis=args.era_analysis,
        save_predictions=args.save_predictions
    )
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 