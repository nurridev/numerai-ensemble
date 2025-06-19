import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import json
import os
from typing import Dict, List, Any
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self):
        self.metrics_history = []
        self.model_performance = {}
        self.ensemble_performance = {}
        self.start_time = datetime.now()
        
    def log_trial_metrics(self, trial_number: int, score: float, 
                         model_scores: Dict[str, float], 
                         predictions: np.ndarray, 
                         targets: np.ndarray,
                         additional_metrics: Dict[str, float] = None):
        """Log metrics for a single trial."""
        
        # Calculate comprehensive metrics
        metrics = {
            "trial_number": trial_number,
            "timestamp": datetime.now().isoformat(),
            "ensemble_score": score,
            "model_scores": model_scores,
            "correlation": np.corrcoef(predictions, targets)[0, 1],
            "mae": np.mean(np.abs(predictions - targets)),
            "mse": np.mean((predictions - targets) ** 2),
            "rmse": np.sqrt(np.mean((predictions - targets) ** 2)),
            "prediction_std": np.std(predictions),
            "target_std": np.std(targets),
        }
        
        # Add Sharpe ratio
        daily_returns = np.diff(predictions)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            metrics["sharpe_ratio"] = np.mean(daily_returns) / np.std(daily_returns)
        else:
            metrics["sharpe_ratio"] = 0.0
            
        # Add max drawdown
        cumulative = np.cumsum(predictions)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max)
        metrics["max_drawdown"] = np.min(drawdown)
        
        # Add additional metrics if provided
        if additional_metrics:
            metrics.update(additional_metrics)
            
        self.metrics_history.append(metrics)
        
        # Log important metrics
        if trial_number % METRICS_CONFIG["update_frequency"] == 0:
            self._log_current_performance(metrics)
            
        # Update plots
        if trial_number % METRICS_CONFIG["plot_frequency"] == 0:
            self._update_plots()
    
    def _log_current_performance(self, metrics: Dict[str, Any]):
        """Log current performance to console."""
        logger.info(f"=== Trial {metrics['trial_number']} Performance ===")
        logger.info(f"Ensemble Score: {metrics['ensemble_score']:.4f}")
        logger.info(f"Correlation: {metrics['correlation']:.4f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
        logger.info(f"RMSE: {metrics['rmse']:.4f}")
        
        # Model individual scores
        logger.info("Model Scores:")
        for model_name, score in metrics['model_scores'].items():
            logger.info(f"  {model_name}: {score:.4f}")
        
        # Time elapsed
        elapsed = datetime.now() - self.start_time
        logger.info(f"Time elapsed: {elapsed}")
        logger.info("=" * 40)
    
    def _update_plots(self):
        """Update real-time performance plots."""
        if len(self.metrics_history) < 2:
            return
            
        try:
            self._create_performance_dashboard()
            self._create_model_comparison()
            logger.info("Performance plots updated")
        except Exception as e:
            logger.warning(f"Could not update plots: {e}")
    
    def _create_performance_dashboard(self):
        """Create a comprehensive performance dashboard."""
        if len(self.metrics_history) < 2:
            return
            
        df = pd.DataFrame(self.metrics_history)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Ensemble Score Over Time', 'Correlation vs Trial', 
                          'Sharpe Ratio Progress', 'Cumulative Max Drawdown'),
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": True}, {"secondary_y": True}]]
        )
        
        # Ensemble score over time
        fig.add_trace(
            go.Scatter(x=df['trial_number'], y=df['ensemble_score'],
                      mode='lines+markers', name='Ensemble Score',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Running best score
        running_best = df['ensemble_score'].cummax()
        fig.add_trace(
            go.Scatter(x=df['trial_number'], y=running_best,
                      mode='lines', name='Best Score',
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # Correlation over time
        fig.add_trace(
            go.Scatter(x=df['trial_number'], y=df['correlation'],
                      mode='lines+markers', name='Correlation',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # Sharpe ratio
        fig.add_trace(
            go.Scatter(x=df['trial_number'], y=df['sharpe_ratio'],
                      mode='lines+markers', name='Sharpe Ratio',
                      line=dict(color='orange', width=2)),
            row=2, col=1
        )
        
        # Max drawdown
        fig.add_trace(
            go.Scatter(x=df['trial_number'], y=df['max_drawdown'],
                      mode='lines+markers', name='Max Drawdown',
                      line=dict(color='red', width=2)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Real-time Ensemble Performance Dashboard',
            showlegend=True,
            height=800,
            template='plotly_white'
        )
        
        # Save plot
        plot_path = os.path.join(PLOTS_DIR, 'performance_dashboard.html')
        fig.write_html(plot_path)
        
        # Also save as static image
        static_path = os.path.join(PLOTS_DIR, 'performance_dashboard.png')
        fig.write_image(static_path, width=1200, height=800)
    
    def _create_model_comparison(self):
        """Create model performance comparison plots."""
        if len(self.metrics_history) < 2:
            return
            
        # Extract model scores over time
        model_data = {}
        trials = []
        
        for metric in self.metrics_history:
            trials.append(metric['trial_number'])
            for model_name, score in metric['model_scores'].items():
                if model_name not in model_data:
                    model_data[model_name] = []
                model_data[model_name].append(score)
        
        # Create model comparison plot
        fig = go.Figure()
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        for i, (model_name, scores) in enumerate(model_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=trials, y=scores,
                    mode='lines+markers',
                    name=model_name,
                    line=dict(color=colors[i % len(colors)], width=2)
                )
            )
        
        fig.update_layout(
            title='Individual Model Performance Over Time',
            xaxis_title='Trial Number',
            yaxis_title='Correlation Score',
            template='plotly_white',
            height=500
        )
        
        # Save plot
        plot_path = os.path.join(PLOTS_DIR, 'model_comparison.html')
        fig.write_html(plot_path)
        
        static_path = os.path.join(PLOTS_DIR, 'model_comparison.png')
        fig.write_image(static_path, width=1000, height=500)
    
    def create_final_report(self):
        """Create a comprehensive final performance report."""
        if not self.metrics_history:
            logger.warning("No metrics available for final report")
            return
            
        df = pd.DataFrame(self.metrics_history)
        
        # Calculate summary statistics
        summary = {
            "total_trials": len(self.metrics_history),
            "best_ensemble_score": df['ensemble_score'].max(),
            "mean_ensemble_score": df['ensemble_score'].mean(),
            "std_ensemble_score": df['ensemble_score'].std(),
            "best_correlation": df['correlation'].max(),
            "mean_correlation": df['correlation'].mean(),
            "best_sharpe_ratio": df['sharpe_ratio'].max(),
            "min_max_drawdown": df['max_drawdown'].min(),
            "training_duration": str(datetime.now() - self.start_time)
        }
        
        # Model performance summary
        model_summary = {}
        for model_name in df['model_scores'].iloc[0].keys():
            model_scores = [metric['model_scores'][model_name] for metric in self.metrics_history]
            model_summary[model_name] = {
                "best_score": max(model_scores),
                "mean_score": np.mean(model_scores),
                "std_score": np.std(model_scores),
                "final_score": model_scores[-1]
            }
        
        # Create comprehensive report
        report = {
            "summary": summary,
            "model_performance": model_summary,
            "trial_history": self.metrics_history,
            "generated_at": datetime.now().isoformat()
        }
        
        # Save report
        report_path = os.path.join(LOGS_DIR, 'final_performance_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create summary visualization
        self._create_final_summary_plot(df, summary, model_summary)
        
        logger.info(f"Final performance report saved: {report_path}")
        return report
    
    def _create_final_summary_plot(self, df: pd.DataFrame, summary: Dict, model_summary: Dict):
        """Create final summary visualization."""
        
        # Create a 2x2 subplot figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Score progression
        ax1.plot(df['trial_number'], df['ensemble_score'], 'b-', alpha=0.7, label='Ensemble Score')
        ax1.plot(df['trial_number'], df['ensemble_score'].cummax(), 'r--', label='Best Score')
        ax1.set_title('Optimization Progress')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Model comparison
        model_names = list(model_summary.keys())
        final_scores = [model_summary[name]['final_score'] for name in model_names]
        colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))
        
        bars = ax2.bar(model_names, final_scores, color=colors)
        ax2.set_title('Final Model Performance')
        ax2.set_ylabel('Correlation Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, final_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 3. Metrics distribution
        metrics_to_plot = ['correlation', 'sharpe_ratio', 'rmse']
        metrics_data = [df[metric].dropna() for metric in metrics_to_plot]
        
        ax3.boxplot(metrics_data, labels=metrics_to_plot)
        ax3.set_title('Metrics Distribution')
        ax3.set_ylabel('Value')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. Training timeline
        time_data = pd.to_datetime(df['timestamp'])
        ax4.plot(time_data, df['ensemble_score'], 'g-', alpha=0.7)
        ax4.set_title('Training Timeline')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Ensemble Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save final summary plot
        summary_path = os.path.join(PLOTS_DIR, 'final_summary.png')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Final summary plot saved: {summary_path}")
    
    def print_live_summary(self):
        """Print a live summary of current performance."""
        if not self.metrics_history:
            print("No metrics available yet.")
            return
        
        latest = self.metrics_history[-1]
        best_score = max([m['ensemble_score'] for m in self.metrics_history])
        
        print("\n" + "="*50)
        print("LIVE PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Current Trial: {latest['trial_number']}")
        print(f"Latest Score: {latest['ensemble_score']:.4f}")
        print(f"Best Score: {best_score:.4f}")
        print(f"Correlation: {latest['correlation']:.4f}")
        print(f"Sharpe Ratio: {latest['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {latest['max_drawdown']:.4f}")
        print(f"RMSE: {latest['rmse']:.4f}")
        print("\nModel Scores:")
        for model, score in latest['model_scores'].items():
            print(f"  {model}: {score:.4f}")
        
        elapsed = datetime.now() - self.start_time
        print(f"\nElapsed Time: {elapsed}")
        print("="*50)
    
    def get_best_trial(self) -> Dict[str, Any]:
        """Get the best trial metrics."""
        if not self.metrics_history:
            return {}
        
        best_trial = max(self.metrics_history, key=lambda x: x['ensemble_score'])
        return best_trial
    
    def save_metrics(self):
        """Save all metrics to file."""
        metrics_path = os.path.join(LOGS_DIR, 'all_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"All metrics saved: {metrics_path}") 