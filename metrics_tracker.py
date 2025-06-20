#!/usr/bin/env python3
"""
H100-Optimized Metrics Tracker
===============================

Advanced real-time metrics tracking and visualization system optimized for H100 GPU performance monitoring.
"""

import os
import sys
import time
import logging
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Import configuration
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class H100MetricsTracker:
    """H100-optimized metrics tracker with real-time monitoring and advanced visualizations."""
    
    def __init__(self, update_frequency: int = 3, save_frequency: int = 15, plot_frequency: int = 5):
        self.update_frequency = update_frequency
        self.save_frequency = save_frequency
        self.plot_frequency = plot_frequency
        
        # Initialize tracking data
        self.trial_metrics = []
        self.h100_performance = []
        self.system_metrics = []
        self.model_performance = {}
        self.feature_importance_history = []
        self.correlation_history = []
        
        # H100-specific metrics
        self.h100_stats = {
            'peak_memory_usage': 0.0,
            'total_compute_time': 0.0,
            'average_utilization': 0.0,
            'temperature_history': [],
            'throughput_history': [],
            'efficiency_score': 0.0
        }
        
        # Create output directories
        self.plots_dir = Path(PLOTS_DIR)
        self.logs_dir = Path(LOGS_DIR)
        self.plots_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Initialize real-time dashboard
        self.dashboard_file = self.plots_dir / "h100_dashboard.html"
        self.last_update_time = time.time()
        
        logger.info("🔥 H100 Metrics Tracker initialized")
        
    def update_metrics(self, trial_number: int, metrics: Dict[str, Any], ensemble: Any = None):
        """Update metrics with H100-specific monitoring."""
        current_time = time.time()
        
        # Add timestamp and trial info
        metrics['timestamp'] = current_time
        metrics['trial_number'] = trial_number
        
        # H100 GPU metrics
        h100_metrics = self._collect_h100_metrics()
        metrics.update(h100_metrics)
        
        # System metrics
        system_metrics = self._collect_system_metrics()
        metrics.update(system_metrics)
        
        # Model-specific metrics
        if ensemble:
            model_metrics = self._collect_model_metrics(ensemble)
            metrics.update(model_metrics)
        
        # Store metrics
        self.trial_metrics.append(metrics)
        
        # Update correlation history
        if 'correlation' in metrics:
            self.correlation_history.append({
                'trial': trial_number,
                'correlation': metrics['correlation'],
                'timestamp': current_time
            })
        
        # Update feature importance if available
        if ensemble and hasattr(ensemble, 'get_feature_importance'):
            try:
                importance = ensemble.get_feature_importance()
                if importance:
                    self.feature_importance_history.append({
                        'trial': trial_number,
                        'importance': importance,
                        'timestamp': current_time
                    })
            except Exception as e:
                logger.warning(f"Could not collect feature importance: {e}")
        
        # Real-time updates
        if trial_number % self.update_frequency == 0:
            self._log_realtime_metrics(trial_number, metrics)
            
        # Dashboard updates
        if trial_number % self.plot_frequency == 0:
            self._update_dashboard()
            
        # Save checkpoints
        if trial_number % self.save_frequency == 0:
            self._save_checkpoint(trial_number)
    
    def _collect_h100_metrics(self) -> Dict[str, Any]:
        """Collect H100-specific performance metrics."""
        h100_metrics = {}
        
        if torch.cuda.is_available():
            # Memory metrics
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            memory_cached = torch.cuda.memory_cached() / 1024**3
            
            h100_metrics.update({
                'h100_memory_allocated_gb': memory_allocated,
                'h100_memory_reserved_gb': memory_reserved,
                'h100_memory_cached_gb': memory_cached,
                'h100_memory_utilization': memory_allocated / (torch.cuda.get_device_properties(0).total_memory / 1024**3)
            })
            
            # Update peak memory
            self.h100_stats['peak_memory_usage'] = max(
                self.h100_stats['peak_memory_usage'], 
                memory_allocated
            )
            
            # GPU utilization and temperature
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        h100_metrics.update({
                            'h100_utilization': gpu.load * 100,
                            'h100_memory_util': gpu.memoryUtil * 100,
                            'h100_temperature': gpu.temperature,
                            'h100_power_draw': getattr(gpu, 'powerDraw', 0),
                            'h100_power_limit': getattr(gpu, 'powerLimit', 0)
                        })
                        
                        # Track temperature history
                        self.h100_stats['temperature_history'].append({
                            'timestamp': time.time(),
                            'temperature': gpu.temperature,
                            'utilization': gpu.load * 100
                        })
                        
                        # Update average utilization
                        if len(self.h100_performance) > 0:
                            avg_util = np.mean([m.get('h100_utilization', 0) for m in self.h100_performance])
                            self.h100_stats['average_utilization'] = avg_util
                            
                except Exception as e:
                    logger.warning(f"Could not collect GPU metrics: {e}")
        
        # Store H100 performance data
        self.h100_performance.append(h100_metrics)
        
        return h100_metrics
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        system_metrics = {}
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        
        # Disk I/O
        disk_io = psutil.disk_io_counters()
        
        # Network I/O
        network_io = psutil.net_io_counters()
        
        system_metrics.update({
            'cpu_utilization': cpu_percent,
            'cpu_count': cpu_count,
            'memory_utilization': memory.percent,
            'memory_available_gb': memory.available / 1024**3,
            'memory_used_gb': memory.used / 1024**3,
            'disk_read_mb': disk_io.read_bytes / 1024**2 if disk_io else 0,
            'disk_write_mb': disk_io.write_bytes / 1024**2 if disk_io else 0,
            'network_sent_mb': network_io.bytes_sent / 1024**2 if network_io else 0,
            'network_recv_mb': network_io.bytes_recv / 1024**2 if network_io else 0
        })
        
        # Store system metrics
        self.system_metrics.append(system_metrics)
        
        return system_metrics
    
    def _collect_model_metrics(self, ensemble: Any) -> Dict[str, Any]:
        """Collect model-specific performance metrics."""
        model_metrics = {}
        
        try:
            # Model count and types
            model_metrics['model_count'] = len(ensemble.models) if hasattr(ensemble, 'models') else 0
            
            if hasattr(ensemble, 'models'):
                model_types = list(ensemble.models.keys())
                model_metrics['model_types'] = model_types
                
                # Individual model scores
                if hasattr(ensemble, 'model_scores'):
                    for model_name, scores in ensemble.model_scores.items():
                        if isinstance(scores, dict):
                            for metric, value in scores.items():
                                model_metrics[f'{model_name}_{metric}'] = value
                        else:
                            model_metrics[f'{model_name}_score'] = scores
            
            # Ensemble weights
            if hasattr(ensemble, 'weights'):
                for model_name, weight in ensemble.weights.items():
                    model_metrics[f'{model_name}_weight'] = weight
            
            # Feature importance
            if hasattr(ensemble, 'get_feature_importance'):
                importance = ensemble.get_feature_importance()
                if importance:
                    # Top 10 most important features
                    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                    model_metrics['top_features'] = dict(sorted_importance[:10])
                    model_metrics['feature_count'] = len(importance)
            
        except Exception as e:
            logger.warning(f"Could not collect model metrics: {e}")
        
        return model_metrics
    
    def _log_realtime_metrics(self, trial_number: int, metrics: Dict[str, Any]):
        """Log real-time metrics with H100 focus."""
        logger.info(f"🔥 H100 Metrics Update - Trial {trial_number}")
        logger.info("=" * 50)
        
        # Primary metrics
        if 'correlation' in metrics:
            logger.info(f"🎯 Correlation: {metrics['correlation']:.4f}")
        if 'training_time' in metrics:
            logger.info(f"⏱️  Training time: {metrics['training_time']:.1f}s")
        
        # H100 metrics
        if 'h100_utilization' in metrics:
            logger.info(f"🚀 H100 utilization: {metrics['h100_utilization']:.1f}%")
        if 'h100_memory_allocated_gb' in metrics:
            logger.info(f"💾 H100 memory: {metrics['h100_memory_allocated_gb']:.1f}GB")
        if 'h100_temperature' in metrics:
            logger.info(f"🌡️  H100 temperature: {metrics['h100_temperature']:.1f}°C")
        
        # System metrics
        if 'cpu_utilization' in metrics:
            logger.info(f"💻 CPU: {metrics['cpu_utilization']:.1f}%")
        if 'memory_utilization' in metrics:
            logger.info(f"🧠 RAM: {metrics['memory_utilization']:.1f}%")
        
        # Model metrics
        if 'model_count' in metrics:
            logger.info(f"🤖 Models: {metrics['model_count']}")
        
        # Performance trends
        if len(self.correlation_history) >= 5:
            recent_corrs = [c['correlation'] for c in self.correlation_history[-5:]]
            trend = np.mean(np.diff(recent_corrs))
            trend_emoji = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            logger.info(f"{trend_emoji} Recent trend: {trend:+.4f}")
        
        logger.info("=" * 50)
    
    def _update_dashboard(self):
        """Update real-time H100 dashboard."""
        try:
            # Create interactive dashboard
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Correlation Progress', 'H100 Performance',
                    'Memory Usage', 'Temperature & Utilization',
                    'Training Times', 'Model Weights'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": True}],
                       [{"secondary_y": False}, {"secondary_y": True}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Correlation progress
            if self.correlation_history:
                trials = [c['trial'] for c in self.correlation_history]
                correlations = [c['correlation'] for c in self.correlation_history]
                
                fig.add_trace(
                    go.Scatter(x=trials, y=correlations, mode='lines+markers',
                             name='Correlation', line=dict(color='blue', width=3)),
                    row=1, col=1
                )
                
                # Add trend line
                if len(correlations) > 5:
                    z = np.polyfit(trials, correlations, 1)
                    trend_line = np.poly1d(z)(trials)
                    fig.add_trace(
                        go.Scatter(x=trials, y=trend_line, mode='lines',
                                 name='Trend', line=dict(color='red', dash='dash')),
                        row=1, col=1
                    )
            
            # H100 Performance
            if self.h100_performance:
                h100_utils = [m.get('h100_utilization', 0) for m in self.h100_performance]
                h100_memory = [m.get('h100_memory_allocated_gb', 0) for m in self.h100_performance]
                trials_h100 = list(range(len(h100_utils)))
                
                fig.add_trace(
                    go.Scatter(x=trials_h100, y=h100_utils, mode='lines',
                             name='H100 Util %', line=dict(color='green')),
                    row=1, col=2
                )
                
                fig.add_trace(
                    go.Scatter(x=trials_h100, y=h100_memory, mode='lines',
                             name='H100 Memory GB', line=dict(color='orange')),
                    row=1, col=2, secondary_y=True
                )
            
            # Memory usage over time
            if self.system_metrics:
                memory_utils = [m.get('memory_utilization', 0) for m in self.system_metrics]
                h100_memory_utils = [m.get('h100_memory_allocated_gb', 0) for m in self.h100_performance]
                trials_mem = list(range(len(memory_utils)))
                
                fig.add_trace(
                    go.Scatter(x=trials_mem, y=memory_utils, mode='lines',
                             name='System RAM %', line=dict(color='purple')),
                    row=2, col=1
                )
                
                if h100_memory_utils:
                    fig.add_trace(
                        go.Scatter(x=list(range(len(h100_memory_utils))), y=h100_memory_utils,
                                 mode='lines', name='H100 Memory GB', line=dict(color='red')),
                        row=2, col=1
                    )
            
            # Temperature and utilization correlation
            if self.h100_stats['temperature_history']:
                temps = [t['temperature'] for t in self.h100_stats['temperature_history']]
                utils = [t['utilization'] for t in self.h100_stats['temperature_history']]
                
                fig.add_trace(
                    go.Scatter(x=utils, y=temps, mode='markers',
                             name='Temp vs Util', marker=dict(color='red', size=8)),
                    row=2, col=2
                )
            
            # Training times
            if self.trial_metrics:
                training_times = [m.get('training_time', 0) for m in self.trial_metrics]
                trial_nums = [m.get('trial_number', i) for i, m in enumerate(self.trial_metrics)]
                
                fig.add_trace(
                    go.Scatter(x=trial_nums, y=training_times, mode='lines+markers',
                             name='Training Time', line=dict(color='brown')),
                    row=3, col=1
                )
            
            # Model weights (latest trial)
            if self.trial_metrics:
                latest_metrics = self.trial_metrics[-1]
                weights = {}
                for key, value in latest_metrics.items():
                    if key.endswith('_weight'):
                        model_name = key.replace('_weight', '')
                        weights[model_name] = value
                
                if weights:
                    fig.add_trace(
                        go.Bar(x=list(weights.keys()), y=list(weights.values()),
                               name='Model Weights', marker_color='lightblue'),
                        row=3, col=2
                    )
            
            # Update layout
            fig.update_layout(
                title=f"H100 Numerai Training Dashboard - {datetime.now().strftime('%H:%M:%S')}",
                showlegend=True,
                height=900,
                template='plotly_dark'
            )
            
            # Save dashboard
            pyo.plot(fig, filename=str(self.dashboard_file), auto_open=False)
            
            logger.info(f"📊 H100 dashboard updated: {self.dashboard_file}")
            
        except Exception as e:
            logger.warning(f"Could not update dashboard: {e}")
    
    def _save_checkpoint(self, trial_number: int):
        """Save comprehensive checkpoint with H100 metrics."""
        checkpoint_data = {
            'trial_number': trial_number,
            'timestamp': datetime.now().isoformat(),
            'trial_metrics': self.trial_metrics[-50:],  # Last 50 trials
            'h100_performance': self.h100_performance[-50:],
            'system_metrics': self.system_metrics[-50:],
            'correlation_history': self.correlation_history[-50:],
            'h100_stats': self.h100_stats,
            'summary': {
                'total_trials': len(self.trial_metrics),
                'best_correlation': max([m.get('correlation', 0) for m in self.trial_metrics]) if self.trial_metrics else 0,
                'average_training_time': np.mean([m.get('training_time', 0) for m in self.trial_metrics]) if self.trial_metrics else 0,
                'peak_h100_memory': self.h100_stats['peak_memory_usage'],
                'average_h100_utilization': self.h100_stats['average_utilization']
            }
        }
        
        checkpoint_path = self.logs_dir / f"h100_checkpoint_trial_{trial_number}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
        
        logger.info(f"💾 H100 checkpoint saved: {checkpoint_path}")
    
    def generate_final_report(self):
        """Generate comprehensive final H100 performance report."""
        logger.info("📊 Generating final H100 performance report...")
        
        # Create comprehensive report
        report = {
            'summary': self._generate_summary_stats(),
            'h100_performance': self._analyze_h100_performance(),
            'model_analysis': self._analyze_model_performance(),
            'efficiency_metrics': self._calculate_efficiency_metrics(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save detailed report
        report_path = self.logs_dir / f"h100_final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate visualizations
        self._create_final_visualizations()
        
        # Log summary
        self._log_final_summary(report)
        
        logger.info(f"📈 Final H100 report saved: {report_path}")
        
        return report
    
    def _generate_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.trial_metrics:
            return {}
        
        correlations = [m.get('correlation', 0) for m in self.trial_metrics]
        training_times = [m.get('training_time', 0) for m in self.trial_metrics]
        
        return {
            'total_trials': len(self.trial_metrics),
            'best_correlation': max(correlations) if correlations else 0,
            'mean_correlation': np.mean(correlations) if correlations else 0,
            'std_correlation': np.std(correlations) if correlations else 0,
            'mean_training_time': np.mean(training_times) if training_times else 0,
            'total_training_time': sum(training_times) if training_times else 0,
            'trials_per_hour': len(self.trial_metrics) / (sum(training_times) / 3600) if training_times else 0
        }
    
    def _analyze_h100_performance(self) -> Dict[str, Any]:
        """Analyze H100-specific performance."""
        if not self.h100_performance:
            return {}
        
        utilizations = [m.get('h100_utilization', 0) for m in self.h100_performance]
        temperatures = [m.get('h100_temperature', 0) for m in self.h100_performance]
        memory_usage = [m.get('h100_memory_allocated_gb', 0) for m in self.h100_performance]
        
        return {
            'average_utilization': np.mean(utilizations) if utilizations else 0,
            'peak_utilization': max(utilizations) if utilizations else 0,
            'average_temperature': np.mean(temperatures) if temperatures else 0,
            'peak_temperature': max(temperatures) if temperatures else 0,
            'average_memory_usage': np.mean(memory_usage) if memory_usage else 0,
            'peak_memory_usage': max(memory_usage) if memory_usage else 0,
            'utilization_efficiency': (np.mean(utilizations) / 100) if utilizations else 0,
            'thermal_efficiency': (90 - np.mean(temperatures)) / 90 if temperatures else 0  # Assuming 90°C as limit
        }
    
    def _analyze_model_performance(self) -> Dict[str, Any]:
        """Analyze model performance patterns."""
        model_analysis = {}
        
        # Analyze individual model performance
        for metrics in self.trial_metrics:
            for key, value in metrics.items():
                if '_correlation' in key or '_score' in key:
                    model_name = key.split('_')[0]
                    if model_name not in model_analysis:
                        model_analysis[model_name] = []
                    model_analysis[model_name].append(value)
        
        # Calculate statistics for each model
        for model_name, scores in model_analysis.items():
            if scores:
                model_analysis[model_name] = {
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'best_score': max(scores),
                    'consistency': 1 - (np.std(scores) / (np.mean(scores) + 1e-8))
                }
        
        return model_analysis
    
    def _calculate_efficiency_metrics(self) -> Dict[str, Any]:
        """Calculate H100 efficiency metrics."""
        if not self.trial_metrics or not self.h100_performance:
            return {}
        
        # Correlation per unit time
        correlations = [m.get('correlation', 0) for m in self.trial_metrics]
        training_times = [m.get('training_time', 0) for m in self.trial_metrics]
        
        efficiency_scores = []
        for i, (corr, time_taken) in enumerate(zip(correlations, training_times)):
            if time_taken > 0:
                efficiency = corr / time_taken  # Correlation per second
                efficiency_scores.append(efficiency)
        
        # H100 utilization efficiency
        utilizations = [m.get('h100_utilization', 0) for m in self.h100_performance]
        memory_utils = [m.get('h100_memory_utilization', 0) for m in self.h100_performance]
        
        return {
            'correlation_per_second': np.mean(efficiency_scores) if efficiency_scores else 0,
            'peak_efficiency': max(efficiency_scores) if efficiency_scores else 0,
            'h100_compute_efficiency': np.mean(utilizations) / 100 if utilizations else 0,
            'h100_memory_efficiency': np.mean(memory_utils) if memory_utils else 0,
            'overall_efficiency_score': np.mean([
                np.mean(utilizations) / 100 if utilizations else 0,
                np.mean(memory_utils) if memory_utils else 0,
                np.mean(efficiency_scores) * 1000 if efficiency_scores else 0  # Scale for visibility
            ])
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on H100 performance."""
        recommendations = []
        
        if not self.h100_performance:
            return ["Insufficient data for recommendations"]
        
        # Analyze utilization
        utilizations = [m.get('h100_utilization', 0) for m in self.h100_performance]
        avg_util = np.mean(utilizations) if utilizations else 0
        
        if avg_util < 70:
            recommendations.append("🚀 H100 utilization is below 70%. Consider increasing batch sizes or model complexity.")
        elif avg_util > 95:
            recommendations.append("⚠️ H100 utilization is very high (>95%). Consider reducing batch sizes to avoid bottlenecks.")
        
        # Analyze memory usage
        memory_usage = [m.get('h100_memory_allocated_gb', 0) for m in self.h100_performance]
        avg_memory = np.mean(memory_usage) if memory_usage else 0
        
        if avg_memory < 20:  # Assuming 80GB total
            recommendations.append("💾 H100 memory usage is low. Consider larger models or batch sizes.")
        elif avg_memory > 70:
            recommendations.append("⚠️ H100 memory usage is high. Monitor for out-of-memory errors.")
        
        # Analyze temperature
        temperatures = [m.get('h100_temperature', 0) for m in self.h100_performance]
        avg_temp = np.mean(temperatures) if temperatures else 0
        
        if avg_temp > 85:
            recommendations.append("🌡️ H100 temperature is high. Ensure adequate cooling.")
        
        # Analyze correlation trends
        if len(self.correlation_history) > 10:
            recent_corrs = [c['correlation'] for c in self.correlation_history[-10:]]
            trend = np.polyfit(range(len(recent_corrs)), recent_corrs, 1)[0]
            
            if trend < -0.001:
                recommendations.append("📉 Correlation is declining. Consider early stopping or hyperparameter adjustment.")
            elif trend < 0.0001:
                recommendations.append("➡️ Correlation has plateaued. Consider new feature engineering or ensemble methods.")
        
        return recommendations
    
    def _create_final_visualizations(self):
        """Create comprehensive final visualizations."""
        try:
            # Set style
            plt.style.use('dark_background')
            sns.set_palette("husl")
            
            # Create figure with subplots
            fig, axes = plt.subplots(3, 2, figsize=(20, 15))
            fig.suptitle('H100 Numerai Training Performance Report', fontsize=20, fontweight='bold')
            
            # 1. Correlation progression
            if self.correlation_history:
                trials = [c['trial'] for c in self.correlation_history]
                correlations = [c['correlation'] for c in self.correlation_history]
                
                axes[0, 0].plot(trials, correlations, 'b-', linewidth=2, label='Correlation')
                axes[0, 0].fill_between(trials, correlations, alpha=0.3)
                axes[0, 0].set_title('Correlation Progression', fontweight='bold')
                axes[0, 0].set_xlabel('Trial')
                axes[0, 0].set_ylabel('Correlation')
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].legend()
            
            # 2. H100 utilization
            if self.h100_performance:
                utils = [m.get('h100_utilization', 0) for m in self.h100_performance]
                axes[0, 1].plot(utils, 'g-', linewidth=2, label='H100 Utilization')
                axes[0, 1].axhline(y=np.mean(utils), color='r', linestyle='--', label=f'Average: {np.mean(utils):.1f}%')
                axes[0, 1].set_title('H100 Utilization Over Time', fontweight='bold')
                axes[0, 1].set_xlabel('Trial')
                axes[0, 1].set_ylabel('Utilization %')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].legend()
            
            # 3. Memory usage
            if self.h100_performance:
                memory = [m.get('h100_memory_allocated_gb', 0) for m in self.h100_performance]
                axes[1, 0].plot(memory, 'orange', linewidth=2, label='H100 Memory')
                axes[1, 0].set_title('H100 Memory Usage', fontweight='bold')
                axes[1, 0].set_xlabel('Trial')
                axes[1, 0].set_ylabel('Memory (GB)')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].legend()
            
            # 4. Temperature vs Utilization
            if self.h100_stats['temperature_history']:
                temps = [t['temperature'] for t in self.h100_stats['temperature_history']]
                utils = [t['utilization'] for t in self.h100_stats['temperature_history']]
                
                scatter = axes[1, 1].scatter(utils, temps, c=range(len(temps)), cmap='viridis', alpha=0.7)
                axes[1, 1].set_title('H100 Temperature vs Utilization', fontweight='bold')
                axes[1, 1].set_xlabel('Utilization %')
                axes[1, 1].set_ylabel('Temperature °C')
                axes[1, 1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[1, 1], label='Time Progress')
            
            # 5. Training time distribution
            if self.trial_metrics:
                training_times = [m.get('training_time', 0) for m in self.trial_metrics]
                axes[2, 0].hist(training_times, bins=20, alpha=0.7, color='purple', edgecolor='white')
                axes[2, 0].set_title('Training Time Distribution', fontweight='bold')
                axes[2, 0].set_xlabel('Training Time (s)')
                axes[2, 0].set_ylabel('Frequency')
                axes[2, 0].grid(True, alpha=0.3)
            
            # 6. Model performance comparison
            model_scores = {}
            for metrics in self.trial_metrics:
                for key, value in metrics.items():
                    if key.endswith('_correlation') or key.endswith('_score'):
                        model_name = key.split('_')[0]
                        if model_name not in model_scores:
                            model_scores[model_name] = []
                        model_scores[model_name].append(value)
            
            if model_scores:
                model_names = list(model_scores.keys())
                mean_scores = [np.mean(scores) for scores in model_scores.values()]
                std_scores = [np.std(scores) for scores in model_scores.values()]
                
                bars = axes[2, 1].bar(model_names, mean_scores, yerr=std_scores, 
                                     capsize=5, alpha=0.8, color='lightblue', edgecolor='white')
                axes[2, 1].set_title('Model Performance Comparison', fontweight='bold')
                axes[2, 1].set_xlabel('Model')
                axes[2, 1].set_ylabel('Mean Score')
                axes[2, 1].grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, score in zip(bars, mean_scores):
                    axes[2, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.plots_dir / f"h100_final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='black')
            plt.close()
            
            logger.info(f"📈 Final visualizations saved: {viz_path}")
            
        except Exception as e:
            logger.warning(f"Could not create final visualizations: {e}")
    
    def _log_final_summary(self, report: Dict[str, Any]):
        """Log final performance summary."""
        logger.info("🏁 H100 FINAL PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        summary = report.get('summary', {})
        h100_perf = report.get('h100_performance', {})
        efficiency = report.get('efficiency_metrics', {})
        
        # General metrics
        logger.info(f"🧪 Total trials: {summary.get('total_trials', 0)}")
        logger.info(f"🎯 Best correlation: {summary.get('best_correlation', 0):.4f}")
        logger.info(f"📊 Mean correlation: {summary.get('mean_correlation', 0):.4f}")
        logger.info(f"⏱️  Total training time: {summary.get('total_training_time', 0)/3600:.2f} hours")
        
        # H100 performance
        logger.info(f"🚀 Average H100 utilization: {h100_perf.get('average_utilization', 0):.1f}%")
        logger.info(f"🔥 Peak H100 utilization: {h100_perf.get('peak_utilization', 0):.1f}%")
        logger.info(f"💾 Peak H100 memory: {h100_perf.get('peak_memory_usage', 0):.1f}GB")
        logger.info(f"🌡️  Average H100 temperature: {h100_perf.get('average_temperature', 0):.1f}°C")
        
        # Efficiency metrics
        logger.info(f"⚡ Correlation/second: {efficiency.get('correlation_per_second', 0):.6f}")
        logger.info(f"🎯 Overall efficiency: {efficiency.get('overall_efficiency_score', 0):.3f}")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            logger.info("\n🎯 OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"   {i}. {rec}")
        
        logger.info("=" * 60)

# Alias for backward compatibility
MetricsTracker = H100MetricsTracker 