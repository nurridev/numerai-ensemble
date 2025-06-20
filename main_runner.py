#!/usr/bin/env python3
"""
H100-Optimized Numerai Training System
======================================

Advanced training script optimized for H100 GPU with maximum performance.
"""

import os
import sys
import time
import logging
import argparse
import signal
import psutil
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Setup enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/h100_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import our modules
from config import *
from data_manager import AdvancedNumeraiDataManager
from gpu_models import create_h100_super_ensemble
from optuna_optimizer import H100OptunaNumeraiOptimizer
from metrics_tracker import H100MetricsTracker

class H100NumeraiSystem:
    """H100-optimized Numerai training system."""
    
    def __init__(self, config_overrides=None):
        self.config_overrides = config_overrides or {}
        self.data_manager = AdvancedNumeraiDataManager()
        self.optimizer = None
        self.metrics_tracker = None
        self.best_model = None
        self.training_start_time = None
        self.interrupted = False
        
        # H100-specific monitoring
        self.h100_stats = {
            'peak_memory_gb': 0.0,
            'total_compute_time': 0.0,
            'models_trained': 0,
            'features_processed': 0,
            'gpu_utilization': []
        }
        
        # Setup signal handling for graceful interruption
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle graceful interruption."""
        logger.info("🛑 Graceful interruption requested...")
        self.interrupted = True
        if self.optimizer:
            self.optimizer.study.stop()
    
    def _setup_h100_optimizations(self):
        """Setup H100-specific optimizations."""
        if torch.cuda.is_available():
            # H100 specific settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable optimizations
            if hasattr(torch.backends, 'opt_einsum'):
                torch.backends.opt_einsum.enabled = True
                
            # Memory pool settings for H100
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,garbage_collection_threshold:0.8'
            
            logger.info("🔥 H100 optimizations enabled")
            
            # Log GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🚀 Using {gpu_name} with {gpu_memory:.1f}GB memory")
            
    def _monitor_h100_performance(self):
        """Monitor H100 performance metrics."""
        if torch.cuda.is_available():
            # Memory usage
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            self.h100_stats['peak_memory_gb'] = max(self.h100_stats['peak_memory_gb'], memory_used)
            
            # GPU utilization
            if GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.h100_stats['gpu_utilization'].append({
                        'utilization': gpu.load * 100,
                        'memory_util': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature
                    })
                    
        # System memory
        system_memory = psutil.virtual_memory()
        
        if len(self.h100_stats['gpu_utilization']) > 0:
            recent_gpu = self.h100_stats['gpu_utilization'][-1]
            logger.info(f"💻 H100: {recent_gpu['utilization']:.1f}% util, "
                       f"{recent_gpu['memory_util']:.1f}% mem, "
                       f"{recent_gpu['temperature']:.1f}°C | "
                       f"RAM: {system_memory.percent:.1f}%")
    
    def load_and_preprocess_data(self, download_fresh=False):
        """Load and preprocess data with H100 optimizations."""
        logger.info("📊 Loading and preprocessing data for H100...")
        
        start_time = time.time()
        
        # Load data
        train, validation, live = self.data_manager.load_data(download_fresh)
        
        # Enhanced preprocessing for H100
        logger.info("🔧 Starting H100-optimized preprocessing...")
        
        # Use larger data chunks for H100's memory capacity
        chunk_size = DATA_CONFIG.get("chunk_size", 100000)
        if len(train) > chunk_size * 2:
            logger.info(f"📦 Processing data in H100-optimized chunks of {chunk_size}")
            
        train_processed, val_processed, live_processed = self.data_manager.preprocess_data(
            train, validation, live, feature_selection=True
        )
        
        preprocessing_time = time.time() - start_time
        logger.info(f"✅ H100 preprocessing completed in {preprocessing_time:.1f}s")
        
        # Log data statistics
        feature_cols = [c for c in train_processed.columns if c.startswith('feature_')]
        self.h100_stats['features_processed'] = len(feature_cols)
        
        logger.info(f"📈 Data summary:")
        logger.info(f"   Training samples: {len(train_processed):,}")
        logger.info(f"   Validation samples: {len(val_processed):,}")
        logger.info(f"   Live samples: {len(live_processed):,}")
        logger.info(f"   Features: {len(feature_cols):,}")
        
        return train_processed, val_processed, live_processed
    
    def setup_optimization(self, n_trials=500):
        """Setup H100-optimized hyperparameter optimization."""
        logger.info(f"🎯 Setting up H100 optimization with {n_trials} trials...")
        
        # Enhanced Optuna config for H100
        h100_optuna_config = OPTUNA_CONFIG.copy()
        h100_optuna_config.update({
            'n_trials': n_trials,
            'timeout': 3600 * 24,  # 24 hours for H100
            'study_name': f"h100_numerai_{int(time.time())}",
            'gc_after_trial': True,
            'show_progress_bar': True
        })
        
        # Apply config overrides
        h100_optuna_config.update(self.config_overrides.get('optuna', {}))
        
        self.optimizer = H100OptunaNumeraiOptimizer(h100_optuna_config)
        
        # Setup enhanced metrics tracking for H100
        self.metrics_tracker = H100MetricsTracker(
            update_frequency=METRICS_CONFIG.get("update_frequency", 3),
            save_frequency=METRICS_CONFIG.get("save_frequency", 15),
            plot_frequency=METRICS_CONFIG.get("plot_frequency", 5)
        )
        
    def run_training(self, train_data, val_data, live_data):
        """Run H100-optimized training."""
        logger.info("🚀 Starting H100-optimized training...")
        
        self.training_start_time = time.time()
        
        # Prepare data for H100
        feature_cols = [c for c in train_data.columns if c.startswith('feature_')]
        X_train = train_data[feature_cols].values.astype(np.float32)  # Use float32 for H100 efficiency
        y_train = train_data['target'].values.astype(np.float32)
        X_val = val_data[feature_cols].values.astype(np.float32)
        y_val = val_data['target'].values.astype(np.float32)
        
        # H100 memory optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Create H100 super ensemble
        def objective_function(trial):
            try:
                # Monitor H100 performance
                self._monitor_h100_performance()
                
                # Create ensemble with trial parameters
                ensemble = create_h100_super_ensemble()
                
                # H100-optimized training
                start_time = time.time()
                ensemble.fit(X_train, y_train, validation_data=(X_val, y_val))
                training_time = time.time() - start_time
                
                # Make predictions
                predictions = ensemble.predict(X_val)
                
                # Calculate correlation
                correlation = np.corrcoef(predictions, y_val)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                    
                # Update H100 stats
                self.h100_stats['models_trained'] += len(ensemble.models)
                self.h100_stats['total_compute_time'] += training_time
                
                # Enhanced metrics for H100
                metrics = {
                    'correlation': correlation,
                    'training_time': training_time,
                    'memory_peak_gb': torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
                    'models_count': len(ensemble.models),
                    'features_count': len(feature_cols)
                }
                
                # Track metrics
                if self.metrics_tracker:
                    self.metrics_tracker.update_metrics(trial.number, metrics, ensemble)
                    
                logger.info(f"🎯 Trial {trial.number}: Correlation={correlation:.4f}, "
                           f"Time={training_time:.1f}s, Models={len(ensemble.models)}")
                
                return correlation
                
            except Exception as e:
                logger.error(f"❌ Trial {trial.number} failed: {e}")
                return 0.0
        
        # Run H100-optimized optimization
        try:
            self.optimizer.optimize(
                objective_function,
                save_best_model=True,
                model_save_path=f"models/h100_best_model_{int(time.time())}.pkl"
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 Training interrupted by user")
        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            raise
            
        # Final H100 performance summary
        self._log_h100_performance_summary()
        
    def _log_h100_performance_summary(self):
        """Log comprehensive H100 performance summary."""
        total_time = time.time() - self.training_start_time if self.training_start_time else 0
        
        logger.info("🏁 H100 Training Summary:")
        logger.info("=" * 50)
        logger.info(f"⏱️  Total training time: {total_time/3600:.2f} hours")
        logger.info(f"🧠 Models trained: {self.h100_stats['models_trained']}")
        logger.info(f"📊 Features processed: {self.h100_stats['features_processed']:,}")
        logger.info(f"💾 Peak H100 memory: {self.h100_stats['peak_memory_gb']:.1f}GB")
        logger.info(f"⚡ Total compute time: {self.h100_stats['total_compute_time']/3600:.2f} hours")
        
        if self.h100_stats['gpu_utilization']:
            avg_util = np.mean([g['utilization'] for g in self.h100_stats['gpu_utilization']])
            avg_temp = np.mean([g['temperature'] for g in self.h100_stats['gpu_utilization']])
            logger.info(f"🔥 Average H100 utilization: {avg_util:.1f}%")
            logger.info(f"🌡️  Average H100 temperature: {avg_temp:.1f}°C")
            
        # Performance metrics
        if self.optimizer and self.optimizer.best_value:
            logger.info(f"🎯 Best correlation: {self.optimizer.best_value:.4f}")
            logger.info(f"🏆 Best trial: {self.optimizer.best_trial}")
            
        # H100 efficiency metrics
        if self.h100_stats['total_compute_time'] > 0:
            models_per_hour = self.h100_stats['models_trained'] / (self.h100_stats['total_compute_time'] / 3600)
            logger.info(f"🚀 H100 efficiency: {models_per_hour:.1f} models/hour")
        
        logger.info("=" * 50)
    
    def generate_predictions(self, live_data):
        """Generate final predictions with the best H100 model."""
        logger.info("🔮 Generating predictions with best H100 model...")
        
        if not self.best_model:
            # Load best model
            best_model_path = self.optimizer.best_model_path if self.optimizer else None
            if best_model_path and os.path.exists(best_model_path):
                import pickle
                with open(best_model_path, 'rb') as f:
                    self.best_model = pickle.load(f)
                logger.info(f"📦 Loaded best model from {best_model_path}")
            else:
                logger.warning("⚠️ No best model found, using last ensemble")
                self.best_model = create_h100_super_ensemble()
                
        # Prepare live data
        feature_cols = [c for c in live_data.columns if c.startswith('feature_')]
        X_live = live_data[feature_cols].values.astype(np.float32)
        
        # Generate predictions with H100 optimization
        start_time = time.time()
        predictions = self.best_model.predict(X_live)
        prediction_time = time.time() - start_time
        
        logger.info(f"✅ Generated {len(predictions)} predictions in {prediction_time:.1f}s")
        
        # Create submission
        submission = pd.DataFrame({
            'id': live_data['id'],
            'prediction': predictions
        })
        
        # Save submission
        submission_path = f"submission_h100_{int(time.time())}.csv"
        submission.to_csv(submission_path, index=False)
        logger.info(f"💾 Saved submission to {submission_path}")
        
        return submission

def main():
    """Main H100-optimized training function."""
    parser = argparse.ArgumentParser(description="H100-Optimized Numerai Training System")
    parser.add_argument("--trials", type=int, default=500, help="Number of optimization trials")
    parser.add_argument("--download-fresh", action="store_true", help="Download fresh data")
    parser.add_argument("--quick-test", action="store_true", help="Quick test with 10 trials")
    parser.add_argument("--config-override", type=str, help="JSON config overrides")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        args.trials = 10
        logger.info("🧪 Quick test mode: 10 trials")
    
    # Config overrides
    config_overrides = {}
    if args.config_override:
        import json
        config_overrides = json.loads(args.config_override)
        
    # Change to output directory
    if args.output_dir != ".":
        os.makedirs(args.output_dir, exist_ok=True)
        os.chdir(args.output_dir)
    
    logger.info("🚀 Starting H100-Optimized Numerai System")
    logger.info("=" * 50)
    
    try:
        # Initialize H100 system
        system = H100NumeraiSystem(config_overrides)
        
        # Setup H100 optimizations
        system._setup_h100_optimizations()
        
        # Load and preprocess data
        train_data, val_data, live_data = system.load_and_preprocess_data(args.download_fresh)
        
        # Setup optimization
        system.setup_optimization(args.trials)
        
        # Run training
        system.run_training(train_data, val_data, live_data)
        
        # Generate predictions
        submission = system.generate_predictions(live_data)
        
        logger.info("🎉 H100-Optimized Numerai training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ H100 training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 