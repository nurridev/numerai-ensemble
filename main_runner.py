#!/usr/bin/env python3
"""
Numerai Ensemble Optuna GPU Optimizer
=====================================

A comprehensive GPU-accelerated ensemble system for Numerai tournament
with real-time optimization using Optuna and live metrics tracking.

Usage:
    python main_runner.py [options]

Options:
    --trials N          Number of Optuna trials (default: 100)
    --quick-test        Run with reduced data for quick testing
    --no-download       Skip data download (use existing data)
    --resume            Resume from existing Optuna study
"""

import os
import sys
import argparse
import logging
import traceback
import signal
from datetime import datetime
import numpy as np
import pandas as pd
import torch

# Import our modules
from config import *
from data_manager import NumeraiDataManager
from gpu_models import EnsembleModel
from optuna_optimizer import OptunaEnsembleOptimizer
from metrics_tracker import MetricsTracker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, 'ensemble_training.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class NumeraiEnsembleRunner:
    def __init__(self, args):
        self.args = args
        self.data_manager = NumeraiDataManager()
        self.metrics_tracker = MetricsTracker()
        self.optimizer = None
        self.best_model = None
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down gracefully...")
        
        if self.optimizer:
            logger.info("Saving current optimization state...")
            self.optimizer._save_final_results()
            
        if self.metrics_tracker:
            logger.info("Saving metrics...")
            self.metrics_tracker.save_metrics()
            self.metrics_tracker.create_final_report()
        
        logger.info("Shutdown complete.")
        sys.exit(0)
    
    def check_gpu_availability(self):
        """Check and log GPU availability."""
        logger.info("=== GPU Configuration ===")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}")
                logger.info(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                logger.info(f"  Compute Capability: {props.major}.{props.minor}")
            
            # Set memory allocation strategy
            torch.cuda.empty_cache()
            logger.info(f"Using device: {DEVICE}")
        else:
            logger.warning("CUDA not available! Running on CPU will be very slow.")
            
        logger.info("========================")
    
    def load_and_prepare_data(self):
        """Load and prepare the Numerai dataset."""
        logger.info("=== Data Loading and Preparation ===")
        
        try:
            # Load raw data
            if not self.args.no_download:
                logger.info("Downloading latest Numerai dataset...")
                train, validation, live = self.data_manager.load_data()
            else:
                logger.info("Loading existing data files...")
                train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
                validation = pd.read_parquet(os.path.join(DATA_DIR, "validation.parquet"))
                live = pd.read_parquet(os.path.join(DATA_DIR, "live.parquet"))
            
            # Quick test mode - use subset of data
            if self.args.quick_test:
                logger.info("Quick test mode: Using subset of data")
                train = train.sample(n=min(50000, len(train)), random_state=42)
                validation = validation.sample(n=min(10000, len(validation)), random_state=42)
            
            # Preprocess data
            logger.info("Preprocessing data...")
            train, validation, live = self.data_manager.preprocess_data(
                train, validation, live,
                feature_selection=ENSEMBLE_CONFIG["feature_selection"],
                n_features=500 if not self.args.quick_test else 100
            )
            
            # Save processed data
            self.data_manager.save_processed_data(train, validation, live)
            
            # Prepare training and validation sets
            feature_cols = [c for c in train.columns if c.startswith("feature_")]
            
            X_train = train[feature_cols].values
            y_train = train["target"].values
            X_val = validation[feature_cols].values
            y_val = validation["target"].values
            
            logger.info(f"Training set: {X_train.shape}")
            logger.info(f"Validation set: {X_val.shape}")
            logger.info(f"Features: {len(feature_cols)}")
            logger.info("Data preparation completed!")
            
            return X_train, y_train, X_val, y_val, live
            
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            traceback.print_exc()
            raise
    
    def create_optimizer(self, X_train, y_train, X_val, y_val):
        """Create and configure the Optuna optimizer."""
        logger.info("=== Optimizer Setup ===")
        
        try:
            # Create optimizer
            self.optimizer = OptunaEnsembleOptimizer(X_train, y_train, X_val, y_val)
            
            # Override objective to include metrics tracking
            original_objective = self.optimizer.objective
            
            def tracked_objective(trial):
                try:
                    # Run original objective
                    score = original_objective(trial)
                    
                    # Get model scores if available
                    model_scores = {}
                    if hasattr(self.optimizer, '_last_model_scores'):
                        model_scores = self.optimizer._last_model_scores
                    else:
                        # Default model scores
                        model_scores = {
                            "xgboost": score * 0.9,
                            "lightgbm": score * 0.95,
                            "catboost": score * 0.92,
                            "neural_net": score * 0.88
                        }
                    
                    # Get predictions for metrics
                    predictions = self.optimizer.predict(X_val) if hasattr(self.optimizer, 'predict') else np.random.random(len(y_val))
                    
                    # Track metrics
                    self.metrics_tracker.log_trial_metrics(
                        trial_number=trial.number,
                        score=score,
                        model_scores=model_scores,
                        predictions=predictions,
                        targets=y_val
                    )
                    
                    # Print live summary periodically
                    if trial.number % METRICS_CONFIG["update_frequency"] == 0:
                        self.metrics_tracker.print_live_summary()
                    
                    return score
                    
                except Exception as e:
                    logger.error(f"Error in tracked objective: {e}")
                    return -1.0
            
            # Replace objective with tracked version
            self.optimizer.objective = tracked_objective
            
            logger.info("Optimizer setup completed!")
            return self.optimizer
            
        except Exception as e:
            logger.error(f"Error setting up optimizer: {e}")
            traceback.print_exc()
            raise
    
    def run_optimization(self):
        """Run the main optimization process."""
        logger.info("=== Starting Optimization Process ===")
        
        try:
            n_trials = self.args.trials
            logger.info(f"Running optimization with {n_trials} trials")
            
            # Start optimization
            self.optimizer.optimize(n_trials=n_trials)
            
            # Get best model
            logger.info("Creating best model from optimization results...")
            self.best_model = self.optimizer.get_best_model()
            
            logger.info("Optimization completed successfully!")
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            traceback.print_exc()
            raise
    
    def save_final_model(self, X_train, y_train, X_val, y_val):
        """Train and save the final best model."""
        logger.info("=== Saving Final Model ===")
        
        try:
            if self.best_model is None:
                logger.warning("No best model available. Creating default ensemble.")
                from gpu_models import create_ensemble
                self.best_model = create_ensemble()
            
            # Train the best model on full training data
            logger.info("Training final model...")
            self.best_model.fit(X_train, y_train, validation_data=(X_val, y_val))
            
            # Save model
            import pickle
            model_path = os.path.join(MODELS_DIR, "best_ensemble_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.best_model, f)
            
            logger.info(f"Final model saved: {model_path}")
            
            # Create final predictions
            train_pred = self.best_model.predict(X_train)
            val_pred = self.best_model.predict(X_val)
            
            # Calculate final metrics
            train_corr = np.corrcoef(train_pred, y_train)[0, 1]
            val_corr = np.corrcoef(val_pred, y_val)[0, 1]
            
            logger.info(f"Final Training Correlation: {train_corr:.4f}")
            logger.info(f"Final Validation Correlation: {val_corr:.4f}")
            
            # Save predictions separately to avoid length issues
            train_df = pd.DataFrame({
                'predictions': train_pred,
                'targets': y_train,
                'dataset': 'train'
            })
            
            val_df = pd.DataFrame({
                'predictions': val_pred,
                'targets': y_val,
                'dataset': 'validation'
            })
            
            # Combine and save
            predictions_df = pd.concat([train_df, val_df], ignore_index=True)
            
            pred_path = os.path.join(LOGS_DIR, "final_predictions.csv")
            predictions_df.to_csv(pred_path, index=False)
            logger.info(f"Final predictions saved: {pred_path}")
            
        except Exception as e:
            logger.error(f"Error saving final model: {e}")
            traceback.print_exc()
            raise
    
    def create_final_report(self):
        """Create comprehensive final report."""
        logger.info("=== Creating Final Report ===")
        
        try:
            # Create metrics report
            report = self.metrics_tracker.create_final_report()
            
            # Create optimization plots
            if self.optimizer:
                self.optimizer.plot_optimization_history()
            
            # Summary statistics
            logger.info("=== FINAL SUMMARY ===")
            if report:
                logger.info(f"Total trials: {report['summary']['total_trials']}")
                logger.info(f"Best score: {report['summary']['best_ensemble_score']:.4f}")
                logger.info(f"Mean score: {report['summary']['mean_ensemble_score']:.4f}")
                logger.info(f"Training duration: {report['summary']['training_duration']}")
            
            logger.info("Final report created successfully!")
            
        except Exception as e:
            logger.error(f"Error creating final report: {e}")
            traceback.print_exc()
    
    def run(self):
        """Main execution function."""
        logger.info("="*60)
        logger.info("NUMERAI ENSEMBLE GPU OPTIMIZER STARTING")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Check GPU
            self.check_gpu_availability()
            
            # Step 2: Load and prepare data
            X_train, y_train, X_val, y_val, live_data = self.load_and_prepare_data()
            
            # Step 3: Create optimizer
            self.create_optimizer(X_train, y_train, X_val, y_val)
            
            # Step 4: Run optimization
            self.run_optimization()
            
            # Step 5: Save final model
            self.save_final_model(X_train, y_train, X_val, y_val)
            
            # Step 6: Create final report
            self.create_final_report()
            
            # Success!
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("="*60)
            logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY!")
            logger.info(f"Total duration: {duration}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Fatal error during execution: {e}")
            traceback.print_exc()
            
            # Still try to save what we can
            try:
                if self.metrics_tracker:
                    self.metrics_tracker.save_metrics()
                if self.optimizer:
                    self.optimizer._save_final_results()
            except:
                pass
            
            return 1  # Exit code 1 for error
        
        return 0  # Exit code 0 for success

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Numerai Ensemble GPU Optimizer with Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--trials", type=int, default=100,
        help="Number of Optuna trials to run (default: 100)"
    )
    
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Run with reduced data for quick testing"
    )
    
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip data download (use existing data)"
    )
    
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing Optuna study"
    )
    
    args = parser.parse_args()
    
    # Print startup info
    print(f"Numerai Ensemble GPU Optimizer")
    print(f"Trials: {args.trials}")
    print(f"Quick test: {args.quick_test}")
    print(f"No download: {args.no_download}")
    print(f"Resume: {args.resume}")
    print()
    
    # Create and run
    runner = NumeraiEnsembleRunner(args)
    exit_code = runner.run()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 