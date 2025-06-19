import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
import logging
from typing import Dict, Any, Callable
import torch
import pickle
import json
from datetime import datetime
import psutil
import GPUtil

from gpu_models import create_ensemble, EnsembleModel, GPUXGBoostRegressor, GPULightGBMRegressor, GPUCatBoostRegressor, GPUNeuralNetRegressor
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptunaEnsembleOptimizer:
    def __init__(self, X_train, y_train, X_val, y_val):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            study_name=OPTUNA_CONFIG["study_name"],
            storage=OPTUNA_CONFIG["storage"],
            load_if_exists=True,
            sampler=getattr(optuna.samplers, OPTUNA_CONFIG["sampler"])(),
            pruner=getattr(optuna.pruners, OPTUNA_CONFIG["pruner"])()
        )
        
        self.best_params = None
        self.best_score = -np.inf
        self.trial_results = []
        
    def objective(self, trial):
        """Optuna objective function."""
        try:
            # Monitor system resources
            self._log_system_stats(trial.number)
            
            # Sample hyperparameters for each model
            params = self._sample_hyperparameters(trial)
            
            # Create models with sampled parameters
            models = self._create_models_with_params(params)
            
            # Create ensemble
            ensemble = EnsembleModel(models, ENSEMBLE_CONFIG["method"])
            
            # Train ensemble
            ensemble.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val))
            
            # Get predictions
            predictions = ensemble.predict(self.X_val)
            
            # Calculate primary metric (correlation)
            score = np.corrcoef(predictions, self.y_val)[0, 1]
            
            if np.isnan(score):
                score = -1.0
                
            # Log trial results
            self._log_trial_results(trial, params, score, predictions)
            
            # Update best parameters
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
                
            return score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return -1.0
    
    def _sample_hyperparameters(self, trial):
        """Sample hyperparameters for all models."""
        params = {}
        
        # XGBoost parameters
        params["xgboost"] = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 2000),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("xgb_reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-8, 10.0, log=True),
        }
        
        # LightGBM parameters
        params["lightgbm"] = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 2000),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 20, 100),
            "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("lgb_feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("lgb_bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("lgb_bagging_freq", 1, 10),
            "min_child_samples": trial.suggest_int("lgb_min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-8, 10.0, log=True),
        }
        
        # CatBoost parameters
        params["catboost"] = {
            "iterations": trial.suggest_int("cb_iterations", 100, 2000),
            "depth": trial.suggest_int("cb_depth", 3, 12),
            "learning_rate": trial.suggest_float("cb_learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("cb_l2_leaf_reg", 1e-8, 10.0, log=True),
            "bootstrap_type": trial.suggest_categorical("cb_bootstrap_type", ["Bayesian", "Bernoulli"]),
        }
        
        if params["catboost"]["bootstrap_type"] == "Bayesian":
            params["catboost"]["bagging_temperature"] = trial.suggest_float("cb_bagging_temperature", 0.0, 10.0)
        else:
            params["catboost"]["subsample"] = trial.suggest_float("cb_subsample", 0.6, 1.0)
        
        # Neural Network parameters
        n_layers = trial.suggest_int("nn_n_layers", 2, 5)
        hidden_sizes = []
        for i in range(n_layers):
            size = trial.suggest_int(f"nn_hidden_size_{i}", 64, 1024)
            hidden_sizes.append(size)
        
        params["neural_net"] = {
            "hidden_sizes": hidden_sizes,
            "dropout": trial.suggest_float("nn_dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("nn_learning_rate", 1e-5, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("nn_batch_size", [256, 512, 1024, 2048]),
            "epochs": trial.suggest_int("nn_epochs", 50, 200),
        }
        
        return params
    
    def _create_models_with_params(self, params):
        """Create models with the given parameters."""
        models = {}
        
        # XGBoost
        xgb_params = MODEL_CONFIGS["xgboost"].copy()
        xgb_params.update(params["xgboost"])
        models["xgboost"] = GPUXGBoostRegressor(**xgb_params)
        
        # LightGBM
        lgb_params = MODEL_CONFIGS["lightgbm"].copy()
        lgb_params.update(params["lightgbm"])
        models["lightgbm"] = GPULightGBMRegressor(**lgb_params)
        
        # CatBoost
        cb_params = MODEL_CONFIGS["catboost"].copy()
        cb_params.update(params["catboost"])
        models["catboost"] = GPUCatBoostRegressor(**cb_params)
        
        # Neural Network
        nn_params = MODEL_CONFIGS["neural_net"].copy()
        nn_params.update(params["neural_net"])
        models["neural_net"] = GPUNeuralNetRegressor(**nn_params)
        
        return models
    
    def _log_system_stats(self, trial_number):
        """Log system resource usage."""
        if trial_number % METRICS_CONFIG["update_frequency"] == 0:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            # GPU
            gpu_stats = ""
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_stats = f"GPU: {gpu.memoryUtil*100:.1f}% memory, {gpu.load*100:.1f}% utilization"
            except:
                gpu_stats = "GPU stats unavailable"
            
            logger.info(f"Trial {trial_number} - CPU: {cpu_percent:.1f}%, "
                       f"Memory: {memory.percent:.1f}%, {gpu_stats}")
    
    def _log_trial_results(self, trial, params, score, predictions):
        """Log detailed trial results."""
        result = {
            "trial_number": trial.number,
            "score": score,
            "params": params,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Calculate additional metrics
        result["metrics"] = self._calculate_additional_metrics(predictions)
        
        self.trial_results.append(result)
        
        # Log every N trials
        if trial.number % METRICS_CONFIG["update_frequency"] == 0:
            logger.info(f"Trial {trial.number}: Score = {score:.4f}")
            logger.info(f"Best score so far: {self.best_score:.4f}")
            
            # Save intermediate results
            if trial.number % METRICS_CONFIG["save_frequency"] == 0:
                self._save_checkpoint(trial.number)
    
    def _calculate_additional_metrics(self, predictions):
        """Calculate additional performance metrics."""
        metrics = {}
        
        try:
            # Correlation (primary metric)
            metrics["correlation"] = np.corrcoef(predictions, self.y_val)[0, 1]
            
            # Sharpe ratio
            daily_returns = np.diff(predictions)
            if len(daily_returns) > 1 and np.std(daily_returns) > 0:
                metrics["sharpe"] = np.mean(daily_returns) / np.std(daily_returns)
            else:
                metrics["sharpe"] = 0.0
            
            # Max drawdown
            cumulative = np.cumsum(predictions)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max)
            metrics["max_drawdown"] = np.min(drawdown)
            
            # Mean absolute error
            metrics["mae"] = np.mean(np.abs(predictions - self.y_val))
            
            # Standard deviation of predictions
            metrics["prediction_std"] = np.std(predictions)
            
        except Exception as e:
            logger.warning(f"Error calculating additional metrics: {e}")
            metrics = {"correlation": -1.0, "sharpe": 0.0, "max_drawdown": 0.0, 
                      "mae": 1.0, "prediction_std": 0.0}
        
        return metrics
    
    def _save_checkpoint(self, trial_number):
        """Save optimization checkpoint."""
        checkpoint = {
            "trial_number": trial_number,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "trial_results": self.trial_results[-50:],  # Keep last 50 results
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"optuna_checkpoint_{trial_number}.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def optimize(self, n_trials=None):
        """Run Optuna optimization."""
        if n_trials is None:
            n_trials = OPTUNA_CONFIG["n_trials"]
        
        logger.info(f"Starting Optuna optimization with {n_trials} trials...")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Training data shape: {self.X_train.shape}")
        logger.info(f"Validation data shape: {self.X_val.shape}")
        
        try:
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                n_jobs=OPTUNA_CONFIG["n_jobs"],
                show_progress_bar=True
            )
            
            # Save final results
            self._save_final_results()
            
            logger.info(f"Optimization completed!")
            logger.info(f"Best score: {self.study.best_value:.4f}")
            logger.info(f"Best parameters: {self.study.best_params}")
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            self._save_final_results()
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _save_final_results(self):
        """Save final optimization results."""
        results = {
            "best_score": self.study.best_value,
            "best_params": self.study.best_params,
            "n_trials": len(self.study.trials),
            "all_trial_results": self.trial_results,
            "study_name": self.study.study_name,
            "timestamp": datetime.now().isoformat()
        }
        
        results_path = os.path.join(LOGS_DIR, "optuna_final_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save study object
        study_path = os.path.join(LOGS_DIR, "optuna_study.pkl")
        with open(study_path, 'wb') as f:
            pickle.dump(self.study, f)
        
        logger.info(f"Final results saved: {results_path}")
    
    def get_best_model(self):
        """Create and return the best model found during optimization."""
        if self.best_params is None:
            raise ValueError("No optimization has been performed yet")
        
        models = self._create_models_with_params(self.best_params)
        ensemble = EnsembleModel(models, ENSEMBLE_CONFIG["method"])
        
        return ensemble
    
    def plot_optimization_history(self):
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=ax1)
            ax1.set_title("Optimization History")
            
            # Parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=ax2)
            ax2.set_title("Parameter Importance")
            
            plt.tight_layout()
            plot_path = os.path.join(PLOTS_DIR, "optuna_optimization.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.show()
            
            logger.info(f"Optimization plots saved: {plot_path}")
            
        except Exception as e:
            logger.warning(f"Could not create plots: {e}") 