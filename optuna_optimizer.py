#!/usr/bin/env python3
"""
H100-Optimized Optuna Hyperparameter Optimizer
===============================================

Advanced hyperparameter optimization system optimized for H100 GPU performance.
"""

import os
import sys
import time
import logging
import pickle
import psutil
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, Optional
from pathlib import Path

# Optuna imports
import optuna
from optuna import Trial
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler, CmaEsSampler
from optuna.storages import RDBStorage
from optuna.integration import PyTorchLightningPruningCallback

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

class H100OptunaNumeraiOptimizer:
    """H100-optimized Optuna hyperparameter optimizer for Numerai."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.study = None
        self.best_value = None
        self.best_trial = None
        self.best_model = None
        self.best_model_path = None
        self.optimization_start_time = None
        
        # H100-specific tracking
        self.h100_metrics = {
            'trials_completed': 0,
            'trials_pruned': 0,
            'average_trial_time': 0.0,
            'peak_gpu_memory': 0.0,
            'total_gpu_time': 0.0,
            'gpu_efficiency': []
        }
        
        # Setup study
        self._setup_study()
        
    def _setup_study(self):
        """Setup H100-optimized Optuna study."""
        logger.info("🔧 Setting up H100-optimized Optuna study...")
        
        # Enhanced sampler for H100
        if self.config.get("sampler") == "TPESampler":
            sampler = TPESampler(
                n_startup_trials=20,  # More startup trials for H100
                n_ei_candidates=48,   # More candidates for H100 parallel processing
                multivariate=True,    # Multivariate optimization
                group=True,           # Group similar parameters
                seed=42
            )
        elif self.config.get("sampler") == "CmaEsSampler":
            sampler = CmaEsSampler(
                n_startup_trials=30,
                independent_sampler=TPESampler(seed=42),
                seed=42
            )
        else:
            sampler = TPESampler(seed=42)
            
        # Enhanced pruner for H100
        if self.config.get("pruner") == "HyperbandPruner":
            pruner = HyperbandPruner(
                min_resource=10,      # Minimum epochs/iterations
                max_resource=200,     # Maximum epochs/iterations  
                reduction_factor=3,   # Aggressive pruning for H100
                bootstrap_count=2
            )
        elif self.config.get("pruner") == "MedianPruner":
            pruner = MedianPruner(
                n_startup_trials=15,
                n_warmup_steps=10,
                interval_steps=5
            )
        else:
            pruner = HyperbandPruner()
            
        # Storage configuration
        storage = self.config.get("storage", "sqlite:///h100_optuna_studies.db")
        
        # Create study with H100 optimizations
        study_name = self.config.get("study_name", f"h100_numerai_{int(time.time())}")
        
        self.study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
            direction="maximize",  # Maximize correlation
            load_if_exists=True
        )
        
        logger.info(f"📊 H100 Optuna study '{study_name}' initialized")
        logger.info(f"   Sampler: {type(sampler).__name__}")
        logger.info(f"   Pruner: {type(pruner).__name__}")
        
    def _suggest_h100_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest H100-optimized hyperparameters."""
        params = {}
        
        # Neural Network H100-specific parameters
        params['neural_net'] = {
            'hidden_sizes': [
                trial.suggest_int('nn_layer1_size', 1024, 4096, step=256),
                trial.suggest_int('nn_layer2_size', 512, 2048, step=128),
                trial.suggest_int('nn_layer3_size', 256, 1024, step=64),
                trial.suggest_int('nn_layer4_size', 128, 512, step=32)
            ],
            'dropout': trial.suggest_float('nn_dropout', 0.2, 0.6),
            'learning_rate': trial.suggest_float('nn_lr', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('nn_batch_size', [4096, 8192, 12288, 16384]),
            'epochs': trial.suggest_int('nn_epochs', 50, 300),
            'use_transformer': trial.suggest_categorical('nn_use_transformer', [True, False]),
            'use_attention': trial.suggest_categorical('nn_use_attention', [True, False]),
            'n_transformer_layers': trial.suggest_int('nn_transformer_layers', 2, 8),
            'activation': trial.suggest_categorical('nn_activation', ['gelu', 'swish', 'mish']),
            'weight_decay': trial.suggest_float('nn_weight_decay', 1e-6, 1e-3, log=True),
            'aux_loss_weight': trial.suggest_float('nn_aux_loss_weight', 0.05, 0.3)
        }
        
        # XGBoost H100-specific parameters
        params['xgboost'] = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 1000, 5000, step=500),
            'max_depth': trial.suggest_int('xgb_max_depth', 8, 15),
            'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1),
            'subsample': trial.suggest_float('xgb_subsample', 0.7, 0.95),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 0.95),
            'colsample_bylevel': trial.suggest_float('xgb_colsample_bylevel', 0.7, 0.95),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.5, 2.0),
            'max_bin': trial.suggest_categorical('xgb_max_bin', [512, 1024, 2048])
        }
        
        # LightGBM H100-specific parameters
        params['lightgbm'] = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 1000, 5000, step=500),
            'max_depth': trial.suggest_int('lgb_max_depth', 8, 15),
            'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.1),
            'num_leaves': trial.suggest_int('lgb_num_leaves', 127, 511),
            'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.7, 0.95),
            'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.7, 0.95),
            'min_data_in_leaf': trial.suggest_int('lgb_min_data_in_leaf', 5, 50),
            'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0.0, 1.0),
            'max_bin': trial.suggest_categorical('lgb_max_bin', [511, 1023, 2047])
        }
        
        # CatBoost H100-specific parameters
        params['catboost'] = {
            'iterations': trial.suggest_int('cb_iterations', 1000, 5000, step=500),
            'depth': trial.suggest_int('cb_depth', 8, 12),
            'learning_rate': trial.suggest_float('cb_lr', 0.01, 0.1),
            'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('cb_random_strength', 0.5, 5.0),
            'bagging_temperature': trial.suggest_float('cb_bagging_temperature', 0.5, 1.5),
            'border_count': trial.suggest_categorical('cb_border_count', [128, 254, 512])
        }
        
        # Ensemble H100-specific parameters
        params['ensemble'] = {
            'method': trial.suggest_categorical('ensemble_method', ['advanced_stacking', 'neural_blend']),
            'cv_folds': trial.suggest_int('ensemble_cv_folds', 10, 20),
            'diversity_bonus': trial.suggest_float('ensemble_diversity_bonus', 0.05, 0.25),
            'blend_method': trial.suggest_categorical('ensemble_blend_method', ['weighted_average', 'rank_averaging'])
        }
        
        # Feature engineering H100-specific parameters
        params['feature_engineering'] = {
            'interaction_top_k': trial.suggest_int('fe_interaction_top_k', 50, 150),
            'clustering_n_clusters': trial.suggest_categorical('fe_clustering_n_clusters', 
                                                              [[10, 20, 50], [20, 50, 100], [50, 100, 200]]),
            'dimensionality_reduction_components': trial.suggest_categorical('fe_dr_components',
                                                                           [[20, 50, 100], [50, 100, 200], [100, 200, 300]]),
            'polynomial_degree': trial.suggest_int('fe_polynomial_degree', 2, 3),
            'target_encoding_smoothing': trial.suggest_float('fe_target_encoding_smoothing', 5.0, 20.0)
        }
        
        return params
    
    def _monitor_h100_trial_performance(self, trial_number: int):
        """Monitor H100 performance during trial."""
        if torch.cuda.is_available():
            # GPU memory
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            self.h100_metrics['peak_gpu_memory'] = max(self.h100_metrics['peak_gpu_memory'], memory_allocated)
            
            # GPU utilization
            if GPU_AVAILABLE:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.h100_metrics['gpu_efficiency'].append({
                        'trial': trial_number,
                        'utilization': gpu.load * 100,
                        'memory_util': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature,
                        'timestamp': time.time()
                    })
                    
        # System resources
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Log performance every 10 trials
        if trial_number % 10 == 0:
            logger.info(f"🔥 H100 Performance (Trial {trial_number}):")
            logger.info(f"   GPU Memory: {memory_allocated:.1f}GB allocated, {memory_reserved:.1f}GB reserved")
            logger.info(f"   System: CPU {cpu_percent:.1f}%, RAM {memory_percent:.1f}%")
            
            if self.h100_metrics['gpu_efficiency']:
                recent_gpu = self.h100_metrics['gpu_efficiency'][-1]
                logger.info(f"   H100: {recent_gpu['utilization']:.1f}% util, {recent_gpu['temperature']:.1f}°C")
    
    def _h100_objective_wrapper(self, objective_func: Callable, trial: Trial) -> float:
        """H100-optimized objective function wrapper."""
        trial_start_time = time.time()
        
        try:
            # Monitor H100 performance
            self._monitor_h100_trial_performance(trial.number)
            
            # Get H100-optimized hyperparameters
            params = self._suggest_h100_hyperparameters(trial)
            
            # Apply parameters to configuration
            self._apply_trial_parameters(params)
            
            # Run objective function
            result = objective_func(trial)
            
            # Calculate trial time
            trial_time = time.time() - trial_start_time
            self.h100_metrics['total_gpu_time'] += trial_time
            self.h100_metrics['trials_completed'] += 1
            
            # Update average trial time
            self.h100_metrics['average_trial_time'] = (
                self.h100_metrics['total_gpu_time'] / self.h100_metrics['trials_completed']
            )
            
            # Report intermediate results for pruning
            if hasattr(trial, 'report'):
                trial.report(result, step=trial.number)
                
            # Check if trial should be pruned
            if trial.should_prune():
                self.h100_metrics['trials_pruned'] += 1
                logger.info(f"✂️ Trial {trial.number} pruned (result: {result:.4f})")
                raise optuna.TrialPruned()
            
            # Update best result
            if self.best_value is None or result > self.best_value:
                self.best_value = result
                self.best_trial = trial.number
                logger.info(f"🏆 New best result: {result:.4f} (Trial {trial.number})")
                
            logger.info(f"✅ Trial {trial.number} completed: {result:.4f} ({trial_time:.1f}s)")
            
            return result
            
        except optuna.TrialPruned:
            self.h100_metrics['trials_pruned'] += 1
            raise
        except Exception as e:
            logger.error(f"❌ Trial {trial.number} failed: {e}")
            return 0.0
        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _apply_trial_parameters(self, params: Dict[str, Any]):
        """Apply trial parameters to global configuration."""
        # Update model configs
        if 'neural_net' in params:
            MODEL_CONFIGS['neural_net'].update(params['neural_net'])
            
        if 'xgboost' in params:
            MODEL_CONFIGS['xgboost'].update(params['xgboost'])
            
        if 'lightgbm' in params:
            MODEL_CONFIGS['lightgbm'].update(params['lightgbm'])
            
        if 'catboost' in params:
            MODEL_CONFIGS['catboost'].update(params['catboost'])
            
        # Update ensemble config
        if 'ensemble' in params:
            ENSEMBLE_CONFIG.update(params['ensemble'])
            
        # Update feature engineering config
        if 'feature_engineering' in params:
            fe_params = params['feature_engineering']
            
            if 'interaction_top_k' in fe_params:
                FEATURE_ENGINEERING['methods']['interactions']['top_k_features'] = fe_params['interaction_top_k']
                
            if 'clustering_n_clusters' in fe_params:
                FEATURE_ENGINEERING['methods']['clustering']['n_clusters'] = fe_params['clustering_n_clusters']
                
            if 'dimensionality_reduction_components' in fe_params:
                FEATURE_ENGINEERING['methods']['dimensionality_reduction']['n_components'] = fe_params['dimensionality_reduction_components']
                
            if 'polynomial_degree' in fe_params:
                FEATURE_ENGINEERING['methods']['polynomial']['degree'] = fe_params['polynomial_degree']
                
            if 'target_encoding_smoothing' in fe_params:
                FEATURE_ENGINEERING['methods']['target_encoding']['smoothing'] = fe_params['target_encoding_smoothing']
    
    def optimize(self, objective_func: Callable, save_best_model: bool = True, 
                model_save_path: str = None) -> optuna.Study:
        """Run H100-optimized hyperparameter optimization."""
        logger.info("🚀 Starting H100-optimized hyperparameter optimization...")
        
        self.optimization_start_time = time.time()
        
        # Setup callbacks
        callbacks = []
        
        # Progress callback
        if self.config.get("show_progress_bar", True):
            def progress_callback(study, trial):
                if trial.number % 10 == 0:
                    logger.info(f"📊 Progress: {trial.number}/{self.config.get('n_trials', 500)} trials")
                    
                    # H100 efficiency metrics
                    if self.h100_metrics['trials_completed'] > 0:
                        avg_time = self.h100_metrics['average_trial_time']
                        pruned_rate = self.h100_metrics['trials_pruned'] / trial.number * 100
                        logger.info(f"⚡ H100 Efficiency: {avg_time:.1f}s/trial, {pruned_rate:.1f}% pruned")
                        
            callbacks.append(progress_callback)
        
        # Early stopping callback
        def early_stopping_callback(study, trial):
            if trial.number > 50:  # After 50 trials
                recent_trials = study.trials[-20:]  # Last 20 trials
                recent_values = [t.value for t in recent_trials if t.value is not None]
                
                if recent_values:
                    improvement = max(recent_values) - min(recent_values)
                    if improvement < 0.001:  # Very small improvement
                        logger.info("🛑 Early stopping: No significant improvement")
                        study.stop()
                        
        callbacks.append(early_stopping_callback)
        
        # Wrapper objective function for H100 optimizations
        def wrapped_objective(trial):
            return self._h100_objective_wrapper(objective_func, trial)
        
        # Run optimization
        try:
            self.study.optimize(
                wrapped_objective,
                n_trials=self.config.get("n_trials", 500),
                timeout=self.config.get("timeout", None),
                n_jobs=self.config.get("n_jobs", 1),
                callbacks=callbacks,
                gc_after_trial=self.config.get("gc_after_trial", True),
                show_progress_bar=False  # We handle progress in callbacks
            )
            
        except KeyboardInterrupt:
            logger.info("🛑 Optimization interrupted by user")
        except Exception as e:
            logger.error(f"❌ Optimization failed: {e}")
            raise
        
        # Save best model if requested
        if save_best_model and self.best_trial is not None:
            if model_save_path is None:
                model_save_path = f"models/h100_best_model_trial_{self.best_trial}.pkl"
                
            self._save_best_model(model_save_path)
        
        # Log final H100 performance summary
        self._log_h100_optimization_summary()
        
        return self.study
    
    def _save_best_model(self, save_path: str):
        """Save the best model from optimization."""
        try:
            # Recreate best model with best parameters
            best_trial = self.study.best_trial
            best_params = self._suggest_h100_hyperparameters(best_trial)
            self._apply_trial_parameters(best_params)
            
            # Create ensemble with best parameters
            from gpu_models import create_h100_super_ensemble
            best_ensemble = create_h100_super_ensemble()
            
            # Save model
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(best_ensemble, f)
                
            self.best_model_path = save_path
            logger.info(f"💾 Best H100 model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save best model: {e}")
    
    def _log_h100_optimization_summary(self):
        """Log comprehensive H100 optimization summary."""
        total_time = time.time() - self.optimization_start_time if self.optimization_start_time else 0
        
        logger.info("🏁 H100 Optimization Summary:")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total optimization time: {total_time/3600:.2f} hours")
        logger.info(f"🧪 Trials completed: {self.h100_metrics['trials_completed']}")
        logger.info(f"✂️  Trials pruned: {self.h100_metrics['trials_pruned']}")
        logger.info(f"⚡ Average trial time: {self.h100_metrics['average_trial_time']:.1f}s")
        logger.info(f"💾 Peak H100 memory: {self.h100_metrics['peak_gpu_memory']:.1f}GB")
        logger.info(f"🔥 Total H100 compute time: {self.h100_metrics['total_gpu_time']/3600:.2f} hours")
        
        if self.best_value is not None:
            logger.info(f"🏆 Best correlation: {self.best_value:.4f}")
            logger.info(f"🎯 Best trial: {self.best_trial}")
            
        # H100 efficiency metrics
        if self.h100_metrics['trials_completed'] > 0:
            trials_per_hour = self.h100_metrics['trials_completed'] / (total_time / 3600)
            pruning_rate = self.h100_metrics['trials_pruned'] / self.h100_metrics['trials_completed'] * 100
            logger.info(f"🚀 H100 efficiency: {trials_per_hour:.1f} trials/hour")
            logger.info(f"📈 Pruning efficiency: {pruning_rate:.1f}%")
            
        # GPU utilization summary
        if self.h100_metrics['gpu_efficiency']:
            avg_utilization = np.mean([g['utilization'] for g in self.h100_metrics['gpu_efficiency']])
            avg_temperature = np.mean([g['temperature'] for g in self.h100_metrics['gpu_efficiency']])
            max_temperature = max([g['temperature'] for g in self.h100_metrics['gpu_efficiency']])
            
            logger.info(f"🔥 Average H100 utilization: {avg_utilization:.1f}%")
            logger.info(f"🌡️  Average H100 temperature: {avg_temperature:.1f}°C")
            logger.info(f"🔥 Peak H100 temperature: {max_temperature:.1f}°C")
            
        logger.info("=" * 60)
    
    def get_best_parameters(self) -> Dict[str, Any]:
        """Get the best hyperparameters found."""
        if self.study.best_trial is None:
            return {}
            
        return self._suggest_h100_hyperparameters(self.study.best_trial)
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame."""
        trials_data = []
        
        for trial in self.study.trials:
            trial_data = {
                'trial_number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            
            # Add parameters
            for key, value in trial.params.items():
                trial_data[f'param_{key}'] = value
                
            trials_data.append(trial_data)
            
        return pd.DataFrame(trials_data)

# Alias for backward compatibility
OptunaNumeraiOptimizer = H100OptunaNumeraiOptimizer 