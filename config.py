import os
import torch

# H100 GPU Configuration - Maximum Performance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_MEMORY_FRACTION = 0.95  # Use 95% of 80GB HBM3
FORCE_GPU = True
H100_OPTIMIZED = True  # H100-specific optimizations

# Numerai Configuration
NUMERAI_PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID", "")
NUMERAI_SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY", "")
TOURNAMENT = "numerai"
DATA_VERSION = "v5.0"

# H100-Optimized Model Configuration
MODEL_CONFIGS = {
    "xgboost": {
        "tree_method": "gpu_hist",
        "device": "cuda:0",
        "max_bin": 2048,  # Much higher for more computation
        "grow_policy": "depthwise",
        "n_estimators": 5000,  # 5x more trees (reduced to prevent crashes)
        "max_depth": 16,  # Still much deeper trees
        "learning_rate": 0.01,  # Lower LR for more iterations
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "colsample_bylevel": 0.85,
        "colsample_bynode": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,  # Use all CPU cores
        "predictor": "gpu_predictor",
        "tree_method_params": {"max_shared_memory_MB": 8192}  # More GPU memory
    },
    "lightgbm": {
        "device": "gpu",
        "gpu_platform_id": 0,
        "gpu_device_id": 0,
        "max_bin": 2047,  # Much higher
        "num_gpu": 1,
        "boosting_type": "gbdt",
        "num_leaves": 1023,  # 4x larger
        "learning_rate": 0.01,  # Lower for more iterations
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "min_data_in_leaf": 5,  # Smaller for more complexity
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "random_state": 42,
        "n_estimators": 5000,  # 5x more trees (reduced to prevent crashes)
        "n_jobs": -1,  # Use all CPU cores
        "gpu_use_dp": True,
        "max_depth": 16,  # Still much deeper
        "force_col_wise": True,  # Better CPU utilization
        "num_threads": 0  # Use all available threads
    },
    "catboost": {
        "task_type": "GPU",
        "devices": "0",
        "gpu_ram_part": 0.95,  # Use almost all GPU memory
        "iterations": 5000,  # 5x more iterations (reduced to prevent crashes)
        "learning_rate": 0.01,  # Lower for more iterations
        "depth": 12,  # Still much deeper trees
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": False,
        "early_stopping_rounds": 200,  # More patience
        "eval_metric": "RMSE",
        "max_ctr_complexity": 16,  # Higher complexity
        "gpu_cat_features_storage": "GpuRam",
        "thread_count": -1,  # Use all CPU cores
        "used_ram_limit": "16GB"  # Use more RAM
    },
    "neural_net": {
        "hidden_sizes": [4096, 2048, 1024, 512, 256, 128, 64],  # Still very large network (reduced from 8192 start)
        "dropout": 0.4,
        "learning_rate": 0.0005,  # Lower for stability
        "batch_size": 16384,  # Large batches for GPU utilization (reduced from 32768)
        "epochs": 500,  # Many more epochs
        "patience": 50,  # More patience
        "weight_decay": 1e-4,
        "use_batch_norm": True,
        "activation": "gelu",
        "use_attention": True,
        "use_transformer": True,
        "n_transformer_layers": 8,  # More transformer layers
        "aux_loss_weight": 0.1
    },
    # Add more model types for computational intensity
    "extra_trees": {
        "n_estimators": 5000,  # Many trees
        "max_depth": 25,  # Very deep
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,  # All CPU cores
        "random_state": 42
    },
    "random_forest": {
        "n_estimators": 5000,  # Many trees
        "max_depth": 25,  # Very deep
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "n_jobs": -1,  # All CPU cores
        "random_state": 42
    },
    "gradient_boosting": {
        "n_estimators": 5000,  # Many trees
        "max_depth": 15,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "max_features": "sqrt",
        "random_state": 42
    }
}

# H100-Enhanced Feature Engineering
FEATURE_ENGINEERING = {
    "enable": True,
    "aggressive_mode": True,  # H100 can handle aggressive feature engineering
    "methods": {
        # Basic transformations - more extensive
        "log_transform": True,
        "sqrt_transform": True,
        "rank_transform": True,
        "standardize": True,
        "normalize": True,
        "power_transform": True,  # Additional transforms for H100
        "quantile_transform": True,
        
        # Statistical features - massively expanded
        "rolling_stats": {
            "enable": True,
            "windows": [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200],  # Many more windows
            "stats": ["mean", "std", "min", "max", "skew", "kurt", "median", "q25", "q75", "q10", "q90", "q95", "q99", "sem", "var", "mad"]  # More statistics
        },
        
        # Interaction features - massive expansion
        "interactions": {
            "enable": True,
            "max_degree": 4,  # 4-way interactions (very expensive)
            "top_k_features": 500,  # Many more features for interactions
            "methods": ["multiply", "add", "subtract", "divide", "power", "log_ratio", "abs_diff", "max_ratio", "min_ratio"]
        },
        
        # Polynomial features - much higher degree
        "polynomial": {
            "enable": True,
            "degree": 4,  # Very high degree (computationally expensive)
            "include_bias": False,
            "interaction_only": False  # Full polynomial expansion
        },
        
        # Clustering features - extensive and expensive
        "clustering": {
            "enable": True,
            "methods": ["kmeans", "gaussian_mixture", "spectral", "hierarchical", "dbscan", "meanshift"],
            "n_clusters": [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500],  # Many more clusters
            "use_gpu": True,
            "gpu_batch_size": 50000,  # Larger batches
            "n_init": 20,  # More initializations (expensive)
            "max_iter": 1000  # More iterations
        },
        
        # Dimensionality reduction - comprehensive and expensive
        "dimensionality_reduction": {
            "enable": True,
            "methods": ["pca", "ica", "tsne", "umap", "truncated_svd", "factor_analysis", "nmf", "sparse_pca"],
            "n_components": [10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000],  # Many more components
            "tsne_perplexity": [5, 10, 30, 50, 100],  # Multiple perplexities
            "umap_neighbors": [5, 15, 30, 50, 100]  # Multiple neighbor counts
        },
        
        # Target encoding - enhanced and expensive
        "target_encoding": {
            "enable": True,
            "smoothing": [5, 10, 20, 50],  # Multiple smoothing values
            "min_samples_leaf": 1,
            "add_noise": True,
            "noise_level": 0.01,
            "cv_folds": 20,  # More folds (expensive)
            "multiple_encodings": True  # Generate multiple encodings
        },
        
        # Advanced statistical features - massively expanded
        "advanced_stats": {
            "enable": True,
            "entropy": True,
            "mutual_information": True,
            "correlation_with_neighbors": True,
            "percentile_ranks": [1, 5, 10, 25, 50, 75, 90, 95, 99],  # More percentiles
            "fourier_features": True,
            "wavelets": True,
            "autocorrelation": True,
            "cross_correlation": True,  # Cross-correlation between features
            "spectral_features": True,  # Spectral analysis features
            "fractal_dimension": True,  # Fractal dimension calculation
            "complexity_measures": True  # Various complexity measures
        },
        
        # H100-Specific Advanced Features - very expensive
        "deep_features": {
            "enable": True,
            "autoencoder_features": True,
            "embedding_features": True,
            "attention_features": True,
            "graph_features": True,  # Graph-based features
            "manifold_features": True,  # Manifold learning features
            "ensemble_features": True  # Features from ensemble predictions
        },
        
        # Time series features (treating rows as time series)
        "time_series": {
            "enable": True,
            "lag_features": [1, 2, 3, 5, 7, 10, 15, 20],  # Lag features
            "difference_features": [1, 2, 3, 5],  # Differencing
            "seasonal_decomposition": True,
            "trend_features": True
        },
        
        # Expensive mathematical transformations
        "mathematical": {
            "enable": True,
            "trigonometric": True,  # Sin, cos, tan transformations
            "hyperbolic": True,  # Sinh, cosh, tanh
            "exponential": True,  # Various exponential functions
            "logarithmic": True,  # Multiple log bases
            "power_series": True,  # Power series expansions
            "special_functions": True  # Gamma, beta, etc.
        }
    }
}

# H100-Enhanced Feature Selection
FEATURE_SELECTION = {
    "enable": True,
    "aggressive_selection": True,
    "methods": {
        # Correlation-based selection
        "correlation": {
            "enable": True,
            "threshold": 0.005,  # Even lower threshold for more features
            "remove_multicollinear": True,
            "multicollinear_threshold": 0.99  # Higher threshold
        },
        
        # Mutual information - massively expanded
        "mutual_information": {
            "enable": True,
            "k_best": 5000,  # 5x more features
            "random_state": 42,
            "n_neighbors": [3, 5, 10, 15],  # Multiple neighbor settings
            "discrete_features": "auto"
        },
        
        # RFE - much more comprehensive
        "rfe": {
            "enable": True,
            "n_features": 3000,  # 4x more features
            "step": 0.02,  # Smaller steps (more expensive)
            "cv": 10,  # More cross-validation
            "scoring": "neg_mean_squared_error",
            "n_jobs": -1  # Parallel processing
        },
        
        # Permutation importance - enhanced
        "permutation_importance": {
            "enable": True,
            "n_repeats": 20,  # More repeats (expensive)
            "random_state": 42,
            "top_k": 2500,  # More features
            "n_jobs": -1,
            "sample_weight": None
        },
        
        # SHAP-based selection - expanded
        "shap": {
            "enable": True,
            "top_k": 2000,  # More features
            "sample_size": 10000,  # Larger sample
            "max_evals": 2000,  # More evaluations
            "check_additivity": False  # Faster computation
        },
        
        # Variance-based selection
        "variance": {
            "enable": True,
            "threshold": 0.001  # Even lower threshold
        },
        
        # Statistical tests - comprehensive
        "statistical": {
            "enable": True,
            "methods": ["f_regression", "chi2", "mutual_info_regression", "f_classif"],
            "k_best": 2500,  # More features
            "alpha": 0.05
        },
        
        # Era stability - enhanced
        "era_stability": {
            "enable": True,
            "min_era_correlation": 0.001,  # Lower threshold
            "consistency_threshold": 0.5,  # Lower threshold
            "stability_metric": "both",  # Correlation and variance
            "min_eras": 10
        },
        
        # H100-Specific Selection Methods - very expensive
        "advanced_selection": {
            "enable": True,
            "boruta": True,  # Boruta feature selection (expensive)
            "lasso_path": True,  # Lasso regularization path
            "elastic_net_path": True,  # ElasticNet path
            "genetic_algorithm": True,  # GA-based selection (very expensive)
            "stability_selection": True,  # Stability selection
            "relief": True,  # ReliefF algorithm
            "mrmr": True,  # Minimum redundancy maximum relevance
            "forward_selection": True,  # Forward stepwise selection
            "backward_elimination": True  # Backward elimination
        },
        
        # Ensemble-based selection
        "ensemble_selection": {
            "enable": True,
            "n_estimators": 1000,  # Many estimators
            "max_features": "sqrt",
            "bootstrap": True,
            "n_jobs": -1
        }
    },
    
    # Final selection strategy - more sophisticated
    "final_selection": {
        "method": "ensemble_voting",  # More sophisticated than intersection
        "min_methods": 3,  # Require multiple methods to agree
        "max_features": 5000,  # Many more features for H100
        "backup_method": "mutual_information",
        "weighting_strategy": "performance_based",
        "use_stacking": True,  # Stack selection methods
        "meta_selector": "elastic_net"  # Meta-selector for final selection
    }
}

# H100-Optimized Hyperparameter Optimization
OPTUNA_CONFIG = {
    "n_trials": 2000,  # 4x more trials
    "n_jobs": 1,
    "study_name": "numerai_h100_ensemble",
    "storage": "sqlite:///optuna_studies.db",
    "sampler": "TPESampler",
    "pruner": "HyperbandPruner",  # Better pruner for many trials
    "timeout": 3600 * 48,  # 48 hours for intensive computation
    "gc_after_trial": True,
    "show_progress_bar": True,
    "parallel_suggest": True,  # H100 can handle parallel suggestions
    "n_startup_trials": 100,  # More startup trials
    "n_ei_candidates": 96  # More candidates for parallel processing
}

# H100-Enhanced Ensemble Configuration
ENSEMBLE_CONFIG = {
    "method": "advanced_stacking",  # More sophisticated stacking
    "meta_learner": "neural_net",  # Neural net meta-learner for H100
    "cv_folds": 20,  # More folds (expensive)
    "validation_split": 0.1,  # Smaller validation for more training
    "neutralization": True,
    "feature_selection": True,
    "feature_selection_method": "ensemble_voting",
    "dynamic_weighting": True,
    "diversity_bonus": 0.2,  # Higher diversity bonus
    "correlation_penalty": 0.03,  # Lower penalty
    "stacking_levels": 3,  # Multi-level stacking (expensive)
    "blend_method": "neural_blending",  # Neural network blending
    "use_feature_importance": True,
    "importance_weighting": True
}

# H100 GPU Memory Management - More Aggressive
GPU_CONFIG = {
    "optimize_memory": True,
    "mixed_precision": True,
    "gradient_accumulation": 1,  # H100 has enough memory
    "pin_memory": True,
    "non_blocking": True,
    "prefetch_factor": 8,  # More prefetching
    "persistent_workers": True,
    "compile_models": True,  # PyTorch 2.0 compilation
    "use_channels_last": True,  # Memory layout optimization
    "enable_flash_attention": True,  # H100-specific attention optimization
    "tensor_parallel": False,  # Not needed for single GPU
    "gradient_checkpointing": False,  # H100 has enough memory
    "amp_backend": "native",  # Use native AMP
    "cudnn_benchmark": True,  # Optimize for consistent input sizes
    "max_memory_fraction": 0.98,  # Use almost all GPU memory
    "enable_tf32": True,  # Enable TF32 for H100
    "allow_tf32": True
}

# H100-Optimized Data Processing - More Intensive
DATA_CONFIG = {
    "use_gpu_preprocessing": True,
    "chunk_size": 500000,  # Much larger chunks for H100
    "parallel_jobs": -1,  # Use all CPU cores
    "memory_map": True,
    "cache_processed": True,
    "validate_data": True,
    "gpu_dataloader": True,  # Use GPU dataloaders
    "persistent_cache": True,
    "compression": "lz4",  # Fast compression for H100 speeds
    "batch_processing": True,
    "prefetch_buffer_size": 16,  # Larger prefetch buffer
    "use_shared_memory": True,  # Use shared memory for data loading
    "pin_memory_device": "cuda:0"
}

# H100-Enhanced Metrics Configuration
METRICS_CONFIG = {
    "primary_metric": "correlation",
    "secondary_metrics": ["sharpe", "max_drawdown", "feature_neutral_mean", "mmc", "fnc", "ic", "rank_ic"],
    "update_frequency": 3,  # More frequent updates
    "save_frequency": 15,   # More frequent saves
    "plot_frequency": 5,    # More frequent plots
    "compute_feature_importance": True,
    "track_era_performance": True,
    "compute_shap_values": True,
    "detailed_logging": True,
    "gpu_monitoring": True,  # Monitor H100 utilization
    "memory_profiling": True
}

# H100-Specific Training Parameters
H100_TRAINING = {
    "use_autocast": True,
    "use_grad_scaler": True,
    "gradient_clip_norm": 1.0,
    "warmup_epochs": 10,
    "cosine_schedule": True,
    "weight_averaging": True,  # SWA/EMA
    "label_smoothing": 0.1,
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "advanced_optimizers": True  # Use latest optimizers
}

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"
CHECKPOINTS_DIR = "checkpoints"
FEATURE_DIR = "features"
CACHE_DIR = "cache"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, CHECKPOINTS_DIR, FEATURE_DIR, CACHE_DIR]:
    os.makedirs(dir_path, exist_ok=True) 