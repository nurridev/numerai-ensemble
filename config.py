import os
import torch

# GPU Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_MEMORY_FRACTION = 0.8

# Numerai Configuration
NUMERAI_PUBLIC_ID = os.getenv("NUMERAI_PUBLIC_ID", "")
NUMERAI_SECRET_KEY = os.getenv("NUMERAI_SECRET_KEY", "")
TOURNAMENT = "numerai"
DATA_VERSION = "v5.0"  # Latest version

# Model Configuration
MODEL_CONFIGS = {
    "xgboost": {
        "tree_method": "hist",
        "device": "cpu",
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    },
    "lightgbm": {
        "device": "cpu",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.01,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "random_state": 42,
        "n_estimators": 500
    },
    "catboost": {
        "task_type": "CPU",
        "iterations": 500,
        "learning_rate": 0.01,
        "depth": 6,
        "random_seed": 42,
        "verbose": False
    },
    "neural_net": {
        "hidden_sizes": [256, 128, 64],
        "dropout": 0.3,
        "learning_rate": 0.001,
        "batch_size": 512,
        "epochs": 50,
        "patience": 10
    }
}

# Optuna Configuration
OPTUNA_CONFIG = {
    "n_trials": 100,
    "n_jobs": 1,  # Single job since we're using GPU
    "study_name": "numerai_ensemble",
    "storage": "sqlite:///optuna_studies.db",
    "sampler": "TPESampler",
    "pruner": "MedianPruner"
}

# Ensemble Configuration
ENSEMBLE_CONFIG = {
    "method": "weighted_average",  # or "stacking"
    "cv_folds": 5,
    "validation_split": 0.2,
    "neutralization": True,
    "feature_selection": True,
    "feature_selection_method": "correlation"
}

# Metrics Configuration
METRICS_CONFIG = {
    "primary_metric": "correlation",
    "secondary_metrics": ["sharpe", "max_drawdown", "feature_neutral_mean"],
    "update_frequency": 10,  # Update metrics every 10 iterations
    "save_frequency": 50,   # Save checkpoint every 50 iterations
    "plot_frequency": 25    # Update plots every 25 iterations
}

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
LOGS_DIR = "logs"
PLOTS_DIR = "plots"
CHECKPOINTS_DIR = "checkpoints"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, LOGS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(dir_path, exist_ok=True) 