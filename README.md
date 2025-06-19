# Numerai GPU Ensemble Optimizer

A high-performance, GPU-accelerated ensemble system for the Numerai tournament using Optuna hyperparameter optimization with real-time metrics tracking.

## Features

- **GPU-Accelerated Models**: XGBoost, LightGBM, CatBoost, and Neural Networks all optimized for GPU
- **Optuna Optimization**: Advanced hyperparameter tuning with TPE sampler and median pruning
- **Real-time Metrics**: Live performance tracking with interactive dashboards
- **Ensemble Methods**: Weighted averaging with dynamic weight optimization
- **Data Pipeline**: Automated Numerai data download and preprocessing
- **Robust Architecture**: Graceful error handling and checkpoint saving

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (recommended: RTX 3060 or better)
- 16GB+ RAM recommended
- 50GB+ free disk space

### Software
- Python 3.8+
- CUDA 11.8+
- Git

## Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd numerai
```

2. **Install CUDA-enabled PyTorch** (replace with your CUDA version):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. **Install requirements**:
```bash
pip install -r requirements.txt
```

4. **Verify GPU setup**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

## Quick Start

### Basic Usage

Run the ensemble optimizer with default settings:
```bash
python main_runner.py
```

### Quick Test Run

Test the system with reduced data:
```bash
python main_runner.py --quick-test --trials 10
```

### Custom Configuration

Run with specific number of trials:
```bash
python main_runner.py --trials 200
```

Resume from existing study:
```bash
python main_runner.py --resume
```

## Configuration

Edit `config.py` to customize:

### GPU Settings
```python
DEVICE = "cuda"  # or "cpu"
GPU_MEMORY_FRACTION = 0.8
```

### Model Parameters
```python
MODEL_CONFIGS = {
    "xgboost": {
        "tree_method": "gpu_hist",
        "n_estimators": 1000,
        "max_depth": 6,
        # ... more parameters
    },
    # ... other models
}
```

### Optuna Settings
```python
OPTUNA_CONFIG = {
    "n_trials": 100,
    "study_name": "numerai_ensemble",
    "sampler": "TPESampler",
    "pruner": "MedianPruner"
}
```

## Components

### Data Manager (`data_manager.py`)
- Downloads latest Numerai dataset
- Handles preprocessing and feature selection
- Provides GPU tensor conversion utilities

### GPU Models (`gpu_models.py`)
- GPU-optimized XGBoost, LightGBM, CatBoost
- PyTorch neural network with batch normalization
- Ensemble model with weighted averaging

### Optuna Optimizer (`optuna_optimizer.py`)
- Hyperparameter optimization for all models
- System resource monitoring
- Checkpoint saving and resumption

### Metrics Tracker (`metrics_tracker.py`)
- Real-time performance monitoring
- Interactive dashboards with Plotly
- Comprehensive final reports

## Output Structure

```
numerai/
├── data/                   # Dataset files
├── models/                 # Saved models
├── logs/                   # Training logs and reports
├── plots/                  # Performance visualizations
├── checkpoints/            # Optimization checkpoints
└── optuna_studies.db      # Optuna study database
```

## Monitoring

### Real-time Metrics

The system displays live metrics every 10 trials:
- Ensemble correlation score
- Individual model performance
- System resource usage
- Training progress

### Visualizations

Generated plots include:
- Performance dashboard (`plots/performance_dashboard.html`)
- Model comparison (`plots/model_comparison.html`)
- Optimization history (`plots/optuna_optimization.png`)
- Final summary (`plots/final_summary.png`)

### Logs

Comprehensive logging to:
- Console output with colored formatting
- `logs/ensemble_training.log` for detailed logs
- `logs/final_performance_report.json` for final metrics

## Performance Optimization

### GPU Memory Management
```python
# Automatic memory cleanup
torch.cuda.empty_cache()

# Memory fraction control
GPU_MEMORY_FRACTION = 0.8
```

### Batch Processing
```python
# Neural network batch sizes optimized for GPU
BATCH_SIZES = [256, 512, 1024, 2048]
```

### Parallel Processing
- GPU-accelerated tree methods
- Vectorized operations
- Efficient data loading

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config.py
"batch_size": 256  # instead of 1024

# Or reduce GPU memory fraction
GPU_MEMORY_FRACTION = 0.6
```

### Data Download Issues
```bash
# Skip download and use existing data
python main_runner.py --no-download
```

### Model Training Failures
- Check CUDA installation: `nvidia-smi`
- Verify PyTorch CUDA: `torch.cuda.is_available()`
- Reduce model complexity in quick test mode

## Advanced Usage

### Custom Objective Function

Modify `optuna_optimizer.py` to add custom metrics:
```python
def custom_objective(self, trial):
    # Your custom optimization logic
    score = your_metric_calculation()
    return score
```

### Additional Models

Add new models to `gpu_models.py`:
```python
class CustomGPUModel(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        # Model initialization
        pass
    
    def fit(self, X, y):
        # Training logic
        return self
    
    def predict(self, X):
        # Prediction logic
        return predictions
```

### Feature Engineering

Extend `data_manager.py` for custom features:
```python
def create_custom_features(self, df):
    # Your feature engineering
    return df
```

## Results Interpretation

### Correlation Score
- Primary metric for Numerai
- Target: > 0.02 for good performance
- Values > 0.05 are excellent

### Sharpe Ratio
- Risk-adjusted returns measure
- Higher values indicate better risk-reward ratio

### Max Drawdown
- Largest peak-to-trough decline
- Lower (more negative) values are worse

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

[License information]

## Support

For issues and questions:
- Check the troubleshooting section
- Review logs in `logs/` directory
- Open an issue on GitHub

## Performance Benchmarks

Typical performance on RTX 3080:
- Data loading: ~5 minutes
- 100 trials: ~2-4 hours
- Memory usage: ~6-8GB GPU
- Final correlation: 0.02-0.04

## Acknowledgments

- Numerai for the tournament and data
- Optuna team for the optimization framework
- XGBoost, LightGBM, CatBoost developers
- PyTorch team for the deep learning framework 