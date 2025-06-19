# Numerai Ensemble Model Backtesting

This document describes the comprehensive backtesting system for evaluating trained Numerai ensemble models.

## 🎯 Overview

The backtesting system provides detailed performance analysis of your trained ensemble models with:

- **Comprehensive Metrics**: Correlation, Sharpe ratio, feature neutrality, and more
- **Era Analysis**: Detailed era-by-era performance breakdown 
- **Visualizations**: Interactive plots and charts
- **Risk Metrics**: Drawdown analysis and volatility measures
- **Model Analysis**: Ensemble weights and component performance

## 📊 Key Features

### Performance Metrics
- **Correlation**: Primary Numerai metric (Pearson & Spearman)
- **Feature Neutral Correlation**: Neutralized against feature exposure
- **Mean Era Correlation**: Average performance across eras
- **Era Sharpe**: Risk-adjusted era performance
- **MAE/RMSE**: Prediction accuracy measures
- **Max Drawdown**: Risk assessment

### Era Analysis
- Era-by-era correlation breakdown
- Era correlation distribution analysis
- Cumulative performance tracking
- Era size vs performance correlation

### Visualizations
- **Prediction Scatter**: Predictions vs targets with correlation line
- **Distribution Plots**: Prediction and target distributions
- **Era Analysis**: 4-panel era performance visualization
- **Cumulative Performance**: Equity curve with drawdown
- **Ensemble Weights**: Model contribution visualization

## 🚀 Quick Start

### Basic Usage

```bash
# Run comprehensive backtesting
python backtest_model.py --era-analysis --save-predictions

# Specify custom paths
python backtest_model.py \
  --model-path models/my_model.pkl \
  --data-path data/my_validation.parquet \
  --output-dir my_backtest_results

# Quick backtesting without era analysis
python backtest_model.py --save-predictions
```

### Using the Python API

```python
from backtest_model import NumeraiBacktester

# Create backtester
backtester = NumeraiBacktester(
    model_path="models/best_ensemble_model.pkl",
    data_path="data/validation.parquet",
    output_dir="backtest_results"
)

# Run complete backtesting
metrics, era_df = backtester.run_backtest(
    era_analysis=True,
    save_predictions=True
)

# Access results
print(f"Correlation: {metrics['correlation']:.4f}")
print(f"Era Sharpe: {metrics['sharpe_era']:.4f}")
```

### Custom Analysis

```python
# Load model and data
backtester.load_model()
backtester.load_data()

# Generate predictions
backtester.make_predictions()

# Calculate custom metrics
metrics = backtester.calculate_metrics()

# Era analysis
era_df = backtester.era_analysis()

# Create visualizations
backtester.create_visualizations(era_df)
```

## 📁 Output Structure

```
backtest_results/
├── predictions.csv              # Detailed predictions with residuals
├── metrics.json                 # Comprehensive performance metrics
├── era_analysis.csv            # Era-by-era performance breakdown
└── figures/
    ├── prediction_scatter.png       # Prediction vs target scatter
    ├── prediction_distribution.png  # Distribution comparison
    ├── era_analysis.png            # Era analysis visualization
    ├── cumulative_performance.png  # Cumulative returns & drawdown
    └── ensemble_weights.png        # Model weights visualization
```

## 📈 Understanding the Metrics

### Core Performance Metrics

| Metric | Description | Good Range | Interpretation |
|--------|-------------|------------|----------------|
| `correlation` | Pearson correlation with targets | > 0.02 | Higher is better |
| `rank_correlation` | Spearman rank correlation | > 0.02 | Rank-based performance |
| `feature_neutral_correlation` | Neutralized correlation | > 0.01 | Risk-adjusted performance |
| `sharpe_ratio` | Risk-adjusted returns | > 1.0 | Return per unit risk |
| `max_drawdown` | Maximum loss period | < -0.1 | Risk assessment |

### Era Analysis Metrics

| Metric | Description | Good Range |
|--------|-------------|------------|
| `mean_era_correlation` | Average era correlation | > 0.02 |
| `std_era_correlation` | Era correlation volatility | < 0.2 |
| `sharpe_era` | Era risk-adjusted performance | > 0.5 |

## 🎛️ Command Line Options

```bash
python backtest_model.py [OPTIONS]

Options:
  --model-path PATH       Path to saved model (default: models/best_ensemble_model.pkl)
  --data-path PATH        Path to validation data (default: data/validation.parquet)
  --output-dir PATH       Output directory (default: backtest_results/)
  --era-analysis          Enable detailed era-by-era analysis
  --save-predictions      Save detailed predictions to CSV
  --help                  Show help message
```

## 📊 Example Results

### Sample Metrics Output

```json
{
  "correlation": 0.0234,
  "rank_correlation": 0.0189,
  "feature_neutral_correlation": 0.0156,
  "mae": 0.2456,
  "rmse": 0.2891,
  "sharpe_ratio": 1.234,
  "max_drawdown": -0.0891,
  "mean_era_correlation": 0.0198,
  "std_era_correlation": 0.1456,
  "sharpe_era": 0.789
}
```

### Sample Era Analysis

| Era | Count | Correlation | MAE | Prediction Mean | Target Mean |
|-----|-------|-------------|-----|-----------------|-------------|
| era001 | 50 | 0.0345 | 0.234 | 0.501 | 0.485 |
| era002 | 48 | -0.0123 | 0.267 | 0.498 | 0.512 |
| era003 | 52 | 0.0567 | 0.223 | 0.503 | 0.489 |

## 🔧 Advanced Usage

### Batch Backtesting

```python
# Backtest multiple models
models = [
    "models/model_v1.pkl",
    "models/model_v2.pkl", 
    "models/model_v3.pkl"
]

results = {}
for model_path in models:
    backtester = NumeraiBacktester(model_path, "data/validation.parquet")
    metrics, _ = backtester.run_backtest()
    results[model_path] = metrics

# Compare results
for model, metrics in results.items():
    print(f"{model}: {metrics['correlation']:.4f}")
```

### Custom Visualization

```python
import matplotlib.pyplot as plt

# Run backtesting
backtester = NumeraiBacktester(model_path, data_path)
backtester.load_model()
backtester.load_data()
backtester.make_predictions()

# Custom plot
plt.figure(figsize=(10, 6))
plt.scatter(backtester.data['target'], backtester.predictions, alpha=0.5)
plt.xlabel('True Targets')
plt.ylabel('Predictions')
plt.title('Custom Prediction Analysis')
plt.show()
```

## 🚨 Troubleshooting

### Common Issues

1. **Model file not found**
   ```bash
   # Ensure model exists
   ls -la models/best_ensemble_model.pkl
   
   # Train a model first if needed
   python main_runner.py --quick-test --trials 5
   ```

2. **Data file not found**
   ```bash
   # Create dummy data for testing
   python create_dummy_data.py
   ```

3. **Memory issues with large datasets**
   ```python
   # Use data subsampling
   data = pd.read_parquet("data/validation.parquet")
   data_subset = data.sample(n=10000, random_state=42)
   data_subset.to_parquet("data/validation_subset.parquet")
   ```

4. **Missing dependencies**
   ```bash
   pip install scipy matplotlib seaborn plotly
   ```

## 🎯 Best Practices

1. **Regular Backtesting**: Run backtests after each model update
2. **Era Analysis**: Always include era analysis for robust evaluation
3. **Multiple Datasets**: Test on different validation periods
4. **Risk Assessment**: Monitor drawdown and volatility metrics
5. **Visualization Review**: Always check the generated plots
6. **Ensemble Analysis**: Understand individual model contributions

## 📚 Integration with Main System

The backtesting system integrates seamlessly with the main training pipeline:

```bash
# Complete workflow
python main_runner.py --trials 50          # Train models
python backtest_model.py --era-analysis    # Backtest results
python demo_backtest.py                    # Run demos
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generated log files
3. Examine the output metrics and visualizations
4. Use the demo script for reference examples

---

*This backtesting system provides comprehensive evaluation capabilities for your Numerai ensemble models, helping you understand performance, risk, and optimization opportunities.* 