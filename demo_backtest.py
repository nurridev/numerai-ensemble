#!/usr/bin/env python3
"""
Demo script showing different ways to use the Numerai backtesting system.
"""

import os
import sys
from backtest_model import NumeraiBacktester

def demo_basic_backtest():
    """Demo basic backtesting functionality."""
    print("🚀 Demo 1: Basic Backtesting")
    print("="*50)
    
    # Create backtester
    backtester = NumeraiBacktester(
        model_path="models/best_ensemble_model.pkl",
        data_path="data/validation.parquet", 
        output_dir="demo_backtest_basic"
    )
    
    # Run basic backtest without era analysis
    metrics, _ = backtester.run_backtest(era_analysis=False, save_predictions=True)
    
    print(f"✅ Basic backtest completed!")
    print(f"   Correlation: {metrics.get('correlation', 'N/A'):.4f}")
    print(f"   Output: demo_backtest_basic/")
    print()

def demo_detailed_backtest():
    """Demo detailed backtesting with era analysis."""
    print("📊 Demo 2: Detailed Backtesting with Era Analysis")
    print("="*50)
    
    # Create backtester
    backtester = NumeraiBacktester(
        model_path="models/best_ensemble_model.pkl",
        data_path="data/validation.parquet",
        output_dir="demo_backtest_detailed"
    )
    
    # Run detailed backtest with era analysis
    metrics, era_df = backtester.run_backtest(era_analysis=True, save_predictions=True)
    
    print(f"✅ Detailed backtest completed!")
    print(f"   Overall Correlation: {metrics.get('correlation', 'N/A'):.4f}")
    print(f"   Mean Era Correlation: {metrics.get('mean_era_correlation', 'N/A'):.4f}")
    print(f"   Era Sharpe: {metrics.get('sharpe_era', 'N/A'):.4f}")
    print(f"   Number of Eras: {len(era_df) if era_df is not None else 0}")
    print(f"   Output: demo_backtest_detailed/")
    print()

def demo_custom_analysis():
    """Demo custom analysis using the backtester components."""
    print("🔬 Demo 3: Custom Analysis")
    print("="*50)
    
    # Create backtester but use components manually
    backtester = NumeraiBacktester(
        model_path="models/best_ensemble_model.pkl",
        data_path="data/validation.parquet",
        output_dir="demo_backtest_custom"
    )
    
    # Load model and data
    backtester.load_model()
    backtester.load_data()
    
    # Generate predictions
    backtester.make_predictions()
    
    # Custom analysis
    print(f"✅ Custom analysis:")
    print(f"   Model type: {type(backtester.model).__name__}")
    print(f"   Data shape: {backtester.data.shape}")
    print(f"   Prediction mean: {backtester.predictions.mean():.4f}")
    print(f"   Prediction std: {backtester.predictions.std():.4f}")
    
    # Calculate subset of metrics
    if 'target' in backtester.data.columns:
        import numpy as np
        correlation = np.corrcoef(backtester.predictions, backtester.data['target'])[0, 1]
        print(f"   Correlation: {correlation:.4f}")
    
    print(f"   Output: demo_backtest_custom/")
    print()

def show_available_outputs():
    """Show what outputs are available from backtesting."""
    print("📁 Available Backtest Outputs")
    print("="*50)
    
    outputs = {
        "predictions.csv": "Detailed predictions with residuals and metadata",
        "metrics.json": "Comprehensive performance metrics",
        "era_analysis.csv": "Era-by-era performance breakdown",
        "figures/prediction_scatter.png": "Prediction vs target scatter plot",
        "figures/prediction_distribution.png": "Distribution comparison plots",
        "figures/era_analysis.png": "Era analysis visualizations",
        "figures/cumulative_performance.png": "Cumulative performance plot",
        "figures/ensemble_weights.png": "Model ensemble weights visualization"
    }
    
    for filename, description in outputs.items():
        print(f"   📄 {filename}")
        print(f"      {description}")
    print()

def main():
    """Run all demos."""
    print("🎯 Numerai Ensemble Backtesting Demos")
    print("="*50)
    print()
    
    # Check if model exists
    if not os.path.exists("models/best_ensemble_model.pkl"):
        print("❌ Model file not found!")
        print("   Please run training first: python main_runner.py --quick-test --trials 5")
        return
    
    # Check if validation data exists
    if not os.path.exists("data/validation.parquet"):
        print("❌ Validation data not found!")
        print("   Please run: python create_dummy_data.py")
        return
    
    try:
        # Show available outputs
        show_available_outputs()
        
        # Run demos
        demo_basic_backtest()
        demo_detailed_backtest()
        demo_custom_analysis()
        
        print("🎉 All demos completed successfully!")
        print()
        print("Next steps:")
        print("- Check the output directories for results")
        print("- View the generated plots")
        print("- Analyze the metrics and era analysis")
        print("- Use: python backtest_model.py --help for full options")
        
    except Exception as e:
        print(f"❌ Error running demos: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 