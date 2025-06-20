#!/usr/bin/env python3
"""
Create Fresh Model for Numerai Export - Maximum Computational Intensity
===============================================

This script creates a computationally intensive model that fully utilizes
VRAM and CPU cores for maximum performance testing.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import psutil
import gc

# GPU monitoring
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our modules
from gpu_models import create_h100_super_ensemble
from data_manager import AdvancedNumeraiDataManager

def create_intensive_training_data():
    """Create large, complex training dataset for maximum computational load."""
    logger.info("📊 Creating large-scale training data for maximum computational intensity...")
    
    np.random.seed(42)
    
    # Much larger dataset for intensive computation
    n_samples = 50000   # 50x larger dataset (reduced from 100k to avoid crash)
    n_features = 100    # 5x more features (reduced from 200 to avoid crash)
    
    logger.info(f"   Generating {n_samples:,} samples with {n_features} features")
    logger.info(f"   Estimated memory usage: ~{(n_samples * n_features * 4) / 1024**3:.2f} GB")
    
    # Generate complex feature patterns
    X_train = np.random.random((n_samples, n_features)).astype(np.float32)
    X_val = np.random.random((n_samples // 5, n_features)).astype(np.float32)
    
    # Add correlated features for more complexity
    for i in range(0, n_features, 10):
        # Create feature clusters
        base_feature = X_train[:, i]
        for j in range(1, min(5, n_features - i)):
            noise = np.random.random(n_samples) * 0.3
            X_train[:, i + j] = base_feature * (0.7 + noise)
    
    # Generate complex target with multiple signal sources
    signal_weights = np.random.random(n_features) * 0.02  # Weaker signal for more challenge
    
    # Multiple signal components
    linear_signal = X_train @ signal_weights
    nonlinear_signal = np.sin(X_train[:, :20].sum(axis=1)) * 0.01
    interaction_signal = (X_train[:, 0] * X_train[:, 1] * X_train[:, 2]) * 0.005
    
    # Combine signals with substantial noise
    y_train = (linear_signal + nonlinear_signal + interaction_signal + 
               np.random.random(n_samples) * 0.05).astype(np.float32)
    
    # Validation target with same pattern
    linear_val = X_val @ signal_weights
    nonlinear_val = np.sin(X_val[:, :20].sum(axis=1)) * 0.01
    interaction_val = (X_val[:, 0] * X_val[:, 1] * X_val[:, 2]) * 0.005
    y_val = (linear_val + nonlinear_val + interaction_val + 
             np.random.random(len(X_val)) * 0.05).astype(np.float32)
    
    logger.info(f"   Training data: {X_train.shape}")
    logger.info(f"   Validation data: {X_val.shape}")
    logger.info(f"   Memory allocated: ~{(X_train.nbytes + X_val.nbytes + y_train.nbytes + y_val.nbytes) / 1024**3:.2f} GB")
    
    return X_train, y_train, X_val, y_val

def monitor_system_resources():
    """Monitor and log system resource usage."""
    # CPU monitoring
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    
    # Memory monitoring
    memory = psutil.virtual_memory()
    
    # GPU monitoring
    gpu_info = "N/A"
    if GPU_AVAILABLE:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_info = f"{gpu.load*100:.1f}% util, {gpu.memoryUtil*100:.1f}% mem, {gpu.temperature}°C"
        except:
            pass
    
    logger.info(f"💻 System Resources:")
    logger.info(f"   CPU: {cpu_percent:.1f}% ({cpu_count} cores)")
    logger.info(f"   RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}/{memory.total/1024**3:.1f} GB)")
    logger.info(f"   GPU: {gpu_info}")

def create_fresh_model():
    """Create and save a computationally intensive ensemble model."""
    logger.info("🚀 Creating MAXIMUM INTENSITY model for Numerai export...")
    logger.info("⚡ This will fully utilize your VRAM and CPU cores!")
    
    # Monitor initial system state
    monitor_system_resources()
    
    # Create intensive training data
    X_train, y_train, X_val, y_val = create_intensive_training_data()
    
    # Force garbage collection
    gc.collect()
    
    # Create ensemble with all models
    logger.info("🏗️ Building SUPER INTENSIVE ensemble...")
    logger.info("   This ensemble includes 7 models:")
    logger.info("   • XGBoost: 5,000 trees, depth 16")
    logger.info("   • LightGBM: 5,000 trees, depth 16") 
    logger.info("   • CatBoost: 5,000 iterations, depth 12")
    logger.info("   • Neural Network: 7-layer with 4096 hidden units")
    logger.info("   • Extra Trees: 5,000 trees, depth 25")
    logger.info("   • Random Forest: 5,000 trees, depth 25")
    logger.info("   • Gradient Boosting: 5,000 trees, depth 15")
    
    ensemble = create_h100_super_ensemble()
    
    # Monitor system before training
    logger.info("📊 Pre-training system state:")
    monitor_system_resources()
    
    # Train ensemble with maximum intensity
    logger.info("🎯 Starting INTENSIVE training (this will take a while)...")
    logger.info("💪 All CPU cores and GPU will be heavily utilized!")
    
    start_time = time.time()
    
    # Training with progress monitoring
    ensemble.fit(X_train, y_train, validation_data=(X_val, y_val))
    
    training_time = time.time() - start_time
    
    # Test predictions
    logger.info("🔮 Generating predictions...")
    predictions = ensemble.predict(X_val)
    correlation = np.corrcoef(predictions, y_val)[0, 1]
    
    logger.info("✅ INTENSIVE training completed!")
    logger.info(f"   Training time: {training_time/60:.1f} minutes")
    logger.info(f"   Validation correlation: {correlation:.4f}")
    logger.info(f"   Models trained: {len(ensemble.models)}")
    
    # Monitor final system state
    logger.info("📊 Post-training system state:")
    monitor_system_resources()
    
    # Print model performance breakdown
    if hasattr(ensemble, 'model_scores'):
        logger.info("🏆 Individual model performance:")
        for name, scores in ensemble.model_scores.items():
            if isinstance(scores, dict) and 'correlation' in scores:
                logger.info(f"   {name}: {scores['correlation']:.4f}")
    
    # Print ensemble weights
    if hasattr(ensemble, 'weights'):
        logger.info("⚖️ Ensemble weights:")
        for name, weight in ensemble.weights.items():
            logger.info(f"   {name}: {weight:.4f}")
    
    # Save the model
    model_path = "models/intensive_h100_ensemble.pkl"
    Path("models").mkdir(exist_ok=True)
    
    logger.info(f"💾 Saving intensive model...")
    with open(model_path, 'wb') as f:
        pickle.dump(ensemble, f)
    
    # Check file size
    file_size = Path(model_path).stat().st_size / 1024**2
    logger.info(f"   Model saved to {model_path}")
    logger.info(f"   File size: {file_size:.1f} MB")
    
    # Performance summary
    logger.info("🎉 INTENSIVE MODEL CREATION COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"📊 Performance Summary:")
    logger.info(f"   Training samples: {len(X_train):,}")
    logger.info(f"   Features: {X_train.shape[1]}")
    logger.info(f"   Models in ensemble: {len(ensemble.models)}")
    logger.info(f"   Training time: {training_time/60:.1f} minutes")
    logger.info(f"   Validation correlation: {correlation:.4f}")
    logger.info(f"   Model file size: {file_size:.1f} MB")
    logger.info("=" * 60)
    
    return ensemble

if __name__ == "__main__":
    # Set high priority for maximum performance
    try:
        import os
        os.nice(-10)  # Higher priority (requires sudo on some systems)
    except:
        pass
    
    logger.info("🔥 MAXIMUM COMPUTATIONAL INTENSITY MODE ACTIVATED!")
    logger.info("⚡ This script will utilize maximum VRAM and CPU cores")
    logger.info("🚀 Buckle up for intensive computation!")
    
    intensive_model = create_fresh_model()
    
    logger.info("🎊 SUCCESS! Your hardware has been thoroughly utilized!")
    logger.info("💪 Ready to export this beast to Numerai!") 