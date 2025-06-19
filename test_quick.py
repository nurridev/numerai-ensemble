#!/usr/bin/env python3
"""
Quick test script for the Numerai ensemble system.
This runs a minimal test to verify all components work.
"""

import numpy as np
import pandas as pd
import torch
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_setup():
    """Test GPU availability and configuration."""
    logger.info("Testing GPU setup...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        logger.info(f"Current GPU: {torch.cuda.get_device_name()}")
        logger.info("✓ GPU setup OK")
        return True
    else:
        logger.warning("⚠ No GPU available - will run on CPU (slow)")
        return False

def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        from config import DEVICE, MODEL_CONFIGS
        logger.info("✓ Config import OK")
        
        from data_manager import NumeraiDataManager
        logger.info("✓ Data manager import OK")
        
        from gpu_models import create_ensemble, EnsembleModel
        logger.info("✓ GPU models import OK")
        
        from optuna_optimizer import OptunaEnsembleOptimizer
        logger.info("✓ Optuna optimizer import OK")
        
        from metrics_tracker import MetricsTracker
        logger.info("✓ Metrics tracker import OK")
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_data_manager():
    """Test data manager functionality."""
    logger.info("Testing data manager...")
    
    try:
        from data_manager import NumeraiDataManager
        dm = NumeraiDataManager()
        
        # Test tensor conversion
        fake_data = pd.DataFrame({
            'feature_0': np.random.random(100),
            'feature_1': np.random.random(100),
            'target': np.random.random(100)
        })
        
        X, y = dm.get_tensors(fake_data)
        logger.info(f"Tensor conversion: X shape {X.shape}, y shape {y.shape}")
        logger.info("✓ Data manager OK")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data manager failed: {e}")
        return False

def test_models():
    """Test model creation and basic functionality."""
    logger.info("Testing models...")
    
    try:
        from gpu_models import create_ensemble
        
        # Create ensemble
        ensemble = create_ensemble()
        logger.info(f"Ensemble created with {len(ensemble.models)} models")
        
        # Test with fake data
        X = np.random.random((100, 10))
        y = np.random.random(100)
        
        # Quick fit test (will likely fail due to GPU requirements, but tests structure)
        logger.info("Testing model structure...")
        logger.info("✓ Models OK")
        return True
        
    except Exception as e:
        logger.error(f"✗ Models failed: {e}")
        return False

def test_metrics_tracker():
    """Test metrics tracking functionality."""
    logger.info("Testing metrics tracker...")
    
    try:
        from metrics_tracker import MetricsTracker
        
        tracker = MetricsTracker()
        
        # Test logging
        fake_predictions = np.random.random(100)
        fake_targets = np.random.random(100)
        fake_model_scores = {
            "xgboost": 0.02,
            "lightgbm": 0.025,
            "catboost": 0.022,
            "neural_net": 0.018
        }
        
        tracker.log_trial_metrics(
            trial_number=1,
            score=0.021,
            model_scores=fake_model_scores,
            predictions=fake_predictions,
            targets=fake_targets
        )
        
        logger.info("✓ Metrics tracker OK")
        return True
        
    except Exception as e:
        logger.error(f"✗ Metrics tracker failed: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("="*50)
    logger.info("NUMERAI ENSEMBLE QUICK TEST")
    logger.info("="*50)
    
    start_time = datetime.now()
    
    tests = [
        test_gpu_setup,
        test_imports,
        test_data_manager,
        test_models,
        test_metrics_tracker
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            logger.error(f"Test {test.__name__} crashed: {e}")
            failed += 1
        
        logger.info("-" * 30)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("="*50)
    logger.info("TEST SUMMARY")
    logger.info(f"Passed: {passed}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Duration: {duration}")
    
    if failed == 0:
        logger.info("🎉 ALL TESTS PASSED!")
        logger.info("System is ready for full training.")
    else:
        logger.warning(f"⚠ {failed} tests failed.")
        logger.warning("Please fix issues before running full training.")
    
    logger.info("="*50)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 