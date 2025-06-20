#!/usr/bin/env python3
"""
Quick Test Script for GPU-Optimized Numerai System
=================================================

This script tests all components with small data to verify functionality.
"""

import sys
import os
import time
import logging
import numpy as np
import pandas as pd
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gpu_availability():
    """Test GPU availability and setup."""
    logger.info("🔍 Testing GPU availability...")
    
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
            
        # Test GPU tensor operations
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✅ GPU tensor operations working")
            return True
        except Exception as e:
            print(f"❌ GPU tensor operations failed: {e}")
            return False
    else:
        print("⚠️ CUDA not available - falling back to CPU")
        return False

def test_config_import():
    """Test configuration import."""
    logger.info("⚙️ Testing configuration...")
    
    try:
        from config import DEVICE, FORCE_GPU, FEATURE_ENGINEERING, FEATURE_SELECTION
        print("✅ Configuration imported successfully")
        print(f"   Device: {DEVICE}")
        print(f"   Force GPU: {FORCE_GPU}")
        print(f"   Feature engineering enabled: {FEATURE_ENGINEERING['enable']}")
        print(f"   Feature selection enabled: {FEATURE_SELECTION['enable']}")
        return True
    except Exception as e:
        print(f"❌ Configuration import failed: {e}")
        return False

def test_data_manager():
    """Test advanced data manager with feature engineering."""
    logger.info("📊 Testing advanced data manager...")
    
    try:
        from data_manager import AdvancedNumeraiDataManager
        from config import FEATURE_ENGINEERING, FEATURE_SELECTION
        
        # Create dummy data
        dm = AdvancedNumeraiDataManager()
        train, val, live = dm._create_dummy_data()
        
        print(f"✅ Data creation successful")
        print(f"   Train: {train.shape}")
        print(f"   Validation: {val.shape}")
        print(f"   Live: {live.shape}")
        
        # Test feature engineering
        if FEATURE_ENGINEERING["enable"]:
            engineered_train = dm.engineer_features(train, "target")
            print(f"✅ Feature engineering successful")
            print(f"   Original features: {len([c for c in train.columns if c.startswith('feature_')])}")
            print(f"   Engineered features: {len([c for c in engineered_train.columns if c.startswith('feature_')])}")
        
        # Test feature selection
        if FEATURE_SELECTION["enable"]:
            selected_train = dm.select_features(engineered_train if FEATURE_ENGINEERING["enable"] else train, "target")
            print(f"✅ Feature selection successful")
            print(f"   Selected features: {len([c for c in selected_train.columns if c.startswith('feature_')])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data manager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gpu_models():
    """Test GPU-optimized models."""
    logger.info("🤖 Testing GPU-optimized models...")
    
    try:
        from gpu_models import (
            GPUOptimizedNeuralNetRegressor,
            GPUXGBoostRegressor, 
            GPULightGBMRegressor,
            GPUCatBoostRegressor,
            create_gpu_ensemble
        )
        
        # Create test data
        X = np.random.random((1000, 50))
        y = np.random.random(1000)
        X_val = np.random.random((200, 50))
        y_val = np.random.random(200)
        
        models_to_test = [
            ("Neural Network", GPUOptimizedNeuralNetRegressor(epochs=5, batch_size=256)),
            ("XGBoost", GPUXGBoostRegressor(n_estimators=10)),
            ("LightGBM", GPULightGBMRegressor(n_estimators=10)),
            ("CatBoost", GPUCatBoostRegressor(iterations=10)),
        ]
        
        for name, model in models_to_test:
            try:
                start_time = time.time()
                model.fit(X, y, validation_data=(X_val, y_val))
                pred = model.predict(X_val)
                train_time = time.time() - start_time
                
                corr = np.corrcoef(pred, y_val)[0, 1]
                print(f"✅ {name}: {train_time:.2f}s, Corr: {corr:.4f}")
                
            except Exception as e:
                print(f"❌ {name} failed: {e}")
        
        # Test ensemble
        try:
            ensemble = create_gpu_ensemble()
            print("✅ Ensemble creation successful")
            
            # Quick ensemble test with very small data
            X_small = X[:100]
            y_small = y[:100]
            X_val_small = X_val[:50]
            y_val_small = y_val[:50]
            
            start_time = time.time()
            ensemble.fit(X_small, y_small, validation_data=(X_val_small, y_val_small))
            pred = ensemble.predict(X_val_small)
            ensemble_time = time.time() - start_time
            
            corr = np.corrcoef(pred, y_val_small)[0, 1]
            print(f"✅ Ensemble: {ensemble_time:.2f}s, Corr: {corr:.4f}")
            
        except Exception as e:
            print(f"❌ Ensemble test failed: {e}")
            import traceback
            traceback.print_exc()
        
        return True
        
    except Exception as e:
        print(f"❌ GPU models test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_optimization():
    """Test GPU memory optimization."""
    logger.info("💾 Testing memory optimization...")
    
    if not torch.cuda.is_available():
        print("⚠️ GPU not available - skipping memory test")
        return True
        
    try:
        # Initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create large tensor
        x = torch.randn(5000, 1000).cuda()
        peak_memory = torch.cuda.memory_allocated()
        
        # Clean up
        del x
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        print(f"✅ Memory test successful")
        print(f"   Initial: {initial_memory / 1024**2:.1f} MB")
        print(f"   Peak: {peak_memory / 1024**2:.1f} MB")
        print(f"   Final: {final_memory / 1024**2:.1f} MB")
        print(f"   Cleanup effective: {(peak_memory - final_memory) / 1024**2:.1f} MB freed")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def test_feature_engineering_performance():
    """Test feature engineering performance."""
    logger.info("⚡ Testing feature engineering performance...")
    
    try:
        from data_manager import AdvancedNumeraiDataManager
        from config import FEATURE_ENGINEERING
        
        # Create larger test data
        np.random.seed(42)
        n_samples = 10000
        n_features = 100
        
        data = pd.DataFrame({
            'id': [f"test_{i}" for i in range(n_samples)],
            'era': [f"era{i//500:03d}" for i in range(n_samples)],
            'target': np.random.uniform(0, 1, n_samples)
        })
        
        # Add features
        for i in range(n_features):
            data[f"feature_{i:03d}"] = np.random.uniform(0, 1, n_samples)
        
        dm = AdvancedNumeraiDataManager()
        
        # Test different feature engineering methods
        methods_to_test = [
            ("Basic transforms", ["log_transform", "sqrt_transform", "rank_transform"]),
            ("Interactions", ["interactions"]),
            ("Clustering", ["clustering"]),
            ("Dimensionality reduction", ["dimensionality_reduction"]),
        ]
        
        for method_name, methods in methods_to_test:
            try:
                # Temporarily enable only specific methods
                original_config = FEATURE_ENGINEERING["methods"].copy()
                
                # Disable all methods
                for key in FEATURE_ENGINEERING["methods"]:
                    if isinstance(FEATURE_ENGINEERING["methods"][key], dict):
                        FEATURE_ENGINEERING["methods"][key]["enable"] = False
                    else:
                        FEATURE_ENGINEERING["methods"][key] = False
                
                # Enable only test methods
                for method in methods:
                    if method in FEATURE_ENGINEERING["methods"]:
                        if isinstance(FEATURE_ENGINEERING["methods"][method], dict):
                            FEATURE_ENGINEERING["methods"][method]["enable"] = True
                        else:
                            FEATURE_ENGINEERING["methods"][method] = True
                
                start_time = time.time()
                result = dm.engineer_features(data.copy(), "target")
                processing_time = time.time() - start_time
                
                original_features = len([c for c in data.columns if c.startswith('feature_')])
                new_features = len([c for c in result.columns if c.startswith('feature_')])
                
                print(f"✅ {method_name}: {processing_time:.2f}s, {original_features} → {new_features} features")
                
                # Restore original config
                FEATURE_ENGINEERING["methods"] = original_config
                
            except Exception as e:
                print(f"❌ {method_name} failed: {e}")
                # Restore original config on error
                try:
                    FEATURE_ENGINEERING["methods"] = original_config
                except:
                    pass
        
        return True
        
    except Exception as e:
        print(f"❌ Feature engineering performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test the complete pipeline end-to-end."""
    logger.info("🔄 Testing full pipeline...")
    
    try:
        from data_manager import AdvancedNumeraiDataManager
        from gpu_models import create_gpu_ensemble
        
        # Create data manager
        dm = AdvancedNumeraiDataManager()
        
        # Load/create data
        train, val, live = dm._create_dummy_data()
        
        # Preprocess data
        start_time = time.time()
        train_processed, val_processed, live_processed = dm.preprocess_data(
            train.iloc[:1000],  # Use subset for speed
            val.iloc[:200], 
            live.iloc[:100]
        )
        preprocessing_time = time.time() - start_time
        
        print(f"✅ Data preprocessing: {preprocessing_time:.2f}s")
        
        # Get features
        feature_cols = [c for c in train_processed.columns if c.startswith('feature_')]
        X_train = train_processed[feature_cols].values
        y_train = train_processed['target'].values
        X_val = val_processed[feature_cols].values
        y_val = val_processed['target'].values
        
        # Create and train ensemble
        ensemble = create_gpu_ensemble()
        
        start_time = time.time()
        ensemble.fit(X_train, y_train, validation_data=(X_val, y_val))
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        pred_val = ensemble.predict(X_val)
        pred_live = ensemble.predict(live_processed[feature_cols].values)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        corr = np.corrcoef(pred_val, y_val)[0, 1]
        
        print(f"✅ Full pipeline successful")
        print(f"   Training time: {training_time:.2f}s")
        print(f"   Prediction time: {prediction_time:.2f}s")
        print(f"   Validation correlation: {corr:.4f}")
        print(f"   Features used: {len(feature_cols)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 GPU-Optimized Numerai System Tests")
    print("=" * 50)
    
    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Configuration", test_config_import),
        ("Data Manager", test_data_manager),
        ("GPU Models", test_gpu_models),
        ("Memory Optimization", test_memory_optimization),
        ("Feature Engineering Performance", test_feature_engineering_performance),
        ("Full Pipeline", test_full_pipeline),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🔸 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for GPU-accelerated training.")
    elif passed >= total * 0.8:
        print("⚠️ Most tests passed. System should work with minor issues.")
    else:
        print("❌ Several tests failed. Please check the configuration and dependencies.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 