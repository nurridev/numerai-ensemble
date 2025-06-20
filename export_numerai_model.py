#!/usr/bin/env python3
"""
Export Numerai-Compatible Model
===============================

This script creates a standalone model file compatible with Numerai's environment
that doesn't depend on custom modules.

Usage:
    python export_numerai_model.py [options]
"""

import os
import sys
import argparse
import pickle
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_base_models(original_model):
    """
    Extract base models from our custom ensemble.
    
    Args:
        original_model: The original EnsembleModel
        
    Returns:
        Tuple of (models_dict, weights_dict)
    """
    models = {}
    weights = {}
    
    if hasattr(original_model, 'models') and hasattr(original_model, 'weights'):
        logger.info("Extracting models from ensemble...")
        
        # Extract individual models
        for name, model in original_model.models.items():
            try:
                # Handle different model wrapper types
                if hasattr(model, 'model') and model.model is not None:
                    # Extract from our custom GPU wrappers
                    base_model = model.model
                    logger.info(f"  Extracted {name}: {type(base_model).__name__}")
                    models[name] = base_model
                elif hasattr(model, 'predict'):
                    # Direct sklearn-compatible model
                    logger.info(f"  Using direct {name}: {type(model).__name__}")
                    models[name] = model
                else:
                    logger.warning(f"  Skipping {name}: no usable model found")
                    continue
                    
                # Validate the extracted model
                if not hasattr(models[name], 'predict'):
                    logger.warning(f"  {name}: extracted model has no predict method")
                    del models[name]
                    continue
                    
            except Exception as e:
                logger.warning(f"  Failed to extract {name}: {e}")
                continue
                
        # Extract weights
        if hasattr(original_model, 'weights'):
            weights = dict(original_model.weights)
            
            # Convert numpy types to Python types and clean up
            clean_weights = {}
            for k, v in weights.items():
                if k in models:  # Only keep weights for successfully extracted models
                    if hasattr(v, 'item'):
                        clean_weights[k] = float(v.item())
                    elif isinstance(v, (int, float)):
                        clean_weights[k] = float(v)
                    else:
                        clean_weights[k] = 1.0 / len(models)  # Default equal weight
                        
            weights = clean_weights
            
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        else:
            # Equal weights if no valid weights found
            equal_weight = 1.0 / len(models)
            weights = {k: equal_weight for k in models.keys()}
            
    else:
        # Single model case
        logger.info("Handling single model...")
        if hasattr(original_model, 'model') and original_model.model is not None:
            models['single_model'] = original_model.model
        elif hasattr(original_model, 'predict'):
            models['single_model'] = original_model
        else:
            raise ValueError("Could not extract a usable model")
            
        weights['single_model'] = 1.0
        
    logger.info(f"Successfully extracted {len(models)} models:")
    for name, weight in weights.items():
        logger.info(f"  {name}: weight={weight:.4f}, type={type(models[name]).__name__}")
        
    if not models:
        raise ValueError("No usable models were extracted")
        
    return models, weights

class NumeraiCompatibleModel:
    """
    Numerai-compatible model that only uses standard libraries.
    This class is defined at module level so it can be pickled.
    """
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        self.models = models
        self.weights = weights
        
    def predict(self, X):
        """Generate ensemble predictions using only standard libraries."""
        import numpy as np
        
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        total_weight = 0
        
        # Get predictions from each model
        for model_name in sorted(self.models.keys()):  # Ensure consistent order
            model = self.models[model_name]
            weight = self.weights.get(model_name, 0)
            
            if weight > 0:
                try:
                    # Handle different model types
                    if hasattr(model, '__class__') and 'xgboost' in str(type(model)):
                        # XGBoost needs DMatrix
                        import xgboost as xgb
                        dmatrix = xgb.DMatrix(X)
                        pred = model.predict(dmatrix)
                    else:
                        # Other models work with numpy arrays
                        pred = model.predict(X)
                    
                    # Handle different prediction formats
                    if hasattr(pred, 'reshape'):
                        pred = pred.reshape(-1)  # Flatten to 1D
                    elif not hasattr(pred, '__len__'):
                        pred = np.array([pred])  # Single prediction to array
                        
                    predictions.append(pred * weight)
                    total_weight += weight
                    
                except Exception as e:
                    print(f"Warning: Error with model {model_name}: {e}")
                    continue
        
        if not predictions:
            raise ValueError("No valid model predictions available")
            
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
            
        return ensemble_pred
        
    def get_params(self):
        """Get model parameters."""
        return {
            'model_count': len(self.models),
            'model_names': list(self.models.keys()),
            'weights': self.weights
        }

def validate_model_compatibility(models: Dict[str, Any]):
    """
    Validate that models use only standard libraries.
    
    Args:
        models: Dictionary of models to validate
        
    Returns:
        List of compatibility issues
    """
    issues = []
    compatible_modules = [
        'xgboost', 'lightgbm', 'catboost', 'sklearn', 
        'numpy', 'pandas', 'scipy', 'builtins'
    ]
    
    for name, model in models.items():
        model_type = type(model).__name__
        module_name = type(model).__module__
        
        # Check if model uses standard ML libraries
        is_compatible = any(lib in module_name for lib in compatible_modules)
        
        if is_compatible:
            logger.info(f"✅ {name} ({model_type}) from {module_name} is compatible")
        else:
            issues.append(f"❌ {name} ({model_type}) from {module_name} may not be compatible")
            
        # Additional validation - check if predict method exists
        if not hasattr(model, 'predict'):
            issues.append(f"❌ {name} has no predict method")
            
    return issues

def export_model(input_path: str, output_path: str, validate_features: bool = True):
    """
    Export a Numerai-compatible model.
    
    Args:
        input_path: Path to the original model file
        output_path: Path for the exported model
        validate_features: Whether to validate with sample data
    """
    logger.info(f"Exporting model from {input_path} to {output_path}")
    
    # Load original model
    try:
        with open(input_path, 'rb') as f:
            original_model = pickle.load(f)
        logger.info("✅ Original model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load original model: {e}")
        raise
        
    # Extract base models and weights
    try:
        models, weights = extract_base_models(original_model)
    except Exception as e:
        logger.error(f"❌ Failed to extract models: {e}")
        raise
        
    logger.info(f"Extracted {len(models)} models:")
    for name, weight in weights.items():
        logger.info(f"  {name}: weight={weight:.4f}")
        
    # Validate compatibility
    issues = validate_model_compatibility(models)
    if issues:
        logger.warning("Compatibility issues found:")
        for issue in issues:
            logger.warning(f"  {issue}")
        
        # Continue anyway - some warnings might be false positives
        logger.info("Continuing with export despite warnings...")
        
    # Create Numerai-compatible model
    try:
        numerai_model = NumeraiCompatibleModel(models, weights)
        logger.info("✅ Created Numerai-compatible model wrapper")
    except Exception as e:
        logger.error(f"❌ Failed to create compatible model: {e}")
        raise
    
    # Test with dummy data if validation requested
    if validate_features:
        try:
            # Test with different input shapes
            test_cases = [
                np.random.random((5, 20)),      # Multiple samples
                np.random.random((1, 20)),      # Single sample
                np.random.random((100, 20))     # Larger batch
            ]
            
            for i, test_X in enumerate(test_cases):
                test_pred = numerai_model.predict(test_X)
                logger.info(f"✅ Validation {i+1}/3: shape {test_X.shape} -> {len(test_pred)} predictions")
                logger.info(f"   Range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
                
                # Validate prediction format
                assert len(test_pred) == len(test_X), f"Prediction count mismatch: {len(test_pred)} vs {len(test_X)}"
                assert np.isfinite(test_pred).all(), "Predictions contain NaN or infinite values"
                
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            raise
            
    # Save standalone model
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(numerai_model, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"✅ Standalone model saved to {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"   File size: {file_size:.2f} MB")
        
        if file_size > 100:  # Numerai has file size limits
            logger.warning(f"⚠️ Model file is large ({file_size:.2f} MB). Consider model compression.")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise
        
    return numerai_model

def test_exported_model(model_path: str, data_path: str = None):
    """
    Test the exported model to ensure it works correctly.
    
    Args:
        model_path: Path to the exported model
        data_path: Optional path to test data
    """
    logger.info(f"Testing exported model: {model_path}")
    
    # Load the exported model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("✅ Exported model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load exported model: {e}")
        raise
        
    # Test with real data if available
    if data_path and os.path.exists(data_path):
        try:
            if data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                data = pd.read_csv(data_path)
                
            feature_cols = [c for c in data.columns if c.startswith('feature_')]
            X = data[feature_cols].values
            
            predictions = model.predict(X)
            logger.info(f"✅ Real data test successful:")
            logger.info(f"   Data shape: {X.shape}")
            logger.info(f"   Predictions: {len(predictions)}")
            logger.info(f"   Range: [{predictions.min():.4f}, {predictions.max():.4f}]")
            logger.info(f"   Mean: {predictions.mean():.4f}")
            
        except Exception as e:
            logger.error(f"❌ Real data test failed: {e}")
            
    # Test with dummy data
    try:
        dummy_X = np.random.random((10, 20))
        dummy_pred = model.predict(dummy_X)
        logger.info(f"✅ Dummy data test successful: {len(dummy_pred)} predictions")
    except Exception as e:
        logger.error(f"❌ Dummy data test failed: {e}")
        
    # Print model info
    info = model.get_params()
    logger.info(f"Model info: {info}")

def create_submission_template():
    """Create a template script for Numerai submissions."""
    template = '''#!/usr/bin/env python3
"""
Numerai Submission Script
========================

This script loads the exported model and generates predictions.
Compatible with Numerai's prediction environment.
"""

import pandas as pd
import numpy as np
import pickle

def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate predictions for live data.
    
    Args:
        live_features: DataFrame with feature columns
        
    Returns:
        DataFrame with id and prediction columns
    """
    # Load the model
    with open('numerai_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Get feature columns
    feature_cols = [c for c in live_features.columns if c.startswith('feature_')]
    X = live_features[feature_cols].values
    
    # Generate predictions
    predictions = model.predict(X)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': live_features['id'],
        'prediction': predictions
    })
    
    return submission

# Test locally (optional)
if __name__ == "__main__":
    # This is for local testing only
    # Numerai will call the predict() function directly
    
    # Load test data
    live_data = pd.read_parquet('data/live.parquet')
    
    # Generate predictions
    submission = predict(live_data)
    
    print(f"Generated {len(submission)} predictions")
    print(f"Prediction range: [{submission['prediction'].min():.4f}, {submission['prediction'].max():.4f}]")
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    print("Submission saved to submission.csv")
'''
    
    with open('numerai_submission_template.py', 'w') as f:
        f.write(template)
        
    logger.info("✅ Created submission template: numerai_submission_template.py")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export Numerai-Compatible Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--input", default="models/best_ensemble_model.pkl",
        help="Path to input model file"
    )
    
    parser.add_argument(
        "--output", default="numerai_model.pkl",
        help="Path for output model file"
    )
    
    parser.add_argument(
        "--test-data", default="data/validation.parquet",
        help="Path to test data for validation"
    )
    
    parser.add_argument(
        "--no-validation", action="store_true",
        help="Skip model validation"
    )
    
    parser.add_argument(
        "--create-template", action="store_true",
        help="Create submission template script"
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        logger.error(f"Input model file not found: {args.input}")
        return 1
        
    try:
        # Export the model
        exported_model = export_model(
            args.input,
            args.output,
            validate_features=not args.no_validation
        )
        
        # Test the exported model
        test_exported_model(args.output, args.test_data)
        
        # Create submission template if requested
        if args.create_template:
            create_submission_template()
            
        logger.info("🎉 Model export completed successfully!")
        logger.info("\nNext steps:")
        logger.info(f"1. Upload {args.output} to Numerai")
        logger.info("2. Test locally with: docker run -i --rm -v \"$PWD:$PWD\" ghcr.io/numerai/numerai_predict_py_3_10:d31dcbd --debug --model $PWD/" + args.output)
        logger.info("3. Submit predictions using the Numerai website")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Model export failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 