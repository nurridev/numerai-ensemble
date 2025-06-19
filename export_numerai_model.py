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

class StandaloneEnsembleModel:
    """
    Standalone ensemble model that only uses standard libraries.
    Compatible with Numerai's environment.
    """
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float]):
        """
        Initialize with trained models and weights.
        
        Args:
            models: Dictionary of trained model objects
            weights: Dictionary of model weights for ensemble
        """
        self.models = models
        self.weights = weights
        self.feature_names = None
        
    def predict(self, X):
        """
        Generate ensemble predictions.
        
        Args:
            X: Feature matrix (numpy array or pandas DataFrame)
            
        Returns:
            numpy array of predictions
        """
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
            
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions = []
        total_weight = 0
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name in self.weights and self.weights[model_name] > 0:
                try:
                    pred = model.predict(X)
                    weight = self.weights[model_name]
                    
                    # Ensure predictions are 1D
                    if pred.ndim > 1:
                        pred = pred.flatten()
                        
                    predictions.append(pred * weight)
                    total_weight += weight
                    
                except Exception as e:
                    logger.warning(f"Error with model {model_name}: {e}")
                    continue
        
        if not predictions:
            raise ValueError("No valid model predictions available")
            
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
            
        return ensemble_pred
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names for validation."""
        self.feature_names = feature_names
        
    def get_model_info(self):
        """Get information about the ensemble."""
        info = {
            'model_count': len(self.models),
            'model_names': list(self.models.keys()),
            'weights': self.weights,
            'feature_count': len(self.feature_names) if self.feature_names else None
        }
        return info

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
        # Extract individual models
        for name, model in original_model.models.items():
            if hasattr(model, 'model'):
                # Extract the actual trained model (e.g., from our GPU wrappers)
                models[name] = model.model
            else:
                # Use the model directly
                models[name] = model
                
        # Extract weights
        weights = dict(original_model.weights)
        
        # Convert numpy types to Python types
        for k, v in weights.items():
            if hasattr(v, 'item'):
                weights[k] = v.item()
                
    else:
        # Single model case
        models['single_model'] = original_model
        weights['single_model'] = 1.0
        
    return models, weights

def validate_model_compatibility(models: Dict[str, Any]):
    """
    Validate that models use only standard libraries.
    
    Args:
        models: Dictionary of models to validate
        
    Returns:
        List of compatibility issues
    """
    issues = []
    
    for name, model in models.items():
        model_type = type(model).__name__
        module_name = type(model).__module__
        
        # Check if model uses standard ML libraries
        if any(lib in module_name for lib in ['xgboost', 'lightgbm', 'catboost', 'sklearn', 'torch']):
            logger.info(f"✅ {name} ({model_type}) is compatible")
        else:
            issues.append(f"❌ {name} ({model_type}) from {module_name} may not be compatible")
            
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
    models, weights = extract_base_models(original_model)
    
    logger.info(f"Extracted {len(models)} models:")
    for name, weight in weights.items():
        logger.info(f"  {name}: weight={weight:.4f}")
        
    # Validate compatibility
    issues = validate_model_compatibility(models)
    if issues:
        logger.warning("Compatibility issues found:")
        for issue in issues:
            logger.warning(f"  {issue}")
            
    # Create standalone ensemble
    standalone_model = StandaloneEnsembleModel(models, weights)
    
    # Test with dummy data if validation requested
    if validate_features:
        try:
            # Create test data (20 features like our dummy data)
            test_X = np.random.random((5, 20))
            test_pred = standalone_model.predict(test_X)
            logger.info(f"✅ Model validation successful - generated {len(test_pred)} predictions")
            logger.info(f"   Prediction range: [{test_pred.min():.4f}, {test_pred.max():.4f}]")
        except Exception as e:
            logger.error(f"❌ Model validation failed: {e}")
            raise
            
    # Save standalone model
    try:
        with open(output_path, 'wb') as f:
            pickle.dump(standalone_model, f)
        logger.info(f"✅ Standalone model saved to {output_path}")
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        logger.info(f"   File size: {file_size:.2f} MB")
        
    except Exception as e:
        logger.error(f"❌ Failed to save model: {e}")
        raise
        
    return standalone_model

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
    info = model.get_model_info()
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