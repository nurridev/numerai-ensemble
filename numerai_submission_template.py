#!/usr/bin/env python3
"""
Numerai Submission Script
========================

This script loads the exported model and generates predictions.
Compatible with Numerai's prediction environment.
"""

import pandas as pd
import numpy as np
import pickle
from typing import Dict, Any

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
                    continue
        
        if not predictions:
            raise ValueError("No valid model predictions available")
            
        # Weighted average
        ensemble_pred = np.sum(predictions, axis=0)
        
        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
            
        return ensemble_pred

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
