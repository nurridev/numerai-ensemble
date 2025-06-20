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
