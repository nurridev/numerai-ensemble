#!/usr/bin/env python3
"""
Create dummy Numerai data for testing the ensemble system.
"""

import pandas as pd
import numpy as np
import os
from config import DATA_DIR

def create_dummy_numerai_data():
    """Create dummy data that mimics Numerai structure."""
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create feature columns (similar to Numerai)
    n_features = 20  # Reduced for testing
    feature_cols = [f"feature_{i:03d}" for i in range(n_features)]
    
    # Training data
    n_train = 1000
    train_data = {
        'id': [f"train_{i}" for i in range(n_train)],
        'era': [f"era{i//50:03d}" for i in range(n_train)],  # ~50 rows per era
        'data_type': ['train'] * n_train,
        'target': np.random.uniform(0.0, 1.0, n_train)
    }
    
    # Add features
    for col in feature_cols:
        train_data[col] = np.random.uniform(0.0, 1.0, n_train)
    
    train_df = pd.DataFrame(train_data)
    
    # Validation data
    n_val = 200
    val_data = {
        'id': [f"val_{i}" for i in range(n_val)],
        'era': [f"era{(i//20)+100:03d}" for i in range(n_val)],  # Different eras
        'data_type': ['validation'] * n_val,
        'target': np.random.uniform(0.0, 1.0, n_val)
    }
    
    # Add features
    for col in feature_cols:
        val_data[col] = np.random.uniform(0.0, 1.0, n_val)
    
    val_df = pd.DataFrame(val_data)
    
    # Live data (no targets)
    n_live = 100
    live_data = {
        'id': [f"live_{i}" for i in range(n_live)],
        'era': [f"era{(i//20)+200:03d}" for i in range(n_live)],  # Future eras
        'data_type': ['live'] * n_live
    }
    
    # Add features
    for col in feature_cols:
        live_data[col] = np.random.uniform(0.0, 1.0, n_live)
    
    live_df = pd.DataFrame(live_data)
    
    # Save to parquet files
    train_df.to_parquet(os.path.join(DATA_DIR, "train.parquet"))
    val_df.to_parquet(os.path.join(DATA_DIR, "validation.parquet"))
    live_df.to_parquet(os.path.join(DATA_DIR, "live.parquet"))
    
    print(f"✅ Created dummy data:")
    print(f"   Train: {train_df.shape}")
    print(f"   Validation: {val_df.shape}")
    print(f"   Live: {live_df.shape}")
    print(f"   Features: {len(feature_cols)}")
    
    return train_df, val_df, live_df

if __name__ == "__main__":
    create_dummy_numerai_data() 