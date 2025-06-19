import os
import pandas as pd
import numpy as np
from numerapi import NumerAPI
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import logging
from typing import Tuple, Optional
import torch
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NumeraiDataManager:
    def __init__(self):
        self.napi = NumerAPI()
        self.scaler = StandardScaler()
        self.feature_selector = None
        
    def download_data(self, force_download: bool = False) -> None:
        """Download the latest Numerai dataset."""
        train_file = os.path.join(DATA_DIR, "train.parquet")
        validation_file = os.path.join(DATA_DIR, "validation.parquet")
        live_file = os.path.join(DATA_DIR, "live.parquet")
        
        if not force_download and all(os.path.exists(f) for f in [train_file, validation_file, live_file]):
            logger.info("Data files already exist. Use force_download=True to re-download.")
            return
            
        logger.info("Downloading Numerai dataset...")
        
        try:
            # Download the latest dataset
            self.napi.download_dataset(filename="train.parquet", dest_path=DATA_DIR)
            self.napi.download_dataset(filename="validation.parquet", dest_path=DATA_DIR)
            self.napi.download_dataset(filename="live.parquet", dest_path=DATA_DIR)
            
            logger.info("Dataset download completed successfully!")
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and return train, validation, and live datasets."""
        self.download_data()
        
        logger.info("Loading datasets...")
        train = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
        validation = pd.read_parquet(os.path.join(DATA_DIR, "validation.parquet"))
        live = pd.read_parquet(os.path.join(DATA_DIR, "live.parquet"))
        
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Validation shape: {validation.shape}")
        logger.info(f"Live shape: {live.shape}")
        
        return train, validation, live
    
    def preprocess_data(self, 
                       train: pd.DataFrame, 
                       validation: pd.DataFrame, 
                       live: pd.DataFrame,
                       feature_selection: bool = True,
                       n_features: int = 500) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Preprocess the data with scaling and feature selection."""
        
        logger.info("Starting data preprocessing...")
        
        # Get feature columns
        feature_cols = [c for c in train.columns if c.startswith("feature_")]
        target_col = "target"
        
        # Handle missing values
        for df in [train, validation, live]:
            df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # Feature selection on training data
        if feature_selection and len(feature_cols) > n_features:
            logger.info(f"Selecting top {n_features} features...")
            
            # Use training data for feature selection
            X_train = train[feature_cols].values
            y_train = train[target_col].values
            
            self.feature_selector = SelectKBest(f_regression, k=n_features)
            self.feature_selector.fit(X_train, y_train)
            
            # Get selected feature names
            selected_indices = self.feature_selector.get_support(indices=True)
            selected_features = [feature_cols[i] for i in selected_indices]
            
            logger.info(f"Selected {len(selected_features)} features")
            feature_cols = selected_features
        
        # Scale features
        logger.info("Scaling features...")
        train_features = train[feature_cols].values
        self.scaler.fit(train_features)
        
        # Apply scaling to all datasets
        for df in [train, validation, live]:
            df.loc[:, feature_cols] = self.scaler.transform(df[feature_cols].values)
        
        # Add era information if available
        if 'era' in train.columns:
            train['era_num'] = train['era'].str.extract(r'(\d+)').astype(int)
            validation['era_num'] = validation['era'].str.extract(r'(\d+)').astype(int)
            if 'era' in live.columns:
                live['era_num'] = live['era'].str.extract(r'(\d+)').astype(int)
        
        logger.info("Data preprocessing completed!")
        
        return train, validation, live
    
    def get_tensors(self, df: pd.DataFrame, target_col: str = "target") -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert dataframe to GPU tensors."""
        feature_cols = [c for c in df.columns if c.startswith("feature_")]
        
        X = torch.FloatTensor(df[feature_cols].values).to(DEVICE)
        
        if target_col in df.columns:
            y = torch.FloatTensor(df[target_col].values).to(DEVICE)
            return X, y
        else:
            return X, None
    
    def neutralize_predictions(self, predictions: np.ndarray, features: np.ndarray, 
                             proportion: float = 1.0) -> np.ndarray:
        """Neutralize predictions against features."""
        try:
            from sklearn.linear_model import LinearRegression
            
            # Fit linear regression
            reg = LinearRegression()
            reg.fit(features, predictions)
            
            # Calculate residuals (neutralized predictions)
            neutralized = predictions - proportion * reg.predict(features)
            
            return neutralized
            
        except Exception as e:
            logger.warning(f"Neutralization failed: {e}")
            return predictions
    
    def calculate_correlation(self, predictions: np.ndarray, targets: np.ndarray, 
                            groupby: Optional[pd.Series] = None) -> float:
        """Calculate correlation score."""
        if groupby is not None:
            # Era-wise correlation
            correlations = []
            for era in groupby.unique():
                era_mask = groupby == era
                if era_mask.sum() > 1:  # Need at least 2 samples
                    era_corr = np.corrcoef(predictions[era_mask], targets[era_mask])[0, 1]
                    if not np.isnan(era_corr):
                        correlations.append(era_corr)
            
            return np.mean(correlations) if correlations else 0.0
        else:
            # Overall correlation
            return np.corrcoef(predictions, targets)[0, 1]
    
    def save_processed_data(self, train: pd.DataFrame, validation: pd.DataFrame, live: pd.DataFrame):
        """Save processed data for quick loading."""
        train.to_parquet(os.path.join(DATA_DIR, "train_processed.parquet"))
        validation.to_parquet(os.path.join(DATA_DIR, "validation_processed.parquet"))
        live.to_parquet(os.path.join(DATA_DIR, "live_processed.parquet"))
        logger.info("Processed data saved!") 