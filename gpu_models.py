import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
import logging
from typing import Dict, Any, List, Tuple
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NeuralNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int] = [512, 256, 128], 
                 dropout: float = 0.3):
        super(NeuralNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class GPUNeuralNetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_sizes=[512, 256, 128], dropout=0.3, learning_rate=0.001,
                 batch_size=1024, epochs=100, patience=10, device=DEVICE):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        
    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y).to(self.device)
            
        # Initialize model
        self.model = NeuralNet(X.shape[1], self.hidden_sizes, self.dropout).to(self.device)
        
        # Setup training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Create data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        return self
    
    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X).to(self.device)
            
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X)
            
        return predictions.cpu().numpy()

class GPUXGBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        # Set GPU parameters
        gpu_params = MODEL_CONFIGS["xgboost"].copy()
        gpu_params.update(params)
        self.params = gpu_params
        self.model = None
        
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.params, dtrain)
        return self
    
    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

class GPULightGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        # Set GPU parameters
        gpu_params = MODEL_CONFIGS["lightgbm"].copy()
        gpu_params.update(params)
        self.params = gpu_params
        self.model = None
        
    def fit(self, X, y):
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class GPUCatBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, **params):
        # Set GPU parameters
        gpu_params = MODEL_CONFIGS["catboost"].copy()
        gpu_params.update(params)
        self.params = gpu_params
        self.model = None
        
    def fit(self, X, y):
        self.model = cb.CatBoostRegressor(**self.params)
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)

class EnsembleModel:
    def __init__(self, models: Dict[str, BaseEstimator], ensemble_method: str = "weighted_average"):
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = None
        self.meta_model = None
        self.is_fitted = False
        
    def fit(self, X, y, validation_data=None):
        """Fit all models in the ensemble."""
        logger.info("Training ensemble models...")
        
        # Fit individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                model.fit(X, y)
                logger.info(f"{name} training completed")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                
        # Calculate ensemble weights
        if validation_data is not None:
            X_val, y_val = validation_data
            self._calculate_weights(X_val, y_val)
        else:
            # Equal weights if no validation data
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            
        self.is_fitted = True
        return self
    
    def _calculate_weights(self, X_val, y_val):
        """Calculate optimal weights based on validation performance."""
        logger.info("Calculating ensemble weights...")
        
        predictions = {}
        correlations = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_val)
                predictions[name] = pred
                corr = np.corrcoef(pred, y_val)[0, 1]
                correlations[name] = corr if not np.isnan(corr) else 0.0
                logger.info(f"{name} validation correlation: {corr:.4f}")
            except Exception as e:
                logger.warning(f"Error calculating correlation for {name}: {e}")
                correlations[name] = 0.0
        
        # Normalize correlations to get weights
        total_corr = sum(max(0, corr) for corr in correlations.values())
        if total_corr > 0:
            self.weights = {name: max(0, corr)/total_corr for name, corr in correlations.items()}
        else:
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            
        logger.info(f"Ensemble weights: {self.weights}")
    
    def predict(self, X):
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
            
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
                weights.append(self.weights[name])
            except Exception as e:
                logger.warning(f"Error getting predictions from {name}: {e}")
                
        if not predictions:
            raise ValueError("No models were able to make predictions")
            
        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize weights
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def cross_validate(self, X, y, cv_folds=5):
        """Perform cross-validation on the ensemble."""
        logger.info(f"Performing {cv_folds}-fold cross-validation...")
        
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            logger.info(f"Training fold {fold + 1}/{cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create fresh models for this fold
            fold_models = {}
            for name, model_class in [
                ("xgboost", GPUXGBoostRegressor),
                ("lightgbm", GPULightGBMRegressor),
                ("catboost", GPUCatBoostRegressor),
                ("neural_net", GPUNeuralNetRegressor)
            ]:
                fold_models[name] = model_class()
            
            # Train ensemble on fold
            ensemble = EnsembleModel(fold_models)
            ensemble.fit(X_train, y_train, validation_data=(X_val, y_val))
            
            # Get predictions and calculate score
            pred = ensemble.predict(X_val)
            score = np.corrcoef(pred, y_val)[0, 1]
            fold_scores.append(score)
            
            logger.info(f"Fold {fold + 1} score: {score:.4f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        logger.info(f"CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
        
        return fold_scores

def create_ensemble() -> EnsembleModel:
    """Create and return a new ensemble model."""
    models = {
        "xgboost": GPUXGBoostRegressor(),
        "lightgbm": GPULightGBMRegressor(),
        "catboost": GPUCatBoostRegressor(),
        "neural_net": GPUNeuralNetRegressor(**MODEL_CONFIGS["neural_net"])
    }
    
    return EnsembleModel(models, ENSEMBLE_CONFIG["method"]) 