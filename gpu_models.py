import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
import logging
import gc
import psutil
from typing import Dict, Any, List, Tuple, Optional
from config import *
import scipy.stats as stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class H100SuperNeuralNet(nn.Module):
    """H100-optimized neural network with massive architecture for maximum GPU utilization."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [8192, 4096, 2048, 1024, 512, 256, 128, 64], 
                 dropout: float = 0.4, use_batch_norm: bool = True, activation: str = "gelu",
                 use_attention: bool = True, use_residual: bool = True):
        super(H100SuperNeuralNet, self).__init__()
        
        self.input_size = input_size
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Massive input processing layer
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Multi-head attention layer for feature interactions
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_sizes[0],
                num_heads=16,  # Many attention heads
                dropout=dropout * 0.3,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Main network layers with residual connections
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i, (in_size, out_size) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
            # Main linear layer
            self.layers.append(nn.Linear(in_size, out_size))
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(out_size))
            else:
                self.batch_norms.append(nn.Identity())
            
            # Adaptive dropout (higher dropout in later layers)
            dropout_rate = dropout * (0.5 + 0.5 * i / len(hidden_sizes))
            self.dropouts.append(nn.Dropout(dropout_rate))
        
        # Residual projection layers for skip connections
        if use_residual:
            self.residual_projections = nn.ModuleList()
            for i, (in_size, out_size) in enumerate(zip(hidden_sizes[:-1], hidden_sizes[1:])):
                if in_size != out_size:
                    self.residual_projections.append(nn.Linear(in_size, out_size))
                else:
                    self.residual_projections.append(nn.Identity())
        
        # Output head with multiple branches
        final_size = hidden_sizes[-1]
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(final_size, final_size // 2),
                nn.GELU(),
                nn.Dropout(dropout * 0.2),
                nn.Linear(final_size // 2, 1)
            ) for _ in range(3)  # 3 output heads for ensemble within network
        ])
        
        # Final combination layer
        self.output_combiner = nn.Linear(3, 1)
        
        # Initialize weights with different strategies for different layer sizes
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Advanced weight initialization optimized for H100."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.weight.size(0) > 2048:  # Very large layers
                    nn.init.xavier_normal_(m.weight, gain=0.5)
                elif m.weight.size(0) > 512:  # Large layers
                    nn.init.xavier_normal_(m.weight, gain=0.8)
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                nn.init.xavier_uniform_(m.in_proj_weight)
                nn.init.xavier_uniform_(m.out_proj.weight)
    
    def forward(self, x):
        # Input projection to large hidden size
        x = self.input_projection(x)
        
        # Attention mechanism for feature interactions
        if self.use_attention:
            # Reshape for attention (batch_size, seq_len=1, features)
            x_att = x.unsqueeze(1)
            attn_out, _ = self.attention(x_att, x_att, x_att)
            x = self.attention_norm(x + attn_out.squeeze(1))
        
        # Forward through main layers with residual connections
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            residual = x
            
            # Main forward pass
            x = layer(x)
            x = bn(x)
            x = F.gelu(x)
            x = dropout(x)
            
            # Residual connection
            if self.use_residual:
                residual_proj = self.residual_projections[i](residual)
                x = x + residual_proj
        
        # Multiple output heads for internal ensemble
        head_outputs = []
        for head in self.output_heads:
            head_outputs.append(head(x))
        
        # Combine outputs
        combined = torch.cat(head_outputs, dim=1)
        output = self.output_combiner(combined)
        
        return output.squeeze()

class H100OptimizedNeuralNetRegressor(BaseEstimator, RegressorMixin):
    """H100-optimized neural network with maximum GPU utilization and computational intensity."""
    
    def __init__(self, hidden_sizes=[8192, 4096, 2048, 1024, 512, 256, 128, 64], dropout=0.4, learning_rate=0.0005,
                 batch_size=32768, epochs=500, patience=50, weight_decay=1e-4,
                 use_batch_norm=True, activation="gelu", device=DEVICE, 
                 use_attention=True, use_transformer=True, n_transformer_layers=8, aux_loss_weight=0.1,
                 gradient_clip_norm=1.0, use_cosine_schedule=True, use_warmup=True):
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size if torch.cuda.is_available() else min(batch_size, 1024)
        self.epochs = epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.device = device
        self.use_attention = use_attention
        self.use_transformer = use_transformer
        self.n_transformer_layers = n_transformer_layers
        self.aux_loss_weight = aux_loss_weight
        self.gradient_clip_norm = gradient_clip_norm
        self.use_cosine_schedule = use_cosine_schedule
        self.use_warmup = use_warmup
        self.model = None
        self.best_score = float('-inf')
        
        # H100 optimizations
        self.use_amp = GPU_CONFIG.get("mixed_precision", True) and torch.cuda.is_available()
        self.compile_model = GPU_CONFIG.get("compile_models", False)
        
        if self.use_amp:
            self.scaler_amp = GradScaler()
            
    def fit(self, X, y, validation_data=None):
        """Fit the H100-optimized neural network with maximum computational intensity."""
        logger.info(f"🚀 Training H100 Super Neural Network on {self.device} with {len(X)} samples")
        logger.info(f"   Architecture: {self.hidden_sizes}")
        logger.info(f"   Batch size: {self.batch_size}")
        logger.info(f"   Total parameters will be: ~{self._estimate_parameters(X.shape[1])/1e6:.1f}M")
        
        # Data validation and preparation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have the same length: {len(X)} vs {len(y)}")
        
        # Adjust batch size if necessary
        effective_batch_size = min(self.batch_size, len(X))
        if effective_batch_size < self.batch_size:
            logger.warning(f"Reducing batch size from {self.batch_size} to {effective_batch_size} due to small dataset")
        
        # Convert to tensors with optimal dtypes for H100
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if isinstance(y, np.ndarray):
            y = torch.FloatTensor(y)
            
        # Handle NaN values
        if torch.isnan(X).any() or torch.isnan(y).any():
            logger.warning("NaN values detected in training data, replacing with zeros")
            X = torch.nan_to_num(X, nan=0.0)
            y = torch.nan_to_num(y, nan=0.0)
            
        # GPU optimization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            X = X.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
        # Initialize massive H100-optimized model
        self.model = H100SuperNeuralNet(
            X.shape[1], self.hidden_sizes, self.dropout, 
            self.use_batch_norm, self.activation, self.use_attention, True
        ).to(self.device)
        
        # Model compilation for H100
        if self.compile_model and hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model)
                logger.info("✅ Model compiled for H100 optimization")
            except:
                logger.warning("Model compilation failed, using eager mode")
        
        # Advanced optimizer with multiple strategies
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-6
        )
        
        # Learning rate scheduler with warmup and cosine decay
        steps_per_epoch = max(1, len(X) // effective_batch_size)
        total_steps = self.epochs * steps_per_epoch
        
        if self.use_cosine_schedule:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.learning_rate * 10,  # Higher peak LR
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                anneal_strategy='cos',
                div_factor=10.0,
                final_div_factor=100.0
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=self.learning_rate * 0.01
            )
        
        # Multiple loss functions for robustness
        mse_loss = nn.MSELoss()
        huber_loss = nn.HuberLoss(delta=1.0)
        
        # Create high-performance data loader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(
            dataset, 
            batch_size=effective_batch_size, 
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=min(8, psutil.cpu_count()),  # Use multiple workers
            persistent_workers=True,
            drop_last=False,
            prefetch_factor=4 if torch.cuda.is_available() else 2
        )
        
        # Validate dataloader
        if len(dataloader) == 0:
            raise ValueError("DataLoader is empty - check your data")
        
        logger.info(f"🎯 Starting intensive training with {len(dataloader)} batches per epoch")
        
        # Training loop with maximum computational intensity
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                if torch.cuda.is_available():
                    batch_X = batch_X.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                
                # Mixed precision training with multiple loss terms
                if self.use_amp:
                    with autocast():
                        outputs = self.model(batch_X)
                        
                        # Multiple loss components for robustness
                        mse = mse_loss(outputs, batch_y)
                        huber = huber_loss(outputs, batch_y)
                        loss = 0.7 * mse + 0.3 * huber
                        
                        # Auxiliary loss for regularization
                        if self.aux_loss_weight > 0:
                            # L2 regularization on large weights
                            l2_reg = torch.tensor(0., device=self.device)
                            for param in self.model.parameters():
                                if param.requires_grad and param.numel() > 1000:  # Only large weight matrices
                                    l2_reg += torch.norm(param, 2)
                            loss += self.aux_loss_weight * l2_reg
                    
                    self.scaler_amp.scale(loss).backward()
                    
                    # Gradient clipping for stability
                    if self.gradient_clip_norm > 0:
                        self.scaler_amp.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                else:
                    outputs = self.model(batch_X)
                    mse = mse_loss(outputs, batch_y)
                    huber = huber_loss(outputs, batch_y)
                    loss = 0.7 * mse + 0.3 * huber
                    
                    loss.backward()
                    
                    # Gradient clipping
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    
                    optimizer.step()
                
                scheduler.step()
                epoch_loss += loss.item()
                batch_count += 1
            
            # Calculate average loss safely
            avg_loss = epoch_loss / max(1, batch_count)
            
            # Validation with detailed metrics
            if validation_data is not None:
                X_val, y_val = validation_data
                val_loss = self._validate(X_val, y_val)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    self.best_score = -val_loss
                else:
                    patience_counter += 1
                    
                if epoch % 20 == 0:
                    current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else self.learning_rate
                    logger.info(f"Epoch {epoch}: Loss={avg_loss:.4f}, Val={val_loss:.4f}, LR={current_lr:.2e}")
            else:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
            # Memory cleanup every 50 epochs
            if epoch % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        logger.info("🎯 H100 Neural Network training completed!")
        return self
    
    def _validate(self, X_val, y_val):
        """Validate the model."""
        self.model.eval()
        
        if isinstance(X_val, np.ndarray):
            X_val = torch.FloatTensor(X_val)
        if isinstance(y_val, np.ndarray):
            y_val = torch.FloatTensor(y_val)
            
        # Check for NaN values
        if torch.isnan(X_val).any() or torch.isnan(y_val).any():
            logger.warning("NaN values detected in validation data, replacing with zeros")
            X_val = torch.nan_to_num(X_val, nan=0.0)
            y_val = torch.nan_to_num(y_val, nan=0.0)
            
        X_val = X_val.to(self.device)
        y_val = y_val.to(self.device)
            
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    predictions = self.model(X_val)
            else:
                predictions = self.model(X_val)
                    
            loss = F.mse_loss(predictions, y_val)
            
        return loss.item()
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
            
        # Check for NaN values
        if torch.isnan(X).any():
            logger.warning("NaN values detected in prediction data, replacing with zeros")
            X = torch.nan_to_num(X, nan=0.0)
            
        if torch.cuda.is_available():
            X = X.to(self.device, non_blocking=True)
            
        self.model.eval()
        predictions = []
        
        # Batch prediction
        batch_size = min(self.batch_size * 4, len(X))
        
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                
                if self.use_amp:
                    with autocast():
                        batch_pred = self.model(batch_X)
                else:
                    batch_pred = self.model(batch_X)
                        
                predictions.append(batch_pred.cpu().numpy())
                
        return np.concatenate(predictions)
    
    def _estimate_parameters(self, input_size):
        """Estimate the number of parameters in the model."""
        total = input_size * self.hidden_sizes[0]  # Input projection
        for i in range(len(self.hidden_sizes) - 1):
            total += self.hidden_sizes[i] * self.hidden_sizes[i + 1]
        total += self.hidden_sizes[-1] * 3 + 3  # Output heads
        return total

class H100XGBoostRegressor(BaseEstimator, RegressorMixin):
    """H100-enhanced XGBoost with maximum GPU optimization."""
    
    def __init__(self, **params):
        # H100-optimized parameters
        h100_params = MODEL_CONFIGS["xgboost"].copy()
        
        if torch.cuda.is_available() and FORCE_GPU:
            try:
                h100_params.update({
                    "tree_method": "gpu_hist",
                    "device": "cuda:0",
                    "max_bin": 1024,
                    "grow_policy": "depthwise",
                    "predictor": "gpu_predictor"
                })
                # Remove gpu_id to avoid conflict with device parameter
                if "gpu_id" in h100_params:
                    del h100_params["gpu_id"]
            except:
                # Fallback to CPU configuration
                h100_params.update({
                    "tree_method": "hist",
                    "device": "cpu",
                    "grow_policy": "depthwise"
                })
                # Remove GPU-specific parameters
                gpu_params = ["gpu_id", "predictor", "tree_method_params"]
                for param in gpu_params:
                    if param in h100_params:
                        del h100_params[param]
        else:
            # CPU configuration
            h100_params.update({
                "tree_method": "hist",
                "device": "cpu"
            })
            # Remove GPU-specific parameters
            gpu_params = ["gpu_id", "predictor", "tree_method_params"]
            for param in gpu_params:
                if param in h100_params:
                    del h100_params[param]
        
        h100_params.update(params)
        self.params = h100_params
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X, y, validation_data=None):
        """Fit XGBoost with H100 optimizations."""
        logger.info(f"🌳 Training H100-Optimized XGBoost with {self.params.get('n_estimators', 3000)} trees")
        
        # Data validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        # Try GPU first, fallback to CPU if needed
        try:
            dtrain = xgb.DMatrix(X, label=y)
            
            eval_list = [(dtrain, 'train')]
            if validation_data is not None:
                X_val, y_val = validation_data
                dval = xgb.DMatrix(X_val, label=y_val)
                eval_list.append((dval, 'validation'))
                
            # H100-optimized training
            self.model = xgb.train(
                self.params,
                dtrain,
                evals=eval_list,
                verbose_eval=False,
                early_stopping_rounds=100 if validation_data else None
            )
            
        except Exception as e:
            if "gpu" in str(e).lower() or "cuda" in str(e).lower() or "device" in str(e).lower():
                logger.warning(f"GPU training failed ({e}), falling back to CPU...")
                
                # Create CPU-only parameters
                cpu_params = self.params.copy()
                cpu_params.update({
                    "tree_method": "hist",
                    "device": "cpu"
                })
                # Remove GPU-specific parameters
                gpu_params = ["gpu_id", "predictor", "tree_method_params"]
                for param in gpu_params:
                    if param in cpu_params:
                        del cpu_params[param]
                
                # Retry with CPU
                dtrain = xgb.DMatrix(X, label=y)
                eval_list = [(dtrain, 'train')]
                if validation_data is not None:
                    X_val, y_val = validation_data
                    dval = xgb.DMatrix(X_val, label=y_val)
                    eval_list.append((dval, 'validation'))
                    
                self.model = xgb.train(
                    cpu_params,
                    dtrain,
                    evals=eval_list,
                    verbose_eval=False,
                    early_stopping_rounds=100 if validation_data else None
                )
                logger.info("✅ XGBoost training completed on CPU")
            else:
                raise e
        
        # Store feature importance
        self.feature_importance_ = self.model.get_score(importance_type='total_gain')
        
        return self
    
    def predict(self, X):
        """Make predictions with H100 optimization."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

class H100LightGBMRegressor(BaseEstimator, RegressorMixin):
    """H100-enhanced LightGBM with maximum GPU optimization."""
    
    def __init__(self, **params):
        h100_params = MODEL_CONFIGS["lightgbm"].copy()
        
        # Only enable GPU if available and explicitly requested
        if torch.cuda.is_available() and FORCE_GPU:
            try:
                h100_params.update({
                    "device": "gpu",
                    "gpu_platform_id": 0,
                    "gpu_device_id": 0,
                    "max_bin": 1023,
                    "num_gpu": 1
                })
            except:
                # Fallback to CPU if GPU setup fails
                h100_params["device"] = "cpu"
        else:
            h100_params["device"] = "cpu"
        
        h100_params.update(params)
        self.params = h100_params
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X, y, validation_data=None):
        """Fit LightGBM with H100 optimizations."""
        logger.info(f"🚀 Training H100-Optimized LightGBM with {self.params.get('n_estimators', 3000)} trees")
        
        # Data validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        fit_params = {
            'feature_name': [f"feature_{i}" for i in range(X.shape[1])],
            'categorical_feature': 'auto'
        }
        
        if validation_data is not None:
            X_val, y_val = validation_data
            fit_params.update({
                'eval_set': [(X_val, y_val)],
                'eval_names': ['validation'],
                'callbacks': [lgb.early_stopping(100), lgb.log_evaluation(0)]
            })
            
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y, **fit_params)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(
                [f"feature_{i}" for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

class H100CatBoostRegressor(BaseEstimator, RegressorMixin):
    """H100-enhanced CatBoost with maximum GPU optimization."""
    
    def __init__(self, **params):
        h100_params = MODEL_CONFIGS["catboost"].copy()
        
        # Conservative GPU setup for CatBoost
        if torch.cuda.is_available() and FORCE_GPU:
            try:
                h100_params.update({
                    "task_type": "GPU",
                    "devices": "0"
                })
                # Remove problematic parameters for GPU mode
                problematic_params = ["bootstrap_type", "bagging_temperature", "sampling_frequency"]
                for param in problematic_params:
                    if param in h100_params:
                        del h100_params[param]
            except:
                # Fallback to CPU
                h100_params["task_type"] = "CPU"
        else:
            h100_params["task_type"] = "CPU"
        
        h100_params.update(params)
        self.params = h100_params
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X, y, validation_data=None):
        """Fit CatBoost with H100 optimizations."""
        logger.info(f"🐱 Training H100-Optimized CatBoost with {self.params.get('iterations', 3000)} iterations")
        
        # Data validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        fit_params = {}
        if validation_data is not None:
            X_val, y_val = validation_data
            fit_params['eval_set'] = (X_val, y_val)
            fit_params['use_best_model'] = True
            fit_params['plot'] = False
            fit_params['verbose'] = False
            
        self.model = cb.CatBoostRegressor(**self.params)
        self.model.fit(X, y, **fit_params)
        
        # Store feature importance
        if hasattr(self.model, 'get_feature_importance'):
            importance_values = self.model.get_feature_importance()
            self.feature_importance_ = dict(zip(
                [f"feature_{i}" for i in range(len(importance_values))],
                importance_values
            ))
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

class H100ExtraTreesRegressor(BaseEstimator, RegressorMixin):
    """H100-optimized Extra Trees with maximum CPU utilization."""
    
    def __init__(self, **params):
        h100_params = MODEL_CONFIGS["extra_trees"].copy()
        h100_params.update(params)
        self.params = h100_params
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X, y, validation_data=None):
        """Fit Extra Trees with maximum CPU utilization."""
        logger.info(f"🌲 Training H100-Optimized Extra Trees with {self.params.get('n_estimators', 5000)} trees")
        
        # Data validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        self.model = ExtraTreesRegressor(**self.params)
        self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(
                [f"feature_{i}" for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

class H100RandomForestRegressor(BaseEstimator, RegressorMixin):
    """H100-optimized Random Forest with maximum CPU utilization."""
    
    def __init__(self, **params):
        h100_params = MODEL_CONFIGS["random_forest"].copy()
        h100_params.update(params)
        self.params = h100_params
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X, y, validation_data=None):
        """Fit Random Forest with maximum CPU utilization."""
        logger.info(f"🌳 Training H100-Optimized Random Forest with {self.params.get('n_estimators', 5000)} trees")
        
        # Data validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(
                [f"feature_{i}" for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

class H100GradientBoostingRegressor(BaseEstimator, RegressorMixin):
    """H100-optimized Gradient Boosting with maximum computational intensity."""
    
    def __init__(self, **params):
        h100_params = MODEL_CONFIGS["gradient_boosting"].copy()
        h100_params.update(params)
        self.params = h100_params
        self.model = None
        self.feature_importance_ = None
        
    def fit(self, X, y, validation_data=None):
        """Fit Gradient Boosting with maximum computational intensity."""
        logger.info(f"🚀 Training H100-Optimized Gradient Boosting with {self.params.get('n_estimators', 5000)} trees")
        
        # Data validation
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Empty training data provided")
        
        self.model = GradientBoostingRegressor(**self.params)
        self.model.fit(X, y)
        
        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = dict(zip(
                [f"feature_{i}" for i in range(X.shape[1])],
                self.model.feature_importances_
            ))
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)

class H100SuperEnsembleModel:
    """H100-optimized super ensemble with advanced strategies."""
    
    def __init__(self, models: Dict[str, BaseEstimator], ensemble_method: str = "advanced_stacking"):
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = None
        self.meta_models = []
        self.is_fitted = False
        self.feature_importance_ = {}
        self.model_scores = {}
        
    def fit(self, X, y, validation_data=None):
        """Fit H100-optimized super ensemble."""
        logger.info(f"🎯 Training H100 Super Ensemble with {len(self.models)} models")
        
        # Fit individual models
        for name, model in self.models.items():
            logger.info(f"🚀 Training H100-optimized {name}...")
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                model.fit(X, y, validation_data=validation_data)
                logger.info(f"✅ {name} training completed")
                
                # Store feature importance
                if hasattr(model, 'feature_importance_'):
                    self.feature_importance_[name] = model.feature_importance_
                    
            except Exception as e:
                logger.error(f"❌ Error training {name}: {e}")
                continue
                
        # Calculate weights
        if validation_data is not None:
            X_val, y_val = validation_data
            self._calculate_weights(X_val, y_val)
        else:
            self.weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            
        self.is_fitted = True
        return self
    
    def _calculate_weights(self, X_val, y_val):
        """Calculate weights based on validation performance."""
        logger.info("⚖️ Calculating H100-optimized ensemble weights...")
        
        predictions = {}
        correlations = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X_val)
                predictions[name] = pred
                
                corr = np.corrcoef(pred, y_val)[0, 1]
                correlations[name] = corr if not np.isnan(corr) else 0.0
                self.model_scores[name] = {'correlation': corr}
                
                logger.info(f"   {name}: Corr={corr:.4f}")
                
            except Exception as e:
                logger.warning(f"Error calculating correlation for {name}: {e}")
                correlations[name] = 0.0
                self.model_scores[name] = {'correlation': 0.0}
        
        # Calculate weights
        weights = {}
        for name in self.models.keys():
            corr_weight = max(0, correlations[name])
            weights[name] = corr_weight
            
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w/total_weight for name, w in weights.items()}
        else:
            weights = {name: 1.0/len(self.models) for name in self.models.keys()}
            
        self.weights = weights
        logger.info(f"📊 H100-optimized weights: {self.weights}")
    
    def predict(self, X):
        """Make H100-optimized ensemble predictions."""
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
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def get_feature_importance(self):
        """Get aggregated feature importance."""
        if not self.feature_importance_:
            return {}
            
        all_features = set()
        for importance_dict in self.feature_importance_.values():
            all_features.update(importance_dict.keys())
            
        aggregated_importance = {}
        for feature in all_features:
            importance_sum = 0
            weight_sum = 0
            
            for model_name, importance_dict in self.feature_importance_.items():
                if feature in importance_dict:
                    model_weight = self.weights.get(model_name, 1.0)
                    importance_sum += importance_dict[feature] * model_weight
                    weight_sum += model_weight
                    
            if weight_sum > 0:
                aggregated_importance[feature] = importance_sum / weight_sum
                
        return aggregated_importance

def create_h100_super_ensemble() -> H100SuperEnsembleModel:
    """Create H100-optimized super ensemble model with maximum computational intensity."""
    logger.info("🚀 Creating H100 Super Ensemble with Maximum Computational Intensity...")
    
    # Create all available models for maximum resource utilization
    models = {
        "h100_xgboost": H100XGBoostRegressor(),
        "h100_lightgbm": H100LightGBMRegressor(),
        "h100_catboost": H100CatBoostRegressor(),
        "h100_neural_net": H100OptimizedNeuralNetRegressor(**MODEL_CONFIGS["neural_net"]),
        "h100_extra_trees": H100ExtraTreesRegressor(),
        "h100_random_forest": H100RandomForestRegressor(),
        "h100_gradient_boosting": H100GradientBoostingRegressor()
    }
    
    logger.info(f"🔥 Created ensemble with {len(models)} models:")
    for name in models.keys():
        logger.info(f"   • {name}")
    
    ensemble = H100SuperEnsembleModel(models, ENSEMBLE_CONFIG["method"])
    
    # H100 memory optimization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction
        if hasattr(torch.cuda, 'set_memory_fraction'):
            try:
                torch.cuda.set_memory_fraction(GPU_CONFIG.get("max_memory_fraction", 0.98))
            except:
                pass
        logger.info("🔥 H100 Super Ensemble ready for maximum performance!")
        
    return ensemble

# Aliases for backward compatibility
GPUOptimizedNeuralNetRegressor = H100OptimizedNeuralNetRegressor
GPUNeuralNetRegressor = H100OptimizedNeuralNetRegressor  # For old pickles
GPUXGBoostRegressor = H100XGBoostRegressor
GPULightGBMRegressor = H100LightGBMRegressor  
GPUCatBoostRegressor = H100CatBoostRegressor
AdvancedEnsembleModel = H100SuperEnsembleModel
EnsembleModel = H100SuperEnsembleModel  # For backward compatibility with old pickles
create_gpu_ensemble = create_h100_super_ensemble
create_ensemble = create_h100_super_ensemble 