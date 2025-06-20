#!/usr/bin/env python3
"""
Advanced Numerai Data Manager with Comprehensive Feature Engineering
====================================================================

This module handles data loading, feature engineering, feature selection,
and preprocessing for the Numerai tournament with GPU acceleration.
"""

import os
import sys
import logging
import warnings
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np

# GPU acceleration (optional)
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("CuPy not available - falling back to CPU for array operations")

# Feature Engineering Libraries
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer,
    PolynomialFeatures, QuantileTransformer
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, chi2, mutual_info_regression,
    RFE, SelectPercentile, VarianceThreshold, SelectFromModel
)
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LassoCV, ElasticNetCV

# Advanced feature engineering
from category_encoders import TargetEncoder
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import networkx as nx

# GPU-accelerated libraries
try:
    import cuml
    from cuml.cluster import KMeans as cuKMeans
    from cuml.decomposition import PCA as cuPCA
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    logging.warning("CuML not available - falling back to CPU for some operations")

# SHAP for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - feature importance will be limited")

# UMAP for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logging.warning("UMAP not available - using alternative dimensionality reduction")

# Configuration
from config import *

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedNumeraiDataManager:
    """
    Advanced data manager with comprehensive feature engineering and selection.
    """
    
    def __init__(self):
        self.feature_cache = {}
        self.scalers = {}
        self.feature_names = None
        self.original_features = None
        self.engineered_features = None
        self.selected_features = None
        self.feature_importance = {}
        
        # Initialize GPU if available
        self.use_gpu = (CUPY_AVAILABLE and GPU_AVAILABLE and 
                       torch.cuda.is_available() and FORCE_GPU)
        if self.use_gpu and CUPY_AVAILABLE:
            cp.cuda.Device(0).use()
            logger.info("🚀 GPU acceleration enabled for feature engineering")
        else:
            logger.info("💻 Using CPU for feature engineering")
        
    def load_data(self, download_fresh: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load Numerai dataset with caching.
        
        Args:
            download_fresh: Whether to download fresh data
            
        Returns:
            Tuple of (train, validation, live) DataFrames
        """
        logger.info("🔄 Loading Numerai dataset...")
        
        # Check cache first
        cache_path = Path(CACHE_DIR) / "raw_data.pkl"
        if cache_path.exists() and not download_fresh:
            logger.info("📦 Loading from cache...")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        try:
            import numerapi
            napi = numerapi.NumerAPI()
            
            # Download dataset
            logger.info("⬇️ Downloading latest dataset...")
            napi.download_dataset("v5.0/train.parquet", f"{DATA_DIR}/train.parquet")
            napi.download_dataset("v5.0/validation.parquet", f"{DATA_DIR}/validation.parquet")
            napi.download_dataset("v5.0/live.parquet", f"{DATA_DIR}/live.parquet")
            
            # Load data
            train = pd.read_parquet(f"{DATA_DIR}/train.parquet")
            validation = pd.read_parquet(f"{DATA_DIR}/validation.parquet")
            live = pd.read_parquet(f"{DATA_DIR}/live.parquet")
            
            logger.info(f"✅ Data loaded: Train={train.shape}, Val={validation.shape}, Live={live.shape}")
            
            # Cache the data
            with open(cache_path, 'wb') as f:
                pickle.dump((train, validation, live), f)
                
            return train, validation, live
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            # Fallback to dummy data
            logger.info("🔄 Creating dummy data for testing...")
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create dummy data for testing."""
        np.random.seed(42)
        
        # Enhanced dummy data with more features
        n_features = 100
        feature_cols = [f"feature_{i:03d}" for i in range(n_features)]
        
        # Training data
        n_train = 50000
        train_data = {
            'id': [f"train_{i}" for i in range(n_train)],
            'era': [f"era{i//1000:03d}" for i in range(n_train)],
            'data_type': ['train'] * n_train,
            'target': np.random.uniform(0.0, 1.0, n_train)
        }
        
        # Create correlated features
        base_features = np.random.uniform(0.0, 1.0, (n_train, 20))
        correlated_features = base_features + np.random.normal(0, 0.1, (n_train, 20))
        
        for i, col in enumerate(feature_cols[:20]):
            train_data[col] = base_features[:, i % 20]
        for i, col in enumerate(feature_cols[20:40]):
            train_data[col] = correlated_features[:, i % 20]
        for i, col in enumerate(feature_cols[40:]):
            train_data[col] = np.random.uniform(0.0, 1.0, n_train)
            
        train_df = pd.DataFrame(train_data)
        
        # Validation data
        n_val = 10000
        val_data = {
            'id': [f"val_{i}" for i in range(n_val)],
            'era': [f"era{(i//200)+100:03d}" for i in range(n_val)],
            'data_type': ['validation'] * n_val,
            'target': np.random.uniform(0.0, 1.0, n_val)
        }
        
        for col in feature_cols:
            val_data[col] = np.random.uniform(0.0, 1.0, n_val)
            
        val_df = pd.DataFrame(val_data)
        
        # Live data
        n_live = 5000
        live_data = {
            'id': [f"live_{i}" for i in range(n_live)],
            'era': [f"era{(i//100)+200:03d}" for i in range(n_live)],
            'data_type': ['live'] * n_live
        }
        
        for col in feature_cols:
            live_data[col] = np.random.uniform(0.0, 1.0, n_live)
            
        live_df = pd.DataFrame(live_data)
        
        logger.info(f"✅ Created dummy data: Train={train_df.shape}, Val={val_df.shape}, Live={live_df.shape}")
        return train_df, val_df, live_df
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Comprehensive feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            target_col: Target column name for supervised feature engineering
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Starting comprehensive feature engineering...")
        
        if not FEATURE_ENGINEERING["enable"]:
            return df
            
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        self.original_features = feature_cols.copy()
        
        # Store original features
        result_df = df.copy()
        
        # Basic transformations
        if FEATURE_ENGINEERING["methods"]["log_transform"]:
            result_df = self._add_log_features(result_df, feature_cols)
            
        if FEATURE_ENGINEERING["methods"]["sqrt_transform"]:
            result_df = self._add_sqrt_features(result_df, feature_cols)
            
        if FEATURE_ENGINEERING["methods"]["rank_transform"]:
            result_df = self._add_rank_features(result_df, feature_cols)
            
        # Statistical features
        if FEATURE_ENGINEERING["methods"]["rolling_stats"]["enable"]:
            result_df = self._add_rolling_statistics(result_df, feature_cols)
            
        # Interaction features
        if FEATURE_ENGINEERING["methods"]["interactions"]["enable"]:
            result_df = self._add_interaction_features(result_df, feature_cols)
            
        # Polynomial features
        if FEATURE_ENGINEERING["methods"]["polynomial"]["enable"]:
            result_df = self._add_polynomial_features(result_df, feature_cols)
            
        # Clustering features
        if FEATURE_ENGINEERING["methods"]["clustering"]["enable"]:
            result_df = self._add_clustering_features(result_df, feature_cols)
            
        # Dimensionality reduction features
        if FEATURE_ENGINEERING["methods"]["dimensionality_reduction"]["enable"]:
            result_df = self._add_dimensionality_reduction_features(result_df, feature_cols)
            
        # Target encoding (if target available)
        if target_col and FEATURE_ENGINEERING["methods"]["target_encoding"]["enable"]:
            result_df = self._add_target_encoding_features(result_df, feature_cols, target_col)
            
        # Advanced statistical features
        if FEATURE_ENGINEERING["methods"]["advanced_stats"]["enable"]:
            result_df = self._add_advanced_statistical_features(result_df, feature_cols)
            
        # Cache engineered features
        new_feature_cols = [c for c in result_df.columns if c.startswith('feature_')]
        self.engineered_features = list(set(new_feature_cols) - set(feature_cols))
        
        logger.info(f"✅ Feature engineering complete: {len(self.original_features)} -> {len(new_feature_cols)} features")
        logger.info(f"   Added {len(self.engineered_features)} new features")
        
        return result_df
    
    def _add_log_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add log-transformed features."""
        logger.info("   📊 Adding log-transformed features...")
        
        for col in feature_cols[:50]:  # Limit to avoid explosion
            # Ensure positive values for log
            min_val = df[col].min()
            if min_val <= 0:
                shifted_col = df[col] - min_val + 1e-8
            else:
                shifted_col = df[col]
                
            df[f"{col}_log"] = np.log1p(shifted_col)
            
        return df
    
    def _add_sqrt_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add square root transformed features."""
        logger.info("   📊 Adding sqrt-transformed features...")
        
        for col in feature_cols[:50]:
            # Ensure non-negative values
            shifted_col = df[col] - df[col].min() + 1e-8
            df[f"{col}_sqrt"] = np.sqrt(shifted_col)
            
        return df
    
    def _add_rank_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add rank-transformed features."""
        logger.info("   📊 Adding rank-transformed features...")
        
        for col in feature_cols[:50]:
            df[f"{col}_rank"] = df[col].rank(pct=True)
            
        return df
    
    def _add_rolling_statistics(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add rolling statistical features."""
        logger.info("   📊 Adding rolling statistical features...")
        
        config = FEATURE_ENGINEERING["methods"]["rolling_stats"]
        
        # Sort by era for meaningful rolling statistics
        if 'era' in df.columns:
            df_sorted = df.sort_values('era').copy()
        else:
            df_sorted = df.copy()
            
        for window in config["windows"]:
            for stat in config["stats"]:
                for col in feature_cols[:20]:  # Limit features to avoid explosion
                    try:
                        if stat == "mean":
                            df_sorted[f"{col}_roll_{window}_{stat}"] = df_sorted[col].rolling(window).mean()
                        elif stat == "std":
                            df_sorted[f"{col}_roll_{window}_{stat}"] = df_sorted[col].rolling(window).std()
                        elif stat == "min":
                            df_sorted[f"{col}_roll_{window}_{stat}"] = df_sorted[col].rolling(window).min()
                        elif stat == "max":
                            df_sorted[f"{col}_roll_{window}_{stat}"] = df_sorted[col].rolling(window).max()
                        elif stat == "skew":
                            df_sorted[f"{col}_roll_{window}_{stat}"] = df_sorted[col].rolling(window).skew()
                        elif stat == "kurt":
                            df_sorted[f"{col}_roll_{window}_{stat}"] = df_sorted[col].rolling(window).kurt()
                    except:
                        continue
                        
        # Fill NaN values
        for col in df_sorted.columns:
            if col.startswith('feature_') and '_roll_' in col:
                df_sorted[col] = df_sorted[col].fillna(df_sorted[col].mean())
                
        return df_sorted
    
    def _add_interaction_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add feature interactions."""
        logger.info("   🔗 Adding interaction features...")
        
        config = FEATURE_ENGINEERING["methods"]["interactions"]
        top_k = min(config["top_k_features"], len(feature_cols))
        
        # Select top features for interactions (based on variance)
        variances = df[feature_cols].var().sort_values(ascending=False)
        top_features = variances.head(top_k).index.tolist()
        
        # Add pairwise interactions
        for i in range(len(top_features)):
            for j in range(i + 1, min(i + 10, len(top_features))):  # Limit combinations
                col1, col2 = top_features[i], top_features[j]
                
                if "multiply" in config["methods"]:
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                    
                if "add" in config["methods"]:
                    df[f"{col1}_add_{col2}"] = df[col1] + df[col2]
                    
                if "subtract" in config["methods"]:
                    df[f"{col1}_sub_{col2}"] = df[col1] - df[col2]
                    
                if "divide" in config["methods"]:
                    # Avoid division by zero
                    denominator = df[col2].replace(0, 1e-8)
                    df[f"{col1}_div_{col2}"] = df[col1] / denominator
                    
        return df
    
    def _add_polynomial_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add polynomial features."""
        logger.info("   📈 Adding polynomial features...")
        
        config = FEATURE_ENGINEERING["methods"]["polynomial"]
        
        # Select subset to avoid memory explosion
        selected_features = feature_cols[:30]
        X = df[selected_features].values
        
        # Create polynomial features
        poly = PolynomialFeatures(
            degree=config["degree"],
            include_bias=config["include_bias"],
            interaction_only=config["interaction_only"]
        )
        
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(selected_features)
        
        # Add polynomial features (excluding original features)
        original_count = len(selected_features)
        for i, name in enumerate(feature_names[original_count:], original_count):
            df[f"poly_{name}"] = X_poly[:, i]
            
        return df
    
    def _add_clustering_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add clustering-based features."""
        logger.info("   🎯 Adding clustering features...")
        
        config = FEATURE_ENGINEERING["methods"]["clustering"]
        X = df[feature_cols].values
        
        # Standardize for clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for method in config["methods"]:
            for n_clusters in config["n_clusters"]:
                try:
                    if method == "kmeans":
                        if self.use_gpu and config["use_gpu"] and CUPY_AVAILABLE:
                            # Use GPU-accelerated KMeans
                            clusterer = cuKMeans(n_clusters=n_clusters, random_state=42)
                            X_gpu = cp.asarray(X_scaled)
                            labels = clusterer.fit_predict(X_gpu).get()
                        else:
                            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                            labels = clusterer.fit_predict(X_scaled)
                            
                        df[f"cluster_kmeans_{n_clusters}"] = labels
                        
                        # Add distance to centroids
                        if hasattr(clusterer, 'cluster_centers_'):
                            centers = clusterer.cluster_centers_
                            if self.use_gpu and hasattr(centers, 'get'):
                                centers = centers.get()
                            distances = np.linalg.norm(X_scaled[:, np.newaxis] - centers, axis=2)
                            df[f"cluster_kmeans_{n_clusters}_dist"] = np.min(distances, axis=1)
                            
                    elif method == "gaussian_mixture":
                        clusterer = GaussianMixture(n_components=n_clusters, random_state=42)
                        labels = clusterer.fit_predict(X_scaled)
                        df[f"cluster_gmm_{n_clusters}"] = labels
                        
                        # Add probabilities
                        probabilities = clusterer.predict_proba(X_scaled)
                        df[f"cluster_gmm_{n_clusters}_prob"] = np.max(probabilities, axis=1)
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ Clustering failed for {method} with {n_clusters} clusters: {e}")
                    continue
                    
        return df
    
    def _add_dimensionality_reduction_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add dimensionality reduction features."""
        logger.info("   📉 Adding dimensionality reduction features...")
        
        config = FEATURE_ENGINEERING["methods"]["dimensionality_reduction"]
        X = df[feature_cols].values
        
        # Standardize for dimensionality reduction
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for method in config["methods"]:
            for n_components in config["n_components"]:
                try:
                    if n_components >= min(X_scaled.shape):
                        continue
                        
                    if method == "pca":
                        if self.use_gpu:
                            reducer = cuPCA(n_components=n_components, random_state=42)
                            X_gpu = cp.asarray(X_scaled)
                            X_reduced = reducer.fit_transform(X_gpu).get()
                        else:
                            reducer = PCA(n_components=n_components, random_state=42)
                            X_reduced = reducer.fit_transform(X_scaled)
                            
                    elif method == "ica":
                        reducer = FastICA(n_components=n_components, random_state=42, max_iter=200)
                        X_reduced = reducer.fit_transform(X_scaled)
                        
                    elif method == "tsne":
                        if n_components <= 3:  # t-SNE is expensive
                            from sklearn.manifold import TSNE
                            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
                            # Use subset for t-SNE
                            sample_size = min(5000, len(X_scaled))
                            indices = np.random.choice(len(X_scaled), sample_size, replace=False)
                            X_sample = X_scaled[indices]
                            X_reduced_sample = reducer.fit_transform(X_sample)
                            
                            # Interpolate for full dataset
                            from sklearn.neighbors import KNeighborsRegressor
                            knn = KNeighborsRegressor(n_neighbors=5)
                            knn.fit(X_sample, X_reduced_sample)
                            X_reduced = knn.predict(X_scaled)
                        else:
                            continue
                            
                    elif method == "umap" and UMAP_AVAILABLE:
                        reducer = umap.UMAP(n_components=n_components, random_state=42)
                        X_reduced = reducer.fit_transform(X_scaled)
                    else:
                        continue
                        
                    # Add reduced features
                    for i in range(n_components):
                        df[f"{method}_{n_components}_{i}"] = X_reduced[:, i]
                        
                except Exception as e:
                    logger.warning(f"   ⚠️ {method} failed with {n_components} components: {e}")
                    continue
                    
        return df
    
    def _add_target_encoding_features(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> pd.DataFrame:
        """Add target encoding features."""
        logger.info("   🎯 Adding target encoding features...")
        
        config = FEATURE_ENGINEERING["methods"]["target_encoding"]
        
        if 'era' not in df.columns:
            return df
            
        # Create categorical features from continuous ones
        categorical_features = []
        for col in feature_cols[:20]:  # Limit to avoid overfitting
            # Bin continuous features
            n_bins = min(10, df[col].nunique())
            if n_bins > 1:
                df[f"{col}_binned"] = pd.cut(df[col], bins=n_bins, labels=False, duplicates='drop')
                categorical_features.append(f"{col}_binned")
                
        # Era-aware target encoding
        for cat_col in categorical_features:
            df[f"{cat_col}_target_enc"] = 0.0
            
            for era in df['era'].unique():
                era_mask = df['era'] == era
                era_data = df[era_mask]
                
                if len(era_data) > 0:
                    # Calculate target mean for each category
                    target_means = era_data.groupby(cat_col)[target_col].mean()
                    global_mean = era_data[target_col].mean()
                    
                    # Apply smoothing
                    smoothed_means = {}
                    for cat in target_means.index:
                        count = (era_data[cat_col] == cat).sum()
                        weight = count / (count + config["smoothing"])
                        smoothed_mean = weight * target_means[cat] + (1 - weight) * global_mean
                        
                        # Add noise to prevent overfitting
                        if config["add_noise"]:
                            noise = np.random.normal(0, config["noise_level"])
                            smoothed_mean += noise
                            
                        smoothed_means[cat] = smoothed_mean
                    
                    # Apply encoding
                    df.loc[era_mask, f"{cat_col}_target_enc"] = df.loc[era_mask, cat_col].map(smoothed_means).fillna(global_mean)
                    
        return df
    
    def _add_advanced_statistical_features(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Add advanced statistical features."""
        logger.info("   📊 Adding advanced statistical features...")
        
        config = FEATURE_ENGINEERING["methods"]["advanced_stats"]
        X = df[feature_cols].values
        
        # Entropy features
        if config["entropy"]:
            for i, col in enumerate(feature_cols[:20]):
                try:
                    # Discretize for entropy calculation
                    hist, _ = np.histogram(df[col], bins=20)
                    hist = hist + 1e-8  # Avoid log(0)
                    prob = hist / hist.sum()
                    entropy = -np.sum(prob * np.log2(prob))
                    df[f"{col}_entropy"] = entropy
                except:
                    continue
                    
        # Percentile rank features
        if config["percentile_ranks"]:
            for percentile in config["percentile_ranks"]:
                for col in feature_cols[:30]:
                    df[f"{col}_pct_{percentile}"] = (df[col] <= df[col].quantile(percentile/100)).astype(int)
                    
        # Correlation with neighbors
        if config["correlation_with_neighbors"]:
            # Calculate pairwise correlations
            correlation_matrix = np.corrcoef(X.T)
            
            for i, col in enumerate(feature_cols[:50]):
                # Find most correlated features
                corr_values = correlation_matrix[i]
                corr_indices = np.argsort(np.abs(corr_values))[::-1][1:6]  # Top 5 excluding self
                
                # Average correlation with top neighbors
                df[f"{col}_neighbor_corr"] = np.mean([corr_values[idx] for idx in corr_indices])
                
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """
        Advanced feature selection pipeline.
        
        Args:
            df: DataFrame with features
            target_col: Target column for supervised selection
            
        Returns:
            DataFrame with selected features
        """
        logger.info("🎯 Starting advanced feature selection...")
        
        if not FEATURE_SELECTION["enable"]:
            return df
            
        feature_cols = [c for c in df.columns if c.startswith('feature_')]
        logger.info(f"   Starting with {len(feature_cols)} features")
        
        selected_features = {}
        
        # Apply each selection method
        if target_col:
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Correlation-based selection
            if FEATURE_SELECTION["methods"]["correlation"]["enable"]:
                selected_features["correlation"] = self._select_by_correlation(df, feature_cols, target_col)
                
            # Mutual information
            if FEATURE_SELECTION["methods"]["mutual_information"]["enable"]:
                selected_features["mutual_information"] = self._select_by_mutual_information(X, y, feature_cols)
                
            # RFE
            if FEATURE_SELECTION["methods"]["rfe"]["enable"]:
                selected_features["rfe"] = self._select_by_rfe(X, y, feature_cols)
                
            # Permutation importance
            if FEATURE_SELECTION["methods"]["permutation_importance"]["enable"]:
                selected_features["permutation"] = self._select_by_permutation_importance(X, y, feature_cols)
                
            # SHAP-based selection
            if FEATURE_SELECTION["methods"]["shap"]["enable"] and SHAP_AVAILABLE:
                selected_features["shap"] = self._select_by_shap(X, y, feature_cols)
                
            # Statistical tests
            if FEATURE_SELECTION["methods"]["statistical"]["enable"]:
                selected_features["statistical"] = self._select_by_statistical_tests(X, y, feature_cols)
                
            # Era stability
            if FEATURE_SELECTION["methods"]["era_stability"]["enable"] and 'era' in df.columns:
                selected_features["era_stability"] = self._select_by_era_stability(df, feature_cols, target_col)
        
        # Variance-based selection (unsupervised)
        if FEATURE_SELECTION["methods"]["variance"]["enable"]:
            selected_features["variance"] = self._select_by_variance(df, feature_cols)
            
        # Combine selections
        final_features = self._combine_feature_selections(selected_features, feature_cols)
        
        # Keep only selected features plus non-feature columns
        non_feature_cols = [c for c in df.columns if not c.startswith('feature_')]
        result_df = df[non_feature_cols + final_features].copy()
        
        self.selected_features = final_features
        logger.info(f"✅ Feature selection complete: {len(feature_cols)} -> {len(final_features)} features")
        
        return result_df
    
    def _select_by_correlation(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> List[str]:
        """Select features by correlation with target."""
        config = FEATURE_SELECTION["methods"]["correlation"]
        
        # Calculate correlations
        correlations = df[feature_cols].corrwith(df[target_col]).abs()
        
        # Remove low correlation features
        good_features = correlations[correlations >= config["threshold"]].index.tolist()
        
        # Remove multicollinear features
        if config["remove_multicollinear"]:
            feature_corr = df[good_features].corr().abs()
            upper_tri = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
            
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > config["multicollinear_threshold"])]
            good_features = [f for f in good_features if f not in to_drop]
            
        logger.info(f"   📊 Correlation selection: {len(good_features)} features")
        return good_features
    
    def _select_by_mutual_information(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> List[str]:
        """Select features by mutual information."""
        config = FEATURE_SELECTION["methods"]["mutual_information"]
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=config["random_state"])
        
        # Select top k features
        top_indices = np.argsort(mi_scores)[::-1][:config["k_best"]]
        selected_features = [feature_cols[i] for i in top_indices]
        
        logger.info(f"   🧠 Mutual information selection: {len(selected_features)} features")
        return selected_features
    
    def _select_by_rfe(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> List[str]:
        """Select features using Recursive Feature Elimination."""
        config = FEATURE_SELECTION["methods"]["rfe"]
        
        # Use RandomForest as estimator
        estimator = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        
        # RFE selection
        selector = RFE(
            estimator=estimator,
            n_features_to_select=config["n_features"],
            step=config["step"]
        )
        
        selector.fit(X, y)
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.support_[i]]
        
        logger.info(f"   🔄 RFE selection: {len(selected_features)} features")
        return selected_features
    
    def _select_by_permutation_importance(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> List[str]:
        """Select features by permutation importance."""
        config = FEATURE_SELECTION["methods"]["permutation_importance"]
        
        # Train a model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X, y)
        
        # Calculate permutation importance
        from sklearn.inspection import permutation_importance
        perm_importance = permutation_importance(
            model, X, y,
            n_repeats=config["n_repeats"],
            random_state=config["random_state"],
            n_jobs=-1
        )
        
        # Select top features
        top_indices = np.argsort(perm_importance.importances_mean)[::-1][:config["top_k"]]
        selected_features = [feature_cols[i] for i in top_indices]
        
        logger.info(f"   🎲 Permutation importance selection: {len(selected_features)} features")
        return selected_features
    
    def _select_by_shap(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> List[str]:
        """Select features using SHAP values."""
        config = FEATURE_SELECTION["methods"]["shap"]
        
        # Sample data for SHAP (expensive)
        sample_size = min(config["sample_size"], len(X))
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample, y_sample = X[indices], y[indices]
        
        # Train model
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_sample, y_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)
        
        # Select top features
        top_indices = np.argsort(feature_importance)[::-1][:config["top_k"]]
        selected_features = [feature_cols[i] for i in top_indices]
        
        logger.info(f"   🔍 SHAP selection: {len(selected_features)} features")
        return selected_features
    
    def _select_by_statistical_tests(self, X: np.ndarray, y: np.ndarray, feature_cols: List[str]) -> List[str]:
        """Select features using statistical tests."""
        config = FEATURE_SELECTION["methods"]["statistical"]
        selected_features = []
        
        for method in config["methods"]:
            try:
                if method == "f_regression":
                    selector = SelectKBest(f_regression, k=config["k_best"])
                elif method == "chi2":
                    # Make features non-negative for chi2
                    X_positive = X - X.min(axis=0) + 1e-8
                    selector = SelectKBest(chi2, k=config["k_best"])
                    X = X_positive
                else:
                    continue
                    
                selector.fit(X, y)
                method_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
                selected_features.extend(method_features)
                
            except Exception as e:
                logger.warning(f"   ⚠️ Statistical test {method} failed: {e}")
                continue
                
        # Remove duplicates
        selected_features = list(set(selected_features))
        
        logger.info(f"   📈 Statistical selection: {len(selected_features)} features")
        return selected_features
    
    def _select_by_era_stability(self, df: pd.DataFrame, feature_cols: List[str], target_col: str) -> List[str]:
        """Select features based on era stability."""
        config = FEATURE_SELECTION["methods"]["era_stability"]
        
        era_correlations = {}
        
        # Calculate correlation for each era
        for era in df['era'].unique():
            era_data = df[df['era'] == era]
            if len(era_data) > 10:  # Minimum samples for correlation
                correlations = era_data[feature_cols].corrwith(era_data[target_col])
                
                for feature in feature_cols:
                    if feature not in era_correlations:
                        era_correlations[feature] = []
                    era_correlations[feature].append(correlations[feature])
        
        # Select stable features
        stable_features = []
        for feature, corrs in era_correlations.items():
            corrs = np.array(corrs)
            corrs = corrs[~np.isnan(corrs)]  # Remove NaN
            
            if len(corrs) > 0:
                mean_corr = np.abs(corrs).mean()
                good_eras = (np.abs(corrs) >= config["min_era_correlation"]).sum()
                consistency = good_eras / len(corrs)
                
                if consistency >= config["consistency_threshold"]:
                    stable_features.append(feature)
        
        logger.info(f"   📅 Era stability selection: {len(stable_features)} features")
        return stable_features
    
    def _select_by_variance(self, df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
        """Select features by variance threshold."""
        config = FEATURE_SELECTION["methods"]["variance"]
        
        selector = VarianceThreshold(threshold=config["threshold"])
        X = df[feature_cols].values
        selector.fit(X)
        
        selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
        
        logger.info(f"   📊 Variance selection: {len(selected_features)} features")
        return selected_features
    
    def _combine_feature_selections(self, selected_features: Dict[str, List[str]], all_features: List[str]) -> List[str]:
        """Combine multiple feature selection results."""
        config = FEATURE_SELECTION["final_selection"]
        
        if not selected_features:
            return all_features[:config["max_features"]]
        
        if config["method"] == "intersection":
            # Find features selected by multiple methods
            feature_counts = {}
            for method, features in selected_features.items():
                for feature in features:
                    feature_counts[feature] = feature_counts.get(feature, 0) + 1
            
            # Select features chosen by at least min_methods
            final_features = [f for f, count in feature_counts.items() if count >= config["min_methods"]]
            
            # If too few features, fall back to backup method
            if len(final_features) < config["max_features"] // 2:
                backup_features = selected_features.get(config["backup_method"], all_features)
                final_features = backup_features[:config["max_features"]]
            
        elif config["method"] == "union":
            # Take union of all methods
            final_features = list(set().union(*selected_features.values()))
            
        else:
            # Default to backup method
            final_features = selected_features.get(config["backup_method"], all_features)
        
        # Limit to max features
        final_features = final_features[:config["max_features"]]
        
        logger.info(f"   🎯 Final selection: {len(final_features)} features from {len(selected_features)} methods")
        return final_features
    
    def preprocess_data(self, train: pd.DataFrame, validation: pd.DataFrame, live: pd.DataFrame,
                       feature_selection: bool = True, n_features: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            train, validation, live: Input DataFrames
            feature_selection: Whether to apply feature selection
            n_features: Number of features to select
            
        Returns:
            Tuple of preprocessed DataFrames
        """
        logger.info("🚀 Starting comprehensive data preprocessing...")
        
        # Feature engineering on training data
        train_processed = self.engineer_features(train, "target")
        
        # Apply same transformations to validation and live data
        validation_processed = self.engineer_features(validation, "target" if "target" in validation.columns else None)
        live_processed = self.engineer_features(live)
        
        # Feature selection
        if feature_selection:
            train_processed = self.select_features(train_processed, "target")
            
            # Apply same selection to validation and live
            feature_cols = [c for c in train_processed.columns if c.startswith('feature_')]
            non_feature_cols = [c for c in validation_processed.columns if not c.startswith('feature_')]
            
            validation_processed = validation_processed[non_feature_cols + feature_cols]
            live_processed = live_processed[[c for c in live_processed.columns if not c.startswith('feature_')] + 
                                          [c for c in feature_cols if c in live_processed.columns]]
        
        # Standardization
        if FEATURE_ENGINEERING["methods"]["standardize"]:
            train_processed, validation_processed, live_processed = self._standardize_features(
                train_processed, validation_processed, live_processed
            )
        
        logger.info("✅ Preprocessing pipeline completed!")
        return train_processed, validation_processed, live_processed
    
    def _standardize_features(self, train: pd.DataFrame, validation: pd.DataFrame, live: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Standardize feature columns."""
        feature_cols = [c for c in train.columns if c.startswith('feature_')]
        
        if self.use_gpu and CUPY_AVAILABLE:
            scaler = cuStandardScaler()
            train_features_gpu = cp.asarray(train[feature_cols].values)
            scaler.fit(train_features_gpu)
            
            train_scaled = scaler.transform(train_features_gpu).get()
            validation_scaled = scaler.transform(cp.asarray(validation[feature_cols].values)).get()
            live_scaled = scaler.transform(cp.asarray(live[feature_cols].values)).get()
        else:
            scaler = StandardScaler()
            train_scaled = scaler.fit_transform(train[feature_cols])
            validation_scaled = scaler.transform(validation[feature_cols])
            live_scaled = scaler.transform(live[feature_cols])
        
        # Update DataFrames
        train_result = train.copy()
        validation_result = validation.copy()
        live_result = live.copy()
        
        train_result[feature_cols] = train_scaled
        validation_result[feature_cols] = validation_scaled
        live_result[feature_cols] = live_scaled
        
        self.scalers['features'] = scaler
        return train_result, validation_result, live_result
    
    def neutralize_predictions(self, predictions: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Neutralize predictions against features."""
        from sklearn.linear_model import LinearRegression
        
        # Fit linear model to predict predictions from features
        model = LinearRegression()
        model.fit(features, predictions)
        
        # Get predictions from features
        feature_predictions = model.predict(features)
        
        # Neutralize by removing feature-based component
        neutralized = predictions - feature_predictions
        
        return neutralized

# For backward compatibility
NumeraiDataManager = AdvancedNumeraiDataManager 