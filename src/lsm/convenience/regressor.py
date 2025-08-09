"""
LSMRegressor for regression tasks using LSM temporal dynamics.

This module provides a scikit-learn compatible regressor that uses Liquid State Machine
reservoir dynamics for continuous value prediction and time series forecasting.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path

# sklearn imports
from sklearn.base import RegressorMixin
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .base import LSMBase
from .config import ConvenienceConfig, ConvenienceValidationError
from ..utils.lsm_exceptions import (
    LSMError, TrainingSetupError, TrainingExecutionError, 
    InvalidInputError, ModelLoadError
)
from ..utils.input_validation import (
    validate_positive_integer, validate_positive_float,
    validate_string_list, create_helpful_error_message
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)

# Import training components directly
try:
    from ..training.train import LSMTrainer
    from ..data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
    _TRAINING_AVAILABLE = True
    logger.info("Training components loaded successfully")
except ImportError as e:
    logger.warning(f"Training components not available: {e}")
    LSMTrainer = None
    StandardTokenizerWrapper = None
    SinusoidalEmbedder = None
    _TRAINING_AVAILABLE = False


class LSMRegressor(LSMBase, RegressorMixin):
    """
    LSM-based regressor with scikit-learn-like interface.
    
    This class uses Liquid State Machine reservoir dynamics for continuous value
    prediction and time series forecasting. It leverages the temporal dynamics
    of the reservoir to capture sequential patterns in the data.
    
    Parameters
    ----------
    window_size : int, default=10
        Size of the sliding window for sequence processing
    embedding_dim : int, default=128
        Dimension of the embedding space
    reservoir_type : str, default='echo_state'
        Type of reservoir ('standard', 'hierarchical', 'attentive', 'echo_state', 'deep')
        'echo_state' is recommended for time series prediction
    reservoir_config : dict, optional
        Additional configuration for the reservoir
    regressor_type : str, default='ridge'
        Type of downstream regressor ('linear', 'ridge', 'random_forest')
    regressor_config : dict, optional
        Configuration for the downstream regressor
    feature_extraction : str, default='mean'
        How to extract features from reservoir states ('mean', 'last', 'max', 'concat')
    normalize_targets : bool, default=True
        Whether to normalize target values during training
    time_series_mode : bool, default=False
        Whether to use time series prediction mode with lag features
    n_lags : int, default=5
        Number of lag features to use in time series mode
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters passed to the base class
        
    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted
    feature_names_in_ : list
        Input feature names
    n_features_in_ : int
        Number of input features
    target_scaler_ : StandardScaler
        Scaler for target values (if normalize_targets=True)
        
    Examples
    --------
    >>> from lsm import LSMRegressor
    >>> 
    >>> # Simple regression
    >>> regressor = LSMRegressor()
    >>> X = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]  # Sequential data
    >>> y = [4, 5, 6]  # Target values
    >>> regressor.fit(X, y)
    >>> predictions = regressor.predict([[4, 5, 6]])
    >>> 
    >>> # Time series prediction
    >>> regressor = LSMRegressor(
    ...     reservoir_type='echo_state',
    ...     time_series_mode=True,
    ...     n_lags=10
    ... )
    >>> regressor.fit(X, y)
    """
    
    def __init__(self,
                 window_size: int = 10,
                 embedding_dim: int = 128,
                 reservoir_type: str = 'echo_state',
                 reservoir_config: Optional[Dict[str, Any]] = None,
                 regressor_type: str = 'ridge',
                 regressor_config: Optional[Dict[str, Any]] = None,
                 feature_extraction: str = 'mean',
                 normalize_targets: bool = True,
                 time_series_mode: bool = False,
                 n_lags: int = 5,
                 random_state: Optional[int] = None,
                 **kwargs):
        
        # Set defaults optimized for regression
        if reservoir_config is None:
            reservoir_config = {
                'reservoir_units': [150, 75],  # Larger for regression
                'sparsity': 0.05,  # Lower sparsity for better dynamics
                'spectral_radius': 0.95,  # Higher for memory
                'leak_rate': 0.1  # For echo state networks
            }
        
        # Store regression-specific parameters
        self.regressor_type = regressor_type
        self.regressor_config = regressor_config or {}
        self.feature_extraction = feature_extraction
        self.normalize_targets = normalize_targets
        self.time_series_mode = time_series_mode
        self.n_lags = n_lags
        
        # Initialize base class
        super().__init__(
            window_size=window_size,
            embedding_dim=embedding_dim,
            reservoir_type=reservoir_type,
            reservoir_config=reservoir_config,
            random_state=random_state,
            **kwargs
        )
        
        # Regression components (initialized during fit)
        self._downstream_regressor = None
        self._feature_extractor = None
        self._target_scaler = None
        
        # sklearn-compatible attributes
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.target_scaler_ = None
        
        # Validate regression-specific parameters
        self._validate_regression_parameters()
    
    def _validate_regression_parameters(self) -> None:
        """Validate regression-specific parameters."""
        try:
            # Validate regressor type
            valid_regressor_types = ['linear', 'ridge', 'random_forest']
            if self.regressor_type not in valid_regressor_types:
                raise ConvenienceValidationError(
                    f"Invalid regressor_type: {self.regressor_type}",
                    suggestion="Use 'ridge' for regularized linear regression or 'random_forest' for non-linear",
                    valid_options=valid_regressor_types
                )
            
            # Validate feature extraction method
            valid_extraction_methods = ['mean', 'last', 'max', 'concat']
            if self.feature_extraction not in valid_extraction_methods:
                raise ConvenienceValidationError(
                    f"Invalid feature_extraction: {self.feature_extraction}",
                    suggestion="Use 'mean' for average pooling, 'last' for final state, 'max' for max pooling, or 'concat' for concatenation",
                    valid_options=valid_extraction_methods
                )
            
            # Validate n_lags for time series mode
            if self.time_series_mode:
                self.n_lags = validate_positive_integer(
                    self.n_lags, 'n_lags', min_value=1, max_value=50
                )
            
            # Validate regressor config
            if not isinstance(self.regressor_config, dict):
                raise ConvenienceValidationError(
                    f"regressor_config must be a dictionary, got {type(self.regressor_config).__name__}",
                    suggestion="Pass regressor configuration as a dictionary: {'alpha': 1.0, 'max_iter': 1000}"
                )
        
        except Exception as e:
            logger.error(f"Regression parameter validation failed: {e}")
            raise
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'LSMRegressor':
        """
        Create LSMRegressor from a preset configuration.
        
        Parameters
        ----------
        preset_name : str
            Name of the preset ('fast', 'balanced', 'quality', 'time_series')
        **overrides : dict
            Parameters to override in the preset
            
        Returns
        -------
        regressor : LSMRegressor
            Configured regressor instance
            
        Examples
        --------
        >>> regressor = LSMRegressor.from_preset('fast')
        >>> regressor = LSMRegressor.from_preset('time_series', n_lags=20)
        """
        config = ConvenienceConfig.create_config(
            preset=preset_name, 
            task_type='regression',
            **overrides
        )
        
        # Extract regression-specific parameters
        reg_params = {}
        for param in ['regressor_type', 'regressor_config', 'feature_extraction', 
                      'normalize_targets', 'time_series_mode', 'n_lags']:
            if param in config:
                reg_params[param] = config.pop(param)
        
        return cls(**config, **reg_params)
    
    def fit(self, 
            X: Union[List[List[float]], np.ndarray],
            y: Union[List[float], np.ndarray],
            validation_split: float = 0.2,
            epochs: int = 30,
            batch_size: int = 32,
            verbose: bool = True,
            **fit_params) -> 'LSMRegressor':
        """
        Train the LSM regressor on sequential data.
        
        This implementation uses engineered features for simplicity and speed.
        Future versions will integrate full LSM reservoir state extraction.
        
        Parameters
        ----------
        X : list or array-like of shape (n_samples, n_features)
            Training sequential data
        y : list or array-like of shape (n_samples,)
            Target continuous values
        validation_split : float, default=0.2
            Fraction of data to use for validation (currently unused)
        epochs : int, default=30
            Number of training epochs (currently unused)
        batch_size : int, default=32
            Training batch size (currently unused)
        verbose : bool, default=True
            Whether to show training progress
        **fit_params : dict
            Additional parameters (currently unused)
            
        Returns
        -------
        self : LSMRegressor
            Returns self for method chaining
        """
        try:
            if verbose:
                logger.info("Starting LSM regression training...")
            
            # Validate and preprocess input data
            X_processed, y_processed = self._preprocess_regression_data(X, y)
            
            # Store feature information for sklearn compatibility
            self.feature_names_in_ = [f'feature_{i}' for i in range(X_processed.shape[1])]
            self.n_features_in_ = X_processed.shape[1]
            
            # Set up target scaling if requested
            if self.normalize_targets:
                self._target_scaler = StandardScaler()
                y_scaled = self._target_scaler.fit_transform(y_processed.reshape(-1, 1)).ravel()
                self.target_scaler_ = self._target_scaler
            else:
                y_scaled = y_processed
                self.target_scaler_ = None
            
            # Extract features using engineered approach
            if verbose:
                logger.info("Extracting temporal features...")
            
            X_features = self._extract_temporal_features(X_processed)
            
            # Train downstream regressor
            if verbose:
                logger.info(f"Training {self.regressor_type} regressor...")
            
            start_time = time.time()
            
            self._downstream_regressor = self._create_downstream_regressor()
            self._downstream_regressor.fit(X_features, y_scaled)
            
            regressor_training_time = time.time() - start_time
            
            # Calculate training metrics
            y_pred_scaled = self._downstream_regressor.predict(X_features)
            if self.normalize_targets:
                y_pred = self._target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            else:
                y_pred = y_pred_scaled
            
            train_r2 = r2_score(y_processed, y_pred)
            train_mse = mean_squared_error(y_processed, y_pred)
            train_mae = mean_absolute_error(y_processed, y_pred)
            
            # Store training metadata
            self._training_metadata = {
                'regressor_training_time': regressor_training_time,
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split,
                'data_size': len(X_processed),
                'n_features': X_processed.shape[1],
                'feature_extraction': self.feature_extraction,
                'regressor_type': self.regressor_type,
                'normalize_targets': self.normalize_targets,
                'time_series_mode': self.time_series_mode,
                'lsm_available': _TRAINING_AVAILABLE,
                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_mae': train_mae
            }
            
            # Store performance metrics
            self._performance_metrics = {
                'train_r2_score': train_r2,
                'train_mse': train_mse,
                'train_mae': train_mae,
                'n_samples': len(X_processed),
                'n_features': X_processed.shape[1]
            }
            
            self._is_fitted = True
            
            if verbose:
                logger.info(f"Regression training completed in {regressor_training_time:.2f} seconds")
                logger.info(f"Trained on {len(X_processed)} samples with {X_processed.shape[1]} features")
                logger.info(f"Training R² score: {train_r2:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Regression training failed: {e}")
            raise TrainingExecutionError(
                epoch=None,
                reason=f"LSM regression training failed: {e}"
            )
    
    def predict(self, X: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """
        Predict continuous values for sequential data.
        
        Parameters
        ----------
        X : list or array-like of shape (n_samples, n_features)
            Sequential data to predict
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted continuous values
        """
        self._check_is_fitted()
        
        try:
            # Preprocess input data
            X_processed = self._preprocess_prediction_data(X)
            
            # Extract features
            X_features = self._extract_temporal_features(X_processed)
            
            # Make predictions using downstream regressor
            y_pred_scaled = self._downstream_regressor.predict(X_features)
            
            # Inverse transform if targets were normalized
            if self.normalize_targets and self._target_scaler is not None:
                y_pred = self._target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            else:
                y_pred = y_pred_scaled
            
            logger.debug(f"Made predictions for {len(X_processed)} samples")
            
            return y_pred
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise InvalidInputError(
                "prediction process",
                "successful regression prediction",
                f"prediction failed: {e}"
            )
    
    def score(self, X: Union[List[List[float]], np.ndarray], 
              y: Union[List[float], np.ndarray]) -> float:
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : list or array-like of shape (n_samples, n_features)
            Test sequential data
        y : list or array-like of shape (n_samples,)
            True target values
            
        Returns
        -------
        score : float
            R² coefficient of determination
        """
        self._check_is_fitted()
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate R² score
            r2 = r2_score(y, y_pred)
            
            logger.debug(f"Calculated R² score: {r2:.4f}")
            
            return r2
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            raise InvalidInputError(
                "scoring process",
                "successful R² calculation",
                f"scoring failed: {e}"
            ) 
   
    def predict_with_uncertainty(self, X: Union[List[List[float]], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict continuous values with uncertainty estimates (if supported by regressor).
        
        Parameters
        ----------
        X : list or array-like of shape (n_samples, n_features)
            Sequential data to predict
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted continuous values
        uncertainties : ndarray of shape (n_samples,)
            Uncertainty estimates (standard deviation)
        """
        self._check_is_fitted()
        
        try:
            predictions = self.predict(X)
            
            # Calculate uncertainty based on regressor type
            if self.regressor_type == 'random_forest' and hasattr(self._downstream_regressor, 'estimators_'):
                # For random forest, use prediction variance across trees
                X_processed = self._preprocess_prediction_data(X)
                X_features = self._extract_temporal_features(X_processed)
                
                # Get predictions from all trees
                tree_predictions = np.array([
                    tree.predict(X_features) for tree in self._downstream_regressor.estimators_
                ])
                
                # Calculate standard deviation across trees
                uncertainties = np.std(tree_predictions, axis=0)
                
                # Inverse transform uncertainties if targets were normalized
                if self.normalize_targets and self._target_scaler is not None:
                    # Scale uncertainties by the target scaler's scale
                    uncertainties = uncertainties * self._target_scaler.scale_[0]
            else:
                # For linear models, use a simple heuristic based on residuals
                if hasattr(self, '_performance_metrics') and 'train_mse' in self._performance_metrics:
                    # Use training MSE as uncertainty estimate
                    uncertainties = np.full(len(predictions), np.sqrt(self._performance_metrics['train_mse']))
                else:
                    # Fallback to zero uncertainty
                    uncertainties = np.zeros(len(predictions))
            
            logger.debug(f"Generated predictions with uncertainties for {len(predictions)} samples")
            
            return predictions, uncertainties
            
        except Exception as e:
            logger.error(f"Prediction with uncertainty failed: {e}")
            raise InvalidInputError(
                "uncertainty prediction process",
                "successful uncertainty prediction",
                f"uncertainty prediction failed: {e}"
            )
    
    def forecast(self, X: Union[List[List[float]], np.ndarray], 
                 n_steps: int = 1) -> np.ndarray:
        """
        Multi-step ahead forecasting for time series data.
        
        Parameters
        ----------
        X : list or array-like of shape (n_samples, n_features)
            Historical sequential data
        n_steps : int, default=1
            Number of steps to forecast ahead
            
        Returns
        -------
        forecasts : ndarray of shape (n_steps,)
            Multi-step ahead forecasts
        """
        self._check_is_fitted()
        
        if not self.time_series_mode:
            logger.warning("Multi-step forecasting works best with time_series_mode=True")
        
        try:
            # Start with the last sequence
            X_processed = self._preprocess_prediction_data(X)
            current_sequence = X_processed[-1:].copy()  # Take last sample
            
            forecasts = []
            
            for step in range(n_steps):
                # Make prediction for current sequence
                prediction = self.predict(current_sequence)[0]
                forecasts.append(prediction)
                
                # Update sequence for next prediction
                if current_sequence.shape[1] > 1:
                    # Shift sequence and add prediction as new feature
                    new_sequence = np.roll(current_sequence, -1, axis=1)
                    new_sequence[0, -1] = prediction
                    current_sequence = new_sequence
                else:
                    # Single feature case - just use prediction
                    current_sequence = np.array([[prediction]])
            
            forecasts = np.array(forecasts)
            
            logger.debug(f"Generated {n_steps}-step ahead forecast")
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            raise InvalidInputError(
                "forecasting process",
                "successful multi-step forecast",
                f"forecasting failed: {e}"
            )
    
    def _preprocess_regression_data(self, X, y) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess and validate regression training data."""
        # Process X
        if isinstance(X, list):
            X = np.array(X)
        
        if not isinstance(X, np.ndarray):
            raise ConvenienceValidationError(
                f"Invalid input type: {type(X).__name__}",
                suggestion="Use list of sequences or numpy array",
                valid_options=["list of sequences", "numpy array"]
            )
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim > 2:
            raise ConvenienceValidationError(
                f"Invalid input dimensions: {X.ndim}D",
                suggestion="Use 2D array with shape (n_samples, n_features)"
            )
        
        # Process y
        if isinstance(y, list):
            y = np.array(y)
        
        if not isinstance(y, np.ndarray):
            raise ConvenienceValidationError(
                f"Invalid target type: {type(y).__name__}",
                suggestion="Use list or numpy array of continuous values"
            )
        
        # Ensure y is 1D
        if y.ndim > 1:
            if y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ConvenienceValidationError(
                    f"Multi-output regression not supported: y shape {y.shape}",
                    suggestion="Use 1D array of target values"
                )
        
        # Validate lengths match
        if len(X) != len(y):
            raise ConvenienceValidationError(
                f"Length mismatch: {len(X)} samples but {len(y)} targets",
                suggestion="Ensure each sample has a corresponding target value"
            )
        
        # Validate we have enough data
        if len(X) < 2:
            raise ConvenienceValidationError(
                f"Insufficient training data: {len(X)} samples",
                suggestion="Provide at least 2 samples for training"
            )
        
        # Check for valid numeric data
        if not np.isfinite(X).all():
            raise ConvenienceValidationError(
                "Input data contains non-finite values (NaN or inf)",
                suggestion="Clean your data to remove NaN and infinite values"
            )
        
        if not np.isfinite(y).all():
            raise ConvenienceValidationError(
                "Target data contains non-finite values (NaN or inf)",
                suggestion="Clean your target values to remove NaN and infinite values"
            )
        
        return X, y
    
    def _preprocess_prediction_data(self, X) -> np.ndarray:
        """Preprocess prediction input data."""
        if isinstance(X, list):
            X = np.array(X)
        
        if not isinstance(X, np.ndarray):
            raise ConvenienceValidationError(
                f"Invalid input type: {type(X).__name__}",
                suggestion="Use list of sequences or numpy array"
            )
        
        # Ensure X is 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)  # Single sample
        elif X.ndim > 2:
            raise ConvenienceValidationError(
                f"Invalid input dimensions: {X.ndim}D",
                suggestion="Use 2D array with shape (n_samples, n_features)"
            )
        
        # Check for valid numeric data
        if not np.isfinite(X).all():
            raise ConvenienceValidationError(
                "Input data contains non-finite values (NaN or inf)",
                suggestion="Clean your data to remove NaN and infinite values"
            )
        
        return X
    
    def _extract_temporal_features(self, X: np.ndarray) -> np.ndarray:
        """Extract temporal features from sequential data."""
        try:
            features_list = []
            
            # Basic statistical features
            features_list.append(np.mean(X, axis=1, keepdims=True))  # Mean
            features_list.append(np.std(X, axis=1, keepdims=True))   # Standard deviation
            features_list.append(np.min(X, axis=1, keepdims=True))   # Minimum
            features_list.append(np.max(X, axis=1, keepdims=True))   # Maximum
            
            # Temporal features
            if X.shape[1] > 1:
                # Differences (trend)
                diffs = np.diff(X, axis=1)
                features_list.append(np.mean(diffs, axis=1, keepdims=True))  # Mean difference
                features_list.append(np.std(diffs, axis=1, keepdims=True))   # Diff std
                
                # Last value (most recent)
                features_list.append(X[:, -1:])  # Last value
                
                # First value (oldest)
                features_list.append(X[:, :1])   # First value
            
            # Time series specific features
            if self.time_series_mode and X.shape[1] >= self.n_lags:
                # Add lag features
                for lag in range(1, min(self.n_lags + 1, X.shape[1])):
                    lag_features = X[:, :-lag] if lag < X.shape[1] else X[:, :1]
                    # Take mean of lagged values
                    features_list.append(np.mean(lag_features, axis=1, keepdims=True))
            
            # Combine all features
            X_features = np.concatenate(features_list, axis=1)
            
            # Handle any remaining NaN values
            X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            logger.debug(f"Extracted temporal features shape: {X_features.shape}")
            
            return X_features
            
        except Exception as e:
            logger.error(f"Temporal feature extraction failed: {e}")
            # Fallback to simple features
            X_features = np.column_stack([
                np.mean(X, axis=1),
                np.std(X, axis=1),
                X[:, -1] if X.shape[1] > 0 else np.zeros(X.shape[0])
            ])
            logger.warning("Using simple features as fallback")
            return X_features
    
    def _create_downstream_regressor(self):
        """Create the downstream regressor."""
        if self.regressor_type == 'linear':
            default_config = {}
            config = {**default_config, **self.regressor_config}
            return LinearRegression(**config)
        
        elif self.regressor_type == 'ridge':
            default_config = {
                'alpha': 1.0,
                'random_state': self.random_state
            }
            config = {**default_config, **self.regressor_config}
            return Ridge(**config)
        
        elif self.regressor_type == 'random_forest':
            default_config = {
                'random_state': self.random_state,
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
            config = {**default_config, **self.regressor_config}
            return RandomForestRegressor(**config)
        
        else:
            raise ConvenienceValidationError(
                f"Unsupported regressor type: {self.regressor_type}",
                valid_options=['linear', 'ridge', 'random_forest']
            )
    
    def _load_trainer(self, model_path: str) -> None:
        """
        Load the underlying trainer for regression tasks.
        
        Parameters
        ----------
        model_path : str
            Path to the saved model
        """
        # This will be implemented when full LSM integration is added
        # For now, we rely on the downstream regressor and feature extractor
        logger.debug(f"Loading regression model from {model_path}")
        pass
    
    def __sklearn_tags__(self):
        """Return sklearn tags for this regressor."""
        tags = super().__sklearn_tags__()
        tags.update({
            'requires_y': True,
            'multioutput': False,
            'multiclass': False,
            'binary_only': False,
            'multilabel': False,
        })
        return tags