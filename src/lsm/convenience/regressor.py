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

# Lazy import flag - components will be imported when needed
_TRAINING_AVAILABLE = None
LSMTrainer = None
StandardTokenizerWrapper = None
SinusoidalEmbedder = None
EnhancedTokenizerWrapper = None

def _check_training_components():
    """Lazy import of training components to avoid circular imports."""
    global _TRAINING_AVAILABLE, LSMTrainer, StandardTokenizerWrapper, SinusoidalEmbedder, EnhancedTokenizerWrapper
    
    if _TRAINING_AVAILABLE is not None:
        return _TRAINING_AVAILABLE
    
    try:
        from ..training.train import LSMTrainer as _LSMTrainer
        from ..data.tokenization import StandardTokenizerWrapper as _StandardTokenizerWrapper
        from ..data.tokenization import SinusoidalEmbedder as _SinusoidalEmbedder
        from ..data.enhanced_tokenization import EnhancedTokenizerWrapper as _EnhancedTokenizerWrapper
        
        # Assign to module-level variables
        LSMTrainer = _LSMTrainer
        StandardTokenizerWrapper = _StandardTokenizerWrapper
        SinusoidalEmbedder = _SinusoidalEmbedder
        EnhancedTokenizerWrapper = _EnhancedTokenizerWrapper
        
        _TRAINING_AVAILABLE = True
        logger.info("Training components loaded successfully")
        
    except ImportError as e:
        logger.warning(f"Training components not available: {e}")
        _TRAINING_AVAILABLE = False
    
    return _TRAINING_AVAILABLE


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
    tokenizer : str, default='gpt2'
        Name of the tokenizer to use ('gpt2', 'bert-base-uncased', etc.)
        Can be any supported tokenizer backend (HuggingFace, OpenAI, spaCy, custom)
    max_length : int, default=512
        Maximum sequence length for tokenization
    embedding_type : str, default='standard'
        Type of embedding to use ('standard', 'sinusoidal', 'configurable_sinusoidal')
    sinusoidal_config : dict, optional
        Configuration for sinusoidal embeddings when embedding_type is 'sinusoidal' or 'configurable_sinusoidal'
    streaming : bool, default=False
        Whether to enable streaming data processing for large datasets
    streaming_config : dict, optional
        Configuration for streaming data processing
    tokenizer_backend_config : dict, optional
        Backend-specific configuration for the tokenizer
    enable_caching : bool, default=True
        Whether to enable intelligent caching for tokenization
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
    >>> # With enhanced sinusoidal embeddings
    >>> regressor = LSMRegressor(
    ...     tokenizer='gpt2',
    ...     embedding_type='configurable_sinusoidal',
    ...     sinusoidal_config={'learnable_frequencies': True, 'base_frequency': 10000.0}
    ... )
    >>> regressor.fit(X, y)
    >>> predictions = regressor.predict([[4, 5, 6]])
    >>> 
    >>> # With streaming for large datasets
    >>> regressor = LSMRegressor(
    ...     streaming=True,
    ...     streaming_config={'batch_size': 1000, 'memory_threshold_mb': 1000.0}
    ... )
    >>> regressor.fit_streaming("path/to/large/dataset.txt", y)
    >>> 
    >>> # With different tokenizer backends
    >>> regressor = LSMRegressor(
    ...     tokenizer='bert-base-uncased',  # HuggingFace tokenizer
    ...     embedding_type='sinusoidal',
    ...     enable_caching=True
    ... )
    >>> regressor.fit(X, y)
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
                 tokenizer: str = 'gpt2',
                 max_length: int = 512,
                 embedding_type: str = 'standard',
                 sinusoidal_config: Optional[Dict[str, Any]] = None,
                 streaming: bool = False,
                 streaming_config: Optional[Dict[str, Any]] = None,
                 tokenizer_backend_config: Optional[Dict[str, Any]] = None,
                 enable_caching: bool = True,
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
        
        # Check if training components are available (lazy import)
        if not _check_training_components():
            raise ImportError(
                "LSM training components are not available. "
                "Please ensure TensorFlow and all dependencies are installed."
            )
        
        # Store regression-specific parameters
        self.regressor_type = regressor_type
        self.regressor_config = regressor_config or {}
        self.feature_extraction = feature_extraction
        self.normalize_targets = normalize_targets
        self.time_series_mode = time_series_mode
        self.n_lags = n_lags
        
        # Enhanced tokenizer parameters
        self.tokenizer_name = tokenizer
        self.max_length = max_length
        self.embedding_type = embedding_type
        self.sinusoidal_config = sinusoidal_config or {}
        self.streaming = streaming
        self.streaming_config = streaming_config or {}
        self.tokenizer_backend_config = tokenizer_backend_config or {}
        self.enable_caching = enable_caching
        
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
        self._enhanced_tokenizer = None
        
        # Store enhanced tokenizer configuration
        self._enhanced_tokenizer_config = {
            'embedding_type': self.embedding_type,
            'sinusoidal_config': self.sinusoidal_config.copy(),
            'streaming': self.streaming,
            'streaming_config': self.streaming_config.copy(),
            'tokenizer_backend_config': self.tokenizer_backend_config.copy(),
            'enable_caching': self.enable_caching,
            'max_length': self.max_length
        }
        
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
            
            # Validate enhanced tokenizer parameters
            self.max_length = validate_positive_integer(
                self.max_length, 'max_length', min_value=1, max_value=2048
            )
            
            # Validate tokenizer name
            if not isinstance(self.tokenizer_name, str):
                raise ConvenienceValidationError(
                    f"tokenizer must be a string, got {type(self.tokenizer_name).__name__}",
                    suggestion="Use a tokenizer name like 'gpt2', 'bert-base-uncased', or any supported backend"
                )
            
            # Validate embedding type
            valid_embedding_types = ['standard', 'sinusoidal', 'configurable_sinusoidal']
            if self.embedding_type not in valid_embedding_types:
                raise ConvenienceValidationError(
                    f"Invalid embedding_type: {self.embedding_type}",
                    suggestion="Use 'standard' for basic embeddings, 'sinusoidal' for sinusoidal embeddings, or 'configurable_sinusoidal' for advanced sinusoidal embeddings",
                    valid_options=valid_embedding_types
                )
            
            # Validate sinusoidal config
            if not isinstance(self.sinusoidal_config, dict):
                raise ConvenienceValidationError(
                    f"sinusoidal_config must be a dictionary, got {type(self.sinusoidal_config).__name__}",
                    suggestion="Pass sinusoidal configuration as a dictionary: {'learnable_frequencies': True, 'base_frequency': 10000.0}"
                )
            
            # Validate streaming config
            if not isinstance(self.streaming_config, dict):
                raise ConvenienceValidationError(
                    f"streaming_config must be a dictionary, got {type(self.streaming_config).__name__}",
                    suggestion="Pass streaming configuration as a dictionary: {'batch_size': 1000, 'memory_threshold_mb': 1000.0}"
                )
            
            # Validate tokenizer backend config
            if not isinstance(self.tokenizer_backend_config, dict):
                raise ConvenienceValidationError(
                    f"tokenizer_backend_config must be a dictionary, got {type(self.tokenizer_backend_config).__name__}",
                    suggestion="Pass backend configuration as a dictionary: {'trust_remote_code': True, 'use_fast': True}"
                )
            
            # Validate streaming flag
            if not isinstance(self.streaming, bool):
                raise ConvenienceValidationError(
                    f"streaming must be boolean, got {type(self.streaming).__name__}",
                    suggestion="Use True to enable streaming for large datasets, False to disable"
                )
            
            # Validate caching flag
            if not isinstance(self.enable_caching, bool):
                raise ConvenienceValidationError(
                    f"enable_caching must be boolean, got {type(self.enable_caching).__name__}",
                    suggestion="Use True to enable intelligent caching (recommended), False to disable"
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
                      'normalize_targets', 'time_series_mode', 'n_lags',
                      'embedding_type', 'sinusoidal_config', 'streaming', 
                      'streaming_config', 'tokenizer_backend_config', 'enable_caching']:
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
            
            # Initialize enhanced tokenizer if needed
            if verbose:
                logger.info("Initializing enhanced tokenizer...")
            
            self._enhanced_tokenizer = self._create_enhanced_tokenizer()
            
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
    
    def predict(self, X: Union[List[List[float]], List[str], np.ndarray]) -> np.ndarray:
        """
        Predict continuous values for sequential data.
        
        Parameters
        ----------
        X : list or array-like of shape (n_samples, n_features) or list of strings
            Sequential data to predict. Can be numerical data or text strings when using enhanced tokenizer
            
        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted continuous values
        """
        self._check_is_fitted()
        
        try:
            # Check if input is text data and we have enhanced tokenizer
            if isinstance(X, list) and len(X) > 0 and isinstance(X[0], str) and self._enhanced_tokenizer is not None:
                # Handle text data with enhanced tokenizer
                token_sequences = self._enhanced_tokenizer.tokenize(
                    X, add_special_tokens=True, padding=True, truncation=True
                )
                X_features = self._extract_enhanced_features(np.array(token_sequences))
                n_samples = len(X)
            else:
                # Handle numerical data
                X_processed = self._preprocess_prediction_data(X)
                X_features = self._extract_temporal_features(X_processed)
                n_samples = len(X_processed)
            
            # Make predictions using downstream regressor
            y_pred_scaled = self._downstream_regressor.predict(X_features)
            
            # Inverse transform if targets were normalized
            if self.normalize_targets and self._target_scaler is not None:
                y_pred = self._target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            else:
                y_pred = y_pred_scaled
            
            logger.debug(f"Made predictions for {n_samples} samples")
            
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
    
    def _create_enhanced_tokenizer(self) -> 'EnhancedTokenizerWrapper':
        """Create an enhanced tokenizer wrapper with the specified configuration."""
        try:
            # Create enhanced tokenizer wrapper
            enhanced_tokenizer = EnhancedTokenizerWrapper(
                tokenizer=self.tokenizer_name,
                embedding_dim=self.embedding_dim,
                max_length=self.max_length,
                backend_specific_config=self.tokenizer_backend_config,
                enable_caching=self.enable_caching
            )
            
            logger.info(f"Created enhanced tokenizer: {enhanced_tokenizer}")
            return enhanced_tokenizer
            
        except Exception as e:
            logger.error(f"Failed to create enhanced tokenizer: {e}")
            raise TrainingSetupError(f"Enhanced tokenizer creation failed: {e}")
    
    def fit_streaming(self, 
                     data_source: Union[str, List[str], 'StreamingDataIterator'],
                     y: Union[List[float], np.ndarray],
                     batch_size: int = 1000,
                     epochs: int = 30,
                     memory_threshold_mb: float = 1000.0,
                     progress_callback: Optional[callable] = None,
                     auto_adjust_batch_size: bool = True,
                     min_batch_size: int = 100,
                     max_batch_size: int = 10000,
                     verbose: bool = True,
                     **fit_params) -> 'LSMRegressor':
        """
        Fit the regressor on streaming data for memory-efficient training.
        
        This method processes large datasets that don't fit in memory by using
        streaming data processing with configurable batch sizes, progress tracking,
        and memory usage monitoring.
        
        Parameters
        ----------
        data_source : str, list, or StreamingDataIterator
            Data source (file path, directory, file list, or StreamingDataIterator)
        y : list or array-like of shape (n_samples,)
            Target continuous values
        batch_size : int, default=1000
            Initial batch size for processing
        epochs : int, default=30
            Number of training epochs
        memory_threshold_mb : float, default=1000.0
            Memory threshold in MB for automatic batch size adjustment
        progress_callback : callable, optional
            Optional callback function for progress updates
        auto_adjust_batch_size : bool, default=True
            Whether to automatically adjust batch size based on memory
        min_batch_size : int, default=100
            Minimum allowed batch size
        max_batch_size : int, default=10000
            Maximum allowed batch size
        verbose : bool, default=True
            Whether to show training progress
        **fit_params : dict
            Additional parameters
            
        Returns
        -------
        self : LSMRegressor
            Returns self for method chaining
            
        Raises
        ------
        InvalidInputError
            If invalid parameters provided
        TrainingExecutionError
            If streaming training fails
        """
        try:
            if verbose:
                logger.info("Starting LSM regression streaming training...")
            
            # Validate parameters
            if batch_size <= 0:
                raise InvalidInputError("batch_size", "positive integer", str(batch_size))
            if epochs <= 0:
                raise InvalidInputError("epochs", "positive integer", str(epochs))
            if memory_threshold_mb <= 0:
                raise InvalidInputError("memory_threshold_mb", "positive float", str(memory_threshold_mb))
            if min_batch_size <= 0 or min_batch_size > max_batch_size:
                raise InvalidInputError("min_batch_size", f"positive integer <= {max_batch_size}", str(min_batch_size))
            
            # Initialize enhanced tokenizer
            if verbose:
                logger.info("Initializing enhanced tokenizer for streaming...")
            
            self._enhanced_tokenizer = self._create_enhanced_tokenizer()
            
            # Handle different data source types
            if isinstance(data_source, list) and isinstance(data_source[0], str):
                # Direct text data - process in batches
                def batch_generator(data, batch_size):
                    for i in range(0, len(data), batch_size):
                        yield data[i:i + batch_size]
                
                streaming_iterator = batch_generator(data_source, batch_size)
                use_streaming_iterator = False
            else:
                # File-based data source - use StreamingDataIterator
                from ..data.streaming_data_iterator import StreamingDataIterator
                
                if isinstance(data_source, StreamingDataIterator):
                    streaming_iterator = data_source
                else:
                    streaming_iterator = StreamingDataIterator(
                        data_source=data_source,
                        batch_size=batch_size,
                        memory_threshold_mb=memory_threshold_mb,
                        auto_adjust_batch_size=auto_adjust_batch_size,
                        progress_callback=progress_callback
                    )
                use_streaming_iterator = True
            
            # Process streaming data and collect features
            all_features = []
            all_targets = []
            
            if isinstance(y, list):
                y = np.array(y)
            
            target_idx = 0
            
            for batch_data in streaming_iterator:
                # Process batch data to extract features
                if isinstance(batch_data, list) and isinstance(batch_data[0], str):
                    # Text data - tokenize and extract features
                    token_sequences = self._enhanced_tokenizer.tokenize(
                        batch_data, add_special_tokens=True, padding=True, truncation=True
                    )
                    batch_features = self._extract_enhanced_features(np.array(token_sequences))
                else:
                    # Numerical data - extract temporal features
                    batch_features = self._extract_temporal_features(np.array(batch_data))
                
                # Get corresponding targets
                batch_size_actual = len(batch_data)
                batch_targets = y[target_idx:target_idx + batch_size_actual]
                target_idx += batch_size_actual
                
                all_features.append(batch_features)
                all_targets.append(batch_targets)
                
                if verbose and progress_callback:
                    progress_callback(f"Processed {target_idx} samples")
            
            # Combine all features and targets
            X_features = np.vstack(all_features)
            y_combined = np.concatenate(all_targets)
            
            # Set up target scaling if requested
            if self.normalize_targets:
                self._target_scaler = StandardScaler()
                y_scaled = self._target_scaler.fit_transform(y_combined.reshape(-1, 1)).ravel()
                self.target_scaler_ = self._target_scaler
            else:
                y_scaled = y_combined
                self.target_scaler_ = None
            
            # Train downstream regressor
            if verbose:
                logger.info(f"Training {self.regressor_type} regressor on streaming data...")
            
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
            
            train_r2 = r2_score(y_combined, y_pred)
            train_mse = mean_squared_error(y_combined, y_pred)
            train_mae = mean_absolute_error(y_combined, y_pred)
            
            # Store training metadata
            self._training_metadata = {
                'regressor_training_time': regressor_training_time,
                'epochs': epochs,
                'batch_size': batch_size,
                'data_size': len(y_combined),
                'n_features': X_features.shape[1],
                'feature_extraction': self.feature_extraction,
                'regressor_type': self.regressor_type,
                'normalize_targets': self.normalize_targets,
                'time_series_mode': self.time_series_mode,
                'streaming': True,
                'train_r2': train_r2,
                'train_mse': train_mse,
                'train_mae': train_mae
            }
            
            # Store performance metrics
            self._performance_metrics = {
                'train_r2_score': train_r2,
                'train_mse': train_mse,
                'train_mae': train_mae,
                'n_samples': len(y_combined),
                'n_features': X_features.shape[1]
            }
            
            # Store feature information for sklearn compatibility
            self.feature_names_in_ = [f'feature_{i}' for i in range(X_features.shape[1])]
            self.n_features_in_ = X_features.shape[1]
            
            self._is_fitted = True
            
            if verbose:
                logger.info(f"Streaming regression training completed in {regressor_training_time:.2f} seconds")
                logger.info(f"Trained on {len(y_combined)} samples with {X_features.shape[1]} features")
                logger.info(f"Training R² score: {train_r2:.4f}")
            
            return self
            
        except Exception as e:
            logger.error(f"Streaming regression training failed: {e}")
            raise TrainingExecutionError(
                epoch=None,
                reason=f"LSM streaming regression training failed: {e}"
            )
    
    def _extract_enhanced_features(self, token_sequences: np.ndarray) -> np.ndarray:
        """Extract features using enhanced tokenizer and sinusoidal embeddings."""
        try:
            # Use sinusoidal embeddings if configured
            if self.embedding_type in ['sinusoidal', 'configurable_sinusoidal']:
                # Create sinusoidal embedder if not exists
                if not hasattr(self, '_sinusoidal_embedder') or self._sinusoidal_embedder is None:
                    if self.embedding_type == 'configurable_sinusoidal':
                        self._sinusoidal_embedder = self._enhanced_tokenizer.create_configurable_sinusoidal_embedder(
                            **self.sinusoidal_config
                        )
                    else:
                        self._sinusoidal_embedder = self._enhanced_tokenizer.create_sinusoidal_embedder()
                        # Fit the sinusoidal embedder on the token sequences
                        self._sinusoidal_embedder.fit(token_sequences)
                
                # Get sinusoidal embeddings
                embeddings = self._sinusoidal_embedder.embed_sequences(token_sequences)
                
                # Extract features from embeddings
                features_list = []
                
                # Statistical features from embeddings
                features_list.append(np.mean(embeddings, axis=1))  # Mean embedding
                features_list.append(np.std(embeddings, axis=1))   # Std embedding
                features_list.append(np.max(embeddings, axis=1))   # Max embedding
                features_list.append(np.min(embeddings, axis=1))   # Min embedding
                
                # Combine features
                X_features = np.column_stack(features_list)
            else:
                # Use standard token-based features
                X_features = self._extract_token_features(token_sequences)
            
            logger.debug(f"Extracted enhanced features shape: {X_features.shape}")
            return X_features
            
        except Exception as e:
            logger.error(f"Enhanced feature extraction failed: {e}")
            # Fallback to token-based features
            return self._extract_token_features(token_sequences)
    
    def _extract_token_features(self, token_sequences: np.ndarray) -> np.ndarray:
        """Extract features from token sequences."""
        try:
            features_list = []
            
            for sequence in token_sequences:
                # Basic sequence statistics
                seq_features = []
                
                # Length features
                seq_features.append(len(sequence))
                seq_features.append(np.count_nonzero(sequence))  # Non-zero tokens
                
                # Statistical features
                seq_features.append(np.mean(sequence))
                seq_features.append(np.std(sequence))
                seq_features.append(np.max(sequence))
                seq_features.append(np.min(sequence))
                
                # Token distribution features
                token_counts = np.bincount(sequence, minlength=min(1000, self._enhanced_tokenizer.get_vocab_size()))
                top_tokens = np.sort(token_counts)[-10:]  # Top 10 token frequencies
                seq_features.extend(top_tokens)
                
                # Positional features
                if len(sequence) > 1:
                    seq_features.append(sequence[0])   # First token
                    seq_features.append(sequence[-1])  # Last token
                    seq_features.append(np.mean(np.diff(sequence)))  # Mean difference
                else:
                    seq_features.extend([0, 0, 0])
                
                features_list.append(seq_features)
            
            X_features = np.array(features_list)
            
            # Handle any NaN values
            X_features = np.nan_to_num(X_features, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return X_features
            
        except Exception as e:
            logger.error(f"Token feature extraction failed: {e}")
            # Minimal fallback
            return np.array([[len(seq), np.mean(seq), np.std(seq)] for seq in token_sequences])
    
    def get_enhanced_tokenizer(self) -> Optional['EnhancedTokenizerWrapper']:
        """
        Get the enhanced tokenizer wrapper.
        
        Returns
        -------
        tokenizer : EnhancedTokenizerWrapper or None
            The enhanced tokenizer wrapper if available
        """
        return self._enhanced_tokenizer
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        Get information about the tokenizer configuration.
        
        Returns
        -------
        info : dict
            Dictionary containing tokenizer information
        """
        info = {
            'tokenizer_name': self.tokenizer_name,
            'max_length': self.max_length,
            'embedding_type': self.embedding_type,
            'sinusoidal_config': self.sinusoidal_config,
            'streaming': self.streaming,
            'streaming_config': self.streaming_config,
            'tokenizer_backend_config': self.tokenizer_backend_config,
            'enable_caching': self.enable_caching,
            'enhanced_tokenizer_available': self._enhanced_tokenizer is not None
        }
        
        if self._enhanced_tokenizer is not None:
            try:
                info.update({
                    'vocab_size': self._enhanced_tokenizer.get_vocab_size(),
                    'adapter_backend': self._enhanced_tokenizer.get_adapter().config.backend,
                    'special_tokens': self._enhanced_tokenizer.get_special_tokens()
                })
            except Exception as e:
                logger.warning(f"Failed to get enhanced tokenizer info: {e}")
        
        return info