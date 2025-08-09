"""
Base classes for the LSM convenience API.

This module provides the abstract base class and common functionality for all
convenience API classes, following sklearn's BaseEstimator pattern.
"""

import os
import json
import pickle
import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path

from ..utils.lsm_exceptions import (
    LSMError, ModelLoadError, ModelSaveError, ModelValidationError,
    InvalidInputError, ConfigurationError
)
from ..utils.input_validation import (
    validate_positive_integer, validate_positive_float, validate_string_list,
    validate_file_path, validate_directory_path, create_helpful_error_message
)
from ..utils.lsm_logging import get_logger
from .config import ConvenienceConfig, ConvenienceValidationError

logger = get_logger(__name__)


class LSMBase(ABC):
    """
    Abstract base class for all LSM convenience API classes.
    
    This class provides the common interface and functionality shared by all
    convenience classes, following sklearn's BaseEstimator pattern for
    parameter management, model persistence, and validation.
    
    Parameters
    ----------
    window_size : int, default=10
        Size of the sliding window for sequence processing
    embedding_dim : int, default=128
        Dimension of the embedding space
    reservoir_type : str, default='standard'
        Type of reservoir to use ('standard', 'hierarchical', 'attentive', 'echo_state', 'deep')
    reservoir_config : dict, optional
        Additional configuration for the reservoir
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters specific to subclasses
    """
    
    def __init__(self, 
                 window_size: int = 10,
                 embedding_dim: int = 128,
                 reservoir_type: str = 'standard',
                 reservoir_config: Optional[Dict[str, Any]] = None,
                 random_state: Optional[int] = None,
                 **kwargs):
        
        # Store all parameters for sklearn compatibility
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.reservoir_type = reservoir_type
        self.reservoir_config = reservoir_config or {}
        self.random_state = random_state
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        # Internal state
        self._is_fitted = False
        self._trainer = None
        self._model_components = {}
        self._training_metadata = {}
        
        # Validate parameters on initialization
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate all parameters with helpful error messages."""
        try:
            # Validate basic parameters
            self.window_size = validate_positive_integer(
                self.window_size, 'window_size', min_value=1, max_value=100
            )
            
            self.embedding_dim = validate_positive_integer(
                self.embedding_dim, 'embedding_dim', min_value=1, max_value=2048
            )
            
            # Validate reservoir type
            valid_reservoir_types = ['standard', 'hierarchical', 'attentive', 'echo_state', 'deep']
            if self.reservoir_type not in valid_reservoir_types:
                raise ConvenienceValidationError(
                    f"Invalid reservoir_type: {self.reservoir_type}",
                    suggestion="Try 'hierarchical' for text generation or 'standard' for classification",
                    valid_options=valid_reservoir_types
                )
            
            # Validate reservoir config if provided
            if self.reservoir_config and not isinstance(self.reservoir_config, dict):
                raise ConvenienceValidationError(
                    f"reservoir_config must be a dictionary, got {type(self.reservoir_config).__name__}",
                    suggestion="Pass reservoir configuration as a dictionary: {'sparsity': 0.1, 'spectral_radius': 0.9}"
                )
            
            # Validate random state
            if self.random_state is not None:
                self.random_state = validate_positive_integer(
                    self.random_state, 'random_state', min_value=0
                )
            
        except InvalidInputError as e:
            logger.error(f"Parameter validation failed: {e}")
            # Convert to ConvenienceValidationError for better user experience
            raise ConvenienceValidationError(
                str(e),
                suggestion="Check parameter values and ensure they are within valid ranges"
            )
        except ConvenienceValidationError as e:
            logger.error(f"Parameter validation failed: {e}")
            raise
    
    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """
        Fit the LSM model to training data.
        
        Parameters
        ----------
        X : array-like or list
            Training data
        y : array-like, optional
            Target values (for supervised learning)
        **fit_params : dict
            Additional fitting parameters
            
        Returns
        -------
        self : object
            Returns self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make predictions using the fitted model.
        
        Parameters
        ----------
        X : array-like or list
            Input data for prediction
            
        Returns
        -------
        predictions : array-like
            Model predictions
        """
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the fitted model to disk in convenience format.
        
        Creates a simplified model directory structure while maintaining 
        compatibility with the full LSM system. Includes automatic model
        validation and integrity checking.
        
        Directory structure:
        model_name/
        ├── convenience_config.json    # Convenience API configuration
        ├── model/                     # Full LSM model (existing format)
        │   ├── reservoir_model/
        │   ├── cnn_model/
        │   ├── tokenizer/
        │   └── config.json
        └── metadata.json             # Training metadata and performance
        
        Parameters
        ----------
        path : str or Path
            Directory path where the model will be saved
            
        Raises
        ------
        ModelSaveError
            If the model cannot be saved or validation fails
        """
        if not self._is_fitted:
            raise ModelSaveError(
                str(path), 
                "Model must be fitted before saving. Call fit() first."
            )
        
        try:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            
            # Validate model integrity before saving
            self._validate_model_integrity()
            
            # Save convenience API configuration
            convenience_config = {
                'class_name': self.__class__.__name__,
                'module': self.__class__.__module__,
                'parameters': self.get_params(),
                'version': '1.0.0',
                'created_at': datetime.datetime.now().isoformat(),
                'lsm_version': self._get_lsm_version()
            }
            
            with open(path / 'convenience_config.json', 'w') as f:
                json.dump(convenience_config, f, indent=2, default=str)
            
            # Save training metadata and performance metrics
            metadata = {
                'training_metadata': self._training_metadata,
                'model_components': list(self._model_components.keys()),
                'performance_metrics': getattr(self, '_performance_metrics', {}),
                'training_history': getattr(self, '_training_history', {}),
                'data_info': getattr(self, '_data_info', {}),
                'model_size_mb': self._calculate_model_size(),
                'save_timestamp': datetime.datetime.now().isoformat()
            }
            
            with open(path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save underlying model components in standard LSM format
            if self._trainer:
                model_path = path / 'model'
                self._trainer.save_model(str(model_path))
            
            # Save additional convenience-specific components
            if self._model_components:
                components_path = path / 'model' / 'convenience_components.pkl'
                components_path.parent.mkdir(parents=True, exist_ok=True)
                with open(components_path, 'wb') as f:
                    pickle.dump(self._model_components, f)
            
            # Verify saved model integrity
            self._verify_saved_model(path)
            
            logger.info(f"Model saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            # Clean up partial save if it failed
            if path.exists():
                import shutil
                try:
                    shutil.rmtree(path)
                except:
                    pass
            raise ModelSaveError(str(path), f"Save operation failed: {e}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LSMBase':
        """
        Load a fitted model from disk with automatic validation and integrity checking.
        
        Loads models saved in the convenience format and performs comprehensive
        validation to ensure model integrity and compatibility.
        
        Parameters
        ----------
        path : str or Path
            Directory path where the model is saved
            
        Returns
        -------
        model : LSMBase
            Loaded model instance
            
        Raises
        ------
        ModelLoadError
            If the model cannot be loaded
        ModelValidationError
            If the loaded model fails validation
        """
        try:
            path = Path(path)
            validate_directory_path(str(path), must_exist=True, description="model directory")
            
            # Validate model directory structure
            cls._validate_model_directory(path)
            
            # Load convenience configuration
            config_path = path / 'convenience_config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Validate configuration compatibility
            cls._validate_model_config(config, cls.__name__)
            
            # Load metadata
            metadata_path = path / 'metadata.json'
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            # Create instance with saved parameters
            parameters = config.get('parameters', {})
            instance = cls(**parameters)
            
            # Load training metadata and performance metrics
            instance._training_metadata = metadata.get('training_metadata', {})
            instance._performance_metrics = metadata.get('performance_metrics', {})
            instance._training_history = metadata.get('training_history', {})
            instance._data_info = metadata.get('data_info', {})
            
            # Load convenience-specific components
            components_path = path / 'model' / 'convenience_components.pkl'
            if components_path.exists():
                with open(components_path, 'rb') as f:
                    instance._model_components = pickle.load(f)
            
            # Load underlying model if it exists
            model_path = path / 'model'
            if model_path.exists():
                # This will be implemented by subclasses to load their specific trainers
                instance._load_trainer(str(model_path))
            
            instance._is_fitted = True
            
            # Validate loaded model integrity
            instance._validate_loaded_model()
            
            logger.info(f"Model loaded successfully from {path}")
            
            return instance
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            if isinstance(e, (ModelLoadError, ModelValidationError)):
                raise
            else:
                raise ModelLoadError(str(path), f"Load operation failed: {e}")
    
    def _load_trainer(self, model_path: str) -> None:
        """
        Load the underlying trainer. To be implemented by subclasses.
        
        Parameters
        ----------
        model_path : str
            Path to the saved model
        """
        # This will be implemented by subclasses
        pass
    
    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """
        Get parameters for this estimator.
        
        This method is required for sklearn compatibility and enables
        the use of LSM models in sklearn pipelines and grid search.
        
        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters for this estimator and contained
            subobjects that are estimators
            
        Returns
        -------
        params : dict
            Parameter names mapped to their values
        """
        params = {}
        
        # Get all parameters that don't start with underscore
        for key in dir(self):
            if not key.startswith('_') and not callable(getattr(self, key)):
                value = getattr(self, key)
                # Only include basic types and None
                if isinstance(value, (int, float, str, bool, list, dict, tuple)) or value is None:
                    params[key] = value
        
        return params
    
    def set_params(self, **params) -> 'LSMBase':
        """
        Set the parameters of this estimator.
        
        This method is required for sklearn compatibility and enables
        parameter tuning with grid search and other sklearn utilities.
        
        Parameters
        ----------
        **params : dict
            Estimator parameters
            
        Returns
        -------
        self : object
            Returns self for method chaining
            
        Raises
        ------
        ConvenienceValidationError
            If invalid parameters are provided
        """
        valid_params = self.get_params(deep=False)
        
        for key, value in params.items():
            if key not in valid_params:
                raise ConvenienceValidationError(
                    f"Invalid parameter: {key}",
                    suggestion=f"Valid parameters are: {list(valid_params.keys())}",
                    valid_options=list(valid_params.keys())
                )
            setattr(self, key, value)
        
        # Re-validate parameters after setting
        self._validate_parameters()
        
        # If model was fitted, it needs to be re-fitted with new parameters
        if self._is_fitted:
            logger.warning("Parameters changed on fitted model. Model needs to be re-fitted.")
            self._is_fitted = False
            self._trainer = None
        
        return self
    
    def __sklearn_tags__(self):
        """
        Return sklearn tags for this estimator.
        
        This method provides metadata about the estimator's capabilities
        for sklearn's introspection system.
        """
        return {
            'requires_fit': True,
            'requires_y': False,  # Will be overridden by supervised classes
            'requires_positive_X': False,
            'allow_nan': False,
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'multilabel': False,
            'multiclass': False,
            'binary_only': False,
            'stateless': False,
            'pairwise': False,
        }
    
    def _check_is_fitted(self) -> None:
        """
        Check if the model is fitted and raise an error if not.
        
        Raises
        ------
        InvalidInputError
            If the model is not fitted
        """
        if not self._is_fitted:
            raise InvalidInputError(
                "model state",
                "fitted model",
                "unfitted model (call fit() first)"
            )
    
    def _estimate_memory_usage(self, data_size: int) -> float:
        """
        Estimate memory usage in MB for given data size.
        
        Parameters
        ----------
        data_size : int
            Size of the input data
            
        Returns
        -------
        estimated_mb : float
            Estimated memory usage in MB
        """
        # Basic estimation based on model parameters
        base_memory = (
            self.window_size * self.embedding_dim * 4 / (1024 * 1024)  # 4 bytes per float32
        )
        
        # Add reservoir memory
        reservoir_units = self.reservoir_config.get('reservoir_units', [100, 50])
        if isinstance(reservoir_units, list):
            total_units = sum(reservoir_units)
        else:
            total_units = reservoir_units
        
        reservoir_memory = total_units * self.embedding_dim * 4 / (1024 * 1024)
        
        # Add data processing memory
        data_memory = data_size * self.embedding_dim * 4 / (1024 * 1024)
        
        return base_memory + reservoir_memory + data_memory
    
    def _create_error_suggestions(self, error: Exception, operation: str) -> List[str]:
        """
        Create helpful suggestions based on the error type and operation.
        
        Parameters
        ----------
        error : Exception
            The error that occurred
        operation : str
            The operation that failed
            
        Returns
        -------
        suggestions : list
            List of helpful suggestions
        """
        suggestions = []
        
        if "memory" in str(error).lower():
            suggestions.extend([
                "Try reducing window_size or embedding_dim",
                "Process data in smaller batches",
                "Close other applications to free memory"
            ])
        
        if "disk" in str(error).lower() or "space" in str(error).lower():
            suggestions.extend([
                "Free up disk space",
                "Choose a different save location",
                "Clean up temporary files"
            ])
        
        if operation == "fit" and "shape" in str(error).lower():
            suggestions.extend([
                "Check input data format and dimensions",
                "Ensure all sequences have consistent format",
                "Verify data preprocessing steps"
            ])
        
        if operation == "predict" and not self._is_fitted:
            suggestions.append("Call fit() before making predictions")
        
        return suggestions
    
    def __repr__(self) -> str:
        """Return string representation of the model."""
        params = self.get_params(deep=False)
        param_str = ", ".join(f"{k}={v}" for k, v in list(params.items())[:3])
        if len(params) > 3:
            param_str += ", ..."
        
        fitted_status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({param_str}) - {fitted_status}"
    
    def __getstate__(self) -> Dict[str, Any]:
        """Support for pickle serialization."""
        state = self.__dict__.copy()
        # Remove unpicklable entries if any
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self.__dict__.update(state)
    
    def _validate_model_integrity(self) -> None:
        """
        Validate model integrity before saving.
        
        Raises
        ------
        ModelValidationError
            If the model fails integrity checks
        """
        try:
            # Check that model is properly fitted
            if not self._is_fitted:
                raise ModelValidationError("Model is not fitted")
            
            # Check that trainer exists if expected (only for models that should have trainers)
            if hasattr(self, '_trainer') and self._trainer is None and hasattr(self, '_requires_trainer'):
                if getattr(self, '_requires_trainer', False):
                    raise ModelValidationError("model", ["Model trainer is None but model claims to be fitted"])
            
            # Validate parameters are still consistent
            self._validate_parameters()
            
            # Check model components are valid
            if hasattr(self, '_model_components') and self._model_components:
                for name, component in self._model_components.items():
                    if component is None:
                        logger.warning(f"Model component '{name}' is None")
            
            logger.debug("Model integrity validation passed")
            
        except Exception as e:
            raise ModelValidationError("model", [f"Model integrity validation failed: {e}"])
    
    def _verify_saved_model(self, path: Path) -> None:
        """
        Verify that the saved model can be loaded correctly.
        
        Parameters
        ----------
        path : Path
            Path to the saved model directory
            
        Raises
        ------
        ModelValidationError
            If the saved model cannot be verified
        """
        try:
            # Check that all required files exist
            required_files = ['convenience_config.json']
            for file_name in required_files:
                file_path = path / file_name
                if not file_path.exists():
                    raise ModelValidationError(f"Required file missing: {file_name}")
            
            # Verify convenience config can be loaded
            config_path = path / 'convenience_config.json'
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Verify metadata if it exists
            metadata_path = path / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    json.load(f)  # Just verify it's valid JSON
            
            logger.debug("Saved model verification passed")
            
        except Exception as e:
            raise ModelValidationError(str(path), [f"Saved model verification failed: {e}"])
    
    @classmethod
    def _validate_model_directory(cls, path: Path) -> None:
        """
        Validate that the model directory has the expected structure.
        
        Parameters
        ----------
        path : Path
            Path to the model directory
            
        Raises
        ------
        ModelLoadError
            If the directory structure is invalid
        """
        required_files = ['convenience_config.json']
        missing_files = []
        
        for file_name in required_files:
            if not (path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise ModelLoadError(
                str(path),
                f"Missing required files: {missing_files}",
                missing_components=missing_files
            )
    
    @classmethod
    def _validate_model_config(cls, config: Dict[str, Any], expected_class: str) -> None:
        """
        Validate model configuration for compatibility.
        
        Parameters
        ----------
        config : dict
            Loaded configuration
        expected_class : str
            Expected class name
            
        Raises
        ------
        ModelLoadError
            If the configuration is incompatible
        """
        # Check class compatibility
        if config.get('class_name') != expected_class:
            raise ModelLoadError(
                "config validation",
                f"Model class mismatch: expected {expected_class}, got {config.get('class_name')}"
            )
        
        # Check version compatibility
        config_version = config.get('version', '0.0.0')
        current_version = '1.0.0'  # This should be imported from a version module
        
        if config_version != current_version:
            logger.warning(f"Version mismatch: model saved with {config_version}, loading with {current_version}")
        
        # Validate required fields
        required_fields = ['class_name', 'parameters', 'version']
        for field in required_fields:
            if field not in config:
                raise ModelLoadError(
                    "config validation",
                    f"Missing required configuration field: {field}"
                )
    
    def _validate_loaded_model(self) -> None:
        """
        Validate the loaded model for integrity and functionality.
        
        Raises
        ------
        ModelValidationError
            If the loaded model fails validation
        """
        try:
            # Validate that model is marked as fitted
            if not self._is_fitted:
                raise ModelValidationError("Loaded model is not marked as fitted")
            
            # Validate parameters
            self._validate_parameters()
            
            # Check that essential components exist
            if hasattr(self, '_trainer') and self._trainer is not None:
                # Trainer-specific validation would go here
                pass
            
            logger.debug("Loaded model validation passed")
            
        except Exception as e:
            raise ModelValidationError("loaded_model", [f"Loaded model validation failed: {e}"])
    
    def _get_lsm_version(self) -> str:
        """
        Get the current LSM package version.
        
        Returns
        -------
        version : str
            Current LSM version
        """
        try:
            # Try to get version from package metadata
            import importlib.metadata
            return importlib.metadata.version('lsm')
        except:
            # Fallback to a default version
            return '1.0.0'
    
    def _calculate_model_size(self) -> float:
        """
        Calculate approximate model size in MB.
        
        Returns
        -------
        size_mb : float
            Approximate model size in megabytes
        """
        try:
            total_size = 0
            
            # Estimate trainer size
            if hasattr(self, '_trainer') and self._trainer is not None:
                # This is a rough estimate - actual implementation would depend on trainer structure
                total_size += self.embedding_dim * self.window_size * 4  # 4 bytes per float32
            
            # Estimate component sizes
            if hasattr(self, '_model_components'):
                import sys
                for component in self._model_components.values():
                    if component is not None:
                        total_size += sys.getsizeof(component)
            
            return total_size / (1024 * 1024)  # Convert to MB
            
        except Exception as e:
            logger.warning(f"Could not calculate model size: {e}")
            return 0.0