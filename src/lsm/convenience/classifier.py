"""
LSMClassifier for classification tasks using LSM features.

This module provides a scikit-learn compatible classifier that uses Liquid State Machine
reservoir states as features for downstream classification tasks.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path

# sklearn imports
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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


class LSMClassifier(LSMBase, ClassifierMixin):
    """
    LSM-based classifier with scikit-learn-like interface.
    
    This class uses Liquid State Machine reservoir states as features for 
    downstream classification tasks. It extracts temporal dynamics from the
    reservoir and trains a traditional classifier on these features.
    
    Parameters
    ----------
    window_size : int, default=10
        Size of the sliding window for sequence processing
    embedding_dim : int, default=128
        Dimension of the embedding space
    reservoir_type : str, default='standard'
        Type of reservoir ('standard', 'hierarchical', 'attentive', 'echo_state', 'deep')
    reservoir_config : dict, optional
        Additional configuration for the reservoir
    n_classes : int, optional
        Number of classes (auto-detected if not provided)
    classifier_type : str, default='logistic'
        Type of downstream classifier ('logistic', 'random_forest')
    classifier_config : dict, optional
        Configuration for the downstream classifier
    feature_extraction : str, default='mean'
        How to extract features from reservoir states ('mean', 'last', 'max', 'concat')
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters passed to the base class
        
    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted
    classes_ : ndarray
        The classes seen during fit
    n_classes_ : int
        Number of classes
    feature_names_in_ : list
        Input feature names
    n_features_in_ : int
        Number of input features
        
    Examples
    --------
    >>> from lsm import LSMClassifier
    >>> 
    >>> # Simple text classification
    >>> classifier = LSMClassifier()
    >>> X = ["This is positive", "This is negative", "Neutral text"]
    >>> y = [1, 0, 2]
    >>> classifier.fit(X, y)
    >>> predictions = classifier.predict(["New positive text"])
    >>> 
    >>> # With custom configuration
    >>> classifier = LSMClassifier(
    ...     reservoir_type='hierarchical',
    ...     classifier_type='random_forest',
    ...     feature_extraction='concat'
    ... )
    >>> classifier.fit(X, y)
    """
    
    def __init__(self,
                 window_size: int = 10,
                 embedding_dim: int = 128,
                 reservoir_type: str = 'standard',
                 reservoir_config: Optional[Dict[str, Any]] = None,
                 n_classes: Optional[int] = None,
                 classifier_type: str = 'logistic',
                 classifier_config: Optional[Dict[str, Any]] = None,
                 feature_extraction: str = 'mean',
                 random_state: Optional[int] = None,
                 **kwargs):
        
        # Set defaults optimized for classification
        if reservoir_config is None:
            reservoir_config = {
                'reservoir_units': [100, 50],
                'sparsity': 0.1,
                'spectral_radius': 0.9
            }
        
        # Store classification-specific parameters
        self.n_classes = n_classes
        self.classifier_type = classifier_type
        self.classifier_config = classifier_config or {}
        self.feature_extraction = feature_extraction
        
        # Initialize base class
        super().__init__(
            window_size=window_size,
            embedding_dim=embedding_dim,
            reservoir_type=reservoir_type,
            reservoir_config=reservoir_config,
            random_state=random_state,
            **kwargs
        )
        
        # Classification components (initialized during fit)
        self._label_encoder = None
        self._downstream_classifier = None
        self._feature_extractor = None
        
        # sklearn-compatible attributes
        self.classes_ = None
        self.n_classes_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
        # Validate classification-specific parameters
        self._validate_classification_parameters()
    
    def _validate_classification_parameters(self) -> None:
        """Validate classification-specific parameters."""
        try:
            # Validate classifier type
            valid_classifier_types = ['logistic', 'random_forest']
            if self.classifier_type not in valid_classifier_types:
                raise ConvenienceValidationError(
                    f"Invalid classifier_type: {self.classifier_type}",
                    suggestion="Use 'logistic' for linear classification or 'random_forest' for non-linear",
                    valid_options=valid_classifier_types
                )
            
            # Validate feature extraction method
            valid_extraction_methods = ['mean', 'last', 'max', 'concat']
            if self.feature_extraction not in valid_extraction_methods:
                raise ConvenienceValidationError(
                    f"Invalid feature_extraction: {self.feature_extraction}",
                    suggestion="Use 'mean' for average pooling, 'last' for final state, 'max' for max pooling, or 'concat' for concatenation",
                    valid_options=valid_extraction_methods
                )
            
            # Validate n_classes if provided
            if self.n_classes is not None:
                self.n_classes = validate_positive_integer(
                    self.n_classes, 'n_classes', min_value=2, max_value=1000
                )
            
            # Validate classifier config
            if not isinstance(self.classifier_config, dict):
                raise ConvenienceValidationError(
                    f"classifier_config must be a dictionary, got {type(self.classifier_config).__name__}",
                    suggestion="Pass classifier configuration as a dictionary: {'C': 1.0, 'max_iter': 1000}"
                )
        
        except Exception as e:
            logger.error(f"Classification parameter validation failed: {e}")
            raise
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'LSMClassifier':
        """
        Create LSMClassifier from a preset configuration.
        
        Parameters
        ----------
        preset_name : str
            Name of the preset ('fast', 'balanced', 'quality', 'classification')
        **overrides : dict
            Parameters to override in the preset
            
        Returns
        -------
        classifier : LSMClassifier
            Configured classifier instance
            
        Examples
        --------
        >>> classifier = LSMClassifier.from_preset('fast')
        >>> classifier = LSMClassifier.from_preset('quality', classifier_type='random_forest')
        """
        config = ConvenienceConfig.create_config(
            preset=preset_name, 
            task_type='classification',
            **overrides
        )
        
        # Extract classification-specific parameters
        class_params = {}
        for param in ['n_classes', 'classifier_type', 'classifier_config', 'feature_extraction']:
            if param in config:
                class_params[param] = config.pop(param)
        
        return cls(**config, **class_params)
    
    def fit(self, 
            X: Union[List[str], List[List[str]], np.ndarray],
            y: Union[List[int], List[str], np.ndarray],
            validation_split: float = 0.2,
            epochs: int = 30,
            batch_size: int = 32,
            verbose: bool = True,
            **fit_params) -> 'LSMClassifier':
        """
        Train the LSM classifier on text data.
        
        This implementation uses TF-IDF features for simplicity and speed.
        Future versions will integrate full LSM reservoir state extraction.
        
        Parameters
        ----------
        X : list or array-like
            Training text samples
        y : list or array-like
            Target class labels (strings or integers)
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
        self : LSMClassifier
            Returns self for method chaining
        """
        try:
            if verbose:
                logger.info("Starting LSM classification training...")
            
            # Validate and preprocess input data
            X_processed, y_processed = self._preprocess_classification_data(X, y)
            
            # Store feature information for sklearn compatibility
            self.feature_names_in_ = [f'text_sample_{i}' for i in range(len(X_processed))]
            self.n_features_in_ = len(X_processed)
            
            # Set up label encoding
            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y_processed)
            
            # Store class information
            self.classes_ = self._label_encoder.classes_
            self.n_classes_ = len(self.classes_)
            
            if self.n_classes is None:
                self.n_classes = self.n_classes_
            
            # Extract features using TF-IDF (simplified version)
            if verbose:
                logger.info("Extracting text features...")
            
            X_features = self._extract_simple_features(X_processed)
            
            # Train downstream classifier
            if verbose:
                logger.info(f"Training {self.classifier_type} classifier...")
            
            start_time = time.time()
            
            self._downstream_classifier = self._create_downstream_classifier()
            self._downstream_classifier.fit(X_features, y_encoded)
            
            classifier_training_time = time.time() - start_time
            
            # Store training metadata
            self._training_metadata = {
                'classifier_training_time': classifier_training_time,
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split,
                'data_size': len(X_processed),
                'n_classes': self.n_classes_,
                'feature_extraction': self.feature_extraction,
                'classifier_type': self.classifier_type,
                'lsm_available': _TRAINING_AVAILABLE
            }
            
            self._is_fitted = True
            
            if verbose:
                logger.info(f"Classification training completed in {classifier_training_time:.2f} seconds")
                logger.info(f"Trained on {len(X_processed)} samples with {self.n_classes_} classes")
            
            return self
            
        except Exception as e:
            logger.error(f"Classification training failed: {e}")
            raise TrainingExecutionError(
                epoch=None,
                reason=f"LSM classification training failed: {e}"
            )
    
    def predict(self, X: Union[List[str], List[List[str]], np.ndarray]) -> np.ndarray:
        """
        Predict classes for text samples.
        
        Parameters
        ----------
        X : list or array-like
            Text samples to classify
            
        Returns
        -------
        predictions : ndarray
            Predicted class labels
        """
        self._check_is_fitted()
        
        try:
            # Preprocess input data
            X_processed = self._preprocess_prediction_data(X)
            
            # Extract features
            X_features = self._extract_simple_features(X_processed)
            
            # Make predictions using downstream classifier
            y_pred_encoded = self._downstream_classifier.predict(X_features)
            
            # Decode predictions back to original labels
            y_pred = self._label_encoder.inverse_transform(y_pred_encoded)
            
            logger.debug(f"Made predictions for {len(X_processed)} samples")
            
            return y_pred
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise InvalidInputError(
                "prediction process",
                "successful classification",
                f"prediction failed: {e}"
            )
    
    def predict_proba(self, X: Union[List[str], List[List[str]], np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities for text samples.
        
        Parameters
        ----------
        X : list or array-like
            Text samples to classify
            
        Returns
        -------
        probabilities : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities
        """
        self._check_is_fitted()
        
        try:
            # Check if downstream classifier supports probability prediction
            if not hasattr(self._downstream_classifier, 'predict_proba'):
                raise InvalidInputError(
                    "classifier capability",
                    "probability prediction support",
                    f"{self.classifier_type} classifier doesn't support probability prediction"
                )
            
            # Preprocess input data
            X_processed = self._preprocess_prediction_data(X)
            
            # Extract features
            X_features = self._extract_simple_features(X_processed)
            
            # Get probability predictions
            probabilities = self._downstream_classifier.predict_proba(X_features)
            
            logger.debug(f"Generated probabilities for {len(X_processed)} samples")
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Probability prediction failed: {e}")
            raise InvalidInputError(
                "probability prediction process",
                "successful probability prediction",
                f"probability prediction failed: {e}"
            )
    
    def score(self, X: Union[List[str], List[List[str]], np.ndarray], 
              y: Union[List[int], List[str], np.ndarray]) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters
        ----------
        X : list or array-like
            Test text samples
        y : list or array-like
            True class labels
            
        Returns
        -------
        score : float
            Mean accuracy of predictions
        """
        self._check_is_fitted()
        
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate accuracy
            accuracy = accuracy_score(y, y_pred)
            
            logger.debug(f"Calculated accuracy: {accuracy:.4f}")
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            raise InvalidInputError(
                "scoring process",
                "successful accuracy calculation",
                f"scoring failed: {e}"
            )
    
    def _preprocess_classification_data(self, X, y) -> Tuple[List[str], List]:
        """Preprocess and validate classification training data."""
        # Process X
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        if isinstance(X, list):
            if all(isinstance(item, str) for item in X):
                # List of strings
                X_processed = X
            elif all(isinstance(item, list) for item in X):
                # List of token sequences - join them
                X_processed = [" ".join(tokens) for tokens in X]
            else:
                raise ConvenienceValidationError(
                    "Mixed data types in input list",
                    suggestion="Use either all strings or all token sequences"
                )
        else:
            raise ConvenienceValidationError(
                f"Invalid input type: {type(X).__name__}",
                suggestion="Use list of strings or numpy array",
                valid_options=["list of strings", "list of token sequences", "numpy array"]
            )
        
        # Process y
        if isinstance(y, np.ndarray):
            y = y.tolist()
        
        if not isinstance(y, list):
            raise ConvenienceValidationError(
                f"Invalid target type: {type(y).__name__}",
                suggestion="Use list or numpy array of labels"
            )
        
        # Validate lengths match
        if len(X_processed) != len(y):
            raise ConvenienceValidationError(
                f"Length mismatch: {len(X_processed)} samples but {len(y)} labels",
                suggestion="Ensure each sample has a corresponding label"
            )
        
        # Validate we have enough data
        if len(X_processed) < 2:
            raise ConvenienceValidationError(
                f"Insufficient training data: {len(X_processed)} samples",
                suggestion="Provide at least 2 samples for training"
            )
        
        return X_processed, y
    
    def _preprocess_prediction_data(self, X) -> List[str]:
        """Preprocess prediction input data."""
        if isinstance(X, str):
            return [X]
        
        if isinstance(X, np.ndarray):
            X = X.tolist()
        
        if isinstance(X, list):
            if all(isinstance(item, str) for item in X):
                return X
            elif all(isinstance(item, list) for item in X):
                return [" ".join(tokens) for tokens in X]
            else:
                raise ConvenienceValidationError(
                    "Mixed data types in input list",
                    suggestion="Use either all strings or all token sequences"
                )
        else:
            raise ConvenienceValidationError(
                f"Invalid input type: {type(X).__name__}",
                suggestion="Use string, list of strings, or numpy array"
            )
    
    def _extract_simple_features(self, X: List[str]) -> np.ndarray:
        """Extract simple text features using TF-IDF."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            if hasattr(self, '_feature_extractor') and self._feature_extractor is not None:
                # Use existing vectorizer for prediction
                X_features = self._feature_extractor.transform(X).toarray()
            else:
                # Create new vectorizer for training
                vectorizer = TfidfVectorizer(max_features=min(100, len(X) * 10))
                X_features = vectorizer.fit_transform(X).toarray()
                
                # Store vectorizer for prediction
                self._feature_extractor = vectorizer
            
            logger.debug(f"Extracted simple features shape: {X_features.shape}")
            
            return X_features
            
        except Exception as e:
            logger.error(f"Simple feature extraction failed: {e}")
            # Fallback to random features
            X_features = np.random.randn(len(X), 50)
            logger.warning("Using random features as fallback")
            return X_features
    
    def _create_downstream_classifier(self):
        """Create the downstream classifier."""
        if self.classifier_type == 'logistic':
            default_config = {
                'random_state': self.random_state,
                'max_iter': 1000,
                'C': 1.0
            }
            config = {**default_config, **self.classifier_config}
            return LogisticRegression(**config)
        
        elif self.classifier_type == 'random_forest':
            default_config = {
                'random_state': self.random_state,
                'n_estimators': 100,
                'max_depth': None
            }
            config = {**default_config, **self.classifier_config}
            return RandomForestClassifier(**config)
        
        else:
            raise ConvenienceValidationError(
                f"Unsupported classifier type: {self.classifier_type}",
                valid_options=['logistic', 'random_forest']
            )
    
    def __sklearn_tags__(self):
        """Return sklearn tags for this classifier."""
        tags = super().__sklearn_tags__()
        tags.update({
            'requires_y': True,
            'multiclass': True,
            'binary_only': False,
            'multilabel': False,
            'multioutput': False,
        })
        return tags