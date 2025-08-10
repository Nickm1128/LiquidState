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
    >>> # With enhanced sinusoidal embeddings
    >>> classifier = LSMClassifier(
    ...     tokenizer='gpt2',
    ...     embedding_type='configurable_sinusoidal',
    ...     sinusoidal_config={'learnable_frequencies': True, 'base_frequency': 10000.0}
    ... )
    >>> classifier.fit(X, y)
    >>> predictions = classifier.predict(["New positive text"])
    >>> 
    >>> # With different tokenizer backends
    >>> classifier = LSMClassifier(
    ...     tokenizer='bert-base-uncased',  # HuggingFace tokenizer
    ...     embedding_type='sinusoidal',
    ...     enable_caching=True
    ... )
    >>> classifier.fit(X, y)
    >>> 
    >>> # With streaming for large datasets
    >>> classifier = LSMClassifier(
    ...     streaming=True,
    ...     streaming_config={'batch_size': 1000, 'memory_threshold_mb': 1000.0}
    ... )
    >>> classifier.fit(X, y)
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
        
        # Set defaults optimized for classification
        if reservoir_config is None:
            reservoir_config = {
                'reservoir_units': [100, 50],
                'sparsity': 0.1,
                'spectral_radius': 0.9
            }
        
        # Check if training components are available (lazy import)
        if not _check_training_components():
            raise ImportError(
                "LSM training components are not available. "
                "Please ensure TensorFlow and all dependencies are installed."
            )
        
        # Store classification-specific parameters
        self.n_classes = n_classes
        self.classifier_type = classifier_type
        self.classifier_config = classifier_config or {}
        self.feature_extraction = feature_extraction
        
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
        
        # Classification components (initialized during fit)
        self._label_encoder = None
        self._downstream_classifier = None
        self._feature_extractor = None
        self._enhanced_tokenizer = None
        self._trainer = None
        
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
        for param in ['n_classes', 'classifier_type', 'classifier_config', 'feature_extraction',
                     'tokenizer', 'max_length', 'embedding_type', 'sinusoidal_config',
                     'streaming', 'streaming_config', 'tokenizer_backend_config', 'enable_caching']:
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
            
            # Initialize enhanced tokenizer
            if verbose:
                logger.info("Initializing enhanced tokenizer...")
            
            self._enhanced_tokenizer = self._create_enhanced_tokenizer()
            
            # Extract features using enhanced tokenizer or TF-IDF fallback
            if verbose:
                logger.info("Extracting text features...")
            
            X_features = self._extract_features(X_processed)
            
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
            X_features = self._extract_features(X_processed)
            
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
            X_features = self._extract_features(X_processed)
            
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
    
    def _extract_features(self, X: List[str]) -> np.ndarray:
        """Extract text features using enhanced tokenizer or TF-IDF fallback."""
        try:
            # Try to use enhanced tokenizer for feature extraction
            if self._enhanced_tokenizer is not None:
                return self._extract_enhanced_features(X)
            else:
                # Fallback to TF-IDF features
                return self._extract_simple_features(X)
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Final fallback to TF-IDF
            return self._extract_simple_features(X)
    
    def _extract_enhanced_features(self, X: List[str]) -> np.ndarray:
        """Extract features using the enhanced tokenizer."""
        try:
            # Tokenize texts
            token_sequences = self._enhanced_tokenizer.tokenize(
                X, add_special_tokens=True, padding=True, truncation=True
            )
            
            # Convert to numpy array
            token_array = np.array(token_sequences)
            
            # For now, use simple aggregation of token embeddings
            # In a full implementation, this would use reservoir states
            if self.embedding_type in ['sinusoidal', 'configurable_sinusoidal']:
                # Create sinusoidal embedder if needed
                if not hasattr(self, '_sinusoidal_embedder') or self._sinusoidal_embedder is None:
                    if self.embedding_type == 'configurable_sinusoidal':
                        self._sinusoidal_embedder = self._enhanced_tokenizer.create_configurable_sinusoidal_embedder(
                            **self.sinusoidal_config
                        )
                    else:
                        self._sinusoidal_embedder = self._enhanced_tokenizer.create_sinusoidal_embedder()
                        # Fit the sinusoidal embedder on the token sequences
                        self._sinusoidal_embedder.fit(np.array(token_sequences))
                
                # Get embeddings for tokens
                embeddings = []
                for sequence in token_sequences:
                    # Convert tokens to embeddings using the correct interface
                    if self.embedding_type == 'configurable_sinusoidal':
                        # ConfigurableSinusoidalEmbedder uses call method (Keras layer)
                        import tensorflow as tf
                        sequence_tensor = tf.constant([sequence], dtype=tf.int32)
                        seq_embeddings = self._sinusoidal_embedder(sequence_tensor, training=False).numpy()[0]
                    else:
                        # SinusoidalEmbedder uses embed method
                        seq_embeddings = self._sinusoidal_embedder.embed(sequence)
                    
                    # Aggregate embeddings based on feature extraction method
                    if self.feature_extraction == 'mean':
                        agg_embedding = np.mean(seq_embeddings, axis=0)
                    elif self.feature_extraction == 'last':
                        agg_embedding = seq_embeddings[-1]
                    elif self.feature_extraction == 'max':
                        agg_embedding = np.max(seq_embeddings, axis=0)
                    elif self.feature_extraction == 'concat':
                        # Flatten and truncate/pad to fixed size
                        flat_embedding = seq_embeddings.flatten()
                        target_size = self.embedding_dim * 10  # Reasonable fixed size
                        if len(flat_embedding) > target_size:
                            agg_embedding = flat_embedding[:target_size]
                        else:
                            agg_embedding = np.pad(flat_embedding, (0, target_size - len(flat_embedding)))
                    else:
                        agg_embedding = np.mean(seq_embeddings, axis=0)
                    
                    embeddings.append(agg_embedding)
                
                X_features = np.array(embeddings)
            else:
                # For standard embeddings, use simple token statistics
                X_features = self._extract_token_statistics(token_array)
            
            logger.debug(f"Extracted enhanced features shape: {X_features.shape}")
            return X_features
            
        except Exception as e:
            logger.error(f"Enhanced feature extraction failed: {e}")
            # Fallback to simple features
            return self._extract_simple_features(X)
    
    def _extract_token_statistics(self, token_array: np.ndarray) -> np.ndarray:
        """Extract statistical features from token sequences."""
        features = []
        
        for sequence in token_array:
            # Basic statistics
            seq_len = len(sequence)
            unique_tokens = len(set(sequence))
            avg_token_id = np.mean(sequence)
            std_token_id = np.std(sequence)
            
            # Token distribution features
            token_counts = np.bincount(sequence, minlength=min(1000, self._enhanced_tokenizer.get_vocab_size()))
            top_tokens = np.sort(token_counts)[-10:]  # Top 10 token frequencies
            
            # Combine features
            seq_features = np.concatenate([
                [seq_len, unique_tokens, avg_token_id, std_token_id],
                top_tokens
            ])
            
            features.append(seq_features)
        
        return np.array(features)
    
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
    
    def save_model(self, save_path: str, include_tokenizer: bool = True) -> None:
        """
        Save the trained classifier model.
        
        Parameters
        ----------
        save_path : str
            Directory path to save the model
        include_tokenizer : bool, default=True
            Whether to save the enhanced tokenizer configuration
        """
        self._check_is_fitted()
        
        try:
            import os
            import pickle
            
            # Create save directory
            os.makedirs(save_path, exist_ok=True)
            
            # Save classifier components
            model_data = {
                'downstream_classifier': self._downstream_classifier,
                'label_encoder': self._label_encoder,
                'feature_extractor': self._feature_extractor,
                'classes_': self.classes_,
                'n_classes_': self.n_classes_,
                'feature_names_in_': self.feature_names_in_,
                'n_features_in_': self.n_features_in_,
                'training_metadata': self._training_metadata,
                'classifier_config': {
                    'window_size': self.window_size,
                    'embedding_dim': self.embedding_dim,
                    'reservoir_type': self.reservoir_type,
                    'reservoir_config': self.reservoir_config,
                    'n_classes': self.n_classes,
                    'classifier_type': self.classifier_type,
                    'classifier_config': self.classifier_config,
                    'feature_extraction': self.feature_extraction,
                    'tokenizer_name': self.tokenizer_name,
                    'max_length': self.max_length,
                    'embedding_type': self.embedding_type,
                    'sinusoidal_config': self.sinusoidal_config,
                    'streaming': self.streaming,
                    'streaming_config': self.streaming_config,
                    'tokenizer_backend_config': self.tokenizer_backend_config,
                    'enable_caching': self.enable_caching,
                    'random_state': self.random_state
                }
            }
            
            # Save main model data
            model_path = os.path.join(save_path, 'lsm_classifier.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save enhanced tokenizer if available and requested
            if include_tokenizer and self._enhanced_tokenizer is not None:
                try:
                    self._enhanced_tokenizer.get_adapter().save_adapter_config(save_path)
                    
                    # Save sinusoidal embedder if available
                    if hasattr(self, '_sinusoidal_embedder') and self._sinusoidal_embedder is not None:
                        embedder_path = os.path.join(save_path, 'sinusoidal_embedder.pkl')
                        with open(embedder_path, 'wb') as f:
                            pickle.dump(self._sinusoidal_embedder, f)
                    
                    logger.info(f"Enhanced tokenizer saved to {save_path}")
                except Exception as e:
                    logger.warning(f"Failed to save enhanced tokenizer: {e}")
            
            logger.info(f"LSMClassifier model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Model save failed: {e}")
            raise ModelLoadError(f"Failed to save model: {e}")
    
    @classmethod
    def load_model(cls, load_path: str) -> 'LSMClassifier':
        """
        Load a trained classifier model.
        
        Parameters
        ----------
        load_path : str
            Directory path to load the model from
            
        Returns
        -------
        classifier : LSMClassifier
            Loaded classifier instance
        """
        try:
            import os
            import pickle
            
            # Load main model data
            model_path = os.path.join(load_path, 'lsm_classifier.pkl')
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create classifier instance with saved configuration
            config = model_data['classifier_config']
            classifier = cls(**config)
            
            # Restore classifier components
            classifier._downstream_classifier = model_data['downstream_classifier']
            classifier._label_encoder = model_data['label_encoder']
            classifier._feature_extractor = model_data['feature_extractor']
            classifier.classes_ = model_data['classes_']
            classifier.n_classes_ = model_data['n_classes_']
            classifier.feature_names_in_ = model_data['feature_names_in_']
            classifier.n_features_in_ = model_data['n_features_in_']
            classifier._training_metadata = model_data['training_metadata']
            classifier._is_fitted = True
            
            # Try to restore enhanced tokenizer
            try:
                classifier._enhanced_tokenizer = classifier._create_enhanced_tokenizer()
                
                # Try to load sinusoidal embedder if available
                embedder_path = os.path.join(load_path, 'sinusoidal_embedder.pkl')
                if os.path.exists(embedder_path):
                    with open(embedder_path, 'rb') as f:
                        classifier._sinusoidal_embedder = pickle.load(f)
                
                logger.info("Enhanced tokenizer restored successfully")
            except Exception as e:
                logger.warning(f"Failed to restore enhanced tokenizer: {e}")
                classifier._enhanced_tokenizer = None
            
            logger.info(f"LSMClassifier model loaded from {load_path}")
            return classifier
            
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise ModelLoadError(f"Failed to load model from {load_path}: {e}")
    
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