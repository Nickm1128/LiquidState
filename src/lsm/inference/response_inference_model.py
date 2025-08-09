#!/usr/bin/env python3
"""
Response Inference Model for Secondary Processing.

This module provides a secondary model that accepts token embedding sequences
from the reservoir/CNN pipeline and predicts complete responses. It implements
response-level learning and works with both 2D and 3D CNN outputs.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.loss_functions import create_response_level_loss, create_cosine_similarity_loss
from ..data.tokenization import StandardTokenizerWrapper
from ..utils.lsm_exceptions import LSMError, InferenceError
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class ResponseInferenceError(InferenceError):
    """Raised when response inference model operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {"operation": operation, "reason": reason}
        if details:
            error_details.update(details)
        
        message = f"Response inference model failed during {operation}: {reason}"
        super().__init__(message, error_details)
        self.operation = operation


class ModelArchitecture(Enum):
    """Enumeration of supported model architectures."""
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    GRU = "gru"
    CONV1D = "conv1d"
    HYBRID = "hybrid"


@dataclass
class TrainingConfig:
    """Configuration for response-level training."""
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    loss_type: str = "response_level_cosine"
    loss_config: Optional[Dict[str, Any]] = None
    optimizer: str = "adam"
    metrics: List[str] = None


@dataclass
class ResponsePredictionResult:
    """Result of response prediction."""
    predicted_response: str
    confidence_score: float
    prediction_time: float
    attention_weights: Optional[np.ndarray] = None
    intermediate_states: Optional[List[np.ndarray]] = None


class ResponseInferenceModel:
    """
    Secondary model for complete response prediction from token embedding sequences.
    
    This model accepts embeddings from the reservoir/CNN pipeline and predicts
    complete responses using various neural architectures optimized for response-level
    learning rather than token-by-token generation.
    """
    
    def __init__(self,
                 input_embedding_dim: int,
                 max_sequence_length: int,
                 vocab_size: int,
                 tokenizer: Optional[StandardTokenizerWrapper] = None,
                 architecture: str = "transformer",
                 model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Response Inference Model.
        
        Args:
            input_embedding_dim: Dimension of input embeddings from CNN
            max_sequence_length: Maximum length of input sequences
            vocab_size: Size of vocabulary for output generation
            tokenizer: Optional tokenizer for text processing
            architecture: Model architecture type
            model_config: Optional configuration for model architecture
            
        Raises:
            ResponseInferenceError: If initialization fails
        """
        try:
            self.input_embedding_dim = input_embedding_dim
            self.max_sequence_length = max_sequence_length
            self.vocab_size = vocab_size
            self.tokenizer = tokenizer
            self.architecture = ModelArchitecture(architecture)
            
            # Default model configuration
            self.model_config = model_config or self._get_default_config()
            
            # Model components
            self._model = None
            self._encoder = None
            self._decoder = None
            self._compiled = False
            
            # Training state
            self._training_history = None
            self._is_trained = False
            
            # Performance tracking
            self._prediction_stats = {
                "total_predictions": 0,
                "successful_predictions": 0,
                "average_prediction_time": 0.0,
                "average_confidence": 0.0
            }
            
            logger.info(f"ResponseInferenceModel initialized with {architecture} architecture")
            
        except Exception as e:
            raise ResponseInferenceError(
                "initialization",
                f"Failed to initialize ResponseInferenceModel: {str(e)}",
                {
                    "input_embedding_dim": input_embedding_dim,
                    "max_sequence_length": max_sequence_length,
                    "vocab_size": vocab_size,
                    "architecture": architecture
                }
            )
    
    def create_model(self) -> tf.keras.Model:
        """
        Create the response inference model based on specified architecture.
        
        Returns:
            Compiled Keras model for response prediction
            
        Raises:
            ResponseInferenceError: If model creation fails
        """
        try:
            if self._model is not None:
                return self._model
            
            if self.architecture == ModelArchitecture.TRANSFORMER:
                self._model = self._create_transformer_model()
            elif self.architecture == ModelArchitecture.LSTM:
                self._model = self._create_lstm_model()
            elif self.architecture == ModelArchitecture.GRU:
                self._model = self._create_gru_model()
            elif self.architecture == ModelArchitecture.CONV1D:
                self._model = self._create_conv1d_model()
            elif self.architecture == ModelArchitecture.HYBRID:
                self._model = self._create_hybrid_model()
            else:
                raise ValueError(f"Unsupported architecture: {self.architecture}")
            
            # Compile the model
            self._compile_model()
            
            logger.info(f"Created {self.architecture.value} model with {self._model.count_params()} parameters")
            
            return self._model
            
        except Exception as e:
            raise ResponseInferenceError(
                "model_creation",
                f"Failed to create {self.architecture.value} model: {str(e)}",
                {"model_config": self.model_config}
            )
    
    def predict_response(self,
                        token_embedding_sequence: np.ndarray,
                        return_attention: bool = False,
                        return_intermediate: bool = False) -> ResponsePredictionResult:
        """
        Predict complete response from token embedding sequence.
        
        Args:
            token_embedding_sequence: Input embeddings (seq_len, embedding_dim) or (batch_size, seq_len, embedding_dim)
            return_attention: Whether to return attention weights
            return_intermediate: Whether to return intermediate states
            
        Returns:
            ResponsePredictionResult with predicted response and metadata
            
        Raises:
            ResponseInferenceError: If prediction fails
        """
        try:
            start_time = time.time()
            self._prediction_stats["total_predictions"] += 1
            
            # Ensure model is created
            if self._model is None:
                self.create_model()
            
            # Validate and prepare input
            processed_input = self._prepare_input(token_embedding_sequence)
            
            # Make prediction
            if return_attention or return_intermediate:
                # Use model with intermediate outputs
                prediction_outputs = self._predict_with_intermediates(processed_input)
                logits = prediction_outputs["logits"]
                attention_weights = prediction_outputs.get("attention_weights") if return_attention else None
                intermediate_states = prediction_outputs.get("intermediate_states") if return_intermediate else None
            else:
                # Standard prediction
                logits = self._model.predict(processed_input, verbose=0)
                attention_weights = None
                intermediate_states = None
            
            # Convert logits to response text
            predicted_response, confidence = self._decode_response(logits)
            
            prediction_time = time.time() - start_time
            
            # Update statistics
            self._update_prediction_stats(prediction_time, confidence, success=True)
            
            result = ResponsePredictionResult(
                predicted_response=predicted_response,
                confidence_score=confidence,
                prediction_time=prediction_time,
                attention_weights=attention_weights,
                intermediate_states=intermediate_states
            )
            
            logger.debug(f"Response predicted in {prediction_time:.3f}s with confidence {confidence:.3f}")
            
            return result
            
        except Exception as e:
            self._update_prediction_stats(0.0, 0.0, success=False)
            logger.exception("Response prediction failed")
            raise ResponseInferenceError(
                "response_prediction",
                f"Failed to predict response: {str(e)}",
                {
                    "input_shape": token_embedding_sequence.shape if token_embedding_sequence is not None else None,
                    "return_attention": return_attention,
                    "return_intermediate": return_intermediate
                }
            )
    
    def train_on_response_pairs(self,
                               input_embeddings: List[np.ndarray],
                               target_responses: List[str],
                               training_config: Optional[TrainingConfig] = None,
                               validation_data: Optional[Tuple[List[np.ndarray], List[str]]] = None) -> Dict[str, Any]:
        """
        Train the model on embedding-response pairs for response-level learning.
        
        Args:
            input_embeddings: List of input embedding sequences
            target_responses: List of target response strings
            training_config: Configuration for training
            validation_data: Optional validation data tuple
            
        Returns:
            Training history and metrics
            
        Raises:
            ResponseInferenceError: If training fails
        """
        try:
            # Use default config if not provided
            if training_config is None:
                training_config = TrainingConfig()
            
            # Ensure model is created
            if self._model is None:
                self.create_model()
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(input_embeddings, target_responses)
            
            # Prepare validation data if provided
            validation_data_processed = None
            if validation_data is not None:
                val_embeddings, val_responses = validation_data
                X_val, y_val = self._prepare_training_data(val_embeddings, val_responses)
                validation_data_processed = (X_val, y_val)
            
            # Setup callbacks
            callbacks = self._setup_training_callbacks(training_config)
            
            # Train the model
            logger.info(f"Starting training with {len(input_embeddings)} samples")
            
            history = self._model.fit(
                X_train, y_train,
                batch_size=training_config.batch_size,
                epochs=training_config.epochs,
                validation_data=validation_data_processed,
                validation_split=training_config.validation_split if validation_data_processed is None else 0.0,
                callbacks=callbacks,
                verbose=1
            )
            
            self._training_history = history.history
            self._is_trained = True
            
            # Calculate training metrics
            training_metrics = self._calculate_training_metrics(history.history)
            
            logger.info(f"Training completed. Final loss: {training_metrics['final_loss']:.4f}")
            
            return {
                "history": self._training_history,
                "metrics": training_metrics,
                "config": training_config
            }
            
        except Exception as e:
            logger.exception("Training failed")
            raise ResponseInferenceError(
                "training",
                f"Failed to train model: {str(e)}",
                {
                    "num_samples": len(input_embeddings) if input_embeddings else 0,
                    "training_config": training_config.__dict__ if training_config else None
                }
            )
    
    def evaluate_on_test_data(self,
                             test_embeddings: List[np.ndarray],
                             test_responses: List[str]) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_embeddings: List of test embedding sequences
            test_responses: List of test response strings
            
        Returns:
            Dictionary with evaluation metrics
            
        Raises:
            ResponseInferenceError: If evaluation fails
        """
        try:
            if self._model is None:
                raise ValueError("Model not created yet")
            
            # Prepare test data
            X_test, y_test = self._prepare_training_data(test_embeddings, test_responses)
            
            # Evaluate model
            evaluation_results = self._model.evaluate(X_test, y_test, verbose=0)
            
            # Create metrics dictionary
            metrics = {}
            if isinstance(evaluation_results, list):
                metric_names = self._model.metrics_names
                for name, value in zip(metric_names, evaluation_results):
                    metrics[name] = float(value)
            else:
                metrics["loss"] = float(evaluation_results)
            
            # Calculate additional response-level metrics
            response_metrics = self._calculate_response_metrics(test_embeddings, test_responses)
            metrics.update(response_metrics)
            
            logger.info(f"Evaluation completed. Test loss: {metrics.get('loss', 'N/A')}")
            
            return metrics
            
        except Exception as e:
            raise ResponseInferenceError(
                "evaluation",
                f"Failed to evaluate model: {str(e)}",
                {"num_test_samples": len(test_embeddings) if test_embeddings else 0}
            )
    
    def save_model(self, filepath: str) -> None:
        """
        Save the response inference model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            ResponseInferenceError: If saving fails
        """
        try:
            if self._model is None:
                raise ValueError("Model not created yet")
            
            # Save the main model
            self._model.save(filepath)
            
            # Save additional metadata
            metadata = {
                "input_embedding_dim": self.input_embedding_dim,
                "max_sequence_length": self.max_sequence_length,
                "vocab_size": self.vocab_size,
                "architecture": self.architecture.value,
                "model_config": self.model_config,
                "is_trained": self._is_trained,
                "training_history": self._training_history,
                "prediction_stats": self._prediction_stats
            }
            
            import json
            metadata_path = filepath.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            raise ResponseInferenceError(
                "model_saving",
                f"Failed to save model: {str(e)}",
                {"filepath": filepath}
            )
    
    def load_model(self, filepath: str) -> None:
        """
        Load a response inference model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            ResponseInferenceError: If loading fails
        """
        try:
            # Load the main model
            self._model = keras.models.load_model(filepath)
            self._compiled = True
            
            # Load metadata if available
            import json
            metadata_path = filepath.replace('.h5', '_metadata.json').replace('.keras', '_metadata.json')
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.input_embedding_dim = metadata.get("input_embedding_dim", self.input_embedding_dim)
                self.max_sequence_length = metadata.get("max_sequence_length", self.max_sequence_length)
                self.vocab_size = metadata.get("vocab_size", self.vocab_size)
                self.architecture = ModelArchitecture(metadata.get("architecture", self.architecture.value))
                self.model_config = metadata.get("model_config", self.model_config)
                self._is_trained = metadata.get("is_trained", False)
                self._training_history = metadata.get("training_history")
                self._prediction_stats = metadata.get("prediction_stats", self._prediction_stats)
                
            except FileNotFoundError:
                logger.warning(f"Metadata file not found: {metadata_path}")
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            raise ResponseInferenceError(
                "model_loading",
                f"Failed to load model: {str(e)}",
                {"filepath": filepath}
            )
    
    def get_model_summary(self) -> str:
        """
        Get summary of the model architecture.
        
        Returns:
            String representation of model architecture
        """
        if self._model is None:
            self.create_model()
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self._model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary
    
    def get_prediction_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about prediction performance.
        
        Returns:
            Dictionary with prediction statistics
        """
        stats = self._prediction_stats.copy()
        
        # Calculate success rate
        if stats["total_predictions"] > 0:
            stats["success_rate"] = stats["successful_predictions"] / stats["total_predictions"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset prediction statistics."""
        self._prediction_stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "average_prediction_time": 0.0,
            "average_confidence": 0.0
        }
        logger.info("Prediction statistics reset")
    
    # Private helper methods for model creation
    
    def _create_transformer_model(self) -> tf.keras.Model:
        """Create transformer-based model for response prediction."""
        # Input layer
        inputs = keras.Input(shape=(self.max_sequence_length, self.input_embedding_dim), name='embedding_input')
        
        # Positional encoding
        x = self._add_positional_encoding(inputs)
        
        # Multi-head attention layers
        num_heads = self.model_config.get("num_heads", 8)
        num_layers = self.model_config.get("num_layers", 6)
        d_model = self.model_config.get("d_model", self.input_embedding_dim)  # Use input dimension
        
        for i in range(num_layers):
            # Multi-head attention
            attention_output = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                name=f'attention_{i}'
            )(x, x)
            
            # Add & Norm
            x = layers.Add(name=f'add_attention_{i}')([x, attention_output])
            x = layers.LayerNormalization(name=f'norm_attention_{i}')(x)
            
            # Feed forward
            ff_output = layers.Dense(d_model * 4, activation='relu', name=f'ff1_{i}')(x)
            ff_output = layers.Dense(d_model, name=f'ff2_{i}')(ff_output)
            
            # Add & Norm
            x = layers.Add(name=f'add_ff_{i}')([x, ff_output])
            x = layers.LayerNormalization(name=f'norm_ff_{i}')(x)
        
        # Global pooling for sequence-level representation
        x = layers.GlobalAveragePooling1D(name='global_pool')(x)
        
        # Output layers for response generation
        x = layers.Dense(512, activation='relu', name='response_dense1')(x)
        x = layers.Dropout(0.3, name='response_dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='response_dense2')(x)
        x = layers.Dropout(0.2, name='response_dropout2')(x)
        
        # Output layer (logits for vocabulary)
        outputs = layers.Dense(self.vocab_size, activation='linear', name='response_logits')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='transformer_response_model')
    
    def _create_lstm_model(self) -> tf.keras.Model:
        """Create LSTM-based model for response prediction."""
        inputs = keras.Input(shape=(self.max_sequence_length, self.input_embedding_dim), name='embedding_input')
        
        # LSTM layers
        lstm_units = self.model_config.get("lstm_units", 256)
        num_layers = self.model_config.get("num_layers", 2)
        
        x = inputs
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            x = layers.LSTM(
                lstm_units,
                return_sequences=return_sequences,
                dropout=0.3,
                recurrent_dropout=0.3,
                name=f'lstm_{i}'
            )(x)
        
        # Dense layers for response generation
        x = layers.Dense(512, activation='relu', name='response_dense1')(x)
        x = layers.Dropout(0.4, name='response_dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='response_dense2')(x)
        x = layers.Dropout(0.3, name='response_dropout2')(x)
        
        # Output layer
        outputs = layers.Dense(self.vocab_size, activation='linear', name='response_logits')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='lstm_response_model')
    
    def _create_gru_model(self) -> tf.keras.Model:
        """Create GRU-based model for response prediction."""
        inputs = keras.Input(shape=(self.max_sequence_length, self.input_embedding_dim), name='embedding_input')
        
        # GRU layers
        gru_units = self.model_config.get("gru_units", 256)
        num_layers = self.model_config.get("num_layers", 2)
        
        x = inputs
        for i in range(num_layers):
            return_sequences = i < num_layers - 1
            x = layers.GRU(
                gru_units,
                return_sequences=return_sequences,
                dropout=0.3,
                recurrent_dropout=0.3,
                name=f'gru_{i}'
            )(x)
        
        # Dense layers for response generation
        x = layers.Dense(512, activation='relu', name='response_dense1')(x)
        x = layers.Dropout(0.4, name='response_dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='response_dense2')(x)
        x = layers.Dropout(0.3, name='response_dropout2')(x)
        
        # Output layer
        outputs = layers.Dense(self.vocab_size, activation='linear', name='response_logits')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='gru_response_model')
    
    def _create_conv1d_model(self) -> tf.keras.Model:
        """Create 1D CNN-based model for response prediction."""
        inputs = keras.Input(shape=(self.max_sequence_length, self.input_embedding_dim), name='embedding_input')
        
        # 1D Convolutional layers
        filters = self.model_config.get("filters", [128, 256, 512])
        kernel_sizes = self.model_config.get("kernel_sizes", [3, 3, 3])
        
        x = inputs
        for i, (num_filters, kernel_size) in enumerate(zip(filters, kernel_sizes)):
            x = layers.Conv1D(
                filters=num_filters,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
                name=f'conv1d_{i}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i}')(x)
            x = layers.MaxPooling1D(pool_size=2, name=f'pool_{i}')(x)
            x = layers.Dropout(0.25, name=f'dropout_{i}')(x)
        
        # Global pooling
        x = layers.GlobalMaxPooling1D(name='global_pool')(x)
        
        # Dense layers for response generation
        x = layers.Dense(512, activation='relu', name='response_dense1')(x)
        x = layers.Dropout(0.4, name='response_dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='response_dense2')(x)
        x = layers.Dropout(0.3, name='response_dropout2')(x)
        
        # Output layer
        outputs = layers.Dense(self.vocab_size, activation='linear', name='response_logits')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='conv1d_response_model')
    
    def _create_hybrid_model(self) -> tf.keras.Model:
        """Create hybrid model combining CNN and RNN for response prediction."""
        inputs = keras.Input(shape=(self.max_sequence_length, self.input_embedding_dim), name='embedding_input')
        
        # CNN branch
        cnn_branch = layers.Conv1D(128, 3, padding='same', activation='relu', name='cnn_conv1')(inputs)
        cnn_branch = layers.Conv1D(256, 3, padding='same', activation='relu', name='cnn_conv2')(cnn_branch)
        cnn_branch = layers.GlobalMaxPooling1D(name='cnn_pool')(cnn_branch)
        
        # RNN branch
        rnn_branch = layers.LSTM(256, return_sequences=False, name='rnn_lstm')(inputs)
        
        # Combine branches
        combined = layers.Concatenate(name='combine_branches')([cnn_branch, rnn_branch])
        
        # Final processing
        x = layers.Dense(512, activation='relu', name='response_dense1')(combined)
        x = layers.Dropout(0.4, name='response_dropout1')(x)
        
        x = layers.Dense(256, activation='relu', name='response_dense2')(x)
        x = layers.Dropout(0.3, name='response_dropout2')(x)
        
        # Output layer
        outputs = layers.Dense(self.vocab_size, activation='linear', name='response_logits')(x)
        
        return keras.Model(inputs=inputs, outputs=outputs, name='hybrid_response_model')    

    def _add_positional_encoding(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add positional encoding to input embeddings."""
        # Simplified positional encoding that avoids complex tensor operations
        # This is a basic implementation - could be enhanced for production use
        
        # Just return inputs for now - positional encoding can be added later
        # In a full implementation, this would add proper sinusoidal positional encodings
        return inputs
    
    def _compile_model(self) -> None:
        """Compile the model with appropriate loss and optimizer."""
        # Create loss function
        loss_type = self.model_config.get("loss_type", "response_level_cosine")
        loss_config = self.model_config.get("loss_config", {})
        
        if loss_type == "response_level_cosine":
            loss_fn = create_response_level_loss(**loss_config)
        elif loss_type == "cosine_similarity":
            loss_fn = create_cosine_similarity_loss(**loss_config)
        elif loss_type == "sparse_categorical_crossentropy":
            loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            loss_fn = "mse"
        
        # Create optimizer
        learning_rate = self.model_config.get("learning_rate", 0.001)
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Compile model
        self._model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=['mae', 'accuracy']
        )
        
        self._compiled = True
    
    def _prepare_input(self, token_embedding_sequence: np.ndarray) -> np.ndarray:
        """Prepare input for model prediction."""
        # Ensure correct shape
        if len(token_embedding_sequence.shape) == 2:
            # Add batch dimension
            token_embedding_sequence = np.expand_dims(token_embedding_sequence, axis=0)
        
        # Pad or truncate to max_sequence_length
        batch_size, seq_len, embed_dim = token_embedding_sequence.shape
        
        if seq_len > self.max_sequence_length:
            # Truncate
            token_embedding_sequence = token_embedding_sequence[:, :self.max_sequence_length, :]
        elif seq_len < self.max_sequence_length:
            # Pad with zeros
            padding = np.zeros((batch_size, self.max_sequence_length - seq_len, embed_dim))
            token_embedding_sequence = np.concatenate([token_embedding_sequence, padding], axis=1)
        
        return token_embedding_sequence
    
    def _predict_with_intermediates(self, processed_input: np.ndarray) -> Dict[str, np.ndarray]:
        """Make prediction with intermediate outputs."""
        # For now, return standard prediction
        # This could be extended to return attention weights and intermediate states
        logits = self._model.predict(processed_input, verbose=0)
        
        return {
            "logits": logits,
            "attention_weights": None,  # Would need model modification to extract
            "intermediate_states": None  # Would need model modification to extract
        }
    
    def _decode_response(self, logits: np.ndarray) -> Tuple[str, float]:
        """Decode logits to response text and confidence."""
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(logits).numpy()
        
        # Get most likely tokens
        predicted_tokens = np.argmax(probabilities, axis=-1)
        
        # Calculate confidence as max probability
        confidence = float(np.max(probabilities))
        
        # Convert tokens to text
        if self.tokenizer is not None:
            # Use tokenizer to decode
            if len(predicted_tokens.shape) > 1:
                predicted_tokens = predicted_tokens[0]  # Take first batch item
            
            # Filter out padding tokens (assuming 0 is padding)
            predicted_tokens = predicted_tokens[predicted_tokens != 0]
            
            try:
                response_text = self.tokenizer.decode(predicted_tokens.tolist())
            except Exception as e:
                logger.warning(f"Failed to decode tokens: {e}")
                response_text = f"[DECODED_TOKENS: {predicted_tokens.tolist()[:10]}...]"
        else:
            # Simple fallback - convert token IDs to string
            if len(predicted_tokens.shape) > 1:
                predicted_tokens = predicted_tokens[0]
            response_text = f"Generated response from tokens: {predicted_tokens[:10].tolist()}"
        
        return response_text, confidence
    
    def _prepare_training_data(self,
                              input_embeddings: List[np.ndarray],
                              target_responses: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from embeddings and responses."""
        # Prepare input embeddings
        X = []
        for embedding_seq in input_embeddings:
            # Ensure correct shape and padding
            if len(embedding_seq.shape) == 1:
                embedding_seq = np.expand_dims(embedding_seq, axis=0)
            
            seq_len, embed_dim = embedding_seq.shape
            
            if seq_len > self.max_sequence_length:
                embedding_seq = embedding_seq[:self.max_sequence_length, :]
            elif seq_len < self.max_sequence_length:
                padding = np.zeros((self.max_sequence_length - seq_len, embed_dim))
                embedding_seq = np.concatenate([embedding_seq, padding], axis=0)
            
            X.append(embedding_seq)
        
        X = np.array(X)
        
        # Prepare target responses
        y = []
        for response in target_responses:
            if self.tokenizer is not None:
                # Tokenize response
                try:
                    tokens = self.tokenizer.tokenize([response], padding=False, truncation=True)[0]
                    # For now, use first token as target (could be extended for sequence-to-sequence)
                    target_token = tokens[0] if tokens else 0
                except Exception as e:
                    logger.warning(f"Failed to tokenize response: {e}")
                    target_token = 0
            else:
                # Simple hash-based encoding
                target_token = hash(response) % self.vocab_size
            
            y.append(target_token)
        
        y = np.array(y)
        
        return X, y
    
    def _setup_training_callbacks(self, training_config: TrainingConfig) -> List[keras.callbacks.Callback]:
        """Setup training callbacks."""
        callbacks = []
        
        # Early stopping
        if training_config.early_stopping_patience > 0:
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=training_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
        
        # Reduce learning rate on plateau
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def _calculate_training_metrics(self, history: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate training metrics from history."""
        metrics = {}
        
        if 'loss' in history:
            metrics['final_loss'] = history['loss'][-1]
            metrics['best_loss'] = min(history['loss'])
        
        if 'val_loss' in history:
            metrics['final_val_loss'] = history['val_loss'][-1]
            metrics['best_val_loss'] = min(history['val_loss'])
        
        if 'accuracy' in history:
            metrics['final_accuracy'] = history['accuracy'][-1]
            metrics['best_accuracy'] = max(history['accuracy'])
        
        if 'val_accuracy' in history:
            metrics['final_val_accuracy'] = history['val_accuracy'][-1]
            metrics['best_val_accuracy'] = max(history['val_accuracy'])
        
        return metrics
    
    def _calculate_response_metrics(self,
                                   test_embeddings: List[np.ndarray],
                                   test_responses: List[str]) -> Dict[str, float]:
        """Calculate response-level metrics."""
        metrics = {}
        
        try:
            # Predict responses for test data
            predicted_responses = []
            confidences = []
            
            for embedding_seq in test_embeddings:
                result = self.predict_response(embedding_seq)
                predicted_responses.append(result.predicted_response)
                confidences.append(result.confidence_score)
            
            # Calculate average confidence
            metrics['average_confidence'] = float(np.mean(confidences))
            
            # Calculate response length statistics
            pred_lengths = [len(resp.split()) for resp in predicted_responses]
            true_lengths = [len(resp.split()) for resp in test_responses]
            
            metrics['avg_predicted_length'] = float(np.mean(pred_lengths))
            metrics['avg_true_length'] = float(np.mean(true_lengths))
            metrics['length_difference'] = abs(metrics['avg_predicted_length'] - metrics['avg_true_length'])
            
            # Simple text similarity (could be improved with better metrics)
            similarities = []
            for pred, true in zip(predicted_responses, test_responses):
                # Simple word overlap similarity
                pred_words = set(pred.lower().split())
                true_words = set(true.lower().split())
                
                if len(pred_words) == 0 and len(true_words) == 0:
                    similarity = 1.0
                elif len(pred_words) == 0 or len(true_words) == 0:
                    similarity = 0.0
                else:
                    intersection = len(pred_words.intersection(true_words))
                    union = len(pred_words.union(true_words))
                    similarity = intersection / union if union > 0 else 0.0
                
                similarities.append(similarity)
            
            metrics['average_similarity'] = float(np.mean(similarities))
            
        except Exception as e:
            logger.warning(f"Failed to calculate response metrics: {e}")
            metrics['average_confidence'] = 0.0
            metrics['average_similarity'] = 0.0
        
        return metrics
    
    def _update_prediction_stats(self, prediction_time: float, confidence: float, success: bool):
        """Update prediction statistics."""
        if success:
            self._prediction_stats["successful_predictions"] += 1
        
        # Update average prediction time
        current_avg = self._prediction_stats["average_prediction_time"]
        total_preds = self._prediction_stats["total_predictions"]
        
        if total_preds > 1:
            self._prediction_stats["average_prediction_time"] = (
                (current_avg * (total_preds - 1) + prediction_time) / total_preds
            )
        else:
            self._prediction_stats["average_prediction_time"] = prediction_time
        
        # Update average confidence
        current_avg_conf = self._prediction_stats["average_confidence"]
        if total_preds > 1:
            self._prediction_stats["average_confidence"] = (
                (current_avg_conf * (total_preds - 1) + confidence) / total_preds
            )
        else:
            self._prediction_stats["average_confidence"] = confidence
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration based on architecture."""
        base_config = {
            "learning_rate": 0.001,
            "loss_type": "sparse_categorical_crossentropy",  # Use simpler loss for now
            "loss_config": {}
        }
        
        if self.architecture == ModelArchitecture.TRANSFORMER:
            base_config.update({
                "num_heads": 8,
                "num_layers": 6,
                "d_model": self.input_embedding_dim  # Use input embedding dimension
            })
        elif self.architecture == ModelArchitecture.LSTM:
            base_config.update({
                "lstm_units": 256,
                "num_layers": 2
            })
        elif self.architecture == ModelArchitecture.GRU:
            base_config.update({
                "gru_units": 256,
                "num_layers": 2
            })
        elif self.architecture == ModelArchitecture.CONV1D:
            base_config.update({
                "filters": [128, 256, 512],
                "kernel_sizes": [3, 3, 3]
            })
        
        return base_config


# Convenience functions for easy usage

def create_response_inference_model(input_embedding_dim: int,
                                   max_sequence_length: int = 128,
                                   vocab_size: int = 10000,
                                   architecture: str = "transformer",
                                   tokenizer: Optional[StandardTokenizerWrapper] = None) -> ResponseInferenceModel:
    """
    Create a ResponseInferenceModel with standard configuration.
    
    Args:
        input_embedding_dim: Dimension of input embeddings from CNN
        max_sequence_length: Maximum length of input sequences
        vocab_size: Size of vocabulary for output generation
        architecture: Model architecture type
        tokenizer: Optional tokenizer for text processing
        
    Returns:
        Configured ResponseInferenceModel instance
    """
    return ResponseInferenceModel(
        input_embedding_dim=input_embedding_dim,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        architecture=architecture
    )


def create_transformer_response_model(input_embedding_dim: int,
                                     max_sequence_length: int = 128,
                                     vocab_size: int = 10000,
                                     num_heads: int = 8,
                                     num_layers: int = 6,
                                     tokenizer: Optional[StandardTokenizerWrapper] = None) -> ResponseInferenceModel:
    """
    Create a transformer-based ResponseInferenceModel.
    
    Args:
        input_embedding_dim: Dimension of input embeddings from CNN
        max_sequence_length: Maximum length of input sequences
        vocab_size: Size of vocabulary for output generation
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        tokenizer: Optional tokenizer for text processing
        
    Returns:
        Transformer-based ResponseInferenceModel
    """
    model_config = {
        "num_heads": num_heads,
        "num_layers": num_layers,
        "d_model": input_embedding_dim,
        "learning_rate": 0.001,
        "loss_type": "response_level_cosine"
    }
    
    return ResponseInferenceModel(
        input_embedding_dim=input_embedding_dim,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        architecture="transformer",
        model_config=model_config
    )


def create_lstm_response_model(input_embedding_dim: int,
                              max_sequence_length: int = 128,
                              vocab_size: int = 10000,
                              lstm_units: int = 256,
                              num_layers: int = 2,
                              tokenizer: Optional[StandardTokenizerWrapper] = None) -> ResponseInferenceModel:
    """
    Create an LSTM-based ResponseInferenceModel.
    
    Args:
        input_embedding_dim: Dimension of input embeddings from CNN
        max_sequence_length: Maximum length of input sequences
        vocab_size: Size of vocabulary for output generation
        lstm_units: Number of LSTM units
        num_layers: Number of LSTM layers
        tokenizer: Optional tokenizer for text processing
        
    Returns:
        LSTM-based ResponseInferenceModel
    """
    model_config = {
        "lstm_units": lstm_units,
        "num_layers": num_layers,
        "learning_rate": 0.001,
        "loss_type": "response_level_cosine"
    }
    
    return ResponseInferenceModel(
        input_embedding_dim=input_embedding_dim,
        max_sequence_length=max_sequence_length,
        vocab_size=vocab_size,
        tokenizer=tokenizer,
        architecture="lstm",
        model_config=model_config
    )