#!/usr/bin/env python3
"""
Embedding Modifier Generator for System Influence.

This module provides a dedicated model to generate embedding modifiers from system
prompts, implementing training methods for modifier generation using backpropagation
and methods to apply modifiers to base embeddings.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
import time

from .system_message_processor import SystemMessageProcessor, SystemMessageContext
from ..data.tokenization import StandardTokenizerWrapper
from ..utils.lsm_exceptions import LSMError, ModelError
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class EmbeddingModifierError(ModelError):
    """Raised when embedding modifier generation fails."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {"operation": operation, "reason": reason}
        if details:
            error_details.update(details)
        
        message = f"Embedding modifier generation failed during {operation}: {reason}"
        super().__init__(message, error_details)
        self.operation = operation


@dataclass
class ModifierConfig:
    """Configuration for embedding modifier generation."""
    system_embedding_dim: int = 256
    base_embedding_dim: int = 512
    modifier_types: List[str] = None
    hidden_dims: List[int] = None
    dropout_rates: List[float] = None
    activation: str = "relu"
    use_batch_norm: bool = True
    learning_rate: float = 0.001
    
    def __post_init__(self):
        if self.modifier_types is None:
            self.modifier_types = ["attention", "feature", "output", "scaling"]
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        if self.dropout_rates is None:
            self.dropout_rates = [0.3, 0.2, 0.1]


@dataclass
class ModifierOutput:
    """Container for generated embedding modifiers."""
    attention_modifiers: np.ndarray
    feature_modifiers: np.ndarray
    output_modifiers: np.ndarray
    scaling_modifiers: np.ndarray
    combined_modifiers: np.ndarray
    confidence_scores: Dict[str, float]
    generation_time: float
    metadata: Dict[str, Any]


@dataclass
class TrainingBatch:
    """Training batch for embedding modifier generation."""
    system_embeddings: np.ndarray
    target_modifiers: Dict[str, np.ndarray]
    base_embeddings: Optional[np.ndarray] = None
    influence_strengths: Optional[np.ndarray] = None


class EmbeddingModifierGenerator:
    """
    Model to generate embedding modifiers from system prompts.
    
    This class creates and trains neural networks that can generate various types
    of embedding modifiers based on system message embeddings, allowing for
    fine-grained control over how system messages influence the model's behavior.
    """
    
    def __init__(self, 
                 config: Optional[ModifierConfig] = None,
                 system_processor: Optional[SystemMessageProcessor] = None):
        """
        Initialize the EmbeddingModifierGenerator.
        
        Args:
            config: Configuration for modifier generation
            system_processor: Optional SystemMessageProcessor for integration
            
        Raises:
            EmbeddingModifierError: If initialization fails
        """
        try:
            self.config = config or ModifierConfig()
            self.system_processor = system_processor
            
            # Model components
            self._modifier_model = None
            self._compiled = False
            
            # Training state
            self._training_history = []
            self._best_loss = float('inf')
            self._training_step = 0
            
            # Statistics
            self._generation_count = 0
            self._total_generation_time = 0.0
            self._modifier_type_usage = {mod_type: 0 for mod_type in self.config.modifier_types}
            
            logger.info(f"Initialized EmbeddingModifierGenerator with config: "
                       f"system_dim={self.config.system_embedding_dim}, "
                       f"base_dim={self.config.base_embedding_dim}")
            
        except Exception as e:
            raise EmbeddingModifierError(
                "initialization",
                f"Failed to initialize EmbeddingModifierGenerator: {str(e)}",
                {"config": config.__dict__ if config else None}
            )
    
    def create_modifier_model(self) -> tf.keras.Model:
        """
        Create the neural network model for generating embedding modifiers.
        
        Returns:
            Compiled Keras model for modifier generation
            
        Raises:
            EmbeddingModifierError: If model creation fails
        """
        try:
            if self._modifier_model is not None:
                return self._modifier_model
            
            # Input layer for system embeddings
            system_input = keras.Input(
                shape=(self.config.system_embedding_dim,),
                name='system_embedding_input'
            )
            
            # Shared feature extraction layers
            x = system_input
            for i, (hidden_dim, dropout_rate) in enumerate(
                zip(self.config.hidden_dims, self.config.dropout_rates)
            ):
                x = layers.Dense(
                    hidden_dim,
                    activation=self.config.activation,
                    name=f'shared_dense_{i+1}'
                )(x)
                
                if self.config.use_batch_norm:
                    x = layers.BatchNormalization(name=f'shared_bn_{i+1}')(x)
                
                x = layers.Dropout(dropout_rate, name=f'shared_dropout_{i+1}')(x)
            
            # Separate heads for different modifier types
            modifier_outputs = {}
            
            # Attention modifiers - control attention weights
            if "attention" in self.config.modifier_types:
                attention_branch = layers.Dense(
                    128, activation='relu', name='attention_dense'
                )(x)
                attention_modifiers = layers.Dense(
                    64,
                    activation='sigmoid',  # Sigmoid for attention weights
                    name='attention_modifiers'
                )(attention_branch)
                modifier_outputs['attention_modifiers'] = attention_modifiers
            
            # Feature modifiers - modify intermediate features
            if "feature" in self.config.modifier_types:
                feature_branch = layers.Dense(
                    256, activation='relu', name='feature_dense'
                )(x)
                feature_modifiers = layers.Dense(
                    128,
                    activation='tanh',  # Tanh for bidirectional modification
                    name='feature_modifiers'
                )(feature_branch)
                modifier_outputs['feature_modifiers'] = feature_modifiers
            
            # Output modifiers - directly modify final embeddings
            if "output" in self.config.modifier_types:
                output_branch = layers.Dense(
                    self.config.base_embedding_dim, 
                    activation='relu', 
                    name='output_dense'
                )(x)
                output_modifiers = layers.Dense(
                    self.config.base_embedding_dim,
                    activation='tanh',  # Tanh for bidirectional modification
                    name='output_modifiers'
                )(output_branch)
                modifier_outputs['output_modifiers'] = output_modifiers
            
            # Scaling modifiers - control influence strength
            if "scaling" in self.config.modifier_types:
                scaling_branch = layers.Dense(
                    64, activation='relu', name='scaling_dense'
                )(x)
                scaling_modifiers = layers.Dense(
                    32,
                    activation='sigmoid',  # Sigmoid for scaling factors
                    name='scaling_modifiers'
                )(scaling_branch)
                modifier_outputs['scaling_modifiers'] = scaling_modifiers
            
            # Combined modifiers for unified processing
            all_modifiers = list(modifier_outputs.values())
            if len(all_modifiers) > 1:
                combined = layers.Concatenate(name='concatenate_modifiers')(all_modifiers)
            else:
                combined = all_modifiers[0]
            
            # Final processing layer for combined modifiers
            combined_processed = layers.Dense(
                256, activation='relu', name='combined_dense'
            )(combined)
            combined_modifiers = layers.Dense(
                128, activation='tanh', name='combined_modifiers'
            )(combined_processed)
            
            modifier_outputs['combined_modifiers'] = combined_modifiers
            
            # Create the model
            self._modifier_model = keras.Model(
                inputs=system_input,
                outputs=modifier_outputs,
                name='embedding_modifier_generator'
            )
            
            logger.info(f"Created modifier model with {len(modifier_outputs)} output heads")
            
            return self._modifier_model
            
        except Exception as e:
            raise EmbeddingModifierError(
                "model_creation",
                f"Failed to create modifier model: {str(e)}",
                {"config": self.config.__dict__}
            )
    
    def compile_model(self, 
                     loss_weights: Optional[Dict[str, float]] = None,
                     custom_losses: Optional[Dict[str, str]] = None) -> None:
        """
        Compile the modifier model with appropriate losses and metrics.
        
        Args:
            loss_weights: Optional weights for different loss components
            custom_losses: Optional custom loss functions for different outputs
            
        Raises:
            EmbeddingModifierError: If compilation fails
        """
        try:
            if self._modifier_model is None:
                self.create_modifier_model()
            
            # Default loss weights
            if loss_weights is None:
                loss_weights = {
                    'attention_modifiers': 1.0,
                    'feature_modifiers': 1.0,
                    'output_modifiers': 2.0,  # Higher weight for output modifiers
                    'scaling_modifiers': 0.5,
                    'combined_modifiers': 1.5
                }
            
            # Default loss functions
            if custom_losses is None:
                custom_losses = {
                    'attention_modifiers': 'binary_crossentropy',  # For attention weights
                    'feature_modifiers': 'mse',
                    'output_modifiers': 'cosine_similarity',  # For embedding similarity
                    'scaling_modifiers': 'mse',
                    'combined_modifiers': 'mse'
                }
            
            # Filter losses and weights based on actual model outputs
            model_outputs = [output.name.split('/')[0] for output in self._modifier_model.outputs]
            filtered_losses = {k: v for k, v in custom_losses.items() if k in model_outputs}
            filtered_weights = {k: v for k, v in loss_weights.items() if k in model_outputs}
            
            # Compile the model
            self._modifier_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
                loss=filtered_losses,
                loss_weights=filtered_weights,
                metrics={
                    output_name: ['mae'] for output_name in model_outputs
                }
            )
            
            self._compiled = True
            logger.info(f"Compiled modifier model with losses: {list(filtered_losses.keys())}")
            
        except Exception as e:
            raise EmbeddingModifierError(
                "model_compilation",
                f"Failed to compile modifier model: {str(e)}",
                {"loss_weights": loss_weights, "custom_losses": custom_losses}
            )
    
    def generate_modifiers(self, 
                          system_prompt: str,
                          influence_strength: float = 1.0) -> ModifierOutput:
        """
        Generate embedding modifiers from a system prompt.
        
        Args:
            system_prompt: System message text
            influence_strength: Strength of system influence (0.0 to 2.0)
            
        Returns:
            ModifierOutput with generated modifiers
            
        Raises:
            EmbeddingModifierError: If modifier generation fails
        """
        try:
            start_time = time.time()
            
            # Ensure model is ready
            if self._modifier_model is None or not self._compiled:
                self.create_modifier_model()
                self.compile_model()
            
            # Process system prompt to get embeddings
            if self.system_processor is None:
                raise ValueError("SystemMessageProcessor required for prompt processing")
            
            system_context = self.system_processor.process_system_message(
                system_prompt, 
                validate=True, 
                create_embeddings=True
            )
            
            system_embeddings = system_context.embeddings
            if len(system_embeddings.shape) == 1:
                system_embeddings = np.expand_dims(system_embeddings, axis=0)
            
            # Generate modifiers
            modifier_predictions = self._modifier_model.predict(
                system_embeddings, 
                verbose=0
            )
            
            # Extract individual modifier types
            attention_modifiers = modifier_predictions.get('attention_modifiers', np.array([]))
            feature_modifiers = modifier_predictions.get('feature_modifiers', np.array([]))
            output_modifiers = modifier_predictions.get('output_modifiers', np.array([]))
            scaling_modifiers = modifier_predictions.get('scaling_modifiers', np.array([]))
            combined_modifiers = modifier_predictions.get('combined_modifiers', np.array([]))
            
            # Apply influence strength scaling
            if influence_strength != 1.0:
                attention_modifiers = attention_modifiers * influence_strength
                feature_modifiers = feature_modifiers * influence_strength
                output_modifiers = output_modifiers * influence_strength
                # Scaling modifiers are adjusted differently
                scaling_modifiers = scaling_modifiers * np.sqrt(influence_strength)
                combined_modifiers = combined_modifiers * influence_strength
            
            # Calculate confidence scores
            confidence_scores = self._calculate_confidence_scores(
                modifier_predictions, system_context
            )
            
            generation_time = time.time() - start_time
            
            # Update statistics
            self._generation_count += 1
            self._total_generation_time += generation_time
            for mod_type in self.config.modifier_types:
                if f"{mod_type}_modifiers" in modifier_predictions:
                    self._modifier_type_usage[mod_type] += 1
            
            # Create metadata
            metadata = {
                "system_prompt_length": len(system_prompt),
                "system_format": system_context.parsed_content.get("format", "unknown"),
                "influence_strength": influence_strength,
                "model_outputs": list(modifier_predictions.keys()),
                "generation_id": self._generation_count
            }
            
            return ModifierOutput(
                attention_modifiers=attention_modifiers[0] if len(attention_modifiers) > 0 else np.array([]),
                feature_modifiers=feature_modifiers[0] if len(feature_modifiers) > 0 else np.array([]),
                output_modifiers=output_modifiers[0] if len(output_modifiers) > 0 else np.array([]),
                scaling_modifiers=scaling_modifiers[0] if len(scaling_modifiers) > 0 else np.array([]),
                combined_modifiers=combined_modifiers[0] if len(combined_modifiers) > 0 else np.array([]),
                confidence_scores=confidence_scores,
                generation_time=generation_time,
                metadata=metadata
            )
            
        except Exception as e:
            raise EmbeddingModifierError(
                "modifier_generation",
                f"Failed to generate modifiers: {str(e)}",
                {"system_prompt": system_prompt[:100] if system_prompt else None}
            )
    
    def apply_modifiers_to_embeddings(self,
                                    base_embeddings: np.ndarray,
                                    modifiers: ModifierOutput,
                                    application_mode: str = "additive") -> np.ndarray:
        """
        Apply generated modifiers to base embeddings.
        
        Args:
            base_embeddings: Base embeddings to modify
            modifiers: Generated modifiers to apply
            application_mode: How to apply modifiers ("additive", "multiplicative", "hybrid")
            
        Returns:
            Modified embeddings with system influence applied
            
        Raises:
            EmbeddingModifierError: If modifier application fails
        """
        try:
            if base_embeddings is None or len(base_embeddings) == 0:
                raise ValueError("Base embeddings cannot be None or empty")
            
            # Ensure base_embeddings is 2D (batch_size, embedding_dim)
            if len(base_embeddings.shape) == 1:
                base_embeddings = np.expand_dims(base_embeddings, axis=0)
            
            batch_size, embedding_dim = base_embeddings.shape
            modified_embeddings = base_embeddings.copy()
            
            # Apply output modifiers (most direct influence)
            if len(modifiers.output_modifiers) > 0:
                output_mods = modifiers.output_modifiers
                
                # Ensure modifier dimensions match
                if len(output_mods) != embedding_dim:
                    # Resize modifiers to match embedding dimension
                    if len(output_mods) > embedding_dim:
                        output_mods = output_mods[:embedding_dim]
                    else:
                        # Pad with zeros or repeat pattern
                        padding_needed = embedding_dim - len(output_mods)
                        if len(output_mods) > 0:
                            # Repeat pattern
                            repeats = (padding_needed // len(output_mods)) + 1
                            extended_mods = np.tile(output_mods, repeats)
                            output_mods = extended_mods[:embedding_dim]
                        else:
                            output_mods = np.zeros(embedding_dim)
                
                # Apply modifiers based on mode
                if application_mode == "additive":
                    modified_embeddings = modified_embeddings + output_mods
                elif application_mode == "multiplicative":
                    # Use sigmoid to ensure positive scaling factors
                    scaling_factors = 1.0 + np.tanh(output_mods) * 0.5
                    modified_embeddings = modified_embeddings * scaling_factors
                elif application_mode == "hybrid":
                    # Combine additive and multiplicative
                    additive_component = output_mods * 0.5
                    multiplicative_component = 1.0 + np.tanh(output_mods) * 0.3
                    modified_embeddings = (modified_embeddings + additive_component) * multiplicative_component
            
            # Apply scaling modifiers for fine-tuned influence
            if len(modifiers.scaling_modifiers) > 0:
                scaling_factors = modifiers.scaling_modifiers
                
                # Apply scaling to different parts of the embedding
                if len(scaling_factors) >= 4:  # Enough for quadrant scaling
                    quarter_size = embedding_dim // 4
                    for i in range(4):
                        start_idx = i * quarter_size
                        end_idx = start_idx + quarter_size if i < 3 else embedding_dim
                        if i < len(scaling_factors):
                            modified_embeddings[:, start_idx:end_idx] *= scaling_factors[i]
                else:
                    # Global scaling
                    global_scale = np.mean(scaling_factors)
                    modified_embeddings *= global_scale
            
            # Apply feature modifiers for intermediate processing
            if len(modifiers.feature_modifiers) > 0:
                feature_mods = modifiers.feature_modifiers
                
                # Apply feature modifications to specific regions
                if len(feature_mods) >= embedding_dim // 4:
                    # Apply to first quarter of embeddings
                    quarter_size = embedding_dim // 4
                    feature_region = feature_mods[:quarter_size]
                    modified_embeddings[:, :quarter_size] += feature_region * 0.3
            
            # Normalize if embeddings become too large
            norms = np.linalg.norm(modified_embeddings, axis=1, keepdims=True)
            max_norm = np.max(norms)
            if max_norm > 10.0:  # Prevent explosion
                modified_embeddings = modified_embeddings / (max_norm / 5.0)
            
            return modified_embeddings
            
        except Exception as e:
            raise EmbeddingModifierError(
                "modifier_application",
                f"Failed to apply modifiers to embeddings: {str(e)}",
                {
                    "base_shape": base_embeddings.shape if base_embeddings is not None else None,
                    "application_mode": application_mode
                }
            )
    
    def train_modifier_model(self,
                           training_data: List[TrainingBatch],
                           validation_data: Optional[List[TrainingBatch]] = None,
                           epochs: int = 50,
                           batch_size: int = 32,
                           callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Train the modifier generation model using backpropagation.
        
        Args:
            training_data: List of training batches
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: Optional Keras callbacks
            
        Returns:
            Training history and metrics
            
        Raises:
            EmbeddingModifierError: If training fails
        """
        try:
            if not training_data:
                raise ValueError("Training data cannot be empty")
            
            # Ensure model is ready
            if self._modifier_model is None or not self._compiled:
                self.create_modifier_model()
                self.compile_model()
            
            # Prepare training data
            X_train, y_train = self._prepare_training_data(training_data)
            X_val, y_val = None, None
            
            if validation_data:
                X_val, y_val = self._prepare_training_data(validation_data)
            
            # Default callbacks
            if callbacks is None:
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss' if validation_data else 'loss',
                        patience=10,
                        restore_best_weights=True
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss' if validation_data else 'loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-6
                    )
                ]
            
            # Train the model
            history = self._modifier_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if validation_data else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Update training state
            self._training_history.append(history.history)
            final_loss = history.history['loss'][-1]
            if final_loss < self._best_loss:
                self._best_loss = final_loss
            self._training_step += epochs
            
            # Calculate training metrics
            training_metrics = {
                "final_loss": final_loss,
                "best_loss": self._best_loss,
                "epochs_trained": epochs,
                "total_training_steps": self._training_step,
                "training_samples": len(training_data),
                "validation_samples": len(validation_data) if validation_data else 0
            }
            
            logger.info(f"Training completed. Final loss: {final_loss:.4f}, "
                       f"Best loss: {self._best_loss:.4f}")
            
            return {
                "history": history.history,
                "metrics": training_metrics,
                "model_summary": self.get_model_summary()
            }
            
        except Exception as e:
            raise EmbeddingModifierError(
                "model_training",
                f"Failed to train modifier model: {str(e)}",
                {
                    "training_samples": len(training_data) if training_data else 0,
                    "epochs": epochs,
                    "batch_size": batch_size
                }
            )
    
    def integrate_with_cnn3d_processor(self, cnn_processor) -> None:
        """
        Integrate with existing CNN3DProcessor embedding modifier functionality.
        
        Args:
            cnn_processor: CNN3DProcessor instance to integrate with
            
        Raises:
            EmbeddingModifierError: If integration fails
        """
        try:
            # Store reference to CNN processor
            self._cnn_processor = cnn_processor
            
            # Override CNN processor's embedding modifier with our model
            if hasattr(cnn_processor, '_embedding_modifier'):
                logger.info("Replacing CNN3DProcessor embedding modifier with EmbeddingModifierGenerator")
                # We'll use our model instead of the CNN's built-in modifier
                cnn_processor._use_external_modifier = True
                cnn_processor._external_modifier = self
            
            # Add method to CNN processor for using our modifier
            def generate_and_apply_modifiers(system_context, base_output):
                """Generate and apply modifiers using EmbeddingModifierGenerator."""
                modifiers = self.generate_modifiers(
                    system_context.message,
                    system_context.influence_strength
                )
                return self.apply_modifiers_to_embeddings(
                    base_output,
                    modifiers,
                    application_mode="hybrid"
                )
            
            cnn_processor.generate_and_apply_modifiers = generate_and_apply_modifiers
            
            logger.info("Successfully integrated EmbeddingModifierGenerator with CNN3DProcessor")
            
        except Exception as e:
            raise EmbeddingModifierError(
                "cnn_integration",
                f"Failed to integrate with CNN3DProcessor: {str(e)}"
            )
    
    def get_model_summary(self) -> str:
        """
        Get summary of the modifier model architecture.
        
        Returns:
            String representation of model architecture
        """
        if self._modifier_model is None:
            return "Model not created yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self._modifier_model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about modifier generation.
        
        Returns:
            Dictionary with generation statistics
        """
        avg_generation_time = (
            self._total_generation_time / max(self._generation_count, 1)
        )
        
        return {
            "total_generations": self._generation_count,
            "total_generation_time": self._total_generation_time,
            "average_generation_time": avg_generation_time,
            "modifier_type_usage": self._modifier_type_usage.copy(),
            "training_steps": self._training_step,
            "best_loss": self._best_loss,
            "model_compiled": self._compiled
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the modifier model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            EmbeddingModifierError: If saving fails
        """
        try:
            if self._modifier_model is None:
                raise ValueError("Model not created yet")
            
            self._modifier_model.save(filepath)
            logger.info(f"Saved modifier model to {filepath}")
            
        except Exception as e:
            raise EmbeddingModifierError(
                "model_saving",
                f"Failed to save model: {str(e)}",
                {"filepath": filepath}
            )
    
    def load_model(self, filepath: str) -> None:
        """
        Load a modifier model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            EmbeddingModifierError: If loading fails
        """
        try:
            self._modifier_model = keras.models.load_model(filepath)
            self._compiled = True
            logger.info(f"Loaded modifier model from {filepath}")
            
        except Exception as e:
            raise EmbeddingModifierError(
                "model_loading",
                f"Failed to load model: {str(e)}",
                {"filepath": filepath}
            )
    
    # Private helper methods
    
    def _calculate_confidence_scores(self,
                                   predictions: Dict[str, np.ndarray],
                                   system_context: SystemMessageContext) -> Dict[str, float]:
        """Calculate confidence scores for generated modifiers."""
        confidence_scores = {}
        
        for modifier_type, values in predictions.items():
            if len(values) == 0:
                confidence_scores[modifier_type] = 0.0
                continue
            
            # Calculate confidence based on prediction variance and magnitude
            values_flat = values.flatten()
            
            # Variance-based confidence (lower variance = higher confidence)
            variance = np.var(values_flat)
            variance_confidence = 1.0 / (1.0 + variance)
            
            # Magnitude-based confidence (moderate magnitudes are more confident)
            mean_magnitude = np.mean(np.abs(values_flat))
            magnitude_confidence = 1.0 - min(mean_magnitude, 1.0)
            
            # System context confidence (higher complexity = lower confidence)
            context_confidence = 1.0 - system_context.parsed_content.get("complexity_score", 0.5)
            
            # Combined confidence score
            combined_confidence = (
                variance_confidence * 0.4 +
                magnitude_confidence * 0.3 +
                context_confidence * 0.3
            )
            
            confidence_scores[modifier_type] = float(np.clip(combined_confidence, 0.0, 1.0))
        
        return confidence_scores
    
    def _prepare_training_data(self, 
                             training_batches: List[TrainingBatch]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data for model fitting."""
        # Collect all system embeddings
        all_system_embeddings = []
        all_targets = {mod_type: [] for mod_type in self.config.modifier_types}
        all_targets["combined_modifiers"] = []
        
        for batch in training_batches:
            all_system_embeddings.append(batch.system_embeddings)
            
            # Collect target modifiers
            for mod_type in self.config.modifier_types:
                target_key = f"{mod_type}_modifiers"
                if target_key in batch.target_modifiers:
                    all_targets[mod_type].append(batch.target_modifiers[target_key])
                else:
                    # Create dummy targets if not provided
                    dummy_shape = self._get_modifier_shape(mod_type)
                    dummy_target = np.zeros(dummy_shape)
                    all_targets[mod_type].append(dummy_target)
            
            # Combined modifiers target
            if "combined_modifiers" in batch.target_modifiers:
                all_targets["combined_modifiers"].append(batch.target_modifiers["combined_modifiers"])
            else:
                # Create dummy combined target
                all_targets["combined_modifiers"].append(np.zeros(128))
        
        # Convert to numpy arrays
        X = np.vstack(all_system_embeddings)
        y = {}
        
        for mod_type, targets in all_targets.items():
            if targets:
                y[f"{mod_type}_modifiers"] = np.vstack(targets)
        
        return X, y
    
    def _get_modifier_shape(self, modifier_type: str) -> Tuple[int, ...]:
        """Get the expected shape for a modifier type."""
        shapes = {
            "attention": (64,),
            "feature": (128,),
            "output": (self.config.base_embedding_dim,),
            "scaling": (32,)
        }
        return shapes.get(modifier_type, (64,))


# Convenience functions for easy usage

def create_embedding_modifier_generator(
    system_embedding_dim: int = 256,
    base_embedding_dim: int = 512,
    tokenizer_name: str = "gpt2"
) -> EmbeddingModifierGenerator:
    """
    Create an EmbeddingModifierGenerator with default configuration.
    
    Args:
        system_embedding_dim: Dimension of system embeddings
        base_embedding_dim: Dimension of base embeddings to modify
        tokenizer_name: Tokenizer to use for system message processing
        
    Returns:
        Configured EmbeddingModifierGenerator instance
    """
    # Create system processor
    tokenizer = StandardTokenizerWrapper(tokenizer_name, max_length=512)
    system_processor = SystemMessageProcessor(tokenizer)
    
    # Create config
    config = ModifierConfig(
        system_embedding_dim=system_embedding_dim,
        base_embedding_dim=base_embedding_dim
    )
    
    return EmbeddingModifierGenerator(config, system_processor)


def create_training_batch_from_prompts(
    system_prompts: List[str],
    target_behaviors: List[np.ndarray],
    system_processor: SystemMessageProcessor
) -> List[TrainingBatch]:
    """
    Create training batches from system prompts and target behaviors.
    
    Args:
        system_prompts: List of system prompt texts
        target_behaviors: List of target behavior embeddings
        system_processor: SystemMessageProcessor for embedding generation
        
    Returns:
        List of TrainingBatch objects
    """
    training_batches = []
    
    for prompt, target in zip(system_prompts, target_behaviors):
        # Process system prompt
        context = system_processor.process_system_message(prompt)
        system_embedding = context.embeddings
        
        # Create target modifiers (simplified - in practice would be more sophisticated)
        target_modifiers = {
            "output_modifiers": target,
            "combined_modifiers": target[:128] if len(target) >= 128 else np.pad(target, (0, max(0, 128 - len(target))))
        }
        
        batch = TrainingBatch(
            system_embeddings=np.expand_dims(system_embedding, axis=0),
            target_modifiers=target_modifiers
        )
        
        training_batches.append(batch)
    
    return training_batches