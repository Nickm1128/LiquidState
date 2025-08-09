#!/usr/bin/env python3
"""
CNN 3D Processor for System Message Integration.

This module provides a specialized processor for 3D CNN architectures that can
integrate system message embeddings with reservoir outputs for enhanced
conversational AI capabilities.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Union
from dataclasses import dataclass

from .cnn_architecture_factory import CNNArchitectureFactory, CNNArchitectureError
from .loss_functions import CNNLossCalculator, create_response_level_loss
from .system_message_processor import SystemMessageProcessor, SystemMessageContext
from .embedding_modifier_generator import EmbeddingModifierGenerator, ModifierOutput
from ..data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from ..utils.lsm_exceptions import LSMError, ModelError


class CNN3DProcessorError(ModelError):
    """Raised when 3D CNN processing fails."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {"operation": operation, "reason": reason}
        if details:
            error_details.update(details)
        
        message = f"3D CNN processing failed during {operation}: {reason}"
        super().__init__(message, error_details)
        self.operation = operation


@dataclass
class SystemContext:
    """Container for system message context and embeddings."""
    message: str
    embeddings: np.ndarray
    modifier_weights: Optional[np.ndarray] = None
    influence_strength: float = 1.0
    processing_mode: str = "3d_cnn"


@dataclass
class ProcessingResult:
    """Result of 3D CNN processing with system context."""
    output_embeddings: np.ndarray
    system_influence: float
    processing_time: float
    intermediate_features: Optional[Dict[str, np.ndarray]] = None
    modifier_details: Optional[Dict[str, Any]] = None
    tokenization_info: Optional[Dict[str, Any]] = None


class CNN3DProcessor:
    """
    Processor for 3D CNN architectures with system message integration.
    
    This class handles the creation and management of 3D CNN models that can
    process reservoir outputs alongside system message embeddings to produce
    context-aware response embeddings.
    """
    
    def __init__(self, 
                 reservoir_shape: Tuple[int, int, int, int],
                 system_embedding_dim: int,
                 output_embedding_dim: int,
                 model_config: Optional[Dict[str, Any]] = None,
                 tokenizer: Optional[StandardTokenizerWrapper] = None,
                 embedder: Optional[SinusoidalEmbedder] = None):
        """
        Initialize the CNN 3D Processor.
        
        Args:
            reservoir_shape: Shape of reservoir output (depth, height, width, channels)
            system_embedding_dim: Dimension of system message embeddings
            output_embedding_dim: Dimension of output embeddings
            model_config: Optional configuration for model architecture
            
        Raises:
            CNN3DProcessorError: If initialization fails
        """
        try:
            self.reservoir_shape = reservoir_shape
            self.system_embedding_dim = system_embedding_dim
            self.output_embedding_dim = output_embedding_dim
            
            # Default model configuration
            self.model_config = model_config or {
                "filters": [32, 64, 128],
                "kernel_sizes": [(3, 3, 3), (3, 3, 3), (3, 3, 3)],
                "dropout_rates": [0.25, 0.25, 0.5],
                "use_batch_norm": True,
                "activation": "relu",
                "use_attention": True,
                "attention_type": "spatial_channel"
            }
            
            # Initialize architecture factory
            self.architecture_factory = CNNArchitectureFactory()
            
            # Model will be created when needed
            self._model = None
            self._compiled = False
            
            # System context processing components
            self._system_processor = None
            self._embedding_modifier = None
            self._embedding_modifier_generator = None
            
            # Tokenization and embedding components
            self.tokenizer = tokenizer
            self.embedder = embedder
            
            # Training components
            self._training_model = None
            self._training_history = []
            
        except Exception as e:
            raise CNN3DProcessorError(
                "initialization",
                f"Failed to initialize CNN3DProcessor: {str(e)}",
                {
                    "reservoir_shape": reservoir_shape,
                    "system_embedding_dim": system_embedding_dim,
                    "output_embedding_dim": output_embedding_dim
                }
            )
    
    def create_model(self) -> tf.keras.Model:
        """
        Create the 3D CNN model with system message integration.
        
        Returns:
            Compiled Keras model with dual inputs (reservoir + system)
            
        Raises:
            CNN3DProcessorError: If model creation fails
        """
        try:
            if self._model is not None:
                return self._model
            
            # Create the base 3D CNN model
            self._model = self.architecture_factory.create_3d_cnn(
                input_shape=self.reservoir_shape,
                output_dim=self.output_embedding_dim,
                system_dim=self.system_embedding_dim,
                filters=self.model_config["filters"],
                kernel_sizes=self.model_config["kernel_sizes"],
                dropout_rates=self.model_config["dropout_rates"],
                use_batch_norm=self.model_config["use_batch_norm"],
                activation=self.model_config["activation"]
            )
            
            # Compile the model with enhanced response-level loss
            self._model = self.architecture_factory.compile_model(
                self._model,
                loss_type="response_level_cosine",
                learning_rate=0.001,
                metrics=['mae'],
                loss_config={
                    "sequence_weight": 1.0,
                    "coherence_weight": 0.1,
                    "diversity_weight": 0.05
                }
            )
            
            self._compiled = True
            
            return self._model
            
        except Exception as e:
            raise CNN3DProcessorError(
                "model_creation",
                f"Failed to create 3D CNN model: {str(e)}",
                {"model_config": self.model_config}
            )
    
    def create_system_processor(self) -> tf.keras.Model:
        """
        Create a dedicated system message processor.
        
        Returns:
            Keras model for processing system messages into embeddings
            
        Raises:
            CNN3DProcessorError: If system processor creation fails
        """
        try:
            if self._system_processor is not None:
                return self._system_processor
            
            # Input for raw system message (tokenized)
            system_input = keras.Input(shape=(None,), name='system_tokens')
            
            # Embedding layer for system tokens
            embedding_layer = layers.Embedding(
                input_dim=10000,  # Vocabulary size - should be configurable
                output_dim=128,
                mask_zero=True,
                name='system_embedding'
            )(system_input)
            
            # LSTM for sequence processing
            lstm_output = layers.LSTM(
                256,
                return_sequences=False,
                dropout=0.3,
                recurrent_dropout=0.3,
                name='system_lstm'
            )(embedding_layer)
            
            # Dense layers for final system embedding
            x = layers.Dense(512, activation='relu', name='system_dense1')(lstm_output)
            x = layers.Dropout(0.4, name='system_dropout1')(x)
            
            x = layers.Dense(256, activation='relu', name='system_dense2')(x)
            x = layers.Dropout(0.3, name='system_dropout2')(x)
            
            # Output system embedding
            system_embedding = layers.Dense(
                self.system_embedding_dim,
                activation='tanh',  # Bounded output for stability
                name='system_embedding_output'
            )(x)
            
            self._system_processor = keras.Model(
                inputs=system_input,
                outputs=system_embedding,
                name='system_message_processor'
            )
            
            # Compile system processor
            self._system_processor.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            return self._system_processor
            
        except Exception as e:
            raise CNN3DProcessorError(
                "system_processor_creation",
                f"Failed to create system processor: {str(e)}"
            )
    
    def create_enhanced_system_processor(self) -> SystemMessageProcessor:
        """
        Create an enhanced system message processor with proper tokenization.
        
        Returns:
            SystemMessageProcessor instance with tokenizer integration
            
        Raises:
            CNN3DProcessorError: If system processor creation fails
        """
        try:
            if self._system_processor is not None and isinstance(self._system_processor, SystemMessageProcessor):
                return self._system_processor
            
            # Create system processor with tokenizer if available
            if self.tokenizer is not None:
                from .system_message_processor import SystemMessageConfig
                
                config = SystemMessageConfig(
                    max_length=512,
                    embedding_dim=self.system_embedding_dim,
                    add_special_tokens=True,
                    validate_format=True
                )
                
                self._system_processor = SystemMessageProcessor(
                    tokenizer=self.tokenizer,
                    config=config
                )
            else:
                # Create a basic tokenizer for fallback
                try:
                    from ..data.tokenization import StandardTokenizerWrapper
                    fallback_tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2', max_length=512)
                    
                    from .system_message_processor import SystemMessageConfig
                    config = SystemMessageConfig(
                        max_length=512,
                        embedding_dim=self.system_embedding_dim,
                        add_special_tokens=True,
                        validate_format=True
                    )
                    
                    self._system_processor = SystemMessageProcessor(
                        tokenizer=fallback_tokenizer,
                        config=config
                    )
                except Exception:
                    # If we can't create a tokenizer, raise an error
                    raise ValueError("No tokenizer available and cannot create fallback tokenizer")
            
            return self._system_processor
            
        except Exception as e:
            raise CNN3DProcessorError(
                "enhanced_system_processor_creation",
                f"Failed to create enhanced system processor: {str(e)}"
            )
    
    def create_embedding_modifier_generator(self) -> EmbeddingModifierGenerator:
        """
        Create an embedding modifier generator for advanced system influence.
        
        Returns:
            EmbeddingModifierGenerator instance
            
        Raises:
            CNN3DProcessorError: If embedding modifier generator creation fails
        """
        try:
            if self._embedding_modifier_generator is not None:
                return self._embedding_modifier_generator
            
            # Ensure system processor exists
            if self._system_processor is None:
                self.create_enhanced_system_processor()
            
            # Create modifier generator with system processor integration
            from .embedding_modifier_generator import ModifierConfig
            
            config = ModifierConfig(
                system_embedding_dim=self.system_embedding_dim,
                base_embedding_dim=self.output_embedding_dim,
                modifier_types=["attention", "feature", "output", "scaling"],
                hidden_dims=[512, 256, 128],
                dropout_rates=[0.3, 0.2, 0.1]
            )
            
            self._embedding_modifier_generator = EmbeddingModifierGenerator(
                config=config,
                system_processor=self._system_processor
            )
            
            return self._embedding_modifier_generator
            
        except Exception as e:
            raise CNN3DProcessorError(
                "embedding_modifier_generator_creation",
                f"Failed to create embedding modifier generator: {str(e)}"
            )

    def create_embedding_modifier(self) -> tf.keras.Model:
        """
        Create embedding modifier that generates modifiers from system context.
        
        Returns:
            Keras model that generates embedding modifiers
            
        Raises:
            CNN3DProcessorError: If embedding modifier creation fails
        """
        try:
            if self._embedding_modifier is not None:
                return self._embedding_modifier
            
            # Input system embedding
            system_input = keras.Input(
                shape=(self.system_embedding_dim,),
                name='system_embedding_input'
            )
            
            # Generate modifiers for different parts of the pipeline
            x = layers.Dense(512, activation='relu', name='modifier_dense1')(system_input)
            x = layers.Dropout(0.3, name='modifier_dropout1')(x)
            
            x = layers.Dense(256, activation='relu', name='modifier_dense2')(x)
            x = layers.Dropout(0.2, name='modifier_dropout2')(x)
            
            # Generate multiple modifier outputs
            # Attention modifiers
            attention_modifiers = layers.Dense(
                64,
                activation='sigmoid',
                name='attention_modifiers'
            )(x)
            
            # Feature modifiers
            feature_modifiers = layers.Dense(
                128,
                activation='tanh',
                name='feature_modifiers'
            )(x)
            
            # Output scaling modifiers
            output_modifiers = layers.Dense(
                self.output_embedding_dim,
                activation='tanh',
                name='output_modifiers'
            )(x)
            
            # Combine all modifiers
            combined_modifiers = layers.Concatenate(name='combined_modifiers')([
                attention_modifiers,
                feature_modifiers,
                output_modifiers
            ])
            
            self._embedding_modifier = keras.Model(
                inputs=system_input,
                outputs={
                    'attention_modifiers': attention_modifiers,
                    'feature_modifiers': feature_modifiers,
                    'output_modifiers': output_modifiers,
                    'combined_modifiers': combined_modifiers
                },
                name='embedding_modifier_generator'
            )
            
            # Compile embedding modifier
            self._embedding_modifier.compile(
                optimizer='adam',
                loss={
                    'attention_modifiers': 'mse',
                    'feature_modifiers': 'mse',
                    'output_modifiers': 'cosine_similarity',
                    'combined_modifiers': 'mse'
                },
                metrics=['mae']
            )
            
            return self._embedding_modifier
            
        except Exception as e:
            raise CNN3DProcessorError(
                "embedding_modifier_creation",
                f"Failed to create embedding modifier: {str(e)}"
            )
    
    def process_reservoir_output_with_modifiers(self,
                                              reservoir_output: np.ndarray,
                                              system_message: str,
                                              influence_strength: float = 1.0,
                                              use_advanced_modifiers: bool = True) -> ProcessingResult:
        """
        Enhanced processing with advanced system modifiers and proper tokenization.
        
        Args:
            reservoir_output: Raw reservoir output
            system_message: System message text
            influence_strength: How strongly system message should influence output
            use_advanced_modifiers: Whether to use EmbeddingModifierGenerator
            
        Returns:
            ProcessingResult with enhanced system-aware embeddings
            
        Raises:
            CNN3DProcessorError: If processing fails
        """
        try:
            import time
            start_time = time.time()
            
            # Ensure model is created
            if self._model is None:
                self.create_model()
            
            # Process system message with enhanced processor
            if self._system_processor is None:
                self.create_enhanced_system_processor()
            
            # Handle empty system messages
            if not system_message or system_message.strip() == "":
                # Create default system context for empty messages
                system_context = SystemContext(
                    message="",
                    embeddings=np.zeros(self.system_embedding_dim, dtype=np.float32),
                    influence_strength=0.0  # No influence for empty messages
                )
                
                # Process through base 3D CNN with zero influence
                base_result = self.process_with_system_context(reservoir_output, system_context)
                base_result.processing_time = time.time() - start_time
                return base_result
            
            # Process system message to get context
            system_context_obj = self._system_processor.process_system_message(
                system_message,
                validate=True,
                create_embeddings=True
            )
            
            # Create SystemContext for compatibility
            system_context = SystemContext(
                message=system_message,
                embeddings=system_context_obj.embeddings,
                influence_strength=influence_strength
            )
            
            # Validate inputs
            self._validate_reservoir_output(reservoir_output)
            self._validate_system_context(system_context)
            
            # Process through base 3D CNN
            base_result = self.process_with_system_context(reservoir_output, system_context)
            
            # Apply advanced modifiers if requested
            if use_advanced_modifiers:
                if self._embedding_modifier_generator is None:
                    self.create_embedding_modifier_generator()
                
                # Generate advanced modifiers
                modifiers = self._embedding_modifier_generator.generate_modifiers(
                    system_message,
                    influence_strength
                )
                
                # Apply modifiers to base output
                enhanced_embeddings = self._embedding_modifier_generator.apply_modifiers_to_embeddings(
                    base_result.output_embeddings,
                    modifiers,
                    application_mode="hybrid"
                )
                
                # Update result with enhanced embeddings
                base_result.output_embeddings = enhanced_embeddings
                base_result.modifier_details = {
                    "modifier_types": list(modifiers.metadata.get("model_outputs", [])),
                    "confidence_scores": modifiers.confidence_scores,
                    "generation_time": modifiers.generation_time
                }
            
            # Add tokenization info if available
            if self.tokenizer is not None:
                base_result.tokenization_info = {
                    "tokenizer_type": self.tokenizer.tokenizer_name,
                    "vocab_size": self.tokenizer.get_vocab_size(),
                    "system_message_length": len(system_message),
                    "tokenized_length": len(self.tokenizer.encode_single(system_message))
                }
            
            processing_time = time.time() - start_time
            base_result.processing_time = processing_time
            
            return base_result
            
        except Exception as e:
            raise CNN3DProcessorError(
                "enhanced_processing",
                f"Failed in enhanced processing pipeline: {str(e)}",
                {
                    "system_message": system_message[:100] if system_message else None,
                    "influence_strength": influence_strength,
                    "use_advanced_modifiers": use_advanced_modifiers
                }
            )

    def process_with_system_context(self,
                                   reservoir_output: np.ndarray,
                                   system_context: SystemContext) -> ProcessingResult:
        """
        Process reservoir output with system message context.
        
        Args:
            reservoir_output: Output from LSM reservoir (batch_size, depth, height, width, channels)
            system_context: System message context and embeddings
            
        Returns:
            ProcessingResult with output embeddings and metadata
            
        Raises:
            CNN3DProcessorError: If processing fails
        """
        try:
            import time
            start_time = time.time()
            
            # Ensure model is created
            if self._model is None:
                self.create_model()
            
            # Validate inputs
            self._validate_reservoir_output(reservoir_output)
            self._validate_system_context(system_context)
            
            # Prepare system embeddings
            system_embeddings = system_context.embeddings
            if len(system_embeddings.shape) == 1:
                system_embeddings = np.expand_dims(system_embeddings, axis=0)
            
            # Match batch size with reservoir output
            batch_size = reservoir_output.shape[0]
            if system_embeddings.shape[0] != batch_size:
                # Repeat system embeddings to match batch size
                system_embeddings = np.repeat(system_embeddings, batch_size, axis=0)
            
            # Apply influence strength
            system_embeddings = system_embeddings * system_context.influence_strength
            
            # Process through 3D CNN
            output_embeddings = self._model.predict([
                reservoir_output,
                system_embeddings
            ], verbose=0)
            
            # Calculate system influence (how much the system context affected output)
            # This is a simplified metric - could be more sophisticated
            system_influence = float(np.mean(np.abs(system_embeddings)))
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                output_embeddings=output_embeddings,
                system_influence=system_influence,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise CNN3DProcessorError(
                "system_context_processing",
                f"Failed to process with system context: {str(e)}",
                {
                    "reservoir_shape": reservoir_output.shape if reservoir_output is not None else None,
                    "system_context": system_context.message if system_context else None
                }
            )
    
    def integrate_embedding_modifiers(self,
                                    base_output: np.ndarray,
                                    system_context: SystemContext) -> np.ndarray:
        """
        Apply embedding modifiers to base CNN output.
        
        Args:
            base_output: Base output from 3D CNN
            system_context: System context with modifier information
            
        Returns:
            Modified embeddings with system influence applied
            
        Raises:
            CNN3DProcessorError: If modifier integration fails
        """
        try:
            # Ensure embedding modifier is created
            if self._embedding_modifier is None:
                self.create_embedding_modifier()
            
            # Generate modifiers from system context
            system_embeddings = system_context.embeddings
            if len(system_embeddings.shape) == 1:
                system_embeddings = np.expand_dims(system_embeddings, axis=0)
            
            # Match batch size with base output
            batch_size = base_output.shape[0]
            if system_embeddings.shape[0] != batch_size:
                # Repeat system embeddings to match batch size
                system_embeddings = np.repeat(system_embeddings, batch_size, axis=0)
            
            modifiers = self._embedding_modifier.predict(system_embeddings, verbose=0)
            
            # Apply output modifiers to base output
            output_modifiers = modifiers['output_modifiers']
            
            # Element-wise modification with influence strength
            modified_output = base_output + (
                output_modifiers * system_context.influence_strength
            )
            
            # Optional: Apply feature modifiers if available
            if system_context.modifier_weights is not None:
                feature_modifiers = modifiers['feature_modifiers']
                # Apply custom modifier weights
                weighted_modifiers = feature_modifiers * system_context.modifier_weights[:feature_modifiers.shape[-1]]
                modified_output = modified_output + weighted_modifiers
            
            return modified_output
            
        except Exception as e:
            raise CNN3DProcessorError(
                "modifier_integration",
                f"Failed to integrate embedding modifiers: {str(e)}",
                {
                    "base_output_shape": base_output.shape if base_output is not None else None,
                    "system_context": system_context.message if system_context else None
                }
            )
    
    def create_training_model(self) -> tf.keras.Model:
        """
        Create a training model for system-aware response generation.
        
        Returns:
            Keras model configured for training with system context
            
        Raises:
            CNN3DProcessorError: If training model creation fails
        """
        try:
            if self._training_model is not None:
                return self._training_model
            
            # Create base model if not exists
            if self._model is None:
                self.create_model()
            
            # Create training-specific architecture
            # Input layers
            reservoir_input = keras.Input(
                shape=self.reservoir_shape,
                name='reservoir_input'
            )
            system_input = keras.Input(
                shape=(self.system_embedding_dim,),
                name='system_input'
            )
            target_response_input = keras.Input(
                shape=(None,),  # Variable length for response tokens
                name='target_response'
            )
            
            # Use the base model for feature extraction
            base_features = self._model([reservoir_input, system_input])
            
            # Add response-level training head
            response_dense = layers.Dense(
                1024, activation='relu', name='response_dense1'
            )(base_features)
            response_dense = layers.Dropout(0.3, name='response_dropout1')(response_dense)
            
            response_dense = layers.Dense(
                512, activation='relu', name='response_dense2'
            )(response_dense)
            response_dense = layers.Dropout(0.2, name='response_dropout2')(response_dense)
            
            # Output layer for response prediction
            response_output = layers.Dense(
                self.output_embedding_dim,
                activation='tanh',
                name='response_output'
            )(response_dense)
            
            # Create training model
            self._training_model = keras.Model(
                inputs=[reservoir_input, system_input, target_response_input],
                outputs=response_output,
                name='system_aware_training_model'
            )
            
            # Compile with response-level loss
            self._training_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='cosine_similarity',
                metrics=['mae', 'mse']
            )
            
            return self._training_model
            
        except Exception as e:
            raise CNN3DProcessorError(
                "training_model_creation",
                f"Failed to create training model: {str(e)}"
            )
    
    def train_system_aware_model(self,
                                training_data: List[Dict[str, Any]],
                                validation_data: Optional[List[Dict[str, Any]]] = None,
                                epochs: int = 50,
                                batch_size: int = 16,
                                callbacks: Optional[List] = None) -> Dict[str, Any]:
        """
        Train the system-aware response generation model.
        
        Args:
            training_data: List of training samples with reservoir_output, system_message, target_response
            validation_data: Optional validation data
            epochs: Number of training epochs
            batch_size: Training batch size
            callbacks: Optional Keras callbacks
            
        Returns:
            Training history and metrics
            
        Raises:
            CNN3DProcessorError: If training fails
        """
        try:
            if not training_data:
                raise ValueError("Training data cannot be empty")
            
            # Ensure training model is created
            if self._training_model is None:
                self.create_training_model()
            
            # Ensure system processor is available
            if self._system_processor is None:
                self.create_enhanced_system_processor()
            
            # Prepare training data
            X_reservoir, X_system, y_target = self._prepare_training_data(training_data)
            
            # Prepare validation data if provided
            X_val_reservoir, X_val_system, y_val_target = None, None, None
            if validation_data:
                X_val_reservoir, X_val_system, y_val_target = self._prepare_training_data(validation_data)
            
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
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath='best_system_aware_model.h5',
                        monitor='val_loss' if validation_data else 'loss',
                        save_best_only=True
                    )
                ]
            
            # Train the model
            history = self._training_model.fit(
                [X_reservoir, X_system, X_reservoir],  # Dummy target input for model structure
                y_target,
                validation_data=([X_val_reservoir, X_val_system, X_val_reservoir], y_val_target) if validation_data else None,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Store training history
            self._training_history.append(history.history)
            
            # Calculate training metrics
            final_loss = history.history['loss'][-1]
            best_val_loss = min(history.history.get('val_loss', [final_loss]))
            
            training_metrics = {
                "final_loss": final_loss,
                "best_val_loss": best_val_loss,
                "epochs_trained": len(history.history['loss']),
                "training_samples": len(training_data),
                "validation_samples": len(validation_data) if validation_data else 0
            }
            
            return {
                "history": history.history,
                "metrics": training_metrics,
                "model_summary": self.get_training_model_summary()
            }
            
        except Exception as e:
            raise CNN3DProcessorError(
                "system_aware_training",
                f"Failed to train system-aware model: {str(e)}",
                {
                    "training_samples": len(training_data) if training_data else 0,
                    "epochs": epochs,
                    "batch_size": batch_size
                }
            )
    
    def _prepare_training_data(self, data: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare training data for system-aware training.
        
        Args:
            data: List of training samples
            
        Returns:
            Tuple of (reservoir_outputs, system_embeddings, target_embeddings)
        """
        reservoir_outputs = []
        system_embeddings = []
        target_embeddings = []
        
        for sample in data:
            # Extract reservoir output
            reservoir_output = sample.get('reservoir_output')
            if reservoir_output is None:
                raise ValueError("Training sample missing 'reservoir_output'")
            reservoir_outputs.append(reservoir_output)
            
            # Process system message
            system_message = sample.get('system_message', '')
            if self._system_processor is None:
                self.create_enhanced_system_processor()
            
            system_context = self._system_processor.process_system_message(
                system_message,
                validate=True,
                create_embeddings=True
            )
            system_embeddings.append(system_context.embeddings)
            
            # Process target response
            target_response = sample.get('target_response', '')
            if self.tokenizer and self.embedder:
                # Use proper tokenization and embedding
                tokens = self.tokenizer.encode_single(target_response)
                target_embedding = np.mean(self.embedder.embed(tokens), axis=0)
            else:
                # Fallback to simple embedding
                target_embedding = np.random.randn(self.output_embedding_dim).astype(np.float32)
            
            target_embeddings.append(target_embedding)
        
        return (
            np.array(reservoir_outputs),
            np.array(system_embeddings),
            np.array(target_embeddings)
        )

    def process_reservoir_output_with_system(self,
                                           reservoir_output: np.ndarray,
                                           system_message: str,
                                           system_tokens: Optional[np.ndarray] = None,
                                           influence_strength: float = 1.0) -> ProcessingResult:
        """
        Complete processing pipeline from reservoir output to system-aware embeddings.
        
        Args:
            reservoir_output: Raw reservoir output
            system_message: System message text
            system_tokens: Pre-tokenized system message (optional)
            influence_strength: How strongly system message should influence output
            
        Returns:
            ProcessingResult with system-aware embeddings
            
        Raises:
            CNN3DProcessorError: If processing pipeline fails
        """
        try:
            # Process system message if tokens not provided
            if system_tokens is None:
                # Use proper tokenization
                system_tokens = self._tokenize_text(system_message)
            
            # Create system processor if needed
            if self._system_processor is None:
                self.create_system_processor()
            
            # Generate system embeddings
            system_embeddings = self._system_processor.predict(
                np.expand_dims(system_tokens, axis=0),
                verbose=0
            )[0]  # Remove batch dimension
            
            # Create system context
            system_context = SystemContext(
                message=system_message,
                embeddings=system_embeddings,
                influence_strength=influence_strength
            )
            
            # Process with system context
            result = self.process_with_system_context(reservoir_output, system_context)
            
            # Apply embedding modifiers for enhanced system integration
            modified_embeddings = self.integrate_embedding_modifiers(
                result.output_embeddings,
                system_context
            )
            
            # Update result with modified embeddings
            result.output_embeddings = modified_embeddings
            
            return result
            
        except Exception as e:
            raise CNN3DProcessorError(
                "complete_processing",
                f"Failed in complete processing pipeline: {str(e)}",
                {
                    "system_message": system_message,
                    "influence_strength": influence_strength
                }
            )
    
    def get_model_summary(self) -> str:
        """
        Get summary of the 3D CNN model architecture.
        
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
    
    def get_training_model_summary(self) -> str:
        """
        Get summary of the training model architecture.
        
        Returns:
            String representation of training model architecture
        """
        if self._training_model is None:
            return "Training model not created yet"
        
        import io
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        try:
            self._training_model.summary()
            summary = buffer.getvalue()
        finally:
            sys.stdout = old_stdout
        
        return summary
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about processing operations.
        
        Returns:
            Dictionary with processing statistics
        """
        stats = {
            "model_created": self._model is not None,
            "training_model_created": self._training_model is not None,
            "system_processor_created": self._system_processor is not None,
            "embedding_modifier_generator_created": self._embedding_modifier_generator is not None,
            "tokenizer_available": self.tokenizer is not None,
            "embedder_available": self.embedder is not None,
            "training_history_length": len(self._training_history)
        }
        
        if self.tokenizer:
            stats["tokenizer_type"] = self.tokenizer.tokenizer_name
            stats["vocab_size"] = self.tokenizer.get_vocab_size()
        
        if self.embedder:
            stats["embedding_dim"] = self.embedder.embedding_dim
            stats["embedder_fitted"] = self.embedder._is_fitted
        
        return stats
    
    def set_tokenizer(self, tokenizer: StandardTokenizerWrapper) -> None:
        """
        Set or update the tokenizer.
        
        Args:
            tokenizer: StandardTokenizerWrapper instance
        """
        self.tokenizer = tokenizer
        
        # Update system processor if it exists
        if self._system_processor is not None:
            self._system_processor = None  # Force recreation with new tokenizer
    
    def set_embedder(self, embedder: SinusoidalEmbedder) -> None:
        """
        Set or update the embedder.
        
        Args:
            embedder: SinusoidalEmbedder instance
        """
        self.embedder = embedder
    
    def save_model(self, filepath: str) -> None:
        """
        Save the 3D CNN model to disk.
        
        Args:
            filepath: Path to save the model
            
        Raises:
            CNN3DProcessorError: If saving fails
        """
        try:
            if self._model is None:
                raise ValueError("Model not created yet")
            
            self._model.save(filepath)
            
        except Exception as e:
            raise CNN3DProcessorError(
                "model_saving",
                f"Failed to save model: {str(e)}",
                {"filepath": filepath}
            )
    
    def load_model(self, filepath: str) -> None:
        """
        Load a 3D CNN model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Raises:
            CNN3DProcessorError: If loading fails
        """
        try:
            self._model = keras.models.load_model(filepath)
            self._compiled = True
            
        except Exception as e:
            raise CNN3DProcessorError(
                "model_loading",
                f"Failed to load model: {str(e)}",
                {"filepath": filepath}
            )
    
    # Private helper methods
    
    def _validate_reservoir_output(self, reservoir_output: np.ndarray) -> None:
        """Validate reservoir output shape and content."""
        if reservoir_output is None:
            raise ValueError("Reservoir output cannot be None")
        
        if len(reservoir_output.shape) != 5:  # batch_size + 4D reservoir shape
            raise ValueError(f"Expected 5D reservoir output, got {len(reservoir_output.shape)}D")
        
        expected_shape = (None,) + self.reservoir_shape
        actual_shape = reservoir_output.shape
        
        if actual_shape[1:] != self.reservoir_shape:
            raise ValueError(f"Reservoir output shape mismatch. Expected {expected_shape}, got {actual_shape}")
    
    def _validate_system_context(self, system_context: SystemContext) -> None:
        """Validate system context object."""
        if system_context is None:
            raise ValueError("System context cannot be None")
        
        if system_context.embeddings is None:
            raise ValueError("System embeddings cannot be None")
        
        if len(system_context.embeddings.shape) == 0:
            raise ValueError("System embeddings must have at least 1 dimension")
        
        # Check embedding dimension
        embedding_dim = system_context.embeddings.shape[-1]
        if embedding_dim != self.system_embedding_dim:
            raise ValueError(f"System embedding dimension mismatch. Expected {self.system_embedding_dim}, got {embedding_dim}")
    
    def _tokenize_text(self, text: str) -> np.ndarray:
        """
        Tokenize text using proper tokenizer or fallback method.
        
        Args:
            text: Text to tokenize
            
        Returns:
            Array of token IDs
        """
        if self.tokenizer is not None:
            # Use proper tokenizer
            tokens = self.tokenizer.encode_single(text, add_special_tokens=True)
            return np.array(tokens, dtype=np.int32)
        else:
            # Fallback to simple tokenization
            words = text.lower().split()
            # Simple hash-based tokenization (not suitable for production)
            tokens = [hash(word) % 10000 for word in words]
            return np.array(tokens, dtype=np.int32)
    
    def _simple_tokenize(self, text: str) -> np.ndarray:
        """
        Simple tokenization for system messages.
        Deprecated: Use _tokenize_text instead.
        """
        return self._tokenize_text(text)


# Convenience functions for easy usage

def create_cnn_3d_processor(reservoir_shape: Tuple[int, int, int, int],
                           system_embedding_dim: int = 256,
                           output_embedding_dim: int = 512,
                           tokenizer: Optional[StandardTokenizerWrapper] = None,
                           embedder: Optional[SinusoidalEmbedder] = None) -> CNN3DProcessor:
    """
    Create a CNN3DProcessor with default configuration.
    
    Args:
        reservoir_shape: Shape of reservoir output (depth, height, width, channels)
        system_embedding_dim: Dimension of system message embeddings
        output_embedding_dim: Dimension of output embeddings
        tokenizer: Optional StandardTokenizerWrapper for proper tokenization
        embedder: Optional SinusoidalEmbedder for optimized embeddings
        
    Returns:
        Configured CNN3DProcessor instance
    """
    return CNN3DProcessor(
        reservoir_shape=reservoir_shape,
        system_embedding_dim=system_embedding_dim,
        output_embedding_dim=output_embedding_dim,
        tokenizer=tokenizer,
        embedder=embedder
    )


def create_system_aware_processor(window_size: int = 64,
                                 channels: int = 1,
                                 system_dim: int = 256,
                                 output_dim: int = 512,
                                 tokenizer: Optional[StandardTokenizerWrapper] = None,
                                 embedder: Optional[SinusoidalEmbedder] = None) -> CNN3DProcessor:
    """
    Create a system-aware 3D CNN processor with standard configuration.
    
    Args:
        window_size: Size of the reservoir window
        channels: Number of channels in reservoir output
        system_dim: Dimension of system embeddings
        output_dim: Dimension of output embeddings
        tokenizer: Optional StandardTokenizerWrapper for proper tokenization
        embedder: Optional SinusoidalEmbedder for optimized embeddings
        
    Returns:
        Configured CNN3DProcessor for system-aware processing
    """
    reservoir_shape = (window_size, window_size, window_size, channels)
    
    return CNN3DProcessor(
        reservoir_shape=reservoir_shape,
        system_embedding_dim=system_dim,
        output_embedding_dim=output_dim,
        tokenizer=tokenizer,
        embedder=embedder,
        model_config={
            "filters": [64, 128, 256],
            "kernel_sizes": [(3, 3, 3), (3, 3, 3), (3, 3, 3)],
            "dropout_rates": [0.2, 0.3, 0.4],
            "use_batch_norm": True,
            "activation": "relu",
            "use_attention": True,
            "attention_type": "spatial_channel"
        }
    )