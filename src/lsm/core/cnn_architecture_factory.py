#!/usr/bin/env python3
"""
CNN Architecture Factory for LSM Training Pipeline Enhancement.

This module provides a factory class for creating different CNN architectures
including 2D CNNs, 3D CNNs for system message integration, and residual CNNs
with various configurations and attention mechanisms.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from enum import Enum

from ..utils.lsm_exceptions import LSMError, ModelError
from .loss_functions import (
    CosineSimilarityLoss, ResponseLevelCosineLoss, CNNLossCalculator,
    LossType, create_cosine_similarity_loss, create_response_level_loss
)


class CNNArchitectureError(ModelError):
    """Raised when CNN architecture creation fails."""
    
    def __init__(self, architecture_type: str, reason: str, config: Optional[Dict[str, Any]] = None):
        details = {"architecture_type": architecture_type, "reason": reason}
        if config:
            details["config"] = config
        
        message = f"Failed to create {architecture_type} CNN architecture: {reason}"
        super().__init__(message, details)
        self.architecture_type = architecture_type


class CNNType(Enum):
    """Enumeration of supported CNN types."""
    CNN_2D = "2d"
    CNN_3D = "3d"
    RESIDUAL_2D = "residual_2d"
    RESIDUAL_3D = "residual_3d"
    MULTI_SCALE_2D = "multi_scale_2d"


class AttentionType(Enum):
    """Enumeration of supported attention mechanisms."""
    NONE = "none"
    SPATIAL = "spatial"
    CHANNEL = "channel"
    SPATIAL_CHANNEL = "spatial_channel"
    SELF_ATTENTION = "self_attention"


class LossType(Enum):
    """Enumeration of supported loss functions."""
    MSE = "mse"
    COSINE_SIMILARITY = "cosine_similarity"
    HUBER = "huber"


class SpatialAttentionBlock(layers.Layer):
    """Spatial attention mechanism for focusing on important regions."""
    
    def __init__(self, kernel_size: int = 7, **kwargs):
        super(SpatialAttentionBlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        
    def build(self, input_shape):
        self.conv_attention = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding='same',
            activation='sigmoid',
            name=f'{self.name}_spatial_conv'
        )
        super(SpatialAttentionBlock, self).build(input_shape)
        
    def call(self, inputs):
        # Compute spatial attention weights
        attention_weights = self.conv_attention(inputs)
        # Apply attention
        attended_features = inputs * attention_weights
        return attended_features, attention_weights


class ChannelAttentionBlock(layers.Layer):
    """Channel attention mechanism for feature channel weighting."""
    
    def __init__(self, reduction_ratio: int = 16, **kwargs):
        super(ChannelAttentionBlock, self).__init__(**kwargs)
        self.reduction_ratio = reduction_ratio
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        self.global_max_pool = layers.GlobalMaxPooling2D()
        
        self.dense1 = layers.Dense(
            channels // self.reduction_ratio,
            activation='relu',
            name=f'{self.name}_dense1'
        )
        self.dense2 = layers.Dense(
            channels,
            activation='sigmoid',
            name=f'{self.name}_dense2'
        )
        
        super(ChannelAttentionBlock, self).build(input_shape)
        
    def call(self, inputs):
        # Average pooling path
        avg_pool = self.global_avg_pool(inputs)
        avg_pool = layers.Reshape((1, 1, -1))(avg_pool)
        avg_pool = self.dense1(avg_pool)
        avg_pool = self.dense2(avg_pool)
        
        # Max pooling path
        max_pool = self.global_max_pool(inputs)
        max_pool = layers.Reshape((1, 1, -1))(max_pool)
        max_pool = self.dense1(max_pool)
        max_pool = self.dense2(max_pool)
        
        # Combine and apply attention
        attention_weights = avg_pool + max_pool
        attended_features = inputs * attention_weights
        
        return attended_features, attention_weights


class CNNArchitectureFactory:
    """
    Factory class for creating different CNN architectures for LSM training.
    
    Supports 2D and 3D CNNs with various configurations including attention
    mechanisms, residual connections, and different loss functions.
    """
    
    def __init__(self):
        """Initialize the CNN architecture factory."""
        self._supported_types = [cnn_type.value for cnn_type in CNNType]
        self._supported_attention = [att_type.value for att_type in AttentionType]
        self._supported_losses = [loss_type.value for loss_type in LossType]
    
    def create_2d_cnn(self, 
                      input_shape: Tuple[int, int, int], 
                      output_dim: int,
                      use_attention: bool = True,
                      attention_type: str = "spatial",
                      filters: List[int] = None,
                      kernel_sizes: List[int] = None,
                      dropout_rates: List[float] = None,
                      use_batch_norm: bool = True,
                      activation: str = "relu") -> tf.keras.Model:
        """
        Create a 2D CNN model for standard processing.
        
        Args:
            input_shape: Shape of input (height, width, channels)
            output_dim: Dimension of output embeddings
            use_attention: Whether to use attention mechanism
            attention_type: Type of attention ("spatial", "channel", "spatial_channel")
            filters: List of filter counts for each conv layer
            kernel_sizes: List of kernel sizes for each conv layer
            dropout_rates: List of dropout rates for each layer
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use
            
        Returns:
            Compiled Keras model
            
        Raises:
            CNNArchitectureError: If model creation fails
        """
        try:
            # Default configurations
            if filters is None:
                filters = [32, 64, 128]
            if kernel_sizes is None:
                kernel_sizes = [5, 3, 3]
            if dropout_rates is None:
                dropout_rates = [0.25, 0.25, 0.5]
            
            # Validate inputs
            self._validate_2d_inputs(input_shape, output_dim, attention_type)
            
            # Input layer
            inputs = keras.Input(shape=input_shape, name='waveform_input')
            x = inputs
            
            # Apply attention if requested
            if use_attention:
                x = self._add_attention_block(x, attention_type)
            
            # Convolutional blocks
            for i, (num_filters, kernel_size, dropout_rate) in enumerate(
                zip(filters, kernel_sizes, dropout_rates)
            ):
                x = layers.Conv2D(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation=activation,
                    name=f'conv2d_{i+1}'
                )(x)
                
                if use_batch_norm:
                    x = layers.BatchNormalization(name=f'bn2d_{i+1}')(x)
                
                x = layers.MaxPool2D(pool_size=2, name=f'pool2d_{i+1}')(x)
                x = layers.Dropout(dropout_rate, name=f'dropout2d_{i+1}')(x)
            
            # Global pooling and dense layers
            x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
            
            # Dense layers for embedding prediction
            x = layers.Dense(256, activation=activation, name='dense1')(x)
            if use_batch_norm:
                x = layers.BatchNormalization(name='bn_dense1')(x)
            x = layers.Dropout(0.5, name='dropout_dense1')(x)
            
            x = layers.Dense(128, activation=activation, name='dense2')(x)
            if use_batch_norm:
                x = layers.BatchNormalization(name='bn_dense2')(x)
            x = layers.Dropout(0.3, name='dropout_dense2')(x)
            
            # Output layer
            outputs = layers.Dense(
                units=output_dim,
                activation='linear',
                name='embedding_output'
            )(x)
            
            # Create model
            model = keras.Model(inputs=inputs, outputs=outputs, name='lsm_2d_cnn')
            
            return model
            
        except Exception as e:
            raise CNNArchitectureError(
                "2D CNN",
                f"Model creation failed: {str(e)}",
                {
                    "input_shape": input_shape,
                    "output_dim": output_dim,
                    "use_attention": use_attention,
                    "attention_type": attention_type
                }
            )    

    def create_3d_cnn(self, 
                      input_shape: Tuple[int, int, int, int], 
                      output_dim: int,
                      system_dim: int,
                      filters: List[int] = None,
                      kernel_sizes: List[Tuple[int, int, int]] = None,
                      dropout_rates: List[float] = None,
                      use_batch_norm: bool = True,
                      activation: str = "relu") -> tf.keras.Model:
        """
        Create a 3D CNN model for system message integration.
        
        Args:
            input_shape: Shape of input (depth, height, width, channels)
            output_dim: Dimension of output embeddings
            system_dim: Dimension of system message embeddings
            filters: List of filter counts for each conv layer
            kernel_sizes: List of 3D kernel sizes for each conv layer
            dropout_rates: List of dropout rates for each layer
            use_batch_norm: Whether to use batch normalization
            activation: Activation function to use
            
        Returns:
            Compiled Keras model with dual inputs (reservoir + system)
            
        Raises:
            CNNArchitectureError: If model creation fails
        """
        try:
            # Default configurations
            if filters is None:
                filters = [32, 64, 128]
            if kernel_sizes is None:
                kernel_sizes = [(3, 3, 3), (3, 3, 3), (3, 3, 3)]
            if dropout_rates is None:
                dropout_rates = [0.25, 0.25, 0.5]
            
            # Validate inputs
            self._validate_3d_inputs(input_shape, output_dim, system_dim)
            
            # Main input (reservoir output)
            reservoir_input = keras.Input(shape=input_shape, name='reservoir_input')
            
            # System message input
            system_input = keras.Input(shape=(system_dim,), name='system_input')
            
            # Process reservoir input through 3D CNN
            x = reservoir_input
            
            # 3D Convolutional blocks
            for i, (num_filters, kernel_size, dropout_rate) in enumerate(
                zip(filters, kernel_sizes, dropout_rates)
            ):
                x = layers.Conv3D(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation=activation,
                    name=f'conv3d_{i+1}'
                )(x)
                
                if use_batch_norm:
                    x = layers.BatchNormalization(name=f'bn3d_{i+1}')(x)
                
                x = layers.MaxPool3D(pool_size=2, name=f'pool3d_{i+1}')(x)
                x = layers.Dropout(dropout_rate, name=f'dropout3d_{i+1}')(x)
            
            # Global pooling for 3D features
            x = layers.GlobalAveragePooling3D(name='global_avg_pool_3d')(x)
            
            # Process system message
            system_processed = layers.Dense(
                128, 
                activation=activation, 
                name='system_dense1'
            )(system_input)
            system_processed = layers.Dropout(0.3, name='system_dropout1')(system_processed)
            
            system_processed = layers.Dense(
                64, 
                activation=activation, 
                name='system_dense2'
            )(system_processed)
            
            # Combine reservoir and system features
            combined = layers.Concatenate(name='combine_features')([x, system_processed])
            
            # Final processing layers
            combined = layers.Dense(256, activation=activation, name='combined_dense1')(combined)
            if use_batch_norm:
                combined = layers.BatchNormalization(name='bn_combined1')(combined)
            combined = layers.Dropout(0.5, name='dropout_combined1')(combined)
            
            combined = layers.Dense(128, activation=activation, name='combined_dense2')(combined)
            if use_batch_norm:
                combined = layers.BatchNormalization(name='bn_combined2')(combined)
            combined = layers.Dropout(0.3, name='dropout_combined2')(combined)
            
            # Output layer
            outputs = layers.Dense(
                units=output_dim,
                activation='linear',
                name='embedding_output'
            )(combined)
            
            # Create model with dual inputs
            model = keras.Model(
                inputs=[reservoir_input, system_input], 
                outputs=outputs, 
                name='lsm_3d_cnn_system'
            )
            
            return model
            
        except Exception as e:
            raise CNNArchitectureError(
                "3D CNN",
                f"Model creation failed: {str(e)}",
                {
                    "input_shape": input_shape,
                    "output_dim": output_dim,
                    "system_dim": system_dim
                }
            )
    
    def create_residual_cnn(self, 
                           input_shape: Tuple[int, int, int], 
                           output_dim: int,
                           depth: int = 2,
                           base_filters: int = 64,
                           use_attention: bool = False,
                           activation: str = "relu") -> tf.keras.Model:
        """
        Create a residual CNN model with skip connections.
        
        Args:
            input_shape: Shape of input (height, width, channels)
            output_dim: Dimension of output embeddings
            depth: Number of residual blocks
            base_filters: Base number of filters (doubles each block)
            use_attention: Whether to use attention mechanism
            activation: Activation function to use
            
        Returns:
            Compiled Keras model with residual connections
            
        Raises:
            CNNArchitectureError: If model creation fails
        """
        try:
            # Validate inputs
            self._validate_2d_inputs(input_shape, output_dim)
            
            inputs = keras.Input(shape=input_shape, name='waveform_input')
            
            # Initial convolution
            x = layers.Conv2D(
                base_filters, 
                7, 
                padding='same', 
                name='initial_conv'
            )(inputs)
            x = layers.BatchNormalization(name='initial_bn')(x)
            x = layers.Activation(activation, name='initial_activation')(x)
            x = layers.MaxPool2D(3, strides=2, padding='same', name='initial_pool')(x)
            
            # Residual blocks
            current_filters = base_filters
            for i in range(depth):
                # Determine if we need to downsample
                downsample = i > 0
                if downsample:
                    current_filters *= 2
                
                x = self._residual_block(
                    x, 
                    current_filters, 
                    downsample=downsample, 
                    block_name=f'res_block_{i+1}',
                    activation=activation
                )
            
            # Apply attention if requested
            if use_attention:
                x = self._add_attention_block(x, "spatial")
            
            # Global pooling and final layers
            x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
            x = layers.Dense(256, activation=activation, name='dense1')(x)
            x = layers.Dropout(0.5, name='dropout1')(x)
            x = layers.Dense(output_dim, activation='linear', name='embedding_output')(x)
            
            model = keras.Model(inputs=inputs, outputs=x, name='residual_lsm_cnn')
            
            return model
            
        except Exception as e:
            raise CNNArchitectureError(
                "Residual CNN",
                f"Model creation failed: {str(e)}",
                {
                    "input_shape": input_shape,
                    "output_dim": output_dim,
                    "depth": depth,
                    "base_filters": base_filters
                }
            )
    
    def create_multi_scale_cnn(self,
                              input_shape: Tuple[int, int, int],
                              output_dim: int,
                              scales: List[int] = None,
                              use_attention: bool = True) -> tf.keras.Model:
        """
        Create a multi-scale CNN that processes input at different scales.
        
        Args:
            input_shape: Shape of input (height, width, channels)
            output_dim: Dimension of output embeddings
            scales: List of kernel sizes for different scales
            use_attention: Whether to use attention mechanism
            
        Returns:
            Compiled Keras model with multi-scale processing
            
        Raises:
            CNNArchitectureError: If model creation fails
        """
        try:
            if scales is None:
                scales = [3, 5, 7]
            
            # Validate inputs
            self._validate_2d_inputs(input_shape, output_dim)
            
            inputs = keras.Input(shape=input_shape, name='waveform_input')
            
            # Different scale branches
            branches = []
            
            for i, scale in enumerate(scales):
                branch = layers.Conv2D(
                    32, 
                    scale, 
                    padding='same', 
                    activation='relu', 
                    name=f'scale_{scale}_conv1'
                )(inputs)
                branch = layers.Conv2D(
                    32, 
                    scale, 
                    padding='same', 
                    activation='relu', 
                    name=f'scale_{scale}_conv2'
                )(branch)
                branch = layers.MaxPool2D(2, name=f'scale_{scale}_pool')(branch)
                branches.append(branch)
            
            # Concatenate multi-scale features
            x = layers.Concatenate(axis=-1, name='multi_scale_concat')(branches)
            
            # Apply attention if requested
            if use_attention:
                x = self._add_attention_block(x, "spatial_channel")
            
            # Process combined features
            x = layers.Conv2D(128, 3, padding='same', activation='relu', name='combined_conv1')(x)
            x = layers.BatchNormalization(name='combined_bn1')(x)
            x = layers.MaxPool2D(2, name='combined_pool')(x)
            x = layers.Dropout(0.25, name='combined_dropout')(x)
            
            x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
            x = layers.Dense(256, activation='relu', name='dense1')(x)
            x = layers.Dropout(0.5, name='dropout1')(x)
            x = layers.Dense(output_dim, activation='linear', name='embedding_output')(x)
            
            model = keras.Model(inputs=inputs, outputs=x, name='multi_scale_lsm_cnn')
            
            return model
            
        except Exception as e:
            raise CNNArchitectureError(
                "Multi-scale CNN",
                f"Model creation failed: {str(e)}",
                {
                    "input_shape": input_shape,
                    "output_dim": output_dim,
                    "scales": scales
                }
            )    

    def compile_model(self, 
                     model: tf.keras.Model, 
                     loss_type: str = "cosine_similarity",
                     learning_rate: float = 0.001,
                     metrics: List[str] = None,
                     loss_config: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
        """
        Compile a CNN model with enhanced loss functions for response-level training.
        
        Args:
            model: Keras model to compile
            loss_type: Type of loss function (supports enhanced cosine similarity variants)
            learning_rate: Learning rate for optimizer
            metrics: List of metrics to track
            loss_config: Optional configuration for loss function parameters
            
        Returns:
            Compiled model
            
        Raises:
            CNNArchitectureError: If compilation fails
        """
        try:
            if metrics is None:
                metrics = ['mae']
            
            if loss_config is None:
                loss_config = {}
            
            # Initialize loss calculator
            loss_calculator = CNNLossCalculator()
            supported_losses = loss_calculator.get_supported_loss_types()
            
            # Validate loss type
            if loss_type not in supported_losses and loss_type not in self._supported_losses:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            # Create optimizer
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Select enhanced loss function
            if loss_type == "mse":
                loss_fn = 'mse'
                if 'mse' not in metrics:
                    metrics.append('mse')
            elif loss_type == "cosine_similarity":
                # Use enhanced cosine similarity loss
                loss_fn = create_cosine_similarity_loss(**loss_config)
                metrics.append(self._cosine_similarity_metric)
            elif loss_type == "cosine_similarity_weighted":
                loss_fn = create_cosine_similarity_loss(weight_factor=2.0, **loss_config)
                metrics.append(self._cosine_similarity_metric)
            elif loss_type == "cosine_similarity_temperature":
                loss_fn = create_cosine_similarity_loss(temperature=0.5, **loss_config)
                metrics.append(self._cosine_similarity_metric)
            elif loss_type == "cosine_similarity_margin":
                loss_fn = create_cosine_similarity_loss(margin=0.1, **loss_config)
                metrics.append(self._cosine_similarity_metric)
            elif loss_type == "response_level_cosine":
                # Use response-level cosine similarity loss
                loss_fn = create_response_level_loss(**loss_config)
                metrics.append(self._cosine_similarity_metric)
            elif loss_type == "huber":
                loss_fn = keras.losses.Huber()
            else:
                # Fallback to legacy implementation
                loss_fn = self._cosine_similarity_loss
                metrics.append(self._cosine_similarity_metric)
            
            # Compile model
            model.compile(
                optimizer=optimizer,
                loss=loss_fn,
                metrics=metrics
            )
            
            return model
            
        except Exception as e:
            raise CNNArchitectureError(
                "Model Compilation",
                f"Compilation failed: {str(e)}",
                {
                    "loss_type": loss_type,
                    "learning_rate": learning_rate,
                    "metrics": metrics,
                    "loss_config": loss_config
                }
            )
    
    def get_supported_architectures(self) -> List[str]:
        """Get list of supported CNN architecture types."""
        return self._supported_types.copy()
    
    def get_supported_attention_types(self) -> List[str]:
        """Get list of supported attention mechanism types."""
        return self._supported_attention.copy()
    
    def get_supported_loss_types(self) -> List[str]:
        """Get list of supported loss function types."""
        # Combine legacy and enhanced loss types
        loss_calculator = CNNLossCalculator()
        enhanced_losses = loss_calculator.get_supported_loss_types()
        return self._supported_losses.copy() + enhanced_losses
    
    def calculate_loss_2d(self, 
                         y_true: tf.Tensor, 
                         y_pred: tf.Tensor,
                         loss_type: str = "cosine_similarity",
                         **kwargs) -> tf.Tensor:
        """
        Calculate loss for 2D CNN architecture with enhanced loss functions.
        
        Args:
            y_true: Ground truth embeddings (batch_size, embedding_dim)
            y_pred: Predicted embeddings (batch_size, embedding_dim)
            loss_type: Type of loss function to use
            **kwargs: Additional arguments for loss function
            
        Returns:
            Computed loss tensor
            
        Raises:
            CNNArchitectureError: If loss calculation fails
        """
        try:
            loss_calculator = CNNLossCalculator()
            return loss_calculator.calculate_loss_2d(y_true, y_pred, loss_type, **kwargs)
            
        except Exception as e:
            raise CNNArchitectureError(
                "2D Loss Calculation",
                f"Failed to calculate 2D CNN loss: {str(e)}",
                {
                    "loss_type": loss_type,
                    "y_true_shape": y_true.shape.as_list() if hasattr(y_true, 'shape') else None,
                    "y_pred_shape": y_pred.shape.as_list() if hasattr(y_pred, 'shape') else None
                }
            )
    
    def calculate_loss_3d(self,
                         y_true: tf.Tensor,
                         y_pred: tf.Tensor,
                         system_context: Optional[tf.Tensor] = None,
                         loss_type: str = "cosine_similarity",
                         **kwargs) -> tf.Tensor:
        """
        Calculate loss for 3D CNN architecture with system context support.
        
        Args:
            y_true: Ground truth embeddings (batch_size, embedding_dim)
            y_pred: Predicted embeddings (batch_size, embedding_dim)
            system_context: Optional system context embeddings for 3D CNN
            loss_type: Type of loss function to use
            **kwargs: Additional arguments for loss function
            
        Returns:
            Computed loss tensor
            
        Raises:
            CNNArchitectureError: If loss calculation fails
        """
        try:
            loss_calculator = CNNLossCalculator()
            return loss_calculator.calculate_loss_3d(
                y_true, y_pred, system_context, loss_type, **kwargs
            )
            
        except Exception as e:
            raise CNNArchitectureError(
                "3D Loss Calculation",
                f"Failed to calculate 3D CNN loss: {str(e)}",
                {
                    "loss_type": loss_type,
                    "y_true_shape": y_true.shape.as_list() if hasattr(y_true, 'shape') else None,
                    "y_pred_shape": y_pred.shape.as_list() if hasattr(y_pred, 'shape') else None,
                    "has_system_context": system_context is not None
                }
            )
    
    # Private helper methods
    
    def _validate_2d_inputs(self, input_shape: Tuple[int, int, int], 
                           output_dim: int, attention_type: str = "spatial"):
        """Validate inputs for 2D CNN creation."""
        if len(input_shape) != 3:
            raise ValueError(f"2D CNN input_shape must have 3 dimensions, got {len(input_shape)}")
        
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        if attention_type not in self._supported_attention:
            raise ValueError(f"Unsupported attention type: {attention_type}")
    
    def _validate_3d_inputs(self, input_shape: Tuple[int, int, int, int], 
                           output_dim: int, system_dim: int):
        """Validate inputs for 3D CNN creation."""
        if len(input_shape) != 4:
            raise ValueError(f"3D CNN input_shape must have 4 dimensions, got {len(input_shape)}")
        
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        if system_dim <= 0:
            raise ValueError(f"system_dim must be positive, got {system_dim}")
    
    def _add_attention_block(self, x, attention_type: str):
        """Add attention block to the model."""
        if attention_type == "spatial":
            attention_block = SpatialAttentionBlock(name='spatial_attention')
            x, _ = attention_block(x)
        elif attention_type == "channel":
            attention_block = ChannelAttentionBlock(name='channel_attention')
            x, _ = attention_block(x)
        elif attention_type == "spatial_channel":
            # Apply both spatial and channel attention
            spatial_block = SpatialAttentionBlock(name='spatial_attention')
            x, _ = spatial_block(x)
            channel_block = ChannelAttentionBlock(name='channel_attention')
            x, _ = channel_block(x)
        
        return x
    
    def _residual_block(self, x, filters: int, downsample: bool = False, 
                       block_name: str = "res_block", activation: str = "relu"):
        """Create a residual block with skip connections."""
        # Determine stride
        stride = 2 if downsample else 1
        
        # Shortcut connection
        if downsample or x.shape[-1] != filters:
            shortcut = layers.Conv2D(
                filters, 
                1, 
                strides=stride, 
                name=f'{block_name}_shortcut_conv'
            )(x)
            shortcut = layers.BatchNormalization(name=f'{block_name}_shortcut_bn')(shortcut)
        else:
            shortcut = x
        
        # Main path
        x = layers.Conv2D(
            filters, 
            3, 
            strides=stride, 
            padding='same', 
            name=f'{block_name}_conv1'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn1')(x)
        x = layers.Activation(activation, name=f'{block_name}_activation1')(x)
        
        x = layers.Conv2D(
            filters, 
            3, 
            padding='same', 
            name=f'{block_name}_conv2'
        )(x)
        x = layers.BatchNormalization(name=f'{block_name}_bn2')(x)
        
        # Add shortcut
        x = layers.Add(name=f'{block_name}_add')([x, shortcut])
        x = layers.Activation(activation, name=f'{block_name}_activation2')(x)
        
        return x
    
    @staticmethod
    def _cosine_similarity_loss(y_true, y_pred):
        """Cosine similarity loss function for response-level training."""
        # Normalize vectors
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        
        # Compute cosine similarity
        cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        
        # Convert to loss (1 - cosine_similarity)
        return 1.0 - cosine_sim
    
    @staticmethod
    def _cosine_similarity_metric(y_true, y_pred):
        """Cosine similarity metric for monitoring."""
        # Normalize vectors
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        
        # Compute cosine similarity
        cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        
        return cosine_sim


# Convenience functions for backward compatibility and ease of use

def create_standard_2d_cnn(window_size: int, embedding_dim: int, 
                          use_attention: bool = True,
                          loss_type: str = "response_level_cosine") -> tf.keras.Model:
    """
    Create a standard 2D CNN with enhanced loss function for response-level training.
    
    Args:
        window_size: Size of the input waveform (window_size x window_size)
        embedding_dim: Dimension of the output token embedding
        use_attention: Whether to use spatial attention
        loss_type: Type of loss function (defaults to response-level cosine)
        
    Returns:
        Compiled Keras CNN model with enhanced loss function
    """
    factory = CNNArchitectureFactory()
    input_shape = (window_size, window_size, 1)
    
    model = factory.create_2d_cnn(
        input_shape=input_shape,
        output_dim=embedding_dim,
        use_attention=use_attention
    )
    
    return factory.compile_model(model, loss_type=loss_type)


def create_system_aware_3d_cnn(window_size: int, embedding_dim: int, 
                              system_dim: int,
                              loss_type: str = "response_level_cosine") -> tf.keras.Model:
    """
    Create a 3D CNN for system message integration with enhanced loss function.
    
    Args:
        window_size: Size of the input waveform
        embedding_dim: Dimension of the output token embedding
        system_dim: Dimension of system message embeddings
        loss_type: Type of loss function (defaults to response-level cosine)
        
    Returns:
        Compiled Keras 3D CNN model with enhanced loss function
    """
    factory = CNNArchitectureFactory()
    input_shape = (window_size, window_size, window_size, 1)
    
    model = factory.create_3d_cnn(
        input_shape=input_shape,
        output_dim=embedding_dim,
        system_dim=system_dim
    )
    
    return factory.compile_model(model, loss_type=loss_type)
    
    return factory.compile_model(model, loss_type="cosine_similarity")


def create_residual_cnn_model(window_size: int, embedding_dim: int, 
                             depth: int = 2) -> tf.keras.Model:
    """
    Create a residual CNN with skip connections.
    
    Args:
        window_size: Size of the input waveform
        embedding_dim: Dimension of the output token embedding
        depth: Number of residual blocks
        
    Returns:
        Compiled Keras residual CNN model
    """
    factory = CNNArchitectureFactory()
    input_shape = (window_size, window_size, 1)
    
    model = factory.create_residual_cnn(
        input_shape=input_shape,
        output_dim=embedding_dim,
        depth=depth
    )
    
    return factory.compile_model(model, loss_type="cosine_similarity")


if __name__ == "__main__":
    # Test CNN architecture factory
    print("Testing CNN Architecture Factory...")
    
    factory = CNNArchitectureFactory()
    
    # Test parameters
    window_size = 10
    embedding_dim = 128
    system_dim = 64
    
    print(f"Supported architectures: {factory.get_supported_architectures()}")
    print(f"Supported attention types: {factory.get_supported_attention_types()}")
    print(f"Supported loss types: {factory.get_supported_loss_types()}")
    
    # Test 2D CNN
    print("\nTesting 2D CNN creation...")
    try:
        model_2d = factory.create_2d_cnn(
            input_shape=(window_size, window_size, 1),
            output_dim=embedding_dim,
            use_attention=True
        )
        model_2d = factory.compile_model(model_2d, loss_type="cosine_similarity")
        print(f"2D CNN created successfully. Parameters: {model_2d.count_params()}")
        
        # Test with dummy data
        dummy_input = np.random.random((4, window_size, window_size, 1)).astype(np.float32)
        output = model_2d(dummy_input)
        print(f"2D CNN output shape: {output.shape}")
        
    except Exception as e:
        print(f"2D CNN test failed: {e}")
    
    # Test 3D CNN
    print("\nTesting 3D CNN creation...")
    try:
        model_3d = factory.create_3d_cnn(
            input_shape=(window_size, window_size, window_size, 1),
            output_dim=embedding_dim,
            system_dim=system_dim
        )
        model_3d = factory.compile_model(model_3d, loss_type="cosine_similarity")
        print(f"3D CNN created successfully. Parameters: {model_3d.count_params()}")
        
        # Test with dummy data
        dummy_reservoir = np.random.random((4, window_size, window_size, window_size, 1)).astype(np.float32)
        dummy_system = np.random.random((4, system_dim)).astype(np.float32)
        output = model_3d([dummy_reservoir, dummy_system])
        print(f"3D CNN output shape: {output.shape}")
        
    except Exception as e:
        print(f"3D CNN test failed: {e}")
    
    # Test Residual CNN
    print("\nTesting Residual CNN creation...")
    try:
        model_residual = factory.create_residual_cnn(
            input_shape=(window_size, window_size, 1),
            output_dim=embedding_dim,
            depth=2
        )
        model_residual = factory.compile_model(model_residual, loss_type="cosine_similarity")
        print(f"Residual CNN created successfully. Parameters: {model_residual.count_params()}")
        
        # Test with dummy data
        dummy_input = np.random.random((4, window_size, window_size, 1)).astype(np.float32)
        output = model_residual(dummy_input)
        print(f"Residual CNN output shape: {output.shape}")
        
    except Exception as e:
        print(f"Residual CNN test failed: {e}")
    
    # Test convenience functions
    print("\nTesting convenience functions...")
    try:
        standard_model = create_standard_2d_cnn(window_size, embedding_dim)
        print(f"Standard 2D CNN created via convenience function")
        
        system_model = create_system_aware_3d_cnn(window_size, embedding_dim, system_dim)
        print(f"System-aware 3D CNN created via convenience function")
        
        residual_model = create_residual_cnn_model(window_size, embedding_dim)
        print(f"Residual CNN created via convenience function")
        
    except Exception as e:
        print(f"Convenience function test failed: {e}")
    
    print("\nCNN Architecture Factory tests completed!")