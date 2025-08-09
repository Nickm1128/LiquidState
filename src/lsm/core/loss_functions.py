#!/usr/bin/env python3
"""
Enhanced Loss Functions for LSM Training Pipeline.

This module provides advanced loss functions optimized for response-level training,
including cosine similarity loss variants for both 2D and 3D CNN architectures.
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional, Dict, Any, Callable, Union
from enum import Enum

from ..utils.lsm_exceptions import LSMError, ModelError


class LossFunctionError(ModelError):
    """Raised when loss function computation fails."""
    
    def __init__(self, loss_type: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {"loss_type": loss_type, "reason": reason}
        if details:
            error_details.update(details)
        
        message = f"Loss function '{loss_type}' computation failed: {reason}"
        super().__init__(message, error_details)
        self.loss_type = loss_type


class LossType(Enum):
    """Enumeration of supported loss function types."""
    MSE = "mse"
    COSINE_SIMILARITY = "cosine_similarity"
    COSINE_SIMILARITY_WEIGHTED = "cosine_similarity_weighted"
    COSINE_SIMILARITY_TEMPERATURE = "cosine_similarity_temperature"
    COSINE_SIMILARITY_MARGIN = "cosine_similarity_margin"
    HUBER = "huber"
    RESPONSE_LEVEL_COSINE = "response_level_cosine"


class CosineSimilarityLoss:
    """
    Enhanced cosine similarity loss function for response-level training.
    
    This class provides various cosine similarity loss variants optimized for
    different CNN architectures and training scenarios.
    """
    
    def __init__(self, 
                 temperature: float = 1.0,
                 margin: float = 0.0,
                 weight_factor: float = 1.0,
                 epsilon: float = 1e-8,
                 reduction: str = 'mean'):
        """
        Initialize cosine similarity loss with configurable parameters.
        
        Args:
            temperature: Temperature scaling factor for softmax-like behavior
            margin: Margin for margin-based cosine similarity loss
            weight_factor: Weighting factor for loss scaling
            epsilon: Small value to prevent division by zero
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        self.temperature = temperature
        self.margin = margin
        self.weight_factor = weight_factor
        self.epsilon = epsilon
        self.reduction = reduction
        
        # Validate parameters
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError("Reduction must be 'mean', 'sum', or 'none'")
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute cosine similarity loss.
        
        Args:
            y_true: Ground truth embeddings
            y_pred: Predicted embeddings
            
        Returns:
            Computed loss tensor
        """
        return self.compute_loss(y_true, y_pred)
    
    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute standard cosine similarity loss.
        
        Args:
            y_true: Ground truth embeddings (batch_size, embedding_dim)
            y_pred: Predicted embeddings (batch_size, embedding_dim)
            
        Returns:
            Loss tensor
        """
        try:
            # Normalize vectors to unit length
            y_true_norm = tf.nn.l2_normalize(y_true, axis=-1, epsilon=self.epsilon)
            y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1, epsilon=self.epsilon)
            
            # Compute cosine similarity
            cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                cosine_sim = cosine_sim / self.temperature
            
            # Convert to loss (1 - cosine_similarity)
            loss = self.weight_factor * (1.0 - cosine_sim)
            
            # Apply reduction
            if self.reduction == 'mean':
                return tf.reduce_mean(loss)
            elif self.reduction == 'sum':
                return tf.reduce_sum(loss)
            else:  # 'none'
                return loss
                
        except Exception as e:
            raise LossFunctionError(
                "cosine_similarity",
                f"Failed to compute cosine similarity loss: {str(e)}",
                {
                    "y_true_shape": y_true.shape.as_list() if hasattr(y_true, 'shape') else None,
                    "y_pred_shape": y_pred.shape.as_list() if hasattr(y_pred, 'shape') else None
                }
            )
    
    def compute_weighted_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                             sample_weights: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Compute weighted cosine similarity loss.
        
        Args:
            y_true: Ground truth embeddings
            y_pred: Predicted embeddings
            sample_weights: Optional per-sample weights
            
        Returns:
            Weighted loss tensor
        """
        try:
            # Compute base loss
            loss = self.compute_loss(y_true, y_pred)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                if self.reduction == 'none':
                    loss = loss * sample_weights
                else:
                    # For reduced losses, we need to recompute with weights
                    y_true_norm = tf.nn.l2_normalize(y_true, axis=-1, epsilon=self.epsilon)
                    y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1, epsilon=self.epsilon)
                    cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
                    
                    if self.temperature != 1.0:
                        cosine_sim = cosine_sim / self.temperature
                    
                    weighted_loss = sample_weights * self.weight_factor * (1.0 - cosine_sim)
                    
                    if self.reduction == 'mean':
                        loss = tf.reduce_sum(weighted_loss) / tf.reduce_sum(sample_weights)
                    elif self.reduction == 'sum':
                        loss = tf.reduce_sum(weighted_loss)
            
            return loss
            
        except Exception as e:
            raise LossFunctionError(
                "weighted_cosine_similarity",
                f"Failed to compute weighted cosine similarity loss: {str(e)}"
            )
    
    def compute_margin_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute margin-based cosine similarity loss.
        
        Args:
            y_true: Ground truth embeddings
            y_pred: Predicted embeddings
            
        Returns:
            Margin-based loss tensor
        """
        try:
            # Normalize vectors
            y_true_norm = tf.nn.l2_normalize(y_true, axis=-1, epsilon=self.epsilon)
            y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1, epsilon=self.epsilon)
            
            # Compute cosine similarity
            cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
            
            # Apply margin: max(0, margin - cosine_similarity)
            margin_loss = tf.maximum(0.0, self.margin - cosine_sim)
            
            # Scale by weight factor
            loss = self.weight_factor * margin_loss
            
            # Apply reduction
            if self.reduction == 'mean':
                return tf.reduce_mean(loss)
            elif self.reduction == 'sum':
                return tf.reduce_sum(loss)
            else:  # 'none'
                return loss
                
        except Exception as e:
            raise LossFunctionError(
                "margin_cosine_similarity",
                f"Failed to compute margin cosine similarity loss: {str(e)}"
            )


class ResponseLevelCosineLoss:
    """
    Specialized cosine similarity loss for response-level training.
    
    This loss function is optimized for training models that generate complete
    responses rather than individual tokens, with enhanced handling for
    sequence-level embeddings.
    """
    
    def __init__(self,
                 sequence_weight: float = 1.0,
                 coherence_weight: float = 0.1,
                 diversity_weight: float = 0.05,
                 temperature: float = 1.0,
                 epsilon: float = 1e-8):
        """
        Initialize response-level cosine loss.
        
        Args:
            sequence_weight: Weight for sequence-level similarity
            coherence_weight: Weight for internal coherence penalty
            diversity_weight: Weight for diversity encouragement
            temperature: Temperature scaling factor
            epsilon: Small value to prevent division by zero
        """
        self.sequence_weight = sequence_weight
        self.coherence_weight = coherence_weight
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        self.epsilon = epsilon
    
    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Compute response-level cosine similarity loss.
        
        Args:
            y_true: Ground truth response embeddings
            y_pred: Predicted response embeddings
            
        Returns:
            Response-level loss tensor
        """
        try:
            # Primary sequence-level cosine similarity loss
            sequence_loss = self._compute_sequence_loss(y_true, y_pred)
            
            # Coherence penalty (encourages internal consistency)
            coherence_penalty = self._compute_coherence_penalty(y_pred)
            
            # Diversity encouragement (prevents mode collapse)
            diversity_bonus = self._compute_diversity_bonus(y_pred)
            
            # Combine losses
            total_loss = (
                self.sequence_weight * sequence_loss +
                self.coherence_weight * coherence_penalty -
                self.diversity_weight * diversity_bonus
            )
            
            return total_loss
            
        except Exception as e:
            raise LossFunctionError(
                "response_level_cosine",
                f"Failed to compute response-level cosine loss: {str(e)}"
            )
    
    def _compute_sequence_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute primary sequence-level cosine similarity loss."""
        # Normalize vectors
        y_true_norm = tf.nn.l2_normalize(y_true, axis=-1, epsilon=self.epsilon)
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1, epsilon=self.epsilon)
        
        # Compute cosine similarity
        cosine_sim = tf.reduce_sum(y_true_norm * y_pred_norm, axis=-1)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            cosine_sim = cosine_sim / self.temperature
        
        # Convert to loss
        return tf.reduce_mean(1.0 - cosine_sim)
    
    def _compute_coherence_penalty(self, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute coherence penalty to encourage consistent predictions."""
        # Compute pairwise similarities within batch
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1, epsilon=self.epsilon)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(y_pred_norm, y_pred_norm, transpose_b=True)
        
        # Compute variance of similarities (lower variance = more coherent)
        similarity_variance = tf.math.reduce_variance(similarity_matrix)
        
        return similarity_variance
    
    def _compute_diversity_bonus(self, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute diversity bonus to prevent mode collapse."""
        # Normalize predictions
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1, epsilon=self.epsilon)
        
        # Compute mean prediction
        mean_pred = tf.reduce_mean(y_pred_norm, axis=0, keepdims=True)
        
        # Compute average distance from mean (higher = more diverse)
        distances = tf.norm(y_pred_norm - mean_pred, axis=-1)
        avg_distance = tf.reduce_mean(distances)
        
        return avg_distance


class CNNLossCalculator:
    """
    Loss calculator for both 2D and 3D CNN architectures.
    
    This class provides unified loss calculation methods that work with
    different CNN architectures and handle various input formats.
    """
    
    def __init__(self):
        """Initialize the CNN loss calculator."""
        self.loss_functions = {
            LossType.MSE.value: self._mse_loss,
            LossType.COSINE_SIMILARITY.value: CosineSimilarityLoss(),
            LossType.COSINE_SIMILARITY_WEIGHTED.value: CosineSimilarityLoss(weight_factor=2.0),
            LossType.COSINE_SIMILARITY_TEMPERATURE.value: CosineSimilarityLoss(temperature=0.5),
            LossType.COSINE_SIMILARITY_MARGIN.value: CosineSimilarityLoss(margin=0.1),
            LossType.RESPONSE_LEVEL_COSINE.value: ResponseLevelCosineLoss(),
            LossType.HUBER.value: self._huber_loss
        }
    
    def calculate_loss_2d(self, 
                         y_true: tf.Tensor, 
                         y_pred: tf.Tensor,
                         loss_type: str = "cosine_similarity",
                         **kwargs) -> tf.Tensor:
        """
        Calculate loss for 2D CNN architecture.
        
        Args:
            y_true: Ground truth embeddings (batch_size, embedding_dim)
            y_pred: Predicted embeddings (batch_size, embedding_dim)
            loss_type: Type of loss function to use
            **kwargs: Additional arguments for loss function
            
        Returns:
            Computed loss tensor
        """
        try:
            # Validate inputs
            self._validate_2d_inputs(y_true, y_pred)
            
            # Get loss function
            if loss_type not in self.loss_functions:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            loss_fn = self.loss_functions[loss_type]
            
            # Compute loss
            if callable(loss_fn):
                # Filter kwargs to only pass supported parameters
                if hasattr(loss_fn, '__call__') and hasattr(loss_fn, '__class__'):
                    # For class instances, only pass y_true and y_pred
                    return loss_fn(y_true, y_pred)
                else:
                    # For functions, pass all kwargs
                    return loss_fn(y_true, y_pred, **kwargs)
            else:
                return loss_fn(y_true, y_pred)
                
        except Exception as e:
            raise LossFunctionError(
                f"2d_cnn_{loss_type}",
                f"Failed to calculate 2D CNN loss: {str(e)}",
                {
                    "y_true_shape": y_true.shape.as_list() if hasattr(y_true, 'shape') else None,
                    "y_pred_shape": y_pred.shape.as_list() if hasattr(y_pred, 'shape') else None,
                    "loss_type": loss_type
                }
            )
    
    def calculate_loss_3d(self,
                         y_true: tf.Tensor,
                         y_pred: tf.Tensor,
                         system_context: Optional[tf.Tensor] = None,
                         loss_type: str = "cosine_similarity",
                         **kwargs) -> tf.Tensor:
        """
        Calculate loss for 3D CNN architecture with optional system context.
        
        Args:
            y_true: Ground truth embeddings (batch_size, embedding_dim)
            y_pred: Predicted embeddings (batch_size, embedding_dim)
            system_context: Optional system context embeddings
            loss_type: Type of loss function to use
            **kwargs: Additional arguments for loss function
            
        Returns:
            Computed loss tensor
        """
        try:
            # Validate inputs
            self._validate_3d_inputs(y_true, y_pred, system_context)
            
            # Get base loss function
            if loss_type not in self.loss_functions:
                raise ValueError(f"Unsupported loss type: {loss_type}")
            
            loss_fn = self.loss_functions[loss_type]
            
            # Compute base loss
            if callable(loss_fn):
                # Filter kwargs to only pass supported parameters
                if hasattr(loss_fn, '__call__') and hasattr(loss_fn, '__class__'):
                    # For class instances, only pass y_true and y_pred
                    base_loss = loss_fn(y_true, y_pred)
                else:
                    # For functions, pass all kwargs
                    base_loss = loss_fn(y_true, y_pred, **kwargs)
            else:
                base_loss = loss_fn(y_true, y_pred)
            
            # Add system context penalty if provided
            if system_context is not None:
                system_penalty = self._compute_system_context_penalty(
                    y_pred, system_context
                )
                # Weight the system penalty (typically small)
                system_weight = kwargs.get('system_weight', 0.1)
                base_loss = base_loss + system_weight * system_penalty
            
            return base_loss
            
        except Exception as e:
            raise LossFunctionError(
                f"3d_cnn_{loss_type}",
                f"Failed to calculate 3D CNN loss: {str(e)}",
                {
                    "y_true_shape": y_true.shape.as_list() if hasattr(y_true, 'shape') else None,
                    "y_pred_shape": y_pred.shape.as_list() if hasattr(y_pred, 'shape') else None,
                    "has_system_context": system_context is not None,
                    "loss_type": loss_type
                }
            )
    
    def get_loss_function(self, loss_type: str) -> Callable:
        """
        Get a loss function by type.
        
        Args:
            loss_type: Type of loss function
            
        Returns:
            Loss function callable
        """
        if loss_type not in self.loss_functions:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return self.loss_functions[loss_type]
    
    def get_supported_loss_types(self) -> list:
        """Get list of supported loss function types."""
        return list(self.loss_functions.keys())
    
    # Private helper methods
    
    def _validate_2d_inputs(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> None:
        """Validate inputs for 2D CNN loss calculation."""
        if y_true is None or y_pred is None:
            raise ValueError("y_true and y_pred cannot be None")
        
        if len(y_true.shape) != 2 or len(y_pred.shape) != 2:
            raise ValueError("2D CNN inputs must be 2-dimensional (batch_size, embedding_dim)")
        
        if y_true.shape != y_pred.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    
    def _validate_3d_inputs(self, y_true: tf.Tensor, y_pred: tf.Tensor, 
                           system_context: Optional[tf.Tensor]) -> None:
        """Validate inputs for 3D CNN loss calculation."""
        # Same validation as 2D for the main tensors
        self._validate_2d_inputs(y_true, y_pred)
        
        # Additional validation for system context
        if system_context is not None:
            if len(system_context.shape) != 2:
                raise ValueError("System context must be 2-dimensional (batch_size, context_dim)")
            
            if system_context.shape[0] != y_true.shape[0]:
                raise ValueError("System context batch size must match y_true batch size")
    
    def _compute_system_context_penalty(self, y_pred: tf.Tensor, 
                                       system_context: tf.Tensor) -> tf.Tensor:
        """Compute penalty term for system context integration."""
        # Normalize both tensors
        y_pred_norm = tf.nn.l2_normalize(y_pred, axis=-1)
        system_norm = tf.nn.l2_normalize(system_context, axis=-1)
        
        # Handle dimension mismatch by projecting to common space
        pred_dim = y_pred_norm.shape[-1]
        system_dim = system_norm.shape[-1]
        
        if pred_dim != system_dim:
            # Project to the smaller dimension space
            min_dim = min(pred_dim, system_dim)
            y_pred_proj = y_pred_norm[:, :min_dim]
            system_proj = system_norm[:, :min_dim]
        else:
            y_pred_proj = y_pred_norm
            system_proj = system_norm
        
        # Compute alignment between prediction and system context
        # We want some alignment but not perfect alignment (to maintain diversity)
        alignment = tf.reduce_sum(y_pred_proj * system_proj, axis=-1)
        
        # Penalty for too little or too much alignment
        # Optimal alignment is around 0.3-0.7
        optimal_alignment = 0.5
        alignment_penalty = tf.square(alignment - optimal_alignment)
        
        return tf.reduce_mean(alignment_penalty)
    
    @staticmethod
    def _mse_loss(y_true: tf.Tensor, y_pred: tf.Tensor, **kwargs) -> tf.Tensor:
        """Mean squared error loss."""
        return tf.reduce_mean(tf.square(y_true - y_pred))
    
    @staticmethod
    def _huber_loss(y_true: tf.Tensor, y_pred: tf.Tensor, delta: float = 1.0, **kwargs) -> tf.Tensor:
        """Huber loss with configurable delta."""
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= delta
        squared_loss = tf.square(error) / 2
        linear_loss = delta * tf.abs(error) - tf.square(delta) / 2
        return tf.reduce_mean(tf.where(is_small_error, squared_loss, linear_loss))


# Convenience functions for easy usage

def create_cosine_similarity_loss(temperature: float = 1.0,
                                 margin: float = 0.0,
                                 weight_factor: float = 1.0) -> CosineSimilarityLoss:
    """
    Create a cosine similarity loss function with specified parameters.
    
    Args:
        temperature: Temperature scaling factor
        margin: Margin for margin-based loss
        weight_factor: Weight scaling factor
        
    Returns:
        Configured CosineSimilarityLoss instance
    """
    return CosineSimilarityLoss(
        temperature=temperature,
        margin=margin,
        weight_factor=weight_factor
    )


def create_response_level_loss(sequence_weight: float = 1.0,
                              coherence_weight: float = 0.1,
                              diversity_weight: float = 0.05) -> ResponseLevelCosineLoss:
    """
    Create a response-level cosine similarity loss function.
    
    Args:
        sequence_weight: Weight for sequence-level similarity
        coherence_weight: Weight for coherence penalty
        diversity_weight: Weight for diversity bonus
        
    Returns:
        Configured ResponseLevelCosineLoss instance
    """
    return ResponseLevelCosineLoss(
        sequence_weight=sequence_weight,
        coherence_weight=coherence_weight,
        diversity_weight=diversity_weight
    )


def get_loss_for_architecture(architecture_type: str, 
                             loss_type: str = "cosine_similarity") -> Callable:
    """
    Get appropriate loss function for CNN architecture type.
    
    Args:
        architecture_type: Type of CNN architecture ('2d' or '3d')
        loss_type: Type of loss function
        
    Returns:
        Loss function callable
    """
    calculator = CNNLossCalculator()
    
    if architecture_type == "2d":
        return lambda y_true, y_pred: calculator.calculate_loss_2d(y_true, y_pred, loss_type)
    elif architecture_type == "3d":
        return lambda y_true, y_pred, system_context=None: calculator.calculate_loss_3d(
            y_true, y_pred, system_context, loss_type
        )
    else:
        raise ValueError(f"Unsupported architecture type: {architecture_type}")