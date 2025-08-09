#!/usr/bin/env python3
"""
Reservoir Manager for Reservoir Strategy Management.

This module provides the ReservoirManager class that handles reservoir strategy
decisions, manages multiple reservoir instances, coordinates reservoir outputs
for response generation, and supports different reservoir types.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import defaultdict

from ..core.reservoir import build_reservoir
from ..core.advanced_reservoir import create_advanced_reservoir
from ..utils.lsm_exceptions import LSMError, InferenceError
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class ReservoirStrategy(Enum):
    """Enumeration of reservoir reuse strategies."""
    REUSE = "reuse"
    SEPARATE = "separate"
    ADAPTIVE = "adaptive"
    HIERARCHICAL = "hierarchical"


class ReservoirType(Enum):
    """Enumeration of supported reservoir types."""
    STANDARD = "standard"
    HIERARCHICAL = "hierarchical"
    ATTENTIVE = "attentive"
    ECHO_STATE = "echo_state"
    DEEP = "deep"


class ReservoirManagerError(InferenceError):
    """Raised when reservoir management operations fail."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {"operation": operation, "reason": reason}
        if details:
            error_details.update(details)
        
        message = f"Reservoir management failed during {operation}: {reason}"
        super().__init__(message, error_details)
        self.operation = operation


@dataclass
class ReservoirInstance:
    """Container for a reservoir instance with metadata."""
    model: tf.keras.Model
    reservoir_type: ReservoirType
    instance_id: str
    creation_time: float
    usage_count: int = 0
    last_used: float = 0.0
    state: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReservoirOutput:
    """Container for reservoir processing output."""
    output: np.ndarray
    instance_id: str
    reservoir_type: str
    processing_time: float
    confidence: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ReservoirCoordinationResult:
    """Result of coordinating multiple reservoir outputs."""
    coordinated_output: np.ndarray
    individual_outputs: List[ReservoirOutput]
    coordination_strategy: str
    total_processing_time: float
    confidence_scores: List[float]


class ReservoirManager:
    """
    Manages reservoir strategy decisions, multiple reservoir instances,
    and coordinates reservoir outputs for response generation.
    
    This class provides intelligent reservoir management including:
    - Strategy decision making (reuse vs separate vs adaptive)
    - Multiple reservoir instance management
    - Output coordination for response generation
    - Support for different reservoir types (standard, hierarchical, etc.)
    """
    
    def __init__(self,
                 default_strategy: str = "adaptive",
                 max_instances: int = 10,
                 instance_timeout: float = 3600.0,  # 1 hour
                 coordination_strategy: str = "weighted_average",
                 enable_caching: bool = True):
        """
        Initialize ReservoirManager.
        
        Args:
            default_strategy: Default reservoir strategy to use
            max_instances: Maximum number of reservoir instances to maintain
            instance_timeout: Time in seconds before unused instances are cleaned up
            coordination_strategy: Strategy for coordinating multiple outputs
            enable_caching: Whether to enable output caching
        """
        self.default_strategy = ReservoirStrategy(default_strategy)
        self.max_instances = max_instances
        self.instance_timeout = instance_timeout
        self.coordination_strategy = coordination_strategy
        self.enable_caching = enable_caching
        
        # Instance management
        self._instances: Dict[str, ReservoirInstance] = {}
        self._instance_counter = 0
        self._lock = threading.Lock()
        
        # Strategy decision factors
        self._strategy_weights = {
            "sequence_length": 0.3,
            "embedding_complexity": 0.25,
            "system_context": 0.2,
            "performance_history": 0.15,
            "resource_usage": 0.1
        }
        
        # Performance tracking
        self._performance_history = defaultdict(list)
        self._strategy_usage_stats = defaultdict(int)
        
        # Output caching
        self._output_cache: Dict[str, ReservoirOutput] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.info(f"ReservoirManager initialized with strategy: {default_strategy}")
    
    def decide_reservoir_strategy(self,
                                input_data: np.ndarray,
                                system_context: Optional[Any] = None,
                                performance_requirements: Optional[Dict[str, float]] = None) -> ReservoirStrategy:
        """
        Decide the optimal reservoir strategy for the given input.
        
        Args:
            input_data: Input data for processing
            system_context: Optional system context information
            performance_requirements: Optional performance requirements
            
        Returns:
            Recommended reservoir strategy
        """
        try:
            # Calculate decision factors
            factors = self._calculate_decision_factors(
                input_data, system_context, performance_requirements
            )
            
            # Apply strategy decision logic
            strategy = self._apply_strategy_decision_logic(factors)
            
            # Update usage statistics
            self._strategy_usage_stats[strategy.value] += 1
            
            logger.debug(f"Selected strategy: {strategy.value} based on factors: {factors}")
            return strategy
            
        except Exception as e:
            logger.warning(f"Failed to decide reservoir strategy: {e}")
            return self.default_strategy
    
    def get_or_create_reservoir(self,
                              reservoir_type: str = "standard",
                              strategy: Optional[str] = None,
                              config: Optional[Dict[str, Any]] = None) -> ReservoirInstance:
        """
        Get an existing reservoir instance or create a new one.
        
        Args:
            reservoir_type: Type of reservoir to get/create
            strategy: Reservoir strategy being used
            config: Configuration for reservoir creation
            
        Returns:
            ReservoirInstance ready for use
            
        Raises:
            ReservoirManagerError: If reservoir creation fails
        """
        try:
            with self._lock:
                # Clean up expired instances
                self._cleanup_expired_instances()
                
                # Determine if we should reuse an existing instance
                strategy_enum = ReservoirStrategy(strategy or self.default_strategy.value)
                
                if strategy_enum == ReservoirStrategy.REUSE:
                    # Try to find existing instance of the same type
                    existing_instance = self._find_reusable_instance(reservoir_type)
                    if existing_instance:
                        existing_instance.usage_count += 1
                        existing_instance.last_used = time.time()
                        logger.debug(f"Reusing reservoir instance: {existing_instance.instance_id}")
                        return existing_instance
                
                # Create new instance
                return self._create_new_reservoir_instance(reservoir_type, config)
                
        except Exception as e:
            raise ReservoirManagerError(
                "get_or_create_reservoir",
                f"Failed to get or create reservoir: {str(e)}",
                {
                    "reservoir_type": reservoir_type,
                    "strategy": strategy,
                    "config": config
                }
            )
    
    def process_with_reservoir(self,
                             input_data: np.ndarray,
                             reservoir_instance: ReservoirInstance,
                             return_state: bool = False) -> ReservoirOutput:
        """
        Process input data through a specific reservoir instance.
        
        Args:
            input_data: Input data to process
            reservoir_instance: Reservoir instance to use
            return_state: Whether to return internal state
            
        Returns:
            ReservoirOutput with processing results
            
        Raises:
            ReservoirManagerError: If processing fails
        """
        try:
            start_time = time.time()
            
            # Check cache if enabled
            if self.enable_caching:
                cache_key = self._generate_cache_key(input_data, reservoir_instance.instance_id)
                cached_output = self._output_cache.get(cache_key)
                if cached_output is not None:
                    self._cache_hits += 1
                    logger.debug(f"Cache hit for reservoir {reservoir_instance.instance_id}")
                    return cached_output
                self._cache_misses += 1
            
            # Process through reservoir
            if hasattr(reservoir_instance.model, 'predict'):
                # Standard Keras model
                output = reservoir_instance.model.predict(input_data, verbose=0)
            else:
                # Custom model or layer
                output = reservoir_instance.model(input_data)
            
            processing_time = max(0.001, time.time() - start_time)  # Ensure minimum time
            
            # Calculate confidence based on output characteristics
            confidence = self._calculate_output_confidence(output)
            
            # Create result
            result = ReservoirOutput(
                output=output,
                instance_id=reservoir_instance.instance_id,
                reservoir_type=reservoir_instance.reservoir_type.value,
                processing_time=processing_time,
                confidence=confidence,
                metadata={
                    "input_shape": input_data.shape,
                    "output_shape": output.shape,
                    "return_state": return_state
                }
            )
            
            # Update instance statistics
            reservoir_instance.usage_count += 1
            reservoir_instance.last_used = time.time()
            
            # Cache result if enabled
            if self.enable_caching:
                self._output_cache[cache_key] = result
                # Limit cache size
                if len(self._output_cache) > 1000:
                    self._cleanup_cache()
            
            # Update performance history
            self._performance_history[reservoir_instance.reservoir_type.value].append({
                "processing_time": processing_time,
                "confidence": confidence,
                "input_size": input_data.size
            })
            
            logger.debug(f"Processed through reservoir {reservoir_instance.instance_id} in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            raise ReservoirManagerError(
                "process_with_reservoir",
                f"Failed to process with reservoir: {str(e)}",
                {
                    "instance_id": reservoir_instance.instance_id,
                    "input_shape": input_data.shape,
                    "reservoir_type": reservoir_instance.reservoir_type.value
                }
            )
    
    def coordinate_multiple_outputs(self,
                                  reservoir_outputs: List[ReservoirOutput],
                                  coordination_strategy: Optional[str] = None) -> ReservoirCoordinationResult:
        """
        Coordinate outputs from multiple reservoir instances.
        
        Args:
            reservoir_outputs: List of outputs from different reservoirs
            coordination_strategy: Strategy for coordination
            
        Returns:
            ReservoirCoordinationResult with coordinated output
            
        Raises:
            ReservoirManagerError: If coordination fails
        """
        try:
            if not reservoir_outputs:
                raise ValueError("No reservoir outputs provided for coordination")
            
            start_time = time.time()
            strategy = coordination_strategy or self.coordination_strategy
            
            # Extract outputs and confidence scores
            outputs = [ro.output for ro in reservoir_outputs]
            confidences = [ro.confidence for ro in reservoir_outputs]
            
            # Apply coordination strategy
            if strategy == "weighted_average":
                coordinated_output = self._weighted_average_coordination(outputs, confidences)
            elif strategy == "max_confidence":
                coordinated_output = self._max_confidence_coordination(outputs, confidences)
            elif strategy == "ensemble_voting":
                coordinated_output = self._ensemble_voting_coordination(outputs, confidences)
            elif strategy == "hierarchical_merge":
                coordinated_output = self._hierarchical_merge_coordination(outputs, confidences)
            else:
                # Default to simple average
                coordinated_output = np.mean(outputs, axis=0)
            
            total_processing_time = time.time() - start_time
            
            result = ReservoirCoordinationResult(
                coordinated_output=coordinated_output,
                individual_outputs=reservoir_outputs,
                coordination_strategy=strategy,
                total_processing_time=total_processing_time,
                confidence_scores=confidences
            )
            
            logger.debug(f"Coordinated {len(reservoir_outputs)} outputs using {strategy}")
            return result
            
        except Exception as e:
            raise ReservoirManagerError(
                "coordinate_multiple_outputs",
                f"Failed to coordinate outputs: {str(e)}",
                {
                    "num_outputs": len(reservoir_outputs),
                    "coordination_strategy": coordination_strategy or self.coordination_strategy
                }
            )
    
    def manage_reservoir_instances(self) -> Dict[str, Any]:
        """
        Perform maintenance on reservoir instances.
        
        Returns:
            Dictionary with maintenance statistics
        """
        try:
            with self._lock:
                initial_count = len(self._instances)
                
                # Clean up expired instances
                expired_count = self._cleanup_expired_instances()
                
                # Clean up cache
                cache_cleaned = 0
                if self.enable_caching:
                    cache_cleaned = self._cleanup_cache()
                
                # Optimize instance allocation
                optimized_count = self._optimize_instance_allocation()
                
                stats = {
                    "initial_instances": initial_count,
                    "current_instances": len(self._instances),
                    "expired_cleaned": expired_count,
                    "cache_entries_cleaned": cache_cleaned,
                    "instances_optimized": optimized_count,
                    "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
                }
                
                logger.info(f"Reservoir maintenance completed: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Failed to manage reservoir instances: {e}")
            return {"error": str(e)}
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics for reservoir management.
        
        Returns:
            Dictionary with performance statistics
        """
        stats = {
            "strategy_usage": dict(self._strategy_usage_stats),
            "active_instances": len(self._instances),
            "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
            "performance_by_type": {}
        }
        
        # Calculate performance statistics by reservoir type
        for reservoir_type, history in self._performance_history.items():
            if history:
                processing_times = [h["processing_time"] for h in history]
                confidences = [h["confidence"] for h in history]
                
                stats["performance_by_type"][reservoir_type] = {
                    "avg_processing_time": np.mean(processing_times),
                    "avg_confidence": np.mean(confidences),
                    "total_uses": len(history),
                    "min_processing_time": np.min(processing_times),
                    "max_processing_time": np.max(processing_times)
                }
        
        return stats
    
    def reset_statistics(self):
        """Reset all performance statistics."""
        self._performance_history.clear()
        self._strategy_usage_stats.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Reservoir manager statistics reset")
    
    # Private helper methods
    
    def _calculate_decision_factors(self,
                                  input_data: np.ndarray,
                                  system_context: Optional[Any],
                                  performance_requirements: Optional[Dict[str, float]]) -> Dict[str, float]:
        """Calculate factors for strategy decision making."""
        factors = {}
        
        # Sequence length factor
        if len(input_data.shape) > 1:
            factors["sequence_length"] = min(1.0, input_data.shape[1] / 1000.0)
        else:
            factors["sequence_length"] = 0.1
        
        # Embedding complexity factor
        factors["embedding_complexity"] = min(1.0, np.std(input_data) / 2.0)
        
        # System context factor
        factors["system_context"] = 1.0 if system_context is not None else 0.0
        
        # Performance history factor
        if self._performance_history:
            avg_times = []
            for history in self._performance_history.values():
                if history:
                    avg_times.append(np.mean([h["processing_time"] for h in history]))
            factors["performance_history"] = np.mean(avg_times) if avg_times else 0.5
        else:
            factors["performance_history"] = 0.5
        
        # Resource usage factor
        factors["resource_usage"] = min(1.0, len(self._instances) / self.max_instances)
        
        return factors
    
    def _apply_strategy_decision_logic(self, factors: Dict[str, float]) -> ReservoirStrategy:
        """Apply decision logic based on calculated factors."""
        # Calculate weighted score for each strategy
        scores = {}
        
        # REUSE strategy - good for simple, short sequences
        scores[ReservoirStrategy.REUSE] = (
            (1.0 - factors["sequence_length"]) * 0.4 +
            (1.0 - factors["embedding_complexity"]) * 0.3 +
            (1.0 - factors["system_context"]) * 0.2 +
            factors["performance_history"] * 0.1
        )
        
        # SEPARATE strategy - good for complex sequences or system context
        scores[ReservoirStrategy.SEPARATE] = (
            factors["embedding_complexity"] * 0.3 +
            factors["system_context"] * 0.4 +
            factors["sequence_length"] * 0.2 +
            (1.0 - factors["resource_usage"]) * 0.1
        )
        
        # ADAPTIVE strategy - balanced approach
        scores[ReservoirStrategy.ADAPTIVE] = (
            0.5 * (factors["sequence_length"] + factors["embedding_complexity"]) * 0.4 +
            factors["performance_history"] * 0.3 +
            (1.0 - factors["resource_usage"]) * 0.3
        )
        
        # HIERARCHICAL strategy - good for very complex sequences
        scores[ReservoirStrategy.HIERARCHICAL] = (
            factors["sequence_length"] * 0.4 +
            factors["embedding_complexity"] * 0.4 +
            factors["system_context"] * 0.2
        )
        
        # Select strategy with highest score
        best_strategy = max(scores.keys(), key=lambda k: scores[k])
        return best_strategy
    
    def _find_reusable_instance(self, reservoir_type: str) -> Optional[ReservoirInstance]:
        """Find an existing instance that can be reused."""
        for instance in self._instances.values():
            if (instance.reservoir_type.value == reservoir_type and
                (instance.last_used == 0.0 or time.time() - instance.last_used < self.instance_timeout)):
                return instance
        return None
    
    def _create_new_reservoir_instance(self,
                                     reservoir_type: str,
                                     config: Optional[Dict[str, Any]]) -> ReservoirInstance:
        """Create a new reservoir instance."""
        # Check instance limit
        if len(self._instances) >= self.max_instances:
            # Remove least recently used instance
            lru_instance_id = min(
                self._instances.keys(),
                key=lambda k: self._instances[k].last_used if self._instances[k].last_used > 0 else self._instances[k].creation_time
            )
            del self._instances[lru_instance_id]
            logger.debug(f"Removed LRU instance: {lru_instance_id}")
        
        # Generate instance ID
        self._instance_counter += 1
        instance_id = f"{reservoir_type}_{self._instance_counter}_{int(time.time())}"
        
        # Create reservoir model
        reservoir_type_enum = ReservoirType(reservoir_type)
        config = config or {}
        
        if reservoir_type_enum == ReservoirType.STANDARD:
            input_dim = config.get("input_dim", 128)
            hidden_units = config.get("hidden_units", [256, 128, 64])
            model = build_reservoir(input_dim, hidden_units)
        else:
            input_dim = config.get("input_dim", 128)
            model = create_advanced_reservoir(reservoir_type, input_dim, **config)
        
        # Create instance
        instance = ReservoirInstance(
            model=model,
            reservoir_type=reservoir_type_enum,
            instance_id=instance_id,
            creation_time=time.time(),
            metadata=config
        )
        
        self._instances[instance_id] = instance
        logger.info(f"Created new reservoir instance: {instance_id} ({reservoir_type})")
        
        return instance
    
    def _cleanup_expired_instances(self) -> int:
        """Clean up expired reservoir instances."""
        current_time = time.time()
        expired_ids = []
        
        for instance_id, instance in self._instances.items():
            # If last_used is 0.0, use creation_time instead
            last_activity = instance.last_used if instance.last_used > 0.0 else instance.creation_time
            if current_time - last_activity > self.instance_timeout:
                expired_ids.append(instance_id)
        
        for instance_id in expired_ids:
            del self._instances[instance_id]
            logger.debug(f"Cleaned up expired instance: {instance_id}")
        
        return len(expired_ids)
    
    def _cleanup_cache(self) -> int:
        """Clean up output cache."""
        if not self.enable_caching:
            return 0
        
        # Remove oldest entries if cache is too large
        if len(self._output_cache) > 500:
            # Keep only the most recent 300 entries
            # This is a simplified cleanup - in practice might use LRU
            cache_items = list(self._output_cache.items())
            self._output_cache = dict(cache_items[-300:])
            return len(cache_items) - 300
        
        return 0
    
    def _optimize_instance_allocation(self) -> int:
        """Optimize reservoir instance allocation."""
        # Simple optimization: remove unused instances
        current_time = time.time()
        optimized_count = 0
        
        for instance_id, instance in list(self._instances.items()):
            if (instance.usage_count == 0 and 
                current_time - instance.creation_time > 300):  # 5 minutes
                del self._instances[instance_id]
                optimized_count += 1
                logger.debug(f"Optimized unused instance: {instance_id}")
        
        return optimized_count
    
    def _generate_cache_key(self, input_data: np.ndarray, instance_id: str) -> str:
        """Generate cache key for input data and instance."""
        # Simple hash-based key generation
        data_hash = hash(input_data.tobytes())
        return f"{instance_id}_{data_hash}_{input_data.shape}"
    
    def _calculate_output_confidence(self, output: np.ndarray) -> float:
        """Calculate confidence score for reservoir output."""
        # Simple confidence calculation based on output characteristics
        output_norm = np.linalg.norm(output)
        output_std = np.std(output)
        
        # Normalize to 0-1 range
        confidence = min(1.0, (output_norm * output_std) / 10.0)
        return max(0.0, confidence)
    
    # Coordination strategies
    
    def _weighted_average_coordination(self,
                                     outputs: List[np.ndarray],
                                     confidences: List[float]) -> np.ndarray:
        """Coordinate outputs using weighted average based on confidence."""
        if not outputs:
            raise ValueError("No outputs to coordinate")
        
        # Normalize confidences to sum to 1
        total_confidence = sum(confidences)
        if total_confidence == 0:
            weights = [1.0 / len(outputs)] * len(outputs)
        else:
            weights = [c / total_confidence for c in confidences]
        
        # Weighted average
        coordinated = np.zeros_like(outputs[0])
        for output, weight in zip(outputs, weights):
            coordinated += output * weight
        
        return coordinated
    
    def _max_confidence_coordination(self,
                                   outputs: List[np.ndarray],
                                   confidences: List[float]) -> np.ndarray:
        """Coordinate outputs by selecting the one with maximum confidence."""
        max_idx = np.argmax(confidences)
        return outputs[max_idx]
    
    def _ensemble_voting_coordination(self,
                                    outputs: List[np.ndarray],
                                    confidences: List[float]) -> np.ndarray:
        """Coordinate outputs using ensemble voting."""
        # Simple ensemble: average of top-confidence outputs
        if len(outputs) <= 2:
            return self._weighted_average_coordination(outputs, confidences)
        
        # Select top 50% by confidence
        sorted_indices = np.argsort(confidences)[::-1]
        top_count = max(1, len(outputs) // 2)
        top_indices = sorted_indices[:top_count]
        
        top_outputs = [outputs[i] for i in top_indices]
        top_confidences = [confidences[i] for i in top_indices]
        
        return self._weighted_average_coordination(top_outputs, top_confidences)
    
    def _hierarchical_merge_coordination(self,
                                       outputs: List[np.ndarray],
                                       confidences: List[float]) -> np.ndarray:
        """Coordinate outputs using hierarchical merging."""
        if len(outputs) == 1:
            return outputs[0]
        
        # Recursively merge pairs of outputs
        while len(outputs) > 1:
            new_outputs = []
            new_confidences = []
            
            for i in range(0, len(outputs), 2):
                if i + 1 < len(outputs):
                    # Merge pair
                    merged = self._weighted_average_coordination(
                        [outputs[i], outputs[i + 1]],
                        [confidences[i], confidences[i + 1]]
                    )
                    merged_confidence = (confidences[i] + confidences[i + 1]) / 2
                else:
                    # Odd one out
                    merged = outputs[i]
                    merged_confidence = confidences[i]
                
                new_outputs.append(merged)
                new_confidences.append(merged_confidence)
            
            outputs = new_outputs
            confidences = new_confidences
        
        return outputs[0]


# Convenience functions

def create_reservoir_manager(strategy: str = "adaptive",
                           max_instances: int = 10,
                           enable_caching: bool = True) -> ReservoirManager:
    """
    Create a ReservoirManager with standard configuration.
    
    Args:
        strategy: Default reservoir strategy
        max_instances: Maximum number of instances to maintain
        enable_caching: Whether to enable output caching
        
    Returns:
        Configured ReservoirManager instance
    """
    return ReservoirManager(
        default_strategy=strategy,
        max_instances=max_instances,
        enable_caching=enable_caching
    )


def create_high_performance_reservoir_manager() -> ReservoirManager:
    """
    Create a ReservoirManager optimized for high performance.
    
    Returns:
        High-performance ReservoirManager instance
    """
    return ReservoirManager(
        default_strategy="adaptive",
        max_instances=20,
        instance_timeout=7200.0,  # 2 hours
        coordination_strategy="ensemble_voting",
        enable_caching=True
    )