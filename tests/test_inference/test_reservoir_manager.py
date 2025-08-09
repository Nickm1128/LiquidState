#!/usr/bin/env python3
"""
Tests for ReservoirManager.

This module contains comprehensive tests for the ReservoirManager class,
including strategy decision making, instance management, output coordination,
and support for different reservoir types.
"""

import pytest
import numpy as np
import tensorflow as tf
import time
from unittest.mock import Mock, patch, MagicMock

from src.lsm.inference.reservoir_manager import (
    ReservoirManager,
    ReservoirStrategy,
    ReservoirType,
    ReservoirInstance,
    ReservoirOutput,
    ReservoirCoordinationResult,
    ReservoirManagerError,
    create_reservoir_manager,
    create_high_performance_reservoir_manager
)


class TestReservoirManager:
    """Test cases for ReservoirManager class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = ReservoirManager(
            default_strategy="adaptive",
            max_instances=5,
            instance_timeout=60.0,
            enable_caching=True
        )
        
        # Create mock reservoir model
        self.mock_model = Mock()
        self.mock_model.predict = Mock(return_value=np.random.random((1, 64)))
        
        # Test data
        self.test_input = np.random.random((1, 128))
        self.test_system_context = {"message": "test system message"}
    
    def test_initialization(self):
        """Test ReservoirManager initialization."""
        assert self.manager.default_strategy == ReservoirStrategy.ADAPTIVE
        assert self.manager.max_instances == 5
        assert self.manager.instance_timeout == 60.0
        assert self.manager.enable_caching is True
        assert len(self.manager._instances) == 0
    
    def test_decide_reservoir_strategy_simple_input(self):
        """Test strategy decision for simple input."""
        # Simple, short input should prefer REUSE
        simple_input = np.ones((1, 10)) * 0.1  # Low complexity
        strategy = self.manager.decide_reservoir_strategy(simple_input)
        
        assert isinstance(strategy, ReservoirStrategy)
        # Should be one of the valid strategies
        assert strategy in [ReservoirStrategy.REUSE, ReservoirStrategy.ADAPTIVE, 
                          ReservoirStrategy.SEPARATE, ReservoirStrategy.HIERARCHICAL]
    
    def test_decide_reservoir_strategy_complex_input(self):
        """Test strategy decision for complex input."""
        # Complex, long input with high variance
        complex_input = np.random.random((1, 1000)) * 10  # High complexity
        strategy = self.manager.decide_reservoir_strategy(complex_input)
        
        assert isinstance(strategy, ReservoirStrategy)
    
    def test_decide_reservoir_strategy_with_system_context(self):
        """Test strategy decision with system context."""
        strategy = self.manager.decide_reservoir_strategy(
            self.test_input, 
            system_context=self.test_system_context
        )
        
        assert isinstance(strategy, ReservoirStrategy)
        # System context should influence strategy selection
    
    def test_decide_reservoir_strategy_with_performance_requirements(self):
        """Test strategy decision with performance requirements."""
        perf_requirements = {"max_latency": 0.1, "min_accuracy": 0.9}
        strategy = self.manager.decide_reservoir_strategy(
            self.test_input,
            performance_requirements=perf_requirements
        )
        
        assert isinstance(strategy, ReservoirStrategy)
    
    @patch('src.lsm.inference.reservoir_manager.build_reservoir')
    def test_get_or_create_reservoir_new_standard(self, mock_build):
        """Test creating new standard reservoir."""
        mock_build.return_value = self.mock_model
        
        instance = self.manager.get_or_create_reservoir(
            reservoir_type="standard",
            strategy="separate",
            config={"input_dim": 128, "hidden_units": [64, 32]}
        )
        
        assert isinstance(instance, ReservoirInstance)
        assert instance.reservoir_type == ReservoirType.STANDARD
        assert instance.usage_count == 0
        assert instance.model == self.mock_model
        mock_build.assert_called_once()
    
    @patch('src.lsm.inference.reservoir_manager.create_advanced_reservoir')
    def test_get_or_create_reservoir_new_hierarchical(self, mock_create):
        """Test creating new hierarchical reservoir."""
        mock_create.return_value = self.mock_model
        
        instance = self.manager.get_or_create_reservoir(
            reservoir_type="hierarchical",
            strategy="separate",
            config={"input_dim": 128}
        )
        
        assert isinstance(instance, ReservoirInstance)
        assert instance.reservoir_type == ReservoirType.HIERARCHICAL
        mock_create.assert_called_once_with("hierarchical", 128, input_dim=128)
    
    @patch('src.lsm.inference.reservoir_manager.build_reservoir')
    def test_get_or_create_reservoir_reuse_existing(self, mock_build):
        """Test reusing existing reservoir instance."""
        mock_build.return_value = self.mock_model
        
        # Create first instance with reuse strategy
        instance1 = self.manager.get_or_create_reservoir(
            reservoir_type="standard",
            strategy="reuse"
        )
        
        # Try to reuse with REUSE strategy again
        instance2 = self.manager.get_or_create_reservoir(
            reservoir_type="standard",
            strategy="reuse"
        )
        
        # Should reuse the same instance
        assert instance2.instance_id == instance1.instance_id
        assert instance2.usage_count == 1  # Incremented from reuse
    
    @patch('src.lsm.inference.reservoir_manager.build_reservoir')
    def test_get_or_create_reservoir_max_instances(self, mock_build):
        """Test behavior when max instances limit is reached."""
        mock_build.return_value = self.mock_model
        
        # Create max instances
        instances = []
        for i in range(self.manager.max_instances):
            instance = self.manager.get_or_create_reservoir(
                reservoir_type="standard",
                strategy="separate"
            )
            instances.append(instance)
        
        # Creating one more should remove LRU instance
        new_instance = self.manager.get_or_create_reservoir(
            reservoir_type="standard",
            strategy="separate"
        )
        
        assert len(self.manager._instances) == self.manager.max_instances
        assert new_instance.instance_id not in [i.instance_id for i in instances]
    
    @patch('src.lsm.inference.reservoir_manager.build_reservoir')
    def test_process_with_reservoir(self, mock_build):
        """Test processing input through reservoir."""
        expected_output = np.random.random((1, 64))
        self.mock_model.predict.return_value = expected_output
        mock_build.return_value = self.mock_model
        
        # Create instance
        instance = self.manager.get_or_create_reservoir("standard")
        
        # Process input
        result = self.manager.process_with_reservoir(self.test_input, instance)
        
        assert isinstance(result, ReservoirOutput)
        assert np.array_equal(result.output, expected_output)
        assert result.instance_id == instance.instance_id
        assert result.reservoir_type == "standard"
        assert result.processing_time > 0
        assert 0 <= result.confidence <= 1
        assert instance.usage_count == 1
    
    @patch('src.lsm.inference.reservoir_manager.build_reservoir')
    def test_process_with_reservoir_caching(self, mock_build):
        """Test output caching functionality."""
        expected_output = np.random.random((1, 64))
        self.mock_model.predict.return_value = expected_output
        mock_build.return_value = self.mock_model
        
        instance = self.manager.get_or_create_reservoir("standard")
        
        # First call - should miss cache
        result1 = self.manager.process_with_reservoir(self.test_input, instance)
        assert self.manager._cache_misses == 1
        assert self.manager._cache_hits == 0
        
        # Second call with same input - should hit cache
        result2 = self.manager.process_with_reservoir(self.test_input, instance)
        assert self.manager._cache_hits == 1
        
        # Results should be identical
        assert np.array_equal(result1.output, result2.output)
    
    def test_coordinate_multiple_outputs_weighted_average(self):
        """Test coordinating outputs using weighted average."""
        outputs = [
            ReservoirOutput(
                output=np.array([[1.0, 2.0, 3.0]]),
                instance_id="test1",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.8
            ),
            ReservoirOutput(
                output=np.array([[2.0, 3.0, 4.0]]),
                instance_id="test2",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.6
            )
        ]
        
        result = self.manager.coordinate_multiple_outputs(
            outputs, 
            coordination_strategy="weighted_average"
        )
        
        assert isinstance(result, ReservoirCoordinationResult)
        assert result.coordination_strategy == "weighted_average"
        assert len(result.individual_outputs) == 2
        assert len(result.confidence_scores) == 2
        
        # Check weighted average calculation
        expected = (np.array([[1.0, 2.0, 3.0]]) * 0.8 + 
                   np.array([[2.0, 3.0, 4.0]]) * 0.6) / (0.8 + 0.6)
        np.testing.assert_array_almost_equal(result.coordinated_output, expected)
    
    def test_coordinate_multiple_outputs_max_confidence(self):
        """Test coordinating outputs using max confidence."""
        outputs = [
            ReservoirOutput(
                output=np.array([[1.0, 2.0, 3.0]]),
                instance_id="test1",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.6
            ),
            ReservoirOutput(
                output=np.array([[2.0, 3.0, 4.0]]),
                instance_id="test2",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.8
            )
        ]
        
        result = self.manager.coordinate_multiple_outputs(
            outputs,
            coordination_strategy="max_confidence"
        )
        
        # Should select output with highest confidence (second one)
        np.testing.assert_array_equal(
            result.coordinated_output, 
            np.array([[2.0, 3.0, 4.0]])
        )
    
    def test_coordinate_multiple_outputs_ensemble_voting(self):
        """Test coordinating outputs using ensemble voting."""
        outputs = [
            ReservoirOutput(
                output=np.array([[1.0, 2.0]]),
                instance_id="test1",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.9
            ),
            ReservoirOutput(
                output=np.array([[2.0, 3.0]]),
                instance_id="test2",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.7
            ),
            ReservoirOutput(
                output=np.array([[3.0, 4.0]]),
                instance_id="test3",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.5
            )
        ]
        
        result = self.manager.coordinate_multiple_outputs(
            outputs,
            coordination_strategy="ensemble_voting"
        )
        
        assert isinstance(result, ReservoirCoordinationResult)
        assert result.coordination_strategy == "ensemble_voting"
    
    def test_coordinate_multiple_outputs_hierarchical_merge(self):
        """Test coordinating outputs using hierarchical merge."""
        outputs = [
            ReservoirOutput(
                output=np.array([[1.0]]),
                instance_id="test1",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.8
            ),
            ReservoirOutput(
                output=np.array([[2.0]]),
                instance_id="test2",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.6
            ),
            ReservoirOutput(
                output=np.array([[3.0]]),
                instance_id="test3",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.7
            )
        ]
        
        result = self.manager.coordinate_multiple_outputs(
            outputs,
            coordination_strategy="hierarchical_merge"
        )
        
        assert isinstance(result, ReservoirCoordinationResult)
        assert result.coordination_strategy == "hierarchical_merge"
    
    def test_coordinate_multiple_outputs_empty_list(self):
        """Test error handling for empty output list."""
        with pytest.raises(ReservoirManagerError):
            self.manager.coordinate_multiple_outputs([])
    
    @patch('src.lsm.inference.reservoir_manager.build_reservoir')
    def test_manage_reservoir_instances(self, mock_build):
        """Test reservoir instance management and cleanup."""
        mock_build.return_value = self.mock_model
        
        # Create some instances
        instance1 = self.manager.get_or_create_reservoir("standard")
        instance2 = self.manager.get_or_create_reservoir("standard", strategy="separate")
        
        # Simulate usage
        self.manager.process_with_reservoir(self.test_input, instance1)
        
        # Run maintenance
        stats = self.manager.manage_reservoir_instances()
        
        assert isinstance(stats, dict)
        assert "initial_instances" in stats
        assert "current_instances" in stats
        assert "cache_hit_rate" in stats
    
    @patch('src.lsm.inference.reservoir_manager.build_reservoir')
    def test_cleanup_expired_instances(self, mock_build):
        """Test cleanup of expired instances."""
        mock_build.return_value = self.mock_model
        
        # Create instance with short timeout
        manager = ReservoirManager(instance_timeout=0.1)  # 0.1 seconds
        instance = manager.get_or_create_reservoir("standard")
        
        assert len(manager._instances) == 1
        
        # Wait for expiration
        time.sleep(0.2)
        
        # Trigger cleanup
        manager.manage_reservoir_instances()
        
        # Instance should be cleaned up
        assert len(manager._instances) == 0
    
    def test_get_performance_statistics(self):
        """Test getting performance statistics."""
        stats = self.manager.get_performance_statistics()
        
        assert isinstance(stats, dict)
        assert "strategy_usage" in stats
        assert "active_instances" in stats
        assert "cache_hit_rate" in stats
        assert "performance_by_type" in stats
    
    def test_reset_statistics(self):
        """Test resetting performance statistics."""
        # Add some usage
        self.manager._strategy_usage_stats["reuse"] = 5
        self.manager._cache_hits = 10
        self.manager._cache_misses = 5
        
        # Reset
        self.manager.reset_statistics()
        
        # Should be cleared
        assert len(self.manager._strategy_usage_stats) == 0
        assert self.manager._cache_hits == 0
        assert self.manager._cache_misses == 0
    
    def test_calculate_decision_factors(self):
        """Test calculation of decision factors."""
        factors = self.manager._calculate_decision_factors(
            self.test_input,
            self.test_system_context,
            {"max_latency": 0.1}
        )
        
        assert isinstance(factors, dict)
        assert "sequence_length" in factors
        assert "embedding_complexity" in factors
        assert "system_context" in factors
        assert "performance_history" in factors
        assert "resource_usage" in factors
        
        # All factors should be between 0 and 1
        for factor_value in factors.values():
            assert 0 <= factor_value <= 1
    
    def test_apply_strategy_decision_logic(self):
        """Test strategy decision logic."""
        # Test with different factor combinations
        factors1 = {
            "sequence_length": 0.1,
            "embedding_complexity": 0.1,
            "system_context": 0.0,
            "performance_history": 0.8,
            "resource_usage": 0.2
        }
        strategy1 = self.manager._apply_strategy_decision_logic(factors1)
        assert isinstance(strategy1, ReservoirStrategy)
        
        factors2 = {
            "sequence_length": 0.9,
            "embedding_complexity": 0.9,
            "system_context": 1.0,
            "performance_history": 0.5,
            "resource_usage": 0.8
        }
        strategy2 = self.manager._apply_strategy_decision_logic(factors2)
        assert isinstance(strategy2, ReservoirStrategy)
    
    def test_calculate_output_confidence(self):
        """Test output confidence calculation."""
        # Test with different output characteristics
        output1 = np.ones((1, 64)) * 0.1  # Low magnitude
        confidence1 = self.manager._calculate_output_confidence(output1)
        assert 0 <= confidence1 <= 1
        
        output2 = np.random.random((1, 64)) * 10  # High magnitude, high variance
        confidence2 = self.manager._calculate_output_confidence(output2)
        assert 0 <= confidence2 <= 1
        
        # Higher magnitude/variance should generally give higher confidence
        # (though this is a simple heuristic)
    
    def test_generate_cache_key(self):
        """Test cache key generation."""
        key1 = self.manager._generate_cache_key(self.test_input, "instance1")
        key2 = self.manager._generate_cache_key(self.test_input, "instance2")
        key3 = self.manager._generate_cache_key(self.test_input, "instance1")
        
        # Same input and instance should generate same key
        assert key1 == key3
        
        # Different instance should generate different key
        assert key1 != key2
    
    def test_error_handling_invalid_reservoir_type(self):
        """Test error handling for invalid reservoir type."""
        with pytest.raises(ReservoirManagerError):
            self.manager.get_or_create_reservoir("invalid_type")
    
    def test_error_handling_processing_failure(self):
        """Test error handling when processing fails."""
        # Create mock that raises exception
        failing_model = Mock()
        failing_model.predict.side_effect = Exception("Processing failed")
        
        instance = ReservoirInstance(
            model=failing_model,
            reservoir_type=ReservoirType.STANDARD,
            instance_id="failing_instance",
            creation_time=time.time()
        )
        
        with pytest.raises(ReservoirManagerError):
            self.manager.process_with_reservoir(self.test_input, instance)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_reservoir_manager(self):
        """Test create_reservoir_manager function."""
        manager = create_reservoir_manager(
            strategy="reuse",
            max_instances=15,
            enable_caching=False
        )
        
        assert isinstance(manager, ReservoirManager)
        assert manager.default_strategy == ReservoirStrategy.REUSE
        assert manager.max_instances == 15
        assert manager.enable_caching is False
    
    def test_create_high_performance_reservoir_manager(self):
        """Test create_high_performance_reservoir_manager function."""
        manager = create_high_performance_reservoir_manager()
        
        assert isinstance(manager, ReservoirManager)
        assert manager.default_strategy == ReservoirStrategy.ADAPTIVE
        assert manager.max_instances == 20
        assert manager.instance_timeout == 7200.0
        assert manager.coordination_strategy == "ensemble_voting"
        assert manager.enable_caching is True


class TestReservoirStrategy:
    """Test ReservoirStrategy enum."""
    
    def test_strategy_values(self):
        """Test strategy enum values."""
        assert ReservoirStrategy.REUSE.value == "reuse"
        assert ReservoirStrategy.SEPARATE.value == "separate"
        assert ReservoirStrategy.ADAPTIVE.value == "adaptive"
        assert ReservoirStrategy.HIERARCHICAL.value == "hierarchical"


class TestReservoirType:
    """Test ReservoirType enum."""
    
    def test_type_values(self):
        """Test type enum values."""
        assert ReservoirType.STANDARD.value == "standard"
        assert ReservoirType.HIERARCHICAL.value == "hierarchical"
        assert ReservoirType.ATTENTIVE.value == "attentive"
        assert ReservoirType.ECHO_STATE.value == "echo_state"
        assert ReservoirType.DEEP.value == "deep"


class TestDataClasses:
    """Test data classes."""
    
    def test_reservoir_instance(self):
        """Test ReservoirInstance dataclass."""
        mock_model = Mock()
        instance = ReservoirInstance(
            model=mock_model,
            reservoir_type=ReservoirType.STANDARD,
            instance_id="test_instance",
            creation_time=time.time(),
            usage_count=5,
            metadata={"test": "data"}
        )
        
        assert instance.model == mock_model
        assert instance.reservoir_type == ReservoirType.STANDARD
        assert instance.instance_id == "test_instance"
        assert instance.usage_count == 5
        assert instance.metadata == {"test": "data"}
    
    def test_reservoir_output(self):
        """Test ReservoirOutput dataclass."""
        output = ReservoirOutput(
            output=np.array([[1, 2, 3]]),
            instance_id="test_instance",
            reservoir_type="standard",
            processing_time=0.1,
            confidence=0.8,
            metadata={"test": "data"}
        )
        
        assert np.array_equal(output.output, np.array([[1, 2, 3]]))
        assert output.instance_id == "test_instance"
        assert output.reservoir_type == "standard"
        assert output.processing_time == 0.1
        assert output.confidence == 0.8
        assert output.metadata == {"test": "data"}
    
    def test_reservoir_coordination_result(self):
        """Test ReservoirCoordinationResult dataclass."""
        individual_outputs = [
            ReservoirOutput(
                output=np.array([[1, 2]]),
                instance_id="test1",
                reservoir_type="standard",
                processing_time=0.1,
                confidence=0.8
            )
        ]
        
        result = ReservoirCoordinationResult(
            coordinated_output=np.array([[1.5, 2.5]]),
            individual_outputs=individual_outputs,
            coordination_strategy="weighted_average",
            total_processing_time=0.2,
            confidence_scores=[0.8]
        )
        
        assert np.array_equal(result.coordinated_output, np.array([[1.5, 2.5]]))
        assert len(result.individual_outputs) == 1
        assert result.coordination_strategy == "weighted_average"
        assert result.total_processing_time == 0.2
        assert result.confidence_scores == [0.8]


if __name__ == "__main__":
    pytest.main([__file__])