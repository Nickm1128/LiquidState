"""
Tests for the PipelineOrchestrator class.

This module tests the main pipeline coordinator functionality including
component management, swapping, and configuration handling.
"""

import pytest
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock

from src.lsm.pipeline.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfiguration,
    ComponentRegistry,
    ComponentType,
    ArchitectureType,
    ComponentSpec,
    PipelineError,
    ComponentSwapError,
    ConfigurationError,
    create_pipeline_orchestrator,
    create_experimental_pipeline
)


class TestComponentRegistry:
    """Test the ComponentRegistry class."""
    
    def test_component_registry_initialization(self):
        """Test that ComponentRegistry initializes correctly."""
        registry = ComponentRegistry()
        
        # Should have some default components registered
        components = registry.list_components()
        assert len(components) > 0
        
        # Should have at least some basic component types
        expected_types = [
            ComponentType.DATASET_LOADER,
            ComponentType.TOKENIZER,
            ComponentType.EMBEDDER
        ]
        
        for component_type in expected_types:
            if component_type in components:
                assert len(components[component_type]) > 0
    
    def test_register_component(self):
        """Test component registration."""
        registry = ComponentRegistry()
        
        # Create a mock component class
        mock_component = Mock()
        
        # Register the component
        registry.register_component(ComponentType.TOKENIZER, "test_tokenizer", mock_component)
        
        # Verify it was registered
        components = registry.list_components(ComponentType.TOKENIZER)
        assert "test_tokenizer" in components[ComponentType.TOKENIZER]
        
        # Verify we can retrieve it
        retrieved = registry.get_component(ComponentType.TOKENIZER, "test_tokenizer")
        assert retrieved == mock_component
    
    def test_get_nonexistent_component(self):
        """Test getting a component that doesn't exist."""
        registry = ComponentRegistry()
        
        with pytest.raises(ComponentSwapError):
            registry.get_component(ComponentType.TOKENIZER, "nonexistent")
    
    def test_get_default_component(self):
        """Test getting default component."""
        registry = ComponentRegistry()
        
        # Register a component as default
        mock_component = Mock()
        registry.register_component(ComponentType.TOKENIZER, "default_test", mock_component)
        
        # Should be able to get it without specifying name
        retrieved = registry.get_component(ComponentType.TOKENIZER)
        assert retrieved is not None


class TestPipelineConfiguration:
    """Test the PipelineConfiguration class."""
    
    def test_pipeline_configuration_creation(self):
        """Test creating a pipeline configuration."""
        config = PipelineConfiguration(
            architecture_type=ArchitectureType.STANDARD_2D,
            experiment_name="test_experiment",
            description="Test configuration"
        )
        
        assert config.architecture_type == ArchitectureType.STANDARD_2D
        assert config.experiment_name == "test_experiment"
        assert config.description == "Test configuration"
        assert isinstance(config.components, dict)
        assert isinstance(config.global_config, dict)


class TestPipelineOrchestrator:
    """Test the PipelineOrchestrator class."""
    
    def test_orchestrator_initialization(self):
        """Test that PipelineOrchestrator initializes correctly."""
        orchestrator = PipelineOrchestrator()
        
        assert orchestrator.configuration is not None
        assert orchestrator.registry is not None
        assert orchestrator._pipeline_state == "initialized"
    
    def test_orchestrator_with_custom_configuration(self):
        """Test orchestrator with custom configuration."""
        config = PipelineConfiguration(
            architecture_type=ArchitectureType.SYSTEM_AWARE_3D,
            experiment_name="custom_test"
        )
        
        orchestrator = PipelineOrchestrator(config)
        
        assert orchestrator.configuration.architecture_type == ArchitectureType.SYSTEM_AWARE_3D
        assert orchestrator.configuration.experiment_name == "custom_test"
    
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator._initialize_components')
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator._validate_dependencies')
    def test_setup_pipeline(self, mock_validate, mock_initialize):
        """Test pipeline setup."""
        orchestrator = PipelineOrchestrator()
        
        orchestrator.setup_pipeline()
        
        mock_initialize.assert_called_once()
        mock_validate.assert_called_once()
        assert orchestrator._pipeline_state == "ready"
    
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator._initialize_components')
    def test_setup_pipeline_with_config(self, mock_initialize):
        """Test pipeline setup with additional config."""
        orchestrator = PipelineOrchestrator()
        
        additional_config = {"test_param": "test_value"}
        orchestrator.setup_pipeline(additional_config)
        
        assert orchestrator.configuration.global_config["test_param"] == "test_value"
    
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator._initialize_components')
    def test_setup_pipeline_failure(self, mock_initialize):
        """Test pipeline setup failure handling."""
        orchestrator = PipelineOrchestrator()
        
        # Make initialization fail
        mock_initialize.side_effect = Exception("Test error")
        
        with pytest.raises(PipelineError):
            orchestrator.setup_pipeline()
        
        assert orchestrator._pipeline_state == "error"
    
    def test_component_swapping(self):
        """Test component swapping functionality."""
        orchestrator = PipelineOrchestrator()
        
        # Create mock components
        old_component = Mock()
        new_component_class = Mock()
        new_component_instance = Mock()
        new_component_class.return_value = new_component_instance
        
        # Set up initial component
        orchestrator._component_instances[ComponentType.TOKENIZER] = old_component
        
        # Register new component in registry
        orchestrator.registry.register_component(
            ComponentType.TOKENIZER, "new_tokenizer", new_component_class
        )
        
        # Swap component
        orchestrator.swap_component(ComponentType.TOKENIZER, "new_tokenizer")
        
        # Verify swap occurred
        assert orchestrator._component_instances[ComponentType.TOKENIZER] == new_component_instance
        new_component_class.assert_called_once()
    
    def test_component_swapping_with_config(self):
        """Test component swapping with configuration."""
        orchestrator = PipelineOrchestrator()
        
        # Create mock component
        new_component_class = Mock()
        new_component_instance = Mock()
        new_component_class.return_value = new_component_instance
        
        # Register component
        orchestrator.registry.register_component(
            ComponentType.TOKENIZER, "configured_tokenizer", new_component_class
        )
        
        # Swap with config
        config = {"param1": "value1", "param2": "value2"}
        orchestrator.swap_component(ComponentType.TOKENIZER, "configured_tokenizer", config)
        
        # Verify component was created with config
        new_component_class.assert_called_once_with(**config)
    
    def test_component_swapping_failure(self):
        """Test component swapping failure handling."""
        orchestrator = PipelineOrchestrator()
        
        with pytest.raises(ComponentSwapError):
            orchestrator.swap_component(ComponentType.TOKENIZER, "nonexistent_component")
    
    def test_get_component(self):
        """Test getting component instances."""
        orchestrator = PipelineOrchestrator()
        
        # Add a mock component
        mock_component = Mock()
        orchestrator._component_instances[ComponentType.TOKENIZER] = mock_component
        
        # Retrieve component
        retrieved = orchestrator.get_component(ComponentType.TOKENIZER)
        assert retrieved == mock_component
    
    def test_get_nonexistent_component(self):
        """Test getting a component that doesn't exist."""
        orchestrator = PipelineOrchestrator()
        
        with pytest.raises(PipelineError):
            orchestrator.get_component(ComponentType.TOKENIZER)
    
    def test_pipeline_status(self):
        """Test getting pipeline status."""
        config = PipelineConfiguration(
            architecture_type=ArchitectureType.SYSTEM_AWARE_3D,
            experiment_name="status_test",
            description="Status test description"
        )
        orchestrator = PipelineOrchestrator(config)
        
        # Add a mock component
        mock_component = Mock()
        mock_component.__class__.__name__ = "MockComponent"
        orchestrator._component_instances[ComponentType.TOKENIZER] = mock_component
        
        status = orchestrator.get_pipeline_status()
        
        assert status["state"] == "initialized"
        assert status["architecture"] == "system_aware_3d"
        assert status["experiment_name"] == "status_test"
        assert status["description"] == "Status test description"
        assert "tokenizer" in status["components"]
        assert status["components"]["tokenizer"] == "MockComponent"
    
    def test_save_configuration(self):
        """Test saving pipeline configuration."""
        config = PipelineConfiguration(
            architecture_type=ArchitectureType.HYBRID,
            experiment_name="save_test",
            description="Save test",
            global_config={"param": "value"}
        )
        orchestrator = PipelineOrchestrator(config)
        
        # Add a mock component
        mock_component = Mock()
        mock_component.__class__.__name__ = "MockComponent"
        mock_component._config = {"component_param": "component_value"}
        orchestrator._component_instances[ComponentType.TOKENIZER] = mock_component
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name
        
        try:
            orchestrator.save_configuration(filepath)
            
            # Verify file was created and contains expected data
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                saved_config = json.load(f)
            
            assert saved_config["architecture_type"] == "hybrid"
            assert saved_config["experiment_name"] == "save_test"
            assert saved_config["description"] == "Save test"
            assert saved_config["global_config"]["param"] == "value"
            assert "tokenizer" in saved_config["components"]
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_load_configuration(self):
        """Test loading pipeline configuration."""
        # Create test configuration file
        test_config = {
            "architecture_type": "standard_2d",
            "experiment_name": "load_test",
            "description": "Load test",
            "global_config": {"param": "value"},
            "components": {
                "tokenizer": {
                    "class_name": "MockTokenizer",
                    "config": {"tokenizer_param": "tokenizer_value"}
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            json.dump(test_config, f)
            filepath = f.name
        
        try:
            orchestrator = PipelineOrchestrator()
            orchestrator.load_configuration(filepath)
            
            assert orchestrator.configuration.architecture_type == ArchitectureType.STANDARD_2D
            assert orchestrator.configuration.experiment_name == "load_test"
            assert orchestrator.configuration.description == "Load test"
            assert orchestrator.configuration.global_config["param"] == "value"
            assert orchestrator._pipeline_state == "initialized"
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator.get_component')
    def test_process_input_not_ready(self, mock_get_component):
        """Test processing input when pipeline is not ready."""
        orchestrator = PipelineOrchestrator()
        # Don't call setup_pipeline, so state remains "initialized"
        
        with pytest.raises(PipelineError):
            orchestrator.process_input("test input")
    
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator.get_component')
    def test_process_standard_input(self, mock_get_component):
        """Test processing input through standard pipeline."""
        orchestrator = PipelineOrchestrator()
        orchestrator._pipeline_state = "ready"
        
        # Mock components
        mock_tokenizer = Mock()
        mock_embedder = Mock()
        mock_reservoir = Mock()
        mock_response_generator = Mock()
        
        mock_tokenizer.tokenize.return_value = [["token1", "token2"]]
        mock_embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_response_generator.generate_complete_response.return_value = "Generated response"
        
        def get_component_side_effect(component_type):
            if component_type == ComponentType.TOKENIZER:
                return mock_tokenizer
            elif component_type == ComponentType.EMBEDDER:
                return mock_embedder
            elif component_type == ComponentType.RESERVOIR:
                return mock_reservoir
            elif component_type == ComponentType.RESPONSE_GENERATOR:
                return mock_response_generator
            else:
                raise PipelineError(f"Component {component_type} not available")
        
        mock_get_component.side_effect = get_component_side_effect
        
        result = orchestrator.process_input("test input", "standard")
        
        assert result == "Generated response"
        mock_tokenizer.tokenize.assert_called_once_with(["test input"])
        mock_embedder.embed.assert_called_once_with(["token1", "token2"])
        mock_response_generator.generate_complete_response.assert_called_once()


class TestFactoryFunctions:
    """Test the factory functions."""
    
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator.setup_pipeline')
    def test_create_pipeline_orchestrator(self, mock_setup):
        """Test creating a pipeline orchestrator."""
        orchestrator = create_pipeline_orchestrator(
            architecture_type=ArchitectureType.SYSTEM_AWARE_3D,
            experiment_name="factory_test",
            description="Factory test description"
        )
        
        assert isinstance(orchestrator, PipelineOrchestrator)
        assert orchestrator.configuration.architecture_type == ArchitectureType.SYSTEM_AWARE_3D
        assert orchestrator.configuration.experiment_name == "factory_test"
        assert orchestrator.configuration.description == "Factory test description"
        mock_setup.assert_called_once()
    
    @patch('src.lsm.pipeline.pipeline_orchestrator.PipelineOrchestrator.setup_pipeline')
    def test_create_experimental_pipeline(self, mock_setup):
        """Test creating an experimental pipeline."""
        # Create mock component specs
        mock_component_class = Mock()
        components = {
            ComponentType.TOKENIZER: ComponentSpec(
                component_type=ComponentType.TOKENIZER,
                component_class=mock_component_class,
                config={"param": "value"}
            )
        }
        
        orchestrator = create_experimental_pipeline(
            components=components,
            experiment_name="experimental_test",
            description="Experimental test description"
        )
        
        assert isinstance(orchestrator, PipelineOrchestrator)
        assert orchestrator.configuration.architecture_type == ArchitectureType.EXPERIMENTAL
        assert orchestrator.configuration.experiment_name == "experimental_test"
        assert orchestrator.configuration.description == "Experimental test description"
        assert ComponentType.TOKENIZER in orchestrator.configuration.components
        mock_setup.assert_called_once()


class TestIntegration:
    """Integration tests for the pipeline orchestrator."""
    
    @patch('src.lsm.data.StandardTokenizerWrapper')
    @patch('src.lsm.data.SinusoidalEmbedder')
    @patch('src.lsm.inference.ResponseGenerator')
    def test_full_pipeline_integration(self, mock_response_gen, mock_embedder, mock_tokenizer):
        """Test full pipeline integration with mocked components."""
        # Set up mocks
        mock_tokenizer_instance = Mock()
        mock_embedder_instance = Mock()
        mock_response_gen_instance = Mock()
        
        mock_tokenizer.return_value = mock_tokenizer_instance
        mock_embedder.return_value = mock_embedder_instance
        mock_response_gen.return_value = mock_response_gen_instance
        
        mock_tokenizer_instance.tokenize.return_value = [["hello", "world"]]
        mock_embedder_instance.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_response_gen_instance.generate_complete_response.return_value = "Hello, world!"
        
        # Create and setup pipeline
        orchestrator = create_pipeline_orchestrator(ArchitectureType.STANDARD_2D)
        
        # Process input
        result = orchestrator.process_input("Hello world")
        
        # Verify the pipeline worked
        assert result == "Hello, world!"
        mock_tokenizer_instance.tokenize.assert_called()
        mock_embedder_instance.embed.assert_called()
        mock_response_gen_instance.generate_complete_response.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])