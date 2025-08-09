"""
Pipeline orchestrator for managing all LSM components in a modular architecture.

This module provides the main pipeline coordinator that manages all components,
supports component swapping and experimentation, and handles configuration
management for different architectures.
"""

import logging
from typing import Dict, Any, Optional, List, Union, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
import json
import copy

from ..utils.lsm_exceptions import LSMError
from ..utils.lsm_logging import get_logger


class PipelineError(LSMError):
    """Base exception for pipeline orchestration errors."""
    pass


class ComponentSwapError(PipelineError):
    """Exception raised when component swapping fails."""
    pass


class ConfigurationError(PipelineError):
    """Exception raised when configuration is invalid."""
    pass


class ComponentType(Enum):
    """Types of components that can be managed by the pipeline."""
    DATASET_LOADER = "dataset_loader"
    TOKENIZER = "tokenizer"
    EMBEDDER = "embedder"
    RESERVOIR = "reservoir"
    CNN_PROCESSOR = "cnn_processor"
    RESPONSE_GENERATOR = "response_generator"
    SYSTEM_MESSAGE_PROCESSOR = "system_message_processor"
    MESSAGE_ANNOTATOR = "message_annotator"
    TRAINER = "trainer"
    INFERENCE_ENGINE = "inference_engine"


class ArchitectureType(Enum):
    """Supported architecture types."""
    STANDARD_2D = "standard_2d"
    SYSTEM_AWARE_3D = "system_aware_3d"
    HYBRID = "hybrid"
    EXPERIMENTAL = "experimental"


@dataclass
class ComponentSpec:
    """Specification for a pipeline component."""
    component_type: ComponentType
    component_class: Type
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[ComponentType] = field(default_factory=list)
    optional: bool = False


@dataclass
class PipelineConfiguration:
    """Configuration for the entire pipeline."""
    architecture_type: ArchitectureType
    components: Dict[ComponentType, ComponentSpec] = field(default_factory=dict)
    global_config: Dict[str, Any] = field(default_factory=dict)
    experiment_name: Optional[str] = None
    description: Optional[str] = None


class ComponentRegistry:
    """Registry for managing available components."""
    
    def __init__(self):
        self._components: Dict[ComponentType, Dict[str, Type]] = {}
        self._default_components: Dict[ComponentType, str] = {}
        self.logger = get_logger(__name__)
        
        # Register default components
        self._register_default_components()
    
    def _register_default_components(self):
        """Register default components from the LSM package."""
        try:
            # Data components
            from ..data import (
                HuggingFaceDatasetLoader, StandardTokenizerWrapper, 
                SinusoidalEmbedder, MessageAnnotator
            )
            self.register_component(ComponentType.DATASET_LOADER, "huggingface", HuggingFaceDatasetLoader)
            self.register_component(ComponentType.TOKENIZER, "standard", StandardTokenizerWrapper)
            self.register_component(ComponentType.EMBEDDER, "sinusoidal", SinusoidalEmbedder)
            self.register_component(ComponentType.MESSAGE_ANNOTATOR, "default", MessageAnnotator)
            
            # Core components
            from ..core import (
                ReservoirLayer, CNN3DProcessor, SystemMessageProcessor,
                EmbeddingModifierGenerator, CNNArchitectureFactory
            )
            self.register_component(ComponentType.RESERVOIR, "standard", ReservoirLayer)
            self.register_component(ComponentType.CNN_PROCESSOR, "3d", CNN3DProcessor)
            self.register_component(ComponentType.SYSTEM_MESSAGE_PROCESSOR, "default", SystemMessageProcessor)
            
            # Inference components
            from ..inference import ResponseGenerator, OptimizedLSMInference
            self.register_component(ComponentType.RESPONSE_GENERATOR, "default", ResponseGenerator)
            self.register_component(ComponentType.INFERENCE_ENGINE, "optimized", OptimizedLSMInference)
            
            # Training components
            from ..training import LSMTrainer
            if LSMTrainer is not None:
                self.register_component(ComponentType.TRAINER, "default", LSMTrainer)
            
            # Set up default configurations for components that require parameters
            self._setup_default_configurations()
            
            self.logger.info("Default components registered successfully")
            
        except ImportError as e:
            self.logger.warning(f"Some default components could not be registered: {e}")
    
    def _setup_default_configurations(self):
        """Set up default configurations for components that require parameters."""
        self._default_configs = {
            ComponentType.RESERVOIR: {
                "units": 100,
                "sparsity": 0.1,
                "frequency": 1.0,
                "amplitude": 1.0,
                "decay": 0.1
            },
            ComponentType.TOKENIZER: {
                "tokenizer_name": "gpt2"
            },
            ComponentType.EMBEDDER: {
                "vocab_size": 50257,  # GPT-2 vocab size
                "embedding_dim": 128
            },
            ComponentType.CNN_PROCESSOR: {
                "reservoir_shape": (1, 10, 10, 100),  # Default 3D shape
                "system_embedding_dim": 64,
                "output_embedding_dim": 128
            },
            ComponentType.RESPONSE_GENERATOR: {
                "reservoir_strategy": "reuse"
            }
        }
    
    def register_component(self, component_type: ComponentType, name: str, component_class: Type):
        """Register a component in the registry."""
        if component_type not in self._components:
            self._components[component_type] = {}
        
        self._components[component_type][name] = component_class
        
        # Set as default if it's the first component of this type
        if component_type not in self._default_components:
            self._default_components[component_type] = name
        
        self.logger.debug(f"Registered component {name} for type {component_type}")
    
    def get_component(self, component_type: ComponentType, name: Optional[str] = None) -> Type:
        """Get a component class from the registry."""
        if component_type not in self._components:
            raise ComponentSwapError(f"No components registered for type {component_type}")
        
        if name is None:
            name = self._default_components.get(component_type)
            if name is None:
                raise ComponentSwapError(f"No default component for type {component_type}")
        
        if name not in self._components[component_type]:
            available = list(self._components[component_type].keys())
            raise ComponentSwapError(
                f"Component '{name}' not found for type {component_type}. "
                f"Available: {available}"
            )
        
        return self._components[component_type][name]
    
    def list_components(self, component_type: Optional[ComponentType] = None) -> Dict[ComponentType, List[str]]:
        """List all available components."""
        if component_type is not None:
            return {component_type: list(self._components.get(component_type, {}).keys())}
        
        return {ct: list(components.keys()) for ct, components in self._components.items()}


class PipelineOrchestrator:
    """
    Main pipeline coordinator that manages all components and supports
    component swapping and experimentation.
    """
    
    def __init__(self, configuration: Optional[PipelineConfiguration] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            configuration: Pipeline configuration. If None, creates a default configuration.
        """
        self.logger = get_logger(__name__)
        self.registry = ComponentRegistry()
        self.configuration = configuration or self._create_default_configuration()
        self._components: Dict[ComponentType, Any] = {}
        self._component_instances: Dict[ComponentType, Any] = {}
        self._pipeline_state = "initialized"
        
        self.logger.info(f"Pipeline orchestrator initialized with architecture: {self.configuration.architecture_type}")
    
    def _create_default_configuration(self) -> PipelineConfiguration:
        """Create a default pipeline configuration."""
        return PipelineConfiguration(
            architecture_type=ArchitectureType.STANDARD_2D,
            description="Default LSM pipeline configuration"
        )
    
    def setup_pipeline(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Set up the pipeline with the current configuration.
        
        Args:
            config: Additional configuration parameters to merge with existing config.
        """
        try:
            self.logger.info("Setting up pipeline...")
            
            # Merge additional config if provided
            if config:
                self.configuration.global_config.update(config)
            
            # Initialize components based on architecture type
            self._initialize_components()
            
            # Validate component dependencies
            self._validate_dependencies()
            
            self._pipeline_state = "ready"
            self.logger.info("Pipeline setup completed successfully")
            
        except Exception as e:
            self._pipeline_state = "error"
            self.logger.error(f"Pipeline setup failed: {e}")
            raise PipelineError(f"Failed to setup pipeline: {e}") from e
    
    def _initialize_components(self) -> None:
        """Initialize components based on the current configuration."""
        architecture = self.configuration.architecture_type
        
        if architecture == ArchitectureType.STANDARD_2D:
            self._setup_standard_2d_pipeline()
        elif architecture == ArchitectureType.SYSTEM_AWARE_3D:
            self._setup_system_aware_3d_pipeline()
        elif architecture == ArchitectureType.HYBRID:
            self._setup_hybrid_pipeline()
        else:
            self._setup_experimental_pipeline()
    
    def _setup_standard_2d_pipeline(self) -> None:
        """Set up a standard 2D CNN pipeline."""
        # Initialize basic components first
        basic_components = [
            ComponentType.DATASET_LOADER,
            ComponentType.TOKENIZER,
            ComponentType.EMBEDDER,
            ComponentType.RESERVOIR
        ]
        
        for component_type in basic_components:
            self._initialize_component(component_type, optional=True)
        
        # Initialize complex components that may have dependencies
        complex_components = [
            ComponentType.CNN_PROCESSOR,
            ComponentType.RESPONSE_GENERATOR,
            ComponentType.INFERENCE_ENGINE
        ]
        
        for component_type in complex_components:
            self._initialize_component(component_type, optional=True)
    
    def _setup_system_aware_3d_pipeline(self) -> None:
        """Set up a system-aware 3D CNN pipeline."""
        # Initialize basic components first
        basic_components = [
            ComponentType.DATASET_LOADER,
            ComponentType.TOKENIZER,
            ComponentType.EMBEDDER,
            ComponentType.RESERVOIR,
            ComponentType.SYSTEM_MESSAGE_PROCESSOR,
            ComponentType.MESSAGE_ANNOTATOR
        ]
        
        for component_type in basic_components:
            self._initialize_component(component_type, optional=True)
        
        # Initialize complex components that may have dependencies
        complex_components = [
            ComponentType.CNN_PROCESSOR,
            ComponentType.RESPONSE_GENERATOR,
            ComponentType.INFERENCE_ENGINE
        ]
        
        for component_type in complex_components:
            self._initialize_component(component_type, optional=True)
    
    def _setup_hybrid_pipeline(self) -> None:
        """Set up a hybrid pipeline with both 2D and 3D capabilities."""
        # Initialize all available components for maximum flexibility
        for component_type in ComponentType:
            try:
                self._initialize_component(component_type, optional=True)
            except ComponentSwapError:
                self.logger.debug(f"Optional component {component_type} not available")
    
    def _setup_experimental_pipeline(self) -> None:
        """Set up an experimental pipeline with custom configuration."""
        # Use component specifications from configuration
        for component_type, spec in self.configuration.components.items():
            self._initialize_component_from_spec(spec)
    
    def _initialize_component(self, component_type: ComponentType, optional: bool = False) -> None:
        """Initialize a single component."""
        try:
            # Get component class from registry
            component_class = self.registry.get_component(component_type)
            
            # Get configuration for this component
            component_config = self.configuration.global_config.get(
                component_type.value, {}
            )
            
            # Merge with default configuration if available
            if hasattr(self.registry, '_default_configs') and component_type in self.registry._default_configs:
                default_config = self.registry._default_configs[component_type].copy()
                default_config.update(component_config)
                component_config = default_config
            
            # Create component instance
            if component_config:
                component_instance = component_class(**component_config)
            else:
                component_instance = component_class()
            
            self._component_instances[component_type] = component_instance
            self.logger.debug(f"Initialized component: {component_type}")
            
        except Exception as e:
            if optional:
                self.logger.debug(f"Optional component {component_type} initialization failed: {e}")
            else:
                raise ComponentSwapError(f"Failed to initialize {component_type}: {e}") from e
    
    def _initialize_component_from_spec(self, spec: ComponentSpec) -> None:
        """Initialize a component from a specification."""
        try:
            component_instance = spec.component_class(**spec.config)
            self._component_instances[spec.component_type] = component_instance
            self.logger.debug(f"Initialized component from spec: {spec.component_type}")
            
        except Exception as e:
            if spec.optional:
                self.logger.debug(f"Optional component {spec.component_type} initialization failed: {e}")
            else:
                raise ComponentSwapError(f"Failed to initialize {spec.component_type}: {e}") from e
    
    def _validate_dependencies(self) -> None:
        """Validate that all component dependencies are satisfied."""
        for component_type, spec in self.configuration.components.items():
            for dependency in spec.dependencies:
                if dependency not in self._component_instances:
                    raise ConfigurationError(
                        f"Component {component_type} requires {dependency} but it's not available"
                    )
    
    def swap_component(self, component_type: ComponentType, new_component_name: str, 
                      config: Optional[Dict[str, Any]] = None) -> None:
        """
        Swap a component with a different implementation.
        
        Args:
            component_type: Type of component to swap
            new_component_name: Name of the new component implementation
            config: Configuration for the new component
        """
        try:
            self.logger.info(f"Swapping component {component_type} to {new_component_name}")
            
            # Get new component class
            new_component_class = self.registry.get_component(component_type, new_component_name)
            
            # Create new instance
            component_config = config or {}
            new_instance = new_component_class(**component_config)
            
            # Store old instance for potential rollback
            old_instance = self._component_instances.get(component_type)
            
            # Swap the component
            self._component_instances[component_type] = new_instance
            
            # Update configuration
            if component_type in self.configuration.components:
                self.configuration.components[component_type].component_class = new_component_class
                self.configuration.components[component_type].config = component_config
            
            self.logger.info(f"Successfully swapped component {component_type}")
            
        except Exception as e:
            self.logger.error(f"Component swap failed: {e}")
            raise ComponentSwapError(f"Failed to swap {component_type}: {e}") from e
    
    def get_component(self, component_type: ComponentType) -> Any:
        """Get a component instance."""
        if component_type not in self._component_instances:
            raise PipelineError(f"Component {component_type} not available in pipeline")
        
        return self._component_instances[component_type]
    
    def process_input(self, input_data: Any, processing_mode: str = "standard") -> Any:
        """
        Process input through the pipeline.
        
        Args:
            input_data: Input data to process
            processing_mode: Processing mode ('standard', 'system_aware', 'experimental')
            
        Returns:
            Processed output
        """
        if self._pipeline_state != "ready":
            raise PipelineError(f"Pipeline not ready. Current state: {self._pipeline_state}")
        
        try:
            self.logger.debug(f"Processing input with mode: {processing_mode}")
            
            if processing_mode == "standard":
                return self._process_standard(input_data)
            elif processing_mode == "system_aware":
                return self._process_system_aware(input_data)
            elif processing_mode == "experimental":
                return self._process_experimental(input_data)
            else:
                raise PipelineError(f"Unknown processing mode: {processing_mode}")
                
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            raise PipelineError(f"Failed to process input: {e}") from e
    
    def _process_standard(self, input_data: Any) -> Any:
        """Process input through standard 2D pipeline."""
        # Get required components
        tokenizer = self.get_component(ComponentType.TOKENIZER)
        embedder = self.get_component(ComponentType.EMBEDDER)
        reservoir = self.get_component(ComponentType.RESERVOIR)
        response_generator = self.get_component(ComponentType.RESPONSE_GENERATOR)
        
        # Process through pipeline
        tokens = tokenizer.tokenize([input_data])
        embeddings = embedder.embed(tokens[0])
        
        # Use response generator for final output
        result = response_generator.generate_complete_response(embeddings)
        
        return result
    
    def _process_system_aware(self, input_data: Any) -> Any:
        """Process input through system-aware 3D pipeline."""
        # Extract system message if present
        system_message = None
        user_input = input_data
        
        if isinstance(input_data, dict):
            system_message = input_data.get('system_message')
            user_input = input_data.get('user_input', input_data)
        
        # Get required components
        tokenizer = self.get_component(ComponentType.TOKENIZER)
        embedder = self.get_component(ComponentType.EMBEDDER)
        system_processor = self.get_component(ComponentType.SYSTEM_MESSAGE_PROCESSOR)
        response_generator = self.get_component(ComponentType.RESPONSE_GENERATOR)
        
        # Process user input
        tokens = tokenizer.tokenize([user_input])
        embeddings = embedder.embed(tokens[0])
        
        # Process system message if present
        system_context = None
        if system_message:
            system_context = system_processor.create_system_context(system_message)
        
        # Generate response with system context
        if hasattr(response_generator, 'generate_system_aware_response'):
            result = response_generator.generate_system_aware_response(embeddings, system_context)
        else:
            result = response_generator.generate_complete_response(embeddings)
        
        return result
    
    def _process_experimental(self, input_data: Any) -> Any:
        """Process input through experimental pipeline configuration."""
        # This would be customized based on the specific experimental setup
        # For now, fall back to standard processing
        return self._process_standard(input_data)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and component information."""
        return {
            "state": self._pipeline_state,
            "architecture": self.configuration.architecture_type.value,
            "components": {
                ct.value: type(instance).__name__ 
                for ct, instance in self._component_instances.items()
            },
            "experiment_name": self.configuration.experiment_name,
            "description": self.configuration.description
        }
    
    def save_configuration(self, filepath: str) -> None:
        """Save current pipeline configuration to file."""
        try:
            config_dict = {
                "architecture_type": self.configuration.architecture_type.value,
                "global_config": self.configuration.global_config,
                "experiment_name": self.configuration.experiment_name,
                "description": self.configuration.description,
                "components": {
                    ct.value: {
                        "class_name": type(instance).__name__,
                        "config": getattr(instance, '_config', {})
                    }
                    for ct, instance in self._component_instances.items()
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to {filepath}")
            
        except Exception as e:
            raise PipelineError(f"Failed to save configuration: {e}") from e
    
    def load_configuration(self, filepath: str) -> None:
        """Load pipeline configuration from file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Create new configuration
            self.configuration = PipelineConfiguration(
                architecture_type=ArchitectureType(config_dict["architecture_type"]),
                global_config=config_dict.get("global_config", {}),
                experiment_name=config_dict.get("experiment_name"),
                description=config_dict.get("description")
            )
            
            # Reset pipeline state
            self._component_instances.clear()
            self._pipeline_state = "initialized"
            
            self.logger.info(f"Configuration loaded from {filepath}")
            
        except Exception as e:
            raise PipelineError(f"Failed to load configuration: {e}") from e


def create_pipeline_orchestrator(architecture_type: ArchitectureType = ArchitectureType.STANDARD_2D,
                               experiment_name: Optional[str] = None,
                               description: Optional[str] = None) -> PipelineOrchestrator:
    """
    Create a pipeline orchestrator with the specified architecture.
    
    Args:
        architecture_type: Type of architecture to use
        experiment_name: Optional name for the experiment
        description: Optional description
        
    Returns:
        Configured PipelineOrchestrator instance
    """
    config = PipelineConfiguration(
        architecture_type=architecture_type,
        experiment_name=experiment_name,
        description=description
    )
    
    orchestrator = PipelineOrchestrator(config)
    orchestrator.setup_pipeline()
    
    return orchestrator


def create_experimental_pipeline(components: Dict[ComponentType, ComponentSpec],
                               experiment_name: str,
                               description: Optional[str] = None) -> PipelineOrchestrator:
    """
    Create an experimental pipeline with custom component specifications.
    
    Args:
        components: Dictionary of component specifications
        experiment_name: Name for the experiment
        description: Optional description
        
    Returns:
        Configured PipelineOrchestrator instance
    """
    config = PipelineConfiguration(
        architecture_type=ArchitectureType.EXPERIMENTAL,
        components=components,
        experiment_name=experiment_name,
        description=description
    )
    
    orchestrator = PipelineOrchestrator(config)
    orchestrator.setup_pipeline()
    
    return orchestrator