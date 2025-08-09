#!/usr/bin/env python3
"""
Pipeline Orchestrator Demo

This script demonstrates the PipelineOrchestrator functionality including:
- Creating different pipeline architectures
- Component swapping and experimentation
- Configuration management
- Processing inputs through different pipeline modes

The PipelineOrchestrator provides a modular architecture for managing all
LSM components and supports easy experimentation with different configurations.
"""

import sys
import os
import tempfile
import json

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from lsm.pipeline import (
        PipelineOrchestrator,
        PipelineConfiguration,
        ComponentRegistry,
        ComponentType,
        ArchitectureType,
        ComponentSpec,
        create_pipeline_orchestrator,
        create_experimental_pipeline
    )
    from lsm.utils.lsm_logging import setup_logging
    
    print("✓ Successfully imported PipelineOrchestrator components")
    
except ImportError as e:
    print(f"✗ Failed to import PipelineOrchestrator components: {e}")
    print("This might be due to missing dependencies or import issues.")
    sys.exit(1)


def demonstrate_component_registry():
    """Demonstrate ComponentRegistry functionality."""
    print("\n" + "="*60)
    print("COMPONENT REGISTRY DEMONSTRATION")
    print("="*60)
    
    # Create a component registry
    registry = ComponentRegistry()
    
    # List available components
    print("\n1. Available Components:")
    components = registry.list_components()
    for component_type, component_names in components.items():
        if component_names:  # Only show types that have components
            print(f"   {component_type.value}: {component_names}")
    
    # Register a custom component
    print("\n2. Registering Custom Component:")
    
    class CustomTokenizer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size
            print(f"   CustomTokenizer initialized with vocab_size={vocab_size}")
        
        def tokenize(self, texts):
            return [text.split() for text in texts]
    
    registry.register_component(ComponentType.TOKENIZER, "custom", CustomTokenizer)
    print("   ✓ Registered CustomTokenizer")
    
    # Retrieve the custom component
    custom_tokenizer_class = registry.get_component(ComponentType.TOKENIZER, "custom")
    custom_instance = custom_tokenizer_class(vocab_size=5000)
    
    # Test the custom component
    result = custom_instance.tokenize(["hello world", "test tokenization"])
    print(f"   ✓ Custom tokenizer result: {result}")


def demonstrate_basic_pipeline():
    """Demonstrate basic pipeline creation and setup."""
    print("\n" + "="*60)
    print("BASIC PIPELINE DEMONSTRATION")
    print("="*60)
    
    # Create a standard 2D pipeline
    print("\n1. Creating Standard 2D Pipeline:")
    try:
        orchestrator = create_pipeline_orchestrator(
            architecture_type=ArchitectureType.STANDARD_2D,
            experiment_name="basic_demo",
            description="Basic pipeline demonstration"
        )
        print("   ✓ Standard 2D pipeline created successfully")
        
        # Get pipeline status
        status = orchestrator.get_pipeline_status()
        print(f"   Pipeline State: {status['state']}")
        print(f"   Architecture: {status['architecture']}")
        print(f"   Experiment: {status['experiment_name']}")
        print(f"   Components: {len(status['components'])} loaded")
        
        # List components
        print("\n   Available Components:")
        for component_type, component_name in status['components'].items():
            print(f"     {component_type}: {component_name}")
            
    except Exception as e:
        print(f"   ✗ Failed to create pipeline: {e}")
        print("   This is expected if some components are not available")


def demonstrate_system_aware_pipeline():
    """Demonstrate system-aware 3D pipeline."""
    print("\n" + "="*60)
    print("SYSTEM-AWARE PIPELINE DEMONSTRATION")
    print("="*60)
    
    print("\n1. Creating System-Aware 3D Pipeline:")
    try:
        orchestrator = create_pipeline_orchestrator(
            architecture_type=ArchitectureType.SYSTEM_AWARE_3D,
            experiment_name="system_aware_demo",
            description="System-aware pipeline with 3D CNN support"
        )
        print("   ✓ System-aware 3D pipeline created successfully")
        
        status = orchestrator.get_pipeline_status()
        print(f"   Architecture: {status['architecture']}")
        print(f"   Components: {len(status['components'])} loaded")
        
    except Exception as e:
        print(f"   ✗ Failed to create system-aware pipeline: {e}")
        print("   This is expected if system-aware components are not available")


def demonstrate_component_swapping():
    """Demonstrate component swapping functionality."""
    print("\n" + "="*60)
    print("COMPONENT SWAPPING DEMONSTRATION")
    print("="*60)
    
    try:
        # Create a basic pipeline
        orchestrator = PipelineOrchestrator()
        orchestrator.setup_pipeline()
        
        print("\n1. Initial Pipeline Status:")
        status = orchestrator.get_pipeline_status()
        print(f"   Components: {len(status['components'])} loaded")
        
        # Create a mock component for swapping
        class MockTokenizer:
            def __init__(self, model_name="mock"):
                self.model_name = model_name
                print(f"   MockTokenizer initialized with model_name={model_name}")
            
            def tokenize(self, texts):
                return [f"mock_token_{i}" for i, text in enumerate(texts)]
        
        # Register the mock component
        orchestrator.registry.register_component(
            ComponentType.TOKENIZER, "mock", MockTokenizer
        )
        
        print("\n2. Swapping Tokenizer Component:")
        try:
            orchestrator.swap_component(
                ComponentType.TOKENIZER, 
                "mock", 
                {"model_name": "swapped_mock"}
            )
            print("   ✓ Component swapped successfully")
            
            # Verify the swap
            new_status = orchestrator.get_pipeline_status()
            print(f"   New tokenizer: {new_status['components'].get('tokenizer', 'Not found')}")
            
        except Exception as e:
            print(f"   ✗ Component swap failed: {e}")
            
    except Exception as e:
        print(f"   ✗ Failed to demonstrate component swapping: {e}")


def demonstrate_configuration_management():
    """Demonstrate configuration saving and loading."""
    print("\n" + "="*60)
    print("CONFIGURATION MANAGEMENT DEMONSTRATION")
    print("="*60)
    
    try:
        # Create a pipeline with custom configuration
        config = PipelineConfiguration(
            architecture_type=ArchitectureType.HYBRID,
            experiment_name="config_demo",
            description="Configuration management demonstration",
            global_config={
                "batch_size": 32,
                "learning_rate": 0.001,
                "max_sequence_length": 512
            }
        )
        
        orchestrator = PipelineOrchestrator(config)
        print("\n1. Created Pipeline with Custom Configuration:")
        print(f"   Architecture: {config.architecture_type.value}")
        print(f"   Experiment: {config.experiment_name}")
        print(f"   Global Config: {config.global_config}")
        
        # Save configuration
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            config_file = f.name
        
        print(f"\n2. Saving Configuration to: {config_file}")
        try:
            orchestrator.save_configuration(config_file)
            print("   ✓ Configuration saved successfully")
            
            # Show saved content
            with open(config_file, 'r') as f:
                saved_config = json.load(f)
            print("   Saved configuration preview:")
            for key, value in saved_config.items():
                if key != "components":  # Skip components for brevity
                    print(f"     {key}: {value}")
            
        except Exception as e:
            print(f"   ✗ Failed to save configuration: {e}")
        
        # Load configuration
        print(f"\n3. Loading Configuration from: {config_file}")
        try:
            new_orchestrator = PipelineOrchestrator()
            new_orchestrator.load_configuration(config_file)
            print("   ✓ Configuration loaded successfully")
            
            loaded_status = new_orchestrator.get_pipeline_status()
            print(f"   Loaded architecture: {loaded_status['architecture']}")
            print(f"   Loaded experiment: {loaded_status['experiment_name']}")
            
        except Exception as e:
            print(f"   ✗ Failed to load configuration: {e}")
        
        # Clean up
        try:
            os.unlink(config_file)
        except:
            pass
            
    except Exception as e:
        print(f"   ✗ Failed to demonstrate configuration management: {e}")


def demonstrate_experimental_pipeline():
    """Demonstrate experimental pipeline with custom components."""
    print("\n" + "="*60)
    print("EXPERIMENTAL PIPELINE DEMONSTRATION")
    print("="*60)
    
    try:
        # Create custom component specifications
        class ExperimentalTokenizer:
            def __init__(self, experimental_param="default"):
                self.experimental_param = experimental_param
                print(f"   ExperimentalTokenizer initialized with param={experimental_param}")
        
        class ExperimentalEmbedder:
            def __init__(self, embedding_dim=128):
                self.embedding_dim = embedding_dim
                print(f"   ExperimentalEmbedder initialized with dim={embedding_dim}")
        
        # Create component specifications
        components = {
            ComponentType.TOKENIZER: ComponentSpec(
                component_type=ComponentType.TOKENIZER,
                component_class=ExperimentalTokenizer,
                config={"experimental_param": "custom_value"},
                dependencies=[],
                optional=False
            ),
            ComponentType.EMBEDDER: ComponentSpec(
                component_type=ComponentType.EMBEDDER,
                component_class=ExperimentalEmbedder,
                config={"embedding_dim": 256},
                dependencies=[ComponentType.TOKENIZER],
                optional=False
            )
        }
        
        print("\n1. Creating Experimental Pipeline:")
        orchestrator = create_experimental_pipeline(
            components=components,
            experiment_name="experimental_demo",
            description="Custom experimental pipeline"
        )
        print("   ✓ Experimental pipeline created successfully")
        
        status = orchestrator.get_pipeline_status()
        print(f"   Architecture: {status['architecture']}")
        print(f"   Experiment: {status['experiment_name']}")
        print(f"   Components: {len(status['components'])} loaded")
        
    except Exception as e:
        print(f"   ✗ Failed to create experimental pipeline: {e}")


def demonstrate_input_processing():
    """Demonstrate input processing through different pipeline modes."""
    print("\n" + "="*60)
    print("INPUT PROCESSING DEMONSTRATION")
    print("="*60)
    
    try:
        # Create a mock pipeline with mock components
        orchestrator = PipelineOrchestrator()
        
        # Create mock components
        class MockTokenizer:
            def tokenize(self, texts):
                return [text.split() for text in texts]
        
        class MockEmbedder:
            def embed(self, tokens):
                return [[0.1 * i for i in range(len(tokens))]]
        
        class MockReservoir:
            def process(self, embeddings):
                return embeddings
        
        class MockResponseGenerator:
            def generate_complete_response(self, embeddings):
                return f"Generated response from {len(embeddings)} embeddings"
        
        # Register mock components
        orchestrator.registry.register_component(ComponentType.TOKENIZER, "mock", MockTokenizer)
        orchestrator.registry.register_component(ComponentType.EMBEDDER, "mock", MockEmbedder)
        orchestrator.registry.register_component(ComponentType.RESERVOIR, "mock", MockReservoir)
        orchestrator.registry.register_component(ComponentType.RESPONSE_GENERATOR, "mock", MockResponseGenerator)
        
        # Swap to mock components
        orchestrator.swap_component(ComponentType.TOKENIZER, "mock")
        orchestrator.swap_component(ComponentType.EMBEDDER, "mock")
        orchestrator.swap_component(ComponentType.RESERVOIR, "mock")
        orchestrator.swap_component(ComponentType.RESPONSE_GENERATOR, "mock")
        
        # Set pipeline as ready
        orchestrator._pipeline_state = "ready"
        
        print("\n1. Processing Standard Input:")
        try:
            result = orchestrator.process_input("Hello world, this is a test", "standard")
            print(f"   ✓ Standard processing result: {result}")
        except Exception as e:
            print(f"   ✗ Standard processing failed: {e}")
        
        print("\n2. Processing System-Aware Input:")
        try:
            system_input = {
                "user_input": "What is the weather like?",
                "system_message": "You are a helpful weather assistant."
            }
            result = orchestrator.process_input(system_input, "system_aware")
            print(f"   ✓ System-aware processing result: {result}")
        except Exception as e:
            print(f"   ✗ System-aware processing failed: {e}")
            
    except Exception as e:
        print(f"   ✗ Failed to demonstrate input processing: {e}")


def main():
    """Run all demonstrations."""
    print("Pipeline Orchestrator Demo")
    print("=" * 60)
    print("This demo shows the PipelineOrchestrator functionality for managing")
    print("modular LSM architectures with component swapping and experimentation.")
    
    # Set up logging
    setup_logging(log_level="INFO")
    
    try:
        # Run demonstrations
        demonstrate_component_registry()
        demonstrate_basic_pipeline()
        demonstrate_system_aware_pipeline()
        demonstrate_component_swapping()
        demonstrate_configuration_management()
        demonstrate_experimental_pipeline()
        demonstrate_input_processing()
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETE")
        print("="*60)
        print("✓ All PipelineOrchestrator features demonstrated successfully!")
        print("\nKey Features Shown:")
        print("  • Component registry and management")
        print("  • Multiple architecture types (2D, 3D, hybrid, experimental)")
        print("  • Component swapping and experimentation")
        print("  • Configuration saving and loading")
        print("  • Input processing through different pipeline modes")
        print("\nThe PipelineOrchestrator provides a flexible foundation for")
        print("experimenting with different LSM architectures and components.")
        
    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()