"""
Demonstration of ColabCompatibilityManager for Google Colab deployment.

This script shows how to use the ColabCompatibilityManager to set up
the LSM environment in Google Colab, create pipelines optimized for Colab,
and run experiments with different architectures.
"""

import sys
import os

# Add the src directory to the path so we can import the LSM modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.pipeline import (
    ColabCompatibilityManager,
    ColabSetupConfig,
    ArchitectureType,
    setup_colab_environment,
    quick_start_colab
)


def demonstrate_environment_detection():
    """Demonstrate environment detection capabilities."""
    print("=== Environment Detection Demo ===")
    
    # Create manager to detect environment
    manager = ColabCompatibilityManager()
    
    # Show environment information
    manager.show_environment_info()
    
    # Get recommended configuration
    recommended_config = manager.get_recommended_config()
    print("\nRecommended Configuration:")
    for key, value in recommended_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 50)


def demonstrate_custom_setup():
    """Demonstrate custom setup configuration."""
    print("=== Custom Setup Demo ===")
    
    # Create custom setup configuration
    custom_config = ColabSetupConfig(
        install_dependencies=False,  # Skip for demo
        setup_gpu=False,  # Skip for demo
        download_sample_data=True,
        create_examples=True,
        workspace_dir="./demo_workspace",
        data_dir="./demo_data",
        models_dir="./demo_models"
    )
    
    # Create manager with custom config
    manager = ColabCompatibilityManager(custom_config)
    
    try:
        # Setup environment
        manager.setup_colab_environment()
        print("Custom environment setup completed successfully!")
        
        # Show setup status
        print(f"Setup complete: {manager.is_setup_complete()}")
        
    except Exception as e:
        print(f"Setup failed: {e}")
    
    print("\n" + "=" * 50)


def demonstrate_pipeline_creation():
    """Demonstrate pipeline creation with different architectures."""
    print("=== Pipeline Creation Demo ===")
    
    manager = ColabCompatibilityManager()
    
    # Test different architectures
    architectures = [
        ArchitectureType.STANDARD_2D,
        ArchitectureType.SYSTEM_AWARE_3D,
        ArchitectureType.HYBRID
    ]
    
    for arch in architectures:
        try:
            print(f"\nCreating {arch.value} pipeline...")
            
            # Create pipeline
            pipeline = manager.create_pipeline(arch)
            
            # Get pipeline status
            status = pipeline.get_pipeline_status()
            print(f"  Status: {status['state']}")
            print(f"  Architecture: {status['architecture']}")
            print(f"  Components: {len(status['components'])}")
            
        except Exception as e:
            print(f"  Failed to create {arch.value} pipeline: {e}")
    
    print("\n" + "=" * 50)


def demonstrate_simple_pipeline():
    """Demonstrate simple pipeline creation and usage."""
    print("=== Simple Pipeline Demo ===")
    
    try:
        manager = ColabCompatibilityManager()
        
        # Create simple pipeline
        pipeline = manager.create_simple_pipeline()
        
        # Test with sample inputs
        test_inputs = [
            "Hello, how are you?",
            "What is machine learning?",
            "Can you help me with coding?",
            "Tell me about neural networks"
        ]
        
        print("Testing simple pipeline:")
        for i, test_input in enumerate(test_inputs, 1):
            try:
                print(f"\n{i}. Input: {test_input}")
                
                # Process input (this might fail if components aren't fully initialized)
                # In a real Colab environment, this would work after proper setup
                print("   Processing... (would work in full Colab environment)")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    except Exception as e:
        print(f"Simple pipeline demo failed: {e}")
    
    print("\n" + "=" * 50)


def demonstrate_quick_experiment():
    """Demonstrate quick experimentation with multiple architectures."""
    print("=== Quick Experiment Demo ===")
    
    try:
        manager = ColabCompatibilityManager()
        
        # Run quick experiment
        test_input = "Hello, how can I help you today?"
        
        print(f"Testing input: '{test_input}'")
        print("Comparing architectures...")
        
        # This would work in a full environment with all components available
        try:
            results = manager.quick_experiment(test_input)
            
            print("\nResults:")
            for arch_name, result in results.items():
                print(f"\n{arch_name.upper()}:")
                if result["success"]:
                    print(f"  Response: {result['response'][:100]}...")
                    print(f"  Time: {result['processing_time']:.3f}s")
                else:
                    print(f"  Error: {result['error']}")
        
        except Exception as e:
            print(f"Quick experiment failed: {e}")
            print("(This is expected in demo environment without full setup)")
    
    except Exception as e:
        print(f"Quick experiment demo failed: {e}")
    
    print("\n" + "=" * 50)


def demonstrate_convenience_functions():
    """Demonstrate convenience functions."""
    print("=== Convenience Functions Demo ===")
    
    try:
        print("1. Testing setup_colab_environment()...")
        
        # This would set up the full environment in Colab
        # manager = setup_colab_environment()
        print("   (Skipped in demo - would setup full environment)")
        
        print("\n2. Testing quick_start_colab()...")
        
        # This would create a ready-to-use pipeline
        # pipeline = quick_start_colab()
        print("   (Skipped in demo - would create ready pipeline)")
        
        print("\n3. Environment detection works in any environment:")
        manager = ColabCompatibilityManager()
        print(f"   Colab detected: {manager.env_info.is_colab}")
        print(f"   GPU available: {manager.env_info.gpu_available}")
        print(f"   Runtime type: {manager.env_info.runtime_type}")
    
    except Exception as e:
        print(f"Convenience functions demo failed: {e}")
    
    print("\n" + "=" * 50)


def demonstrate_colab_specific_features():
    """Demonstrate Colab-specific features."""
    print("=== Colab-Specific Features Demo ===")
    
    manager = ColabCompatibilityManager()
    
    print("1. Environment Information:")
    print(f"   Is Colab: {manager.env_info.is_colab}")
    print(f"   Python Version: {manager.env_info.python_version}")
    print(f"   Available Memory: {manager.env_info.available_memory_gb:.1f} GB")
    print(f"   Disk Space: {manager.env_info.disk_space_gb:.1f} GB")
    
    print("\n2. Recommended Configuration:")
    config = manager.get_recommended_config()
    for key, value in config.items():
        print(f"   {key}: {value}")
    
    print("\n3. Architecture Comparison:")
    try:
        manager.compare_architectures("Test input for comparison")
    except Exception as e:
        print(f"   Comparison failed: {e}")
        print("   (Expected in demo environment)")
    
    print("\n4. Interactive Demo:")
    try:
        # This would create interactive widgets in Colab
        print("   Interactive demo would be available in Colab with ipywidgets")
        print("   Features: Architecture selection, input text area, process button")
    except Exception as e:
        print(f"   Interactive demo setup failed: {e}")
    
    print("\n" + "=" * 50)


def main():
    """Run all demonstrations."""
    print("ColabCompatibilityManager Demonstration")
    print("=" * 60)
    print()
    print("This demo shows the capabilities of the ColabCompatibilityManager")
    print("for deploying and using LSM in Google Colab environments.")
    print()
    
    # Run demonstrations
    demonstrate_environment_detection()
    demonstrate_custom_setup()
    demonstrate_pipeline_creation()
    demonstrate_simple_pipeline()
    demonstrate_quick_experiment()
    demonstrate_convenience_functions()
    demonstrate_colab_specific_features()
    
    print("Demo completed!")
    print()
    print("In a real Google Colab environment, you would:")
    print("1. Clone the LSM repository")
    print("2. Run: manager = ColabCompatibilityManager()")
    print("3. Run: manager.setup_colab_environment()")
    print("4. Run: pipeline = manager.create_simple_pipeline()")
    print("5. Use: response = pipeline.process_input('Your message')")
    print()
    print("Or use the quick start:")
    print("pipeline = quick_start_colab()")


if __name__ == "__main__":
    main()