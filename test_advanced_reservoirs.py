#!/usr/bin/env python3
"""
Test script for advanced LSM reservoir architectures.

This script demonstrates how to use different reservoir types with 
configurations to validate their functionality.
"""

import json
import numpy as np
import tensorflow as tf
from advanced_reservoir import create_advanced_reservoir

def test_reservoir_architecture(reservoir_type: str, config: dict, input_dim: int = 64):
    """Test a specific reservoir architecture."""
    print(f"\n{'='*60}")
    print(f"Testing {reservoir_type.upper()} Reservoir Architecture")
    print(f"{'='*60}")
    
    try:
        # Create the reservoir model
        model = create_advanced_reservoir(
            architecture_type=reservoir_type,
            input_dim=input_dim,
            **config
        )
        
        print(f"✓ Model created successfully")
        print(f"  Input shape: {model.input.shape}")
        print(f"  Output shape: {model.output.shape}")
        
        # Test with sample data
        batch_size = 8
        test_input = tf.random.normal((batch_size, input_dim))
        
        output = model(test_input)
        print(f"✓ Forward pass successful")
        print(f"  Test input shape: {test_input.shape}")
        print(f"  Test output shape: {output.shape}")
        
        # Display model summary
        print("\nModel Summary:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing {reservoir_type} reservoir: {e}")
        return False

def main():
    """Test all advanced reservoir architectures."""
    print("ADVANCED LSM RESERVOIR ARCHITECTURE TESTING")
    print("Testing different reservoir types with various configurations...\n")
    
    input_dim = 64
    successes = 0
    total_tests = 0
    
    # Test configurations for each reservoir type
    test_configs = {
        'hierarchical': {
            'scales': [
                {'units': 128, 'sparsity': 0.1, 'time_constant': 0.05, 'frequency_range': (0.5, 1.0)},
                {'units': 96, 'sparsity': 0.08, 'time_constant': 0.1, 'frequency_range': (1.0, 2.0)},
                {'units': 64, 'sparsity': 0.06, 'time_constant': 0.2, 'frequency_range': (2.0, 4.0)}
            ],
            'global_connectivity': 0.05
        },
        
        'attentive': {
            'units': 256,
            'num_heads': 4,
            'sparsity': 0.1,
            'attention_dim': 64
        },
        
        'echo_state': {
            'units': 256,
            'spectral_radius': 0.9,
            'sparsity': 0.1,
            'input_scaling': 1.0
        },
        
        'deep': {
            'layer_configs': [
                {'units': 256, 'sparsity': 0.1, 'frequency': 1.0, 'amplitude': 1.0, 'decay': 0.1},
                {'units': 128, 'sparsity': 0.08, 'frequency': 1.5, 'amplitude': 0.8, 'decay': 0.15},
                {'units': 64, 'sparsity': 0.06, 'frequency': 2.0, 'amplitude': 0.6, 'decay': 0.2}
            ],
            'use_skip_connections': True
        }
    }
    
    # Test each reservoir type
    for reservoir_type, config in test_configs.items():
        total_tests += 1
        if test_reservoir_architecture(reservoir_type, config, input_dim):
            successes += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TESTING SUMMARY")
    print(f"{'='*60}")
    print(f"Successful tests: {successes}/{total_tests}")
    print(f"Success rate: {(successes/total_tests)*100:.1f}%")
    
    if successes == total_tests:
        print("✓ All advanced reservoir architectures are working correctly!")
    else:
        print(f"✗ {total_tests - successes} architecture(s) failed testing")
    
    # Example CLI usage
    print(f"\n{'='*60}")
    print(f"CLI USAGE EXAMPLES")
    print(f"{'='*60}")
    print("You can now use these advanced reservoirs in training:")
    print()
    
    def convert_to_serializable(obj):
        """Recursively convert tuples to lists for JSON serialization."""
        if isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        else:
            return obj
    
    for reservoir_type, config in test_configs.items():
        config_serializable = convert_to_serializable(config)
        config_json = json.dumps(config_serializable, separators=(',', ':'))
        print(f"# {reservoir_type.title()} Reservoir:")
        print(f"python main.py train --reservoir-type {reservoir_type} \\")
        print(f"    --reservoir-config '{config_json}' \\")
        print(f"    --window-size 5 --batch-size 8 --epochs 3")
        print()

if __name__ == "__main__":
    main()