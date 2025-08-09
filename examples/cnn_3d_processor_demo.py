#!/usr/bin/env python3
"""
Demonstration of CNN 3D Processor for system message integration.

This example shows how to use the CNN3DProcessor to create 3D CNN models
that can integrate system message embeddings with reservoir outputs for
enhanced conversational AI capabilities.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any

# Import LSM components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.core.cnn_3d_processor import (
    CNN3DProcessor,
    SystemContext,
    create_cnn_3d_processor,
    create_system_aware_processor
)
from src.lsm.utils.lsm_logging import get_logger

# Set up logging
logger = get_logger(__name__)


def demonstrate_basic_3d_cnn_creation():
    """Demonstrate basic 3D CNN processor creation and model building."""
    print("\n" + "="*60)
    print("BASIC 3D CNN PROCESSOR CREATION")
    print("="*60)
    
    # Define processor parameters
    reservoir_shape = (32, 32, 32, 1)  # depth, height, width, channels
    system_embedding_dim = 256
    output_embedding_dim = 512
    
    print(f"Reservoir shape: {reservoir_shape}")
    print(f"System embedding dimension: {system_embedding_dim}")
    print(f"Output embedding dimension: {output_embedding_dim}")
    
    # Create processor
    processor = CNN3DProcessor(
        reservoir_shape=reservoir_shape,
        system_embedding_dim=system_embedding_dim,
        output_embedding_dim=output_embedding_dim
    )
    
    print(f"✓ Created CNN3DProcessor")
    
    # Create the 3D CNN model
    try:
        model = processor.create_model()
        print(f"✓ Created 3D CNN model with {model.count_params():,} parameters")
        
        # Display model summary
        print("\nModel Architecture Summary:")
        print("-" * 40)
        model.summary()
        
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return None
    
    return processor


def demonstrate_system_message_processing():
    """Demonstrate system message processing and embedding generation."""
    print("\n" + "="*60)
    print("SYSTEM MESSAGE PROCESSING")
    print("="*60)
    
    # Create processor
    processor = create_system_aware_processor(
        window_size=32,
        channels=1,
        system_dim=256,
        output_dim=512
    )
    
    # Create system processor
    try:
        system_processor = processor.create_system_processor()
        print(f"✓ Created system message processor")
        
        # Sample system messages
        system_messages = [
            "You are a helpful and friendly assistant.",
            "Respond in a professional and formal tone.",
            "Be creative and use humor in your responses.",
            "Focus on technical accuracy and precision."
        ]
        
        print(f"\nProcessing {len(system_messages)} system messages:")
        
        for i, message in enumerate(system_messages, 1):
            print(f"\n{i}. Message: '{message}'")
            
            # Simple tokenization (in practice, use proper tokenizer)
            tokens = processor._simple_tokenize(message)
            print(f"   Tokens: {tokens[:10]}... (showing first 10)")
            
            # Generate system embeddings
            system_embeddings = system_processor.predict(
                np.expand_dims(tokens, axis=0),
                verbose=0
            )[0]
            
            print(f"   Generated embedding shape: {system_embeddings.shape}")
            print(f"   Embedding norm: {np.linalg.norm(system_embeddings):.4f}")
            
    except Exception as e:
        print(f"✗ Failed to process system messages: {e}")
        return None
    
    return processor


def demonstrate_embedding_modifier_generation():
    """Demonstrate embedding modifier generation from system context."""
    print("\n" + "="*60)
    print("EMBEDDING MODIFIER GENERATION")
    print("="*60)
    
    # Create processor
    processor = create_cnn_3d_processor(
        reservoir_shape=(32, 32, 32, 1),
        system_embedding_dim=256,
        output_embedding_dim=512
    )
    
    try:
        # Create embedding modifier
        modifier_model = processor.create_embedding_modifier()
        print(f"✓ Created embedding modifier model")
        
        # Generate sample system embeddings
        system_embeddings = np.random.randn(1, 256).astype(np.float32)
        print(f"Input system embedding shape: {system_embeddings.shape}")
        
        # Generate modifiers
        modifiers = modifier_model.predict(system_embeddings, verbose=0)
        
        print(f"\nGenerated modifiers:")
        for modifier_type, modifier_values in modifiers.items():
            print(f"  {modifier_type}: shape {modifier_values.shape}, "
                  f"range [{modifier_values.min():.4f}, {modifier_values.max():.4f}]")
        
        # Demonstrate modifier application
        base_output = np.random.randn(1, 512).astype(np.float32)
        output_modifiers = modifiers['output_modifiers']
        
        # Apply modifiers
        modified_output = base_output + output_modifiers
        
        print(f"\nModifier application:")
        print(f"  Base output norm: {np.linalg.norm(base_output):.4f}")
        print(f"  Modified output norm: {np.linalg.norm(modified_output):.4f}")
        print(f"  Modification strength: {np.linalg.norm(output_modifiers):.4f}")
        
    except Exception as e:
        print(f"✗ Failed to generate embedding modifiers: {e}")
        return None
    
    return processor


def demonstrate_complete_processing_pipeline():
    """Demonstrate the complete processing pipeline with system context."""
    print("\n" + "="*60)
    print("COMPLETE PROCESSING PIPELINE")
    print("="*60)
    
    # Create processor
    processor = create_system_aware_processor(
        window_size=32,
        channels=1,
        system_dim=256,
        output_dim=512
    )
    
    try:
        # Generate sample reservoir output
        batch_size = 2
        reservoir_output = np.random.randn(batch_size, 32, 32, 32, 1).astype(np.float32)
        print(f"Generated reservoir output: {reservoir_output.shape}")
        
        # System messages with different influence strengths
        test_cases = [
            ("You are a helpful assistant.", 1.0),
            ("Be very formal and professional.", 0.8),
            ("Use creative and playful language.", 1.2),
            ("Focus on technical precision.", 0.6)
        ]
        
        print(f"\nTesting {len(test_cases)} system message scenarios:")
        
        results = []
        
        for i, (system_message, influence_strength) in enumerate(test_cases, 1):
            print(f"\n{i}. System Message: '{system_message}'")
            print(f"   Influence Strength: {influence_strength}")
            
            # Process with system context
            result = processor.process_reservoir_output_with_system(
                reservoir_output=reservoir_output,
                system_message=system_message,
                influence_strength=influence_strength
            )
            
            print(f"   ✓ Processing completed in {result.processing_time:.4f}s")
            print(f"   Output shape: {result.output_embeddings.shape}")
            print(f"   System influence: {result.system_influence:.4f}")
            print(f"   Output norm: {np.linalg.norm(result.output_embeddings):.4f}")
            
            results.append(result)
        
        # Compare outputs to show system message influence
        print(f"\nComparison of system message effects:")
        print("-" * 50)
        
        base_result = results[0]
        for i, result in enumerate(results[1:], 2):
            similarity = np.dot(
                base_result.output_embeddings.flatten(),
                result.output_embeddings.flatten()
            ) / (
                np.linalg.norm(base_result.output_embeddings) *
                np.linalg.norm(result.output_embeddings)
            )
            
            print(f"Similarity between case 1 and case {i}: {similarity:.4f}")
        
    except Exception as e:
        print(f"✗ Failed in complete processing pipeline: {e}")
        return None
    
    return processor, results


def demonstrate_system_context_variations():
    """Demonstrate different system context configurations."""
    print("\n" + "="*60)
    print("SYSTEM CONTEXT VARIATIONS")
    print("="*60)
    
    # Create processor
    processor = create_cnn_3d_processor(
        reservoir_shape=(16, 16, 16, 1),  # Smaller for faster processing
        system_embedding_dim=128,
        output_embedding_dim=256
    )
    
    try:
        # Create sample reservoir output
        reservoir_output = np.random.randn(1, 16, 16, 16, 1).astype(np.float32)
        
        # Different system context configurations
        contexts = [
            SystemContext(
                message="Standard helpful assistant",
                embeddings=np.random.randn(128).astype(np.float32),
                influence_strength=1.0,
                processing_mode="3d_cnn"
            ),
            SystemContext(
                message="High influence creative assistant",
                embeddings=np.random.randn(128).astype(np.float32),
                influence_strength=2.0,
                processing_mode="3d_cnn"
            ),
            SystemContext(
                message="Low influence technical assistant",
                embeddings=np.random.randn(128).astype(np.float32),
                influence_strength=0.3,
                processing_mode="3d_cnn"
            ),
            SystemContext(
                message="Custom weighted assistant",
                embeddings=np.random.randn(128).astype(np.float32),
                modifier_weights=np.random.randn(64).astype(np.float32),
                influence_strength=1.5,
                processing_mode="3d_cnn"
            )
        ]
        
        print(f"Testing {len(contexts)} different system contexts:")
        
        for i, context in enumerate(contexts, 1):
            print(f"\n{i}. Context: '{context.message}'")
            print(f"   Influence strength: {context.influence_strength}")
            print(f"   Has custom weights: {context.modifier_weights is not None}")
            
            # Process with this context
            result = processor.process_with_system_context(
                reservoir_output=reservoir_output,
                system_context=context
            )
            
            print(f"   ✓ Processing completed in {result.processing_time:.4f}s")
            print(f"   System influence: {result.system_influence:.4f}")
            print(f"   Output norm: {np.linalg.norm(result.output_embeddings):.4f}")
        
    except Exception as e:
        print(f"✗ Failed in system context variations: {e}")
        return None
    
    return processor


def demonstrate_model_persistence():
    """Demonstrate saving and loading 3D CNN models."""
    print("\n" + "="*60)
    print("MODEL PERSISTENCE")
    print("="*60)
    
    # Create processor
    processor = create_cnn_3d_processor(
        reservoir_shape=(16, 16, 16, 1),
        system_embedding_dim=128,
        output_embedding_dim=256
    )
    
    try:
        # Create model
        model = processor.create_model()
        print(f"✓ Created model with {model.count_params():,} parameters")
        
        # Get model summary
        summary = processor.get_model_summary()
        print(f"✓ Generated model summary ({len(summary)} characters)")
        
        # Save model (in practice, you would save to a real file)
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_3d_cnn_model.h5")
            
            processor.save_model(model_path)
            print(f"✓ Saved model to {model_path}")
            
            # Create new processor and load model
            new_processor = create_cnn_3d_processor(
                reservoir_shape=(16, 16, 16, 1),
                system_embedding_dim=128,
                output_embedding_dim=256
            )
            
            new_processor.load_model(model_path)
            print(f"✓ Loaded model into new processor")
            
            # Verify models produce same output
            test_input = [
                np.random.randn(1, 16, 16, 16, 1).astype(np.float32),
                np.random.randn(1, 128).astype(np.float32)
            ]
            
            original_output = processor._model.predict(test_input, verbose=0)
            loaded_output = new_processor._model.predict(test_input, verbose=0)
            
            difference = np.mean(np.abs(original_output - loaded_output))
            print(f"✓ Output difference after loading: {difference:.8f}")
            
            if difference < 1e-6:
                print("✓ Models are identical after save/load")
            else:
                print("⚠ Models differ slightly after save/load")
        
    except Exception as e:
        print(f"✗ Failed in model persistence: {e}")
        return None
    
    return processor


def main():
    """Run all demonstrations."""
    print("CNN 3D PROCESSOR DEMONSTRATION")
    print("=" * 80)
    print("This demo shows the capabilities of the CNN3DProcessor for")
    print("system message integration in LSM-based conversational AI.")
    print("=" * 80)
    
    try:
        # Run demonstrations
        processor1 = demonstrate_basic_3d_cnn_creation()
        processor2 = demonstrate_system_message_processing()
        processor3 = demonstrate_embedding_modifier_generation()
        processor4, results = demonstrate_complete_processing_pipeline()
        processor5 = demonstrate_system_context_variations()
        processor6 = demonstrate_model_persistence()
        
        # Summary
        print("\n" + "="*60)
        print("DEMONSTRATION SUMMARY")
        print("="*60)
        
        successful_demos = sum([
            processor1 is not None,
            processor2 is not None,
            processor3 is not None,
            processor4 is not None,
            processor5 is not None,
            processor6 is not None
        ])
        
        print(f"✓ Successfully completed {successful_demos}/6 demonstrations")
        
        if successful_demos == 6:
            print("🎉 All demonstrations completed successfully!")
            print("\nThe CNN3DProcessor is ready for:")
            print("  • 3D CNN model creation with system message integration")
            print("  • System message processing and embedding generation")
            print("  • Embedding modifier generation and application")
            print("  • Complete processing pipeline with system context")
            print("  • Flexible system context configurations")
            print("  • Model persistence and loading")
        else:
            print(f"⚠ {6 - successful_demos} demonstrations failed")
            print("Check the error messages above for details")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"\n✗ Demonstration failed with error: {e}")


if __name__ == "__main__":
    main()