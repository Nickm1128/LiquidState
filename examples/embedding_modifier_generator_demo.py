#!/usr/bin/env python3
"""
Embedding Modifier Generator Demo.

This script demonstrates the usage of the EmbeddingModifierGenerator for
creating and applying embedding modifiers based on system prompts.
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.core.embedding_modifier_generator import (
    EmbeddingModifierGenerator,
    ModifierConfig,
    TrainingBatch,
    create_embedding_modifier_generator,
    create_training_batch_from_prompts
)
from lsm.core.system_message_processor import create_system_message_processor
from lsm.data.tokenization import StandardTokenizerWrapper
from lsm.utils.lsm_logging import setup_logging

# Setup logging
logger = setup_logging(__name__)


def demonstrate_basic_usage():
    """Demonstrate basic usage of EmbeddingModifierGenerator."""
    print("=== Basic EmbeddingModifierGenerator Usage ===")
    
    try:
        # Create a generator with default settings
        generator = create_embedding_modifier_generator(
            system_embedding_dim=256,
            base_embedding_dim=512,
            tokenizer_name="gpt2"
        )
        
        print(f"Created generator with config:")
        print(f"  System embedding dim: {generator.config.system_embedding_dim}")
        print(f"  Base embedding dim: {generator.config.base_embedding_dim}")
        print(f"  Modifier types: {generator.config.modifier_types}")
        
        # Generate modifiers from a system prompt
        system_prompt = "You are a helpful and creative assistant. Be engaging and informative."
        
        print(f"\nGenerating modifiers for prompt: '{system_prompt}'")
        modifiers = generator.generate_modifiers(system_prompt, influence_strength=0.8)
        
        print(f"Generated modifiers:")
        print(f"  Attention modifiers shape: {modifiers.attention_modifiers.shape}")
        print(f"  Feature modifiers shape: {modifiers.feature_modifiers.shape}")
        print(f"  Output modifiers shape: {modifiers.output_modifiers.shape}")
        print(f"  Scaling modifiers shape: {modifiers.scaling_modifiers.shape}")
        print(f"  Combined modifiers shape: {modifiers.combined_modifiers.shape}")
        print(f"  Generation time: {modifiers.generation_time:.4f}s")
        print(f"  Confidence scores: {modifiers.confidence_scores}")
        
        return generator, modifiers
        
    except Exception as e:
        print(f"Error in basic usage demo: {e}")
        return None, None


def demonstrate_modifier_application():
    """Demonstrate applying modifiers to base embeddings."""
    print("\n=== Modifier Application Demo ===")
    
    try:
        # Create generator
        generator = create_embedding_modifier_generator()
        
        # Generate modifiers
        system_prompt = "Be concise and technical in your responses."
        modifiers = generator.generate_modifiers(system_prompt)
        
        # Create some base embeddings to modify
        base_embeddings = np.random.randn(3, 512)  # 3 samples, 512 dimensions
        print(f"Base embeddings shape: {base_embeddings.shape}")
        print(f"Base embeddings mean: {np.mean(base_embeddings):.4f}")
        print(f"Base embeddings std: {np.std(base_embeddings):.4f}")
        
        # Apply modifiers in different modes
        modes = ["additive", "multiplicative", "hybrid"]
        
        for mode in modes:
            modified_embeddings = generator.apply_modifiers_to_embeddings(
                base_embeddings, modifiers, application_mode=mode
            )
            
            print(f"\n{mode.capitalize()} mode results:")
            print(f"  Modified embeddings shape: {modified_embeddings.shape}")
            print(f"  Modified embeddings mean: {np.mean(modified_embeddings):.4f}")
            print(f"  Modified embeddings std: {np.std(modified_embeddings):.4f}")
            print(f"  Change magnitude: {np.mean(np.abs(modified_embeddings - base_embeddings)):.4f}")
        
    except Exception as e:
        print(f"Error in modifier application demo: {e}")


def demonstrate_custom_configuration():
    """Demonstrate creating generator with custom configuration."""
    print("\n=== Custom Configuration Demo ===")
    
    try:
        # Create custom configuration
        config = ModifierConfig(
            system_embedding_dim=128,
            base_embedding_dim=256,
            modifier_types=["attention", "output"],  # Only specific types
            hidden_dims=[256, 128, 64],
            dropout_rates=[0.2, 0.3, 0.4],
            activation="tanh",
            use_batch_norm=False,
            learning_rate=0.01
        )
        
        # Create system processor
        tokenizer = StandardTokenizerWrapper("gpt2", max_length=256)
        system_processor = create_system_message_processor("gpt2", max_length=256, embedding_dim=128)
        
        # Create generator with custom config
        generator = EmbeddingModifierGenerator(config, system_processor)
        
        print(f"Custom generator configuration:")
        print(f"  System embedding dim: {config.system_embedding_dim}")
        print(f"  Base embedding dim: {config.base_embedding_dim}")
        print(f"  Modifier types: {config.modifier_types}")
        print(f"  Hidden dims: {config.hidden_dims}")
        print(f"  Dropout rates: {config.dropout_rates}")
        print(f"  Activation: {config.activation}")
        print(f"  Use batch norm: {config.use_batch_norm}")
        print(f"  Learning rate: {config.learning_rate}")
        
        # Generate modifiers with custom generator
        system_prompt = "You are a technical expert. Provide detailed explanations."
        modifiers = generator.generate_modifiers(system_prompt, influence_strength=1.2)
        
        print(f"\nGenerated modifiers with custom config:")
        print(f"  Available modifier types: {[k for k, v in modifiers.__dict__.items() if isinstance(v, np.ndarray) and len(v) > 0]}")
        
    except Exception as e:
        print(f"Error in custom configuration demo: {e}")


def demonstrate_training_preparation():
    """Demonstrate preparing training data for the modifier model."""
    print("\n=== Training Data Preparation Demo ===")
    
    try:
        # Create system processor for training data preparation
        system_processor = create_system_message_processor("gpt2")
        
        # Define system prompts and target behaviors
        system_prompts = [
            "You are a helpful assistant. Be friendly and supportive.",
            "You are a technical expert. Be precise and detailed.",
            "You are a creative writer. Be imaginative and engaging.",
            "You are a teacher. Be clear and educational."
        ]
        
        # Create target behavior embeddings (in practice, these would be learned)
        target_behaviors = [
            np.random.randn(512) * 0.5,  # Gentle modifications
            np.random.randn(512) * 1.2,  # Strong technical influence
            np.random.randn(512) * 0.8,  # Creative modifications
            np.random.randn(512) * 0.6   # Educational influence
        ]
        
        print(f"Creating training batches from {len(system_prompts)} prompts...")
        
        # Create training batches
        training_batches = create_training_batch_from_prompts(
            system_prompts, target_behaviors, system_processor
        )
        
        print(f"Created {len(training_batches)} training batches")
        
        for i, batch in enumerate(training_batches):
            print(f"  Batch {i+1}:")
            print(f"    System embeddings shape: {batch.system_embeddings.shape}")
            print(f"    Target modifiers: {list(batch.target_modifiers.keys())}")
            for mod_type, mod_values in batch.target_modifiers.items():
                print(f"      {mod_type}: {mod_values.shape}")
        
        return training_batches
        
    except Exception as e:
        print(f"Error in training preparation demo: {e}")
        return []


def demonstrate_model_training():
    """Demonstrate training the modifier model."""
    print("\n=== Model Training Demo ===")
    
    try:
        # Create generator
        generator = create_embedding_modifier_generator()
        
        # Prepare training data
        training_batches = demonstrate_training_preparation()
        
        if not training_batches:
            print("No training data available, skipping training demo")
            return
        
        # Split into training and validation
        split_idx = len(training_batches) // 2
        train_data = training_batches[:split_idx] if split_idx > 0 else training_batches
        val_data = training_batches[split_idx:] if split_idx > 0 and len(training_batches) > 1 else None
        
        print(f"Training with {len(train_data)} batches, validation with {len(val_data) if val_data else 0} batches")
        
        # Train the model (small number of epochs for demo)
        print("Starting training...")
        training_result = generator.train_modifier_model(
            training_data=train_data,
            validation_data=val_data,
            epochs=3,  # Small number for demo
            batch_size=2
        )
        
        print("Training completed!")
        print(f"Final loss: {training_result['metrics']['final_loss']:.4f}")
        print(f"Best loss: {training_result['metrics']['best_loss']:.4f}")
        print(f"Training samples: {training_result['metrics']['training_samples']}")
        print(f"Validation samples: {training_result['metrics']['validation_samples']}")
        
        # Test the trained model
        print("\nTesting trained model...")
        test_prompt = "You are an expert advisor. Provide thoughtful guidance."
        modifiers = generator.generate_modifiers(test_prompt)
        
        print(f"Generated modifiers after training:")
        print(f"  Confidence scores: {modifiers.confidence_scores}")
        print(f"  Generation time: {modifiers.generation_time:.4f}s")
        
    except Exception as e:
        print(f"Error in model training demo: {e}")


def demonstrate_cnn3d_integration():
    """Demonstrate integration with CNN3DProcessor."""
    print("\n=== CNN3D Integration Demo ===")
    
    try:
        # Create generator
        generator = create_embedding_modifier_generator()
        
        # Create mock CNN3D processor for demonstration
        class MockCNN3DProcessor:
            def __init__(self):
                self._embedding_modifier = None
                self.name = "MockCNN3DProcessor"
        
        mock_cnn = MockCNN3DProcessor()
        
        print(f"Integrating with {mock_cnn.name}...")
        
        # Integrate with CNN processor
        generator.integrate_with_cnn3d_processor(mock_cnn)
        
        print("Integration successful!")
        print(f"CNN processor now has external modifier: {hasattr(mock_cnn, '_external_modifier')}")
        print(f"CNN processor has modifier method: {hasattr(mock_cnn, 'generate_and_apply_modifiers')}")
        
        # Test the integrated functionality
        if hasattr(mock_cnn, 'generate_and_apply_modifiers'):
            print("\nTesting integrated modifier functionality...")
            
            # Create mock system context
            class MockSystemContext:
                def __init__(self):
                    self.message = "You are a helpful assistant."
                    self.influence_strength = 0.8
            
            system_context = MockSystemContext()
            base_output = np.random.randn(2, 512)
            
            try:
                modified_output = mock_cnn.generate_and_apply_modifiers(system_context, base_output)
                print(f"Successfully applied modifiers through CNN integration")
                print(f"Original output shape: {base_output.shape}")
                print(f"Modified output shape: {modified_output.shape}")
                print(f"Modification magnitude: {np.mean(np.abs(modified_output - base_output)):.4f}")
            except Exception as e:
                print(f"Error testing integrated functionality: {e}")
        
    except Exception as e:
        print(f"Error in CNN3D integration demo: {e}")


def demonstrate_statistics_and_monitoring():
    """Demonstrate statistics and monitoring capabilities."""
    print("\n=== Statistics and Monitoring Demo ===")
    
    try:
        # Create generator
        generator = create_embedding_modifier_generator()
        
        # Generate several modifiers to build up statistics
        system_prompts = [
            "Be helpful and friendly.",
            "Be technical and precise.",
            "Be creative and engaging.",
            "Be educational and clear.",
            "Be concise and direct."
        ]
        
        print("Generating modifiers to build statistics...")
        
        for i, prompt in enumerate(system_prompts):
            modifiers = generator.generate_modifiers(prompt, influence_strength=0.5 + i * 0.1)
            print(f"  Generated modifiers {i+1}/5")
        
        # Get generation statistics
        stats = generator.get_generation_statistics()
        
        print(f"\nGeneration Statistics:")
        print(f"  Total generations: {stats['total_generations']}")
        print(f"  Total generation time: {stats['total_generation_time']:.4f}s")
        print(f"  Average generation time: {stats['average_generation_time']:.4f}s")
        print(f"  Training steps: {stats['training_steps']}")
        print(f"  Best loss: {stats['best_loss']}")
        print(f"  Model compiled: {stats['model_compiled']}")
        
        print(f"\nModifier Type Usage:")
        for mod_type, count in stats['modifier_type_usage'].items():
            print(f"  {mod_type}: {count} times")
        
        print(f"\nConfiguration:")
        for key, value in stats['config'].items():
            print(f"  {key}: {value}")
        
        # Get model summary
        print(f"\nModel Architecture Summary:")
        summary = generator.get_model_summary()
        if "Model not created yet" not in summary:
            # Print first few lines of summary
            summary_lines = summary.split('\n')[:10]
            for line in summary_lines:
                print(f"  {line}")
            if len(summary.split('\n')) > 10:
                print("  ... (truncated)")
        else:
            print(f"  {summary}")
        
    except Exception as e:
        print(f"Error in statistics demo: {e}")


def demonstrate_model_persistence():
    """Demonstrate saving and loading models."""
    print("\n=== Model Persistence Demo ===")
    
    try:
        # Create and train a simple generator
        generator = create_embedding_modifier_generator()
        
        # Generate some modifiers to ensure model is created
        modifiers = generator.generate_modifiers("Test prompt for model creation")
        print("Created and initialized model")
        
        # Save model (in practice, you'd use a real file path)
        model_path = "/tmp/test_modifier_model.h5"
        
        try:
            generator.save_model(model_path)
            print(f"Model saved to {model_path}")
            
            # Create new generator and load model
            new_generator = create_embedding_modifier_generator()
            new_generator.load_model(model_path)
            print(f"Model loaded from {model_path}")
            
            # Test loaded model
            test_modifiers = new_generator.generate_modifiers("Test prompt for loaded model")
            print("Successfully generated modifiers with loaded model")
            print(f"  Generation time: {test_modifiers.generation_time:.4f}s")
            
            # Clean up
            import os
            if os.path.exists(model_path):
                os.remove(model_path)
                print("Cleaned up temporary model file")
                
        except Exception as e:
            print(f"Model persistence test failed (this is expected in some environments): {e}")
        
    except Exception as e:
        print(f"Error in model persistence demo: {e}")


def main():
    """Run all demonstrations."""
    print("Embedding Modifier Generator Demo")
    print("=" * 50)
    
    try:
        # Run all demonstrations
        demonstrate_basic_usage()
        demonstrate_modifier_application()
        demonstrate_custom_configuration()
        demonstrate_training_preparation()
        demonstrate_model_training()
        demonstrate_cnn3d_integration()
        demonstrate_statistics_and_monitoring()
        demonstrate_model_persistence()
        
        print("\n" + "=" * 50)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()