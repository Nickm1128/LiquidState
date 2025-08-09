#!/usr/bin/env python3
"""Simple test for 3D CNN training with system messages."""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_functionality():
    """Test basic functionality without complex dependencies."""
    print("Testing basic 3D CNN training functionality...")
    
    try:
        # Import the trainer
        from lsm.training.train import LSMTrainer
        
        # Add the methods from task 8.3
        import task_8_3_methods
        task_8_3_methods.add_methods_to_lsm_trainer()
        
        # Create trainer with minimal configuration
        trainer = LSMTrainer(
            window_size=4,
            embedding_dim=32,
            reservoir_units=[16, 8]
        )
        
        print("✓ LSMTrainer created successfully")
        
        # Test initialization of response-level training
        trainer.initialize_response_level_training(
            use_3d_cnn=True,
            system_message_support=True
        )
        
        print("✓ Response-level training initialized")
        
        # Test system message processing
        if trainer.system_message_processor:
            test_message = "You are a helpful assistant."
            context = trainer.system_message_processor.process_system_message(test_message)
            print(f"✓ System message processed: {context.parsed_content['format']}")
        
        # Test model configuration update
        trainer.update_model_configuration_for_3d_cnn(
            enable_3d_cnn=True,
            system_message_support=True
        )
        
        print("✓ Model configuration updated for 3D CNN")
        
        # Create sample data for testing
        X_train = np.random.randn(8, 4, 32).astype(np.float32)
        y_train = np.random.randn(8, 32).astype(np.float32)
        X_test = np.random.randn(4, 4, 32).astype(np.float32)
        y_test = np.random.randn(4, 32).astype(np.float32)
        
        # Test data preparation
        train_embeddings, train_responses, test_embeddings, test_responses = trainer.prepare_response_level_data(
            X_train, y_train, X_test, y_test
        )
        
        print(f"✓ Data prepared: {len(train_embeddings)} training, {len(test_embeddings)} test sequences")
        
        # Test system influence calculation
        system_messages = [
            "You are a helpful assistant.",
            "Please be concise in your responses.",
            "Act as a technical expert.",
            "Provide detailed explanations."
        ]
        
        system_influence = trainer._calculate_system_influence(test_embeddings, system_messages)
        print(f"✓ System influence calculated: {system_influence:.4f}")
        
        # Test validation
        validation_results = trainer.validate_system_aware_generation(
            test_embeddings, system_messages
        )
        
        print(f"✓ Validation completed: {validation_results.get('success_rate', 0.0):.2f} success rate")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)