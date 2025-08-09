#!/usr/bin/env python3
"""
Test script for response-level training loop implementation.

This script tests the new response-level training functionality including:
- Response-level training loop with cosine similarity loss
- System message integration in training
- Enhanced CNN architectures (2D/3D) support
"""

import os
import sys
import numpy as np
import tensorflow as tf
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from lsm.training.train import LSMTrainer
    from lsm.inference.response_inference_model import TrainingConfig
    from lsm.utils.lsm_logging import get_logger
    
    # Add the methods from task 8.3
    import task_8_3_methods
    task_8_3_methods.add_methods_to_lsm_trainer()
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Trying alternative import path...")
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    from lsm.training.train import LSMTrainer
    from lsm.inference.response_inference_model import TrainingConfig
    from lsm.utils.lsm_logging import get_logger
    
    # Add the methods from task 8.3
    import task_8_3_methods
    task_8_3_methods.add_methods_to_lsm_trainer()

logger = get_logger(__name__)


def create_sample_data(num_samples: int = 100, window_size: int = 10, embedding_dim: int = 128):
    """Create sample data for testing response-level training."""
    np.random.seed(42)
    
    # Create sample input sequences
    X = np.random.randn(num_samples, window_size, embedding_dim).astype(np.float32)
    
    # Create sample target embeddings
    y = np.random.randn(num_samples, embedding_dim).astype(np.float32)
    
    # Split into train/test
    split_idx = int(0.8 * num_samples)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    
    return X_train, y_train, X_test, y_test


def create_sample_system_messages() -> List[str]:
    """Create sample system messages for testing."""
    return [
        "You are a helpful assistant that provides clear and concise responses.",
        "Act as a professional consultant providing expert advice.",
        "Your task is to explain complex topics in simple terms.",
        "Please respond in a friendly and encouraging manner.",
        "Provide detailed technical explanations with examples."
    ]


def test_response_level_training_initialization():
    """Test initialization of response-level training components."""
    print("\n=== Testing Response-Level Training Initialization ===")
    
    try:
        # Create trainer with minimal configuration
        trainer = LSMTrainer(
            window_size=8,
            embedding_dim=64,
            reservoir_units=[32, 16]
        )
        
        # Initialize response-level training
        trainer.initialize_response_level_training(
            use_3d_cnn=False,
            system_message_support=True,
            response_inference_architecture="lstm"  # Use simpler architecture for testing
        )
        
        # Verify components are initialized
        assert trainer.response_level_mode == True, "Response level mode not enabled"
        assert trainer.cnn_architecture_factory is not None, "CNN architecture factory not initialized"
        assert trainer.system_message_processor is not None, "System message processor not initialized"
        assert trainer.response_inference_model is not None, "Response inference model not initialized"
        
        print("✓ Response-level training initialization successful")
        return True
        
    except Exception as e:
        print(f"✗ Response-level training initialization failed: {e}")
        return False


def test_response_level_data_preparation():
    """Test preparation of data for response-level training."""
    print("\n=== Testing Response-Level Data Preparation ===")
    
    try:
        # Create trainer
        trainer = LSMTrainer(
            window_size=8,
            embedding_dim=64
        )
        
        # Initialize response-level training
        trainer.initialize_response_level_training(system_message_support=True)
        
        # Create sample data
        X_train, y_train, X_test, y_test = create_sample_data(
            num_samples=20, window_size=8, embedding_dim=64
        )
        
        # Create sample system messages
        system_messages = create_sample_system_messages()
        
        # Prepare response-level data
        train_embeddings, train_responses, test_embeddings, test_responses = trainer.prepare_response_level_data(
            X_train, y_train, X_test, y_test, system_messages
        )
        
        # Verify data preparation
        assert len(train_embeddings) == len(X_train), "Training embeddings count mismatch"
        assert len(train_responses) == len(X_train), "Training responses count mismatch"
        assert len(test_embeddings) == len(X_test), "Test embeddings count mismatch"
        assert len(test_responses) == len(X_test), "Test responses count mismatch"
        
        # Verify embedding sequences have correct shape
        for emb_seq in train_embeddings[:3]:  # Check first 3
            assert emb_seq.shape[1] == 64, f"Embedding dimension mismatch: {emb_seq.shape}"
            assert emb_seq.shape[0] == 9, f"Sequence length mismatch: {emb_seq.shape[0]} (expected 9 = 8 + 1)"
        
        # Verify responses are strings
        for response in train_responses[:3]:
            assert isinstance(response, str), f"Response is not string: {type(response)}"
            assert len(response) > 0, "Empty response generated"
        
        print("✓ Response-level data preparation successful")
        print(f"  - Prepared {len(train_embeddings)} training sequences")
        print(f"  - Prepared {len(test_embeddings)} test sequences")
        print(f"  - Sample response: '{train_responses[0][:50]}...'")
        return True
        
    except Exception as e:
        print(f"✗ Response-level data preparation failed: {e}")
        return False


def test_cosine_similarity_loss_integration():
    """Test cosine similarity loss integration in training."""
    print("\n=== Testing Cosine Similarity Loss Integration ===")
    
    try:
        # Create trainer
        trainer = LSMTrainer(
            window_size=6,
            embedding_dim=32,
            reservoir_units=[16, 8],
            use_huggingface_data=False
        )
        
        # Initialize components
        trainer.initialize_response_level_training()
        
        # Create minimal sample data
        X_train, y_train, X_test, y_test = create_sample_data(
            num_samples=10, window_size=6, embedding_dim=32
        )
        
        # Test enhanced CNN training with cosine similarity loss
        cnn_results = trainer._train_enhanced_cnn_models(
            X_train, y_train, X_test, y_test,
            use_3d_cnn=False,
            epochs=2,  # Minimal epochs for testing
            batch_size=4
        )
        
        # Verify results
        assert '2d_cnn' in cnn_results, "2D CNN results not found"
        assert 'model' in cnn_results['2d_cnn'], "CNN model not in results"
        assert 'history' in cnn_results['2d_cnn'], "Training history not in results"
        assert 'final_loss' in cnn_results['2d_cnn'], "Final loss not in results"
        
        print("✓ Cosine similarity loss integration successful")
        print(f"  - Final training loss: {cnn_results['2d_cnn']['final_loss']:.6f}")
        return True
        
    except Exception as e:
        print(f"✗ Cosine similarity loss integration failed: {e}")
        return False


def test_system_message_training_support():
    """Test system message support in training."""
    print("\n=== Testing System Message Training Support ===")
    
    try:
        # Create trainer with system message support
        trainer = LSMTrainer(
            window_size=6,
            embedding_dim=32,
            use_huggingface_data=False
        )
        
        # Initialize with system message support
        trainer.initialize_response_level_training(
            use_3d_cnn=True,  # Enable 3D CNN for system messages
            system_message_support=True
        )
        
        # Create sample data
        X_train, y_train, X_test, y_test = create_sample_data(
            num_samples=8, window_size=6, embedding_dim=32
        )
        
        # Create system messages
        system_messages = create_sample_system_messages()[:8]  # Match sample count
        
        # Test system message processing
        if trainer.system_message_processor:
            for msg in system_messages[:2]:  # Test first 2 messages
                context = trainer.system_message_processor.process_system_message(msg)
                assert context.validation_status == True, f"System message validation failed: {msg}"
                assert context.embeddings is not None, "System message embeddings not created"
                assert len(context.embeddings) > 0, "Empty system message embeddings"
        
        # Test system influence calculation
        test_embeddings = [np.random.randn(7, 32) for _ in range(len(X_test))]
        system_influence = trainer._calculate_system_influence(test_embeddings, system_messages[:len(X_test)])
        
        assert isinstance(system_influence, float), f"System influence should be float, got {type(system_influence)}"
        assert 0.0 <= system_influence <= 1.0, f"System influence out of range: {system_influence}"
        
        print("✓ System message training support successful")
        print(f"  - Processed {len(system_messages)} system messages")
        print(f"  - System influence score: {system_influence:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ System message training support failed: {e}")
        return False


def test_full_response_level_training():
    """Test complete response-level training pipeline."""
    print("\n=== Testing Full Response-Level Training Pipeline ===")
    
    try:
        # Create trainer
        trainer = LSMTrainer(
            window_size=6,
            embedding_dim=32,
            reservoir_units=[16, 8],
            use_huggingface_data=False
        )
        
        # Create sample data
        X_train, y_train, X_test, y_test = create_sample_data(
            num_samples=12, window_size=6, embedding_dim=32
        )
        
        # Create system messages
        system_messages = create_sample_system_messages()[:12]
        
        # Configure training
        training_config = TrainingConfig(
            batch_size=4,
            epochs=2,  # Minimal for testing
            learning_rate=0.01,
            validation_split=0.2,
            loss_type="response_level_cosine"
        )
        
        # Run response-level training
        results = trainer.train_response_level(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            system_messages=system_messages,
            training_config=training_config,
            use_3d_cnn=False,  # Use 2D CNN for simpler testing
            epochs=2,
            batch_size=4
        )
        
        # Verify results
        assert 'response_training_results' in results, "Response training results missing"
        assert 'cnn_training_results' in results, "CNN training results missing"
        assert 'coherence_metrics' in results, "Coherence metrics missing"
        assert 'system_influence' in results, "System influence missing"
        assert 'history' in results, "Training history missing"
        
        # Verify coherence metrics
        coherence = results['coherence_metrics']
        assert 'average_coherence' in coherence, "Average coherence missing"
        assert isinstance(coherence['average_coherence'], (int, float)), "Invalid coherence type"
        
        # Verify system influence
        system_influence = results['system_influence']
        assert isinstance(system_influence, (int, float)), "Invalid system influence type"
        
        print("✓ Full response-level training pipeline successful")
        print(f"  - Average coherence: {coherence['average_coherence']:.4f}")
        print(f"  - System influence: {system_influence:.4f}")
        print(f"  - Training completed with {len(results['history']['train_cosine_loss'])} cosine loss values")
        return True
        
    except Exception as e:
        print(f"✗ Full response-level training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all response-level training tests."""
    print("Starting Response-Level Training Tests...")
    print("=" * 60)
    
    # Set TensorFlow to use CPU only for testing
    tf.config.set_visible_devices([], 'GPU')
    
    tests = [
        test_response_level_training_initialization,
        test_response_level_data_preparation,
        test_cosine_similarity_loss_integration,
        test_system_message_training_support,
        test_full_response_level_training
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All response-level training tests passed!")
        return True
    else:
        print(f"❌ {failed} tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)