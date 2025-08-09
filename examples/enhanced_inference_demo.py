#!/usr/bin/env python3
"""
Enhanced Inference Demo

This script demonstrates the new enhanced inference system with:
- StandardTokenizerWrapper integration
- SinusoidalEmbedder for optimized embeddings
- Response-level inference instead of token-level
- Support for system messages

The enhanced system provides better text processing and more coherent responses.
"""

import os
import sys
import tempfile
import numpy as np
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lsm.inference.inference import EnhancedLSMInference
from src.lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from src.lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)


def create_mock_model_directory():
    """Create a temporary directory with mock model files for demonstration."""
    temp_dir = tempfile.mkdtemp()
    
    # Create mock config file
    config_content = '''
{
    "window_size": 5,
    "embedding_dim": 128,
    "reservoir_type": "standard",
    "model_version": "enhanced_demo"
}
'''
    
    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        f.write(config_content)
    
    logger.info(f"Created mock model directory: {temp_dir}")
    return temp_dir


def create_mock_trainer():
    """Create a mock trainer for demonstration."""
    trainer = Mock()
    trainer.window_size = 5
    trainer.model = Mock()
    
    # Mock model prediction - returns random reservoir output
    def mock_predict(inputs):
        batch_size = inputs.shape[0] if len(inputs.shape) > 2 else 1
        return np.random.normal(0, 0.1, (batch_size, 128)).astype(np.float32)
    
    trainer.model.predict = mock_predict
    trainer.get_model_info = Mock(return_value={
        'architecture': {
            'reservoir_type': 'standard',
            'window_size': 5,
            'input_dim': 128,
            'reservoir_size': 256
        }
    })
    
    return trainer


def create_mock_legacy_tokenizer():
    """Create a mock legacy tokenizer for backward compatibility."""
    tokenizer = Mock()
    tokenizer.is_fitted = True
    tokenizer.encode = Mock(return_value=np.random.normal(0, 0.1, (5, 128)).astype(np.float32))
    tokenizer.decode_embedding = Mock(return_value="mock legacy response")
    return tokenizer


def demonstrate_enhanced_inference():
    """Demonstrate the enhanced inference system capabilities."""
    print("🚀 Enhanced LSM Inference Demonstration")
    print("=" * 50)
    
    # Create mock model directory
    model_dir = create_mock_model_directory()
    
    try:
        # Initialize enhanced inference system
        print("\n1. Initializing Enhanced Inference System")
        print("-" * 40)
        
        inference = EnhancedLSMInference(
            model_path=model_dir,
            use_response_level=True,
            tokenizer_name='gpt2',
            lazy_load=False,  # Load immediately for demo
            cache_size=100
        )
        
        # Mock the trainer and tokenizer for demonstration
        inference.trainer = create_mock_trainer()
        inference.legacy_tokenizer = create_mock_legacy_tokenizer()
        inference._model_loaded = True
        inference._tokenizer_loaded = True
        
        print("✓ Enhanced inference system initialized")
        print(f"  - Model path: {model_dir}")
        print(f"  - Response-level inference: {inference.use_response_level}")
        print(f"  - Tokenizer: {inference.tokenizer_name}")
        
        # Demonstrate enhanced tokenization
        print("\n2. Enhanced Tokenization System")
        print("-" * 40)
        
        # This will create new tokenizer and embedder
        inference._load_enhanced_components()
        
        print("✓ Enhanced components loaded:")
        print(f"  - StandardTokenizerWrapper: {inference.standard_tokenizer.tokenizer_name}")
        print(f"  - Vocabulary size: {inference.standard_tokenizer.get_vocab_size()}")
        print(f"  - SinusoidalEmbedder: {inference.sinusoidal_embedder.embedding_dim}D embeddings")
        
        # Demonstrate response generation
        print("\n3. Response Generation")
        print("-" * 40)
        
        test_inputs = [
            "Hello, how are you today?",
            "What is the weather like?",
            "Can you help me with a question?",
            "Tell me about artificial intelligence."
        ]
        
        for i, input_text in enumerate(test_inputs, 1):
            print(f"\nTest {i}: '{input_text}'")
            
            try:
                # Generate response using enhanced system
                response = inference.generate_response(input_text)
                print(f"  Response: '{response}'")
                
                # Also test enhanced tokenizer prediction
                prediction, confidence = inference.predict_with_enhanced_tokenizer(input_text)
                print(f"  Enhanced prediction: '{prediction}' (confidence: {confidence:.3f})")
                
            except Exception as e:
                print(f"  Error: {e}")
        
        # Demonstrate system message support
        print("\n4. System Message Support")
        print("-" * 40)
        
        system_message = "You are a helpful and friendly assistant."
        user_input = "How can I learn programming?"
        
        print(f"System message: '{system_message}'")
        print(f"User input: '{user_input}'")
        
        try:
            response = inference.generate_response(user_input, system_message)
            print(f"System-aware response: '{response}'")
        except Exception as e:
            print(f"System message error: {e}")
        
        # Demonstrate model information
        print("\n5. Enhanced Model Information")
        print("-" * 40)
        
        info = inference.get_enhanced_model_info()
        
        print("Model Architecture:")
        if 'architecture' in info:
            arch = info['architecture']
            print(f"  - Reservoir type: {arch.get('reservoir_type', 'Unknown')}")
            print(f"  - Window size: {arch.get('window_size', 'Unknown')}")
        
        print("Enhanced Components:")
        if 'enhanced_components' in info:
            comp = info['enhanced_components']
            print(f"  - Tokenizer: {comp['standard_tokenizer']['name']}")
            print(f"  - Vocab size: {comp['standard_tokenizer']['vocab_size']}")
            print(f"  - Embedder fitted: {comp['sinusoidal_embedder']['is_fitted']}")
            print(f"  - Response generator: {'✓' if comp['response_generator'] else '✗'}")
        
        print("Performance:")
        if 'performance' in info:
            perf = info['performance']
            print(f"  - Response-level mode: {'✓' if perf['use_response_level'] else '✗'}")
            print(f"  - Cache sizes: {perf['cache_sizes']}")
        
        # Demonstrate backward compatibility
        print("\n6. Backward Compatibility")
        print("-" * 40)
        
        # Test legacy prediction method
        dialogue_sequence = ["hello", "how", "are", "you", "today"]
        
        try:
            legacy_prediction = inference.predict_next_token(dialogue_sequence)
            print(f"Legacy prediction: '{legacy_prediction}'")
            print("✓ Backward compatibility maintained")
        except Exception as e:
            print(f"Legacy prediction error: {e}")
        
        print("\n✅ Enhanced Inference Demonstration Complete!")
        print("\nKey Features Demonstrated:")
        print("  ✓ StandardTokenizerWrapper integration")
        print("  ✓ SinusoidalEmbedder for optimized embeddings")
        print("  ✓ Response-level inference")
        print("  ✓ System message support")
        print("  ✓ Enhanced model information")
        print("  ✓ Backward compatibility with legacy methods")
        
    except Exception as e:
        logger.exception("Demonstration failed")
        print(f"❌ Error during demonstration: {e}")
        return False
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)
    
    return True


def demonstrate_comparison():
    """Demonstrate comparison between legacy and enhanced inference."""
    print("\n🔄 Legacy vs Enhanced Comparison")
    print("=" * 50)
    
    # This would show the differences between the old and new systems
    print("\nLegacy System:")
    print("  - DialogueTokenizer (custom implementation)")
    print("  - Token-level prediction")
    print("  - Limited vocabulary")
    print("  - No system message support")
    
    print("\nEnhanced System:")
    print("  - StandardTokenizerWrapper (GPT-2, BERT, etc.)")
    print("  - SinusoidalEmbedder (optimized for LSM)")
    print("  - Response-level generation")
    print("  - Large vocabulary (50K+ tokens)")
    print("  - System message integration")
    print("  - Better caching and performance")


def main():
    """Main demonstration function."""
    print("Enhanced LSM Inference System Demo")
    print("This demo shows the new tokenization and response generation capabilities")
    print()
    
    try:
        # Run the main demonstration
        success = demonstrate_enhanced_inference()
        
        if success:
            # Show comparison
            demonstrate_comparison()
            
            print("\n" + "=" * 60)
            print("Demo completed successfully!")
            print("The enhanced inference system provides:")
            print("  • Better tokenization with standard models")
            print("  • Optimized embeddings for sine-activated LSM")
            print("  • Complete response generation")
            print("  • System message support")
            print("  • Improved performance and caching")
            return 0
        else:
            print("❌ Demo failed")
            return 1
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
        return 0
    except Exception as e:
        logger.exception("Demo failed")
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())