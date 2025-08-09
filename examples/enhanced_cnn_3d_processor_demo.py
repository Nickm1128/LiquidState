#!/usr/bin/env python3
"""
Enhanced CNN 3D Processor Demo.

This demo showcases the enhanced CNN3DProcessor with improved system message
integration, proper tokenization support, and training pipeline for system-aware
response generation.
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.core.cnn_3d_processor import (
    CNN3DProcessor, 
    SystemContext, 
    create_system_aware_processor
)
from lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)


def create_sample_data():
    """Create sample data for demonstration."""
    # Sample reservoir output (batch_size=2, depth=32, height=32, width=32, channels=1)
    reservoir_output = np.random.randn(2, 32, 32, 32, 1).astype(np.float32)
    
    # Sample system messages
    system_messages = [
        "You are a helpful and friendly assistant. Respond with enthusiasm.",
        "You are a technical expert. Provide detailed and accurate information.",
        "You are a creative writer. Use vivid language and imagery.",
        "You are a teacher. Explain concepts clearly and patiently."
    ]
    
    # Sample training data
    training_data = []
    for i in range(10):
        training_data.append({
            'reservoir_output': np.random.randn(32, 32, 32, 1).astype(np.float32),
            'system_message': system_messages[i % len(system_messages)],
            'target_response': f"This is a sample response {i} that demonstrates the system-aware behavior."
        })
    
    return reservoir_output, system_messages, training_data


def demo_basic_enhanced_processing():
    """Demonstrate basic enhanced processing capabilities."""
    print("\n" + "="*60)
    print("ENHANCED CNN 3D PROCESSOR - BASIC PROCESSING DEMO")
    print("="*60)
    
    try:
        # Create sample data
        reservoir_output, system_messages, _ = create_sample_data()
        
        # Create enhanced processor
        processor = create_system_aware_processor(
            window_size=32,
            channels=1,
            system_dim=256,
            output_dim=512
        )
        
        print(f"Created enhanced CNN3DProcessor:")
        print(f"  - Reservoir shape: {processor.reservoir_shape}")
        print(f"  - System embedding dim: {processor.system_embedding_dim}")
        print(f"  - Output embedding dim: {processor.output_embedding_dim}")
        
        # Test enhanced processing
        system_message = system_messages[0]
        print(f"\nProcessing with system message: '{system_message[:50]}...'")
        
        result = processor.process_reservoir_output_with_modifiers(
            reservoir_output,
            system_message,
            influence_strength=0.8,
            use_advanced_modifiers=True
        )
        
        print(f"\nProcessing Results:")
        print(f"  - Output shape: {result.output_embeddings.shape}")
        print(f"  - System influence: {result.system_influence:.4f}")
        print(f"  - Processing time: {result.processing_time:.4f}s")
        
        if result.modifier_details:
            print(f"  - Modifier types: {result.modifier_details.get('modifier_types', [])}")
            print(f"  - Confidence scores: {list(result.modifier_details.get('confidence_scores', {}).keys())}")
        
        if result.tokenization_info:
            print(f"  - Tokenization info available: {bool(result.tokenization_info)}")
        
        return True
        
    except Exception as e:
        print(f"Error in basic enhanced processing demo: {e}")
        return False


def demo_tokenizer_integration():
    """Demonstrate tokenizer integration."""
    print("\n" + "="*60)
    print("ENHANCED CNN 3D PROCESSOR - TOKENIZER INTEGRATION DEMO")
    print("="*60)
    
    try:
        # Create tokenizer and embedder
        print("Creating StandardTokenizerWrapper...")
        tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2', max_length=256)
        
        print("Creating SinusoidalEmbedder...")
        embedder = SinusoidalEmbedder(
            vocab_size=tokenizer.get_vocab_size(),
            embedding_dim=256
        )
        
        # Fit embedder on sample data
        sample_texts = [
            "Hello world this is a test",
            "The quick brown fox jumps",
            "Machine learning is fascinating",
            "Natural language processing works"
        ]
        
        tokenized_samples = []
        for text in sample_texts:
            tokens = tokenizer.encode_single(text)
            tokenized_samples.append(tokens[:50])  # Limit length
        
        # Pad sequences to same length
        max_len = max(len(seq) for seq in tokenized_samples)
        padded_samples = []
        for seq in tokenized_samples:
            padded = seq + [tokenizer.get_special_tokens()['pad_token_id']] * (max_len - len(seq))
            padded_samples.append(padded[:max_len])
        
        training_tokens = np.array(padded_samples)
        embedder.fit(training_tokens, epochs=20)
        
        print(f"Tokenizer vocab size: {tokenizer.get_vocab_size()}")
        print(f"Embedder fitted: {embedder._is_fitted}")
        
        # Create processor with tokenizer and embedder
        processor = create_system_aware_processor(
            window_size=32,
            channels=1,
            system_dim=256,
            output_dim=512,
            tokenizer=tokenizer,
            embedder=embedder
        )
        
        print(f"\nProcessor statistics:")
        stats = processor.get_processing_statistics()
        for key, value in stats.items():
            print(f"  - {key}: {value}")
        
        # Test processing with proper tokenization
        reservoir_output, system_messages, _ = create_sample_data()
        system_message = system_messages[1]
        
        print(f"\nProcessing with tokenizer integration...")
        print(f"System message: '{system_message[:50]}...'")
        
        result = processor.process_reservoir_output_with_modifiers(
            reservoir_output,
            system_message,
            influence_strength=1.0,
            use_advanced_modifiers=True
        )
        
        print(f"\nResults with tokenizer:")
        print(f"  - Output shape: {result.output_embeddings.shape}")
        print(f"  - System influence: {result.system_influence:.4f}")
        print(f"  - Processing time: {result.processing_time:.4f}s")
        
        if result.tokenization_info:
            print(f"  - Tokenizer type: {result.tokenization_info.get('tokenizer_type')}")
            print(f"  - Vocab size: {result.tokenization_info.get('vocab_size')}")
            print(f"  - Message length: {result.tokenization_info.get('system_message_length')}")
            print(f"  - Tokenized length: {result.tokenization_info.get('tokenized_length')}")
        
        return True
        
    except Exception as e:
        print(f"Error in tokenizer integration demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_training_pipeline():
    """Demonstrate training pipeline for system-aware response generation."""
    print("\n" + "="*60)
    print("ENHANCED CNN 3D PROCESSOR - TRAINING PIPELINE DEMO")
    print("="*60)
    
    try:
        # Create processor
        processor = create_system_aware_processor(
            window_size=32,
            channels=1,
            system_dim=256,
            output_dim=512
        )
        
        # Create training data
        _, _, training_data = create_sample_data()
        
        print(f"Created training data with {len(training_data)} samples")
        
        # Create training model
        print("Creating training model...")
        training_model = processor.create_training_model()
        
        print("Training model summary:")
        print(processor.get_training_model_summary())
        
        # Prepare a small subset for quick demo
        demo_training_data = training_data[:5]
        demo_validation_data = training_data[5:8]
        
        print(f"\nTraining on {len(demo_training_data)} samples...")
        print(f"Validating on {len(demo_validation_data)} samples...")
        
        # Train the model (just a few epochs for demo)
        training_result = processor.train_system_aware_model(
            training_data=demo_training_data,
            validation_data=demo_validation_data,
            epochs=3,  # Small number for demo
            batch_size=2
        )
        
        print(f"\nTraining completed!")
        print(f"Training metrics:")
        for key, value in training_result['metrics'].items():
            print(f"  - {key}: {value}")
        
        # Test trained model
        print(f"\nTesting trained model...")
        test_reservoir = np.random.randn(1, 32, 32, 32, 1).astype(np.float32)
        test_system_message = "You are a helpful assistant."
        
        result = processor.process_reservoir_output_with_modifiers(
            test_reservoir,
            test_system_message,
            influence_strength=0.9,
            use_advanced_modifiers=True
        )
        
        print(f"Test result shape: {result.output_embeddings.shape}")
        print(f"System influence: {result.system_influence:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error in training pipeline demo: {e}")
        import traceback
        traceback.print_exc()
        return False


def demo_system_message_variations():
    """Demonstrate processing with different system message variations."""
    print("\n" + "="*60)
    print("ENHANCED CNN 3D PROCESSOR - SYSTEM MESSAGE VARIATIONS DEMO")
    print("="*60)
    
    try:
        # Create processor
        processor = create_system_aware_processor(
            window_size=32,
            channels=1,
            system_dim=256,
            output_dim=512
        )
        
        # Create sample reservoir output
        reservoir_output = np.random.randn(1, 32, 32, 32, 1).astype(np.float32)
        
        # Test different system messages and influence strengths
        test_cases = [
            ("You are a helpful assistant.", 0.5),
            ("You are a technical expert. Be precise and detailed.", 1.0),
            ("You are creative and imaginative. Use vivid language.", 1.5),
            ("", 0.0),  # Empty system message
        ]
        
        results = []
        
        for system_message, influence_strength in test_cases:
            print(f"\nTesting: '{system_message[:40]}...' (influence: {influence_strength})")
            
            result = processor.process_reservoir_output_with_modifiers(
                reservoir_output,
                system_message,
                influence_strength=influence_strength,
                use_advanced_modifiers=True
            )
            
            results.append({
                'message': system_message,
                'influence': influence_strength,
                'system_influence_score': result.system_influence,
                'output_norm': np.linalg.norm(result.output_embeddings),
                'processing_time': result.processing_time
            })
            
            print(f"  - System influence score: {result.system_influence:.4f}")
            print(f"  - Output norm: {np.linalg.norm(result.output_embeddings):.4f}")
            print(f"  - Processing time: {result.processing_time:.4f}s")
        
        # Compare results
        print(f"\n" + "-"*50)
        print("COMPARISON OF SYSTEM MESSAGE EFFECTS:")
        print("-"*50)
        
        for i, result in enumerate(results):
            print(f"{i+1}. Message: '{result['message'][:30]}...'")
            print(f"   Influence setting: {result['influence']}")
            print(f"   Actual influence: {result['system_influence_score']:.4f}")
            print(f"   Output magnitude: {result['output_norm']:.4f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"Error in system message variations demo: {e}")
        return False


def main():
    """Run all enhanced CNN 3D processor demonstrations."""
    print("ENHANCED CNN 3D PROCESSOR COMPREHENSIVE DEMO")
    print("=" * 80)
    
    demos = [
        ("Basic Enhanced Processing", demo_basic_enhanced_processing),
        ("Tokenizer Integration", demo_tokenizer_integration),
        ("Training Pipeline", demo_training_pipeline),
        ("System Message Variations", demo_system_message_variations),
    ]
    
    results = {}
    
    for demo_name, demo_func in demos:
        print(f"\nRunning {demo_name}...")
        try:
            success = demo_func()
            results[demo_name] = "✓ PASSED" if success else "✗ FAILED"
        except Exception as e:
            print(f"Demo failed with error: {e}")
            results[demo_name] = "✗ ERROR"
    
    # Summary
    print("\n" + "="*80)
    print("DEMO RESULTS SUMMARY")
    print("="*80)
    
    for demo_name, result in results.items():
        print(f"{demo_name:.<50} {result}")
    
    # Overall success
    passed_count = sum(1 for result in results.values() if "PASSED" in result)
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} demos passed")
    
    if passed_count == total_count:
        print("🎉 All enhanced CNN 3D processor demos completed successfully!")
    else:
        print("⚠️  Some demos had issues. Check the output above for details.")


if __name__ == "__main__":
    main()