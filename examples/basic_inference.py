#!/usr/bin/env python3
"""
Basic Inference Examples
========================

This script demonstrates basic usage of the enhanced LSM inference system.
Shows simple prediction, confidence scoring, and top-k predictions.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

# Import from the LSM package
try:
    from lsm import OptimizedLSMInference, ModelManager
    from lsm.utils.lsm_exceptions import ModelLoadError, InferenceError
except ImportError as e:
    # Handle TensorFlow import issues gracefully
    if "tensorflow" in str(e).lower() or "dll" in str(e).lower():
        print("❌ TensorFlow import error detected.")
        print("This example requires TensorFlow to be properly installed.")
        print("Please check your TensorFlow installation and try again.")
        print("\nFor installation help, see: https://www.tensorflow.org/install")
        sys.exit(1)
    else:
        print(f"❌ Import error: {e}")
        print("Please ensure the LSM package is properly installed.")
        sys.exit(1)

def find_example_model():
    """Find an available model for demonstration."""
    manager = ModelManager()
    models = manager.list_available_models()
    
    if not models:
        print("❌ No trained models found!")
        print("Please train a model first using: python main.py train")
        return None
    
    # Use the most recent model
    model = models[0]
    print(f"✅ Using model: {model['path']}")
    print(f"   Created: {model.get('created_at', 'Unknown')}")
    print(f"   Performance: MSE={model.get('test_mse', 'N/A')}")
    print()
    
    return model['path']

def basic_prediction_example(inference):
    """Demonstrate basic next-token prediction."""
    print("🔮 Basic Prediction Example")
    print("=" * 40)
    
    # Example dialogue sequences
    examples = [
        ["Hello", "How are you?"],
        ["Good morning", "Nice weather today"],
        ["What's your name?", "I'm Alice", "Nice to meet you"],
        ["How was your day?", "It was great", "What about yours?"],
        ["I love programming", "Me too", "What language do you prefer?"]
    ]
    
    for i, dialogue in enumerate(examples, 1):
        try:
            print(f"Example {i}:")
            print(f"  Input: {' → '.join(dialogue)}")
            
            # Get prediction
            next_token = inference.predict_next_token(dialogue)
            print(f"  Prediction: {next_token}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

def confidence_scoring_example(inference):
    """Demonstrate prediction with confidence scores."""
    print("📊 Confidence Scoring Example")
    print("=" * 40)
    
    examples = [
        ["Hello"],
        ["How are you?", "I'm fine"],
        ["What's the weather like?"],
        ["I'm feeling happy today"],
        ["Can you help me?"]
    ]
    
    for i, dialogue in enumerate(examples, 1):
        try:
            print(f"Example {i}:")
            print(f"  Input: {' → '.join(dialogue)}")
            
            # Get prediction with confidence
            next_token, confidence = inference.predict_with_confidence(dialogue)
            
            # Interpret confidence level
            if confidence > 0.8:
                confidence_level = "Very High"
            elif confidence > 0.6:
                confidence_level = "High"
            elif confidence > 0.4:
                confidence_level = "Medium"
            elif confidence > 0.2:
                confidence_level = "Low"
            else:
                confidence_level = "Very Low"
            
            print(f"  Prediction: {next_token}")
            print(f"  Confidence: {confidence:.3f} ({confidence_level})")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

def top_k_predictions_example(inference):
    """Demonstrate top-k predictions."""
    print("🏆 Top-K Predictions Example")
    print("=" * 40)
    
    examples = [
        ["Hello", "How are you?"],
        ["What's your favorite", "color?"],
        ["I'm going to the", "store"],
        ["The weather is", "really nice"],
        ["Can you tell me", "about yourself?"]
    ]
    
    for i, dialogue in enumerate(examples, 1):
        try:
            print(f"Example {i}:")
            print(f"  Input: {' → '.join(dialogue)}")
            
            # Get top-5 predictions
            top_predictions = inference.predict_top_k(dialogue, k=5)
            
            print("  Top 5 Predictions:")
            for rank, (token, score) in enumerate(top_predictions, 1):
                print(f"    {rank}. {token} (score: {score:.3f})")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()

def model_info_example(inference):
    """Demonstrate model information retrieval."""
    print("ℹ️  Model Information Example")
    print("=" * 40)
    
    try:
        info = inference.get_model_info()
        
        print("Model Configuration:")
        config = info.get('configuration', {})
        print(f"  Window Size: {config.get('window_size', 'N/A')}")
        print(f"  Embedding Dimension: {config.get('embedding_dim', 'N/A')}")
        print(f"  Reservoir Type: {config.get('reservoir_type', 'N/A')}")
        print(f"  Sparsity: {config.get('sparsity', 'N/A')}")
        print()
        
        print("Training Information:")
        metadata = info.get('metadata', {})
        print(f"  Training Duration: {metadata.get('training_duration_seconds', 'N/A')} seconds")
        print(f"  Final Test MSE: {metadata.get('performance_metrics', {}).get('final_test_mse', 'N/A')}")
        print(f"  Dataset: {metadata.get('dataset_info', {}).get('source', 'N/A')}")
        print()
        
        print("System Information:")
        system_info = metadata.get('system_info', {})
        print(f"  Python Version: {system_info.get('python_version', 'N/A')}")
        print(f"  TensorFlow Version: {system_info.get('tensorflow_version', 'N/A')}")
        print(f"  Platform: {system_info.get('platform', 'N/A')}")
        print()
        
    except Exception as e:
        print(f"❌ Error retrieving model info: {e}")
        print()

def cache_statistics_example(inference):
    """Demonstrate cache statistics monitoring."""
    print("📈 Cache Statistics Example")
    print("=" * 40)
    
    # Make some predictions to populate cache
    test_dialogues = [
        ["Hello", "How are you?"],
        ["Good morning"],
        ["Hello", "How are you?"],  # Repeat to show caching
        ["What's up?"],
        ["Hello", "How are you?"]   # Another repeat
    ]
    
    print("Making predictions to populate cache...")
    for dialogue in test_dialogues:
        try:
            inference.predict_next_token(dialogue)
        except:
            pass  # Ignore errors for this demo
    
    # Get cache statistics
    try:
        stats = inference.get_cache_stats()
        
        print("\nCache Statistics:")
        print(f"  Prediction Cache Size: {stats.get('prediction_cache_size', 'N/A')}")
        print(f"  Embedding Cache Size: {stats.get('embedding_cache_size', 'N/A')}")
        print(f"  Cache Hit Rate: {stats.get('hit_rate', 0):.2%}")
        print(f"  Total Requests: {stats.get('total_requests', 'N/A')}")
        print(f"  Cache Hits: {stats.get('cache_hits', 'N/A')}")
        
        if 'memory_mb' in stats:
            print(f"  Memory Usage: {stats['memory_mb']:.1f} MB")
        
        print()
        
    except Exception as e:
        print(f"❌ Error retrieving cache stats: {e}")
        print()

def main():
    """Run all basic inference examples."""
    print("🚀 LSM Basic Inference Examples")
    print("=" * 50)
    print()
    
    # Find a model to use
    model_path = find_example_model()
    if not model_path:
        return
    
    try:
        # Initialize inference with optimizations
        print("🔧 Initializing inference system...")
        inference = OptimizedLSMInference(
            model_path=model_path,
            lazy_load=True,
            cache_size=1000
        )
        print("✅ Inference system ready!")
        print()
        
        # Run examples
        basic_prediction_example(inference)
        confidence_scoring_example(inference)
        top_k_predictions_example(inference)
        model_info_example(inference)
        cache_statistics_example(inference)
        
        print("🎉 All examples completed successfully!")
        print()
        print("Next steps:")
        print("- Try the interactive mode: python inference.py --interactive")
        print("- Explore batch processing: python examples/batch_processing.py")
        print("- Check model management: python examples/model_management.py")
        
    except ModelLoadError as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure the model directory contains all required files.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check the troubleshooting guide in README.md for help.")

if __name__ == "__main__":
    main()