#!/usr/bin/env python3
"""
Interactive Demo
================

This script provides an enhanced interactive demonstration of the LSM inference system.
Shows advanced interactive features and conversation management.
"""

import sys
import os
import time
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference import OptimizedLSMInference
from lsm_exceptions import ModelLoadError, InferenceError
from src.lsm.management.model_manager import ModelManager

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
    print()
    
    return model['path']

def guided_conversation_demo(inference):
    """Demonstrate guided conversation with the model."""
    print("🗣️  Guided Conversation Demo")
    print("=" * 40)
    print("This demo shows how to have a structured conversation with the model.")
    print("We'll build a conversation step by step and see how context affects predictions.")
    print()
    
    # Start with a greeting
    conversation = []
    
    print("Starting conversation...")
    print()
    
    # Step 1: Initial greeting
    conversation.append("Hello")
    print(f"Human: {conversation[-1]}")
    
    try:
        response = inference.predict_next_token(conversation)
        conversation.append(response)
        print(f"Model: {response}")
        print()
        
        # Step 2: Follow up
        conversation.append("How are you today?")
        print(f"Human: {conversation[-1]}")
        
        response = inference.predict_next_token(conversation)
        conversation.append(response)
        print(f"Model: {response}")
        print()
        
        # Step 3: Show context effect
        print("Let's see how the model responds to the same question with different context:")
        print()
        
        # Context 1: Positive conversation
        positive_context = ["Hello", "I'm having a great day", "How are you?"]
        response1 = inference.predict_next_token(positive_context)
        print(f"Positive context: {' → '.join(positive_context)} → {response1}")
        
        # Context 2: Neutral conversation  
        neutral_context = ["Hello", "How are you?"]
        response2 = inference.predict_next_token(neutral_context)
        print(f"Neutral context: {' → '.join(neutral_context)} → {response2}")
        
        # Context 3: Question-focused conversation
        question_context = ["What's your name?", "I'm Alice", "How are you?"]
        response3 = inference.predict_next_token(question_context)
        print(f"Question context: {' → '.join(question_context)} → {response3}")
        
        print()
        print("Notice how the same question gets different responses based on context!")
        print()
        
    except Exception as e:
        print(f"❌ Conversation demo failed: {e}")
        print()

def confidence_exploration_demo(inference):
    """Demonstrate confidence scoring and interpretation."""
    print("📊 Confidence Exploration Demo")
    print("=" * 40)
    print("This demo explores how confidence scores work and what they mean.")
    print()
    
    # Test different types of inputs
    test_cases = [
        {
            "name": "Common greeting",
            "dialogue": ["Hello"],
            "expected": "High confidence - common pattern"
        },
        {
            "name": "Complete conversation",
            "dialogue": ["Hello", "How are you?", "I'm fine", "What about you?"],
            "expected": "Medium confidence - established context"
        },
        {
            "name": "Unusual sequence",
            "dialogue": ["Quantum", "Physics", "Elephant"],
            "expected": "Low confidence - unusual pattern"
        },
        {
            "name": "Single word",
            "dialogue": ["Yes"],
            "expected": "Variable confidence - depends on training"
        },
        {
            "name": "Question sequence",
            "dialogue": ["What's your name?", "I'm Alice", "Nice to meet you"],
            "expected": "High confidence - common social pattern"
        }
    ]
    
    print("Testing confidence scores for different input types:")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            dialogue = test_case["dialogue"]
            expected = test_case["expected"]
            
            print(f"{i}. {test_case['name']}:")
            print(f"   Input: {' → '.join(dialogue)}")
            
            # Get prediction with confidence
            prediction, confidence = inference.predict_with_confidence(dialogue)
            
            # Interpret confidence level
            if confidence > 0.8:
                confidence_level = "Very High 🟢"
            elif confidence > 0.6:
                confidence_level = "High 🟡"
            elif confidence > 0.4:
                confidence_level = "Medium 🟠"
            elif confidence > 0.2:
                confidence_level = "Low 🔴"
            else:
                confidence_level = "Very Low ⚫"
            
            print(f"   Prediction: {prediction}")
            print(f"   Confidence: {confidence:.3f} ({confidence_level})")
            print(f"   Expected: {expected}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

def top_k_exploration_demo(inference):
    """Demonstrate top-k predictions and their interpretation."""
    print("🏆 Top-K Predictions Exploration Demo")
    print("=" * 40)
    print("This demo shows how to interpret multiple prediction candidates.")
    print()
    
    test_dialogues = [
        {
            "name": "Open-ended greeting",
            "dialogue": ["Hello", "How are you?"],
            "analysis": "Should show variety in possible responses"
        },
        {
            "name": "Specific question",
            "dialogue": ["What's your name?"],
            "analysis": "Should show name-related responses"
        },
        {
            "name": "Continuation prompt",
            "dialogue": ["I'm going to the", "store to buy"],
            "analysis": "Should show shopping-related items"
        }
    ]
    
    for i, test in enumerate(test_dialogues, 1):
        try:
            dialogue = test["dialogue"]
            
            print(f"{i}. {test['name']}:")
            print(f"   Input: {' → '.join(dialogue)}")
            print(f"   Analysis: {test['analysis']}")
            
            # Get top-5 predictions
            top_predictions = inference.predict_top_k(dialogue, k=5)
            
            print("   Top 5 Predictions:")
            for rank, (token, score) in enumerate(top_predictions, 1):
                # Add visual indicator for score ranges
                if score > 0.8:
                    indicator = "🟢"
                elif score > 0.6:
                    indicator = "🟡"
                elif score > 0.4:
                    indicator = "🟠"
                else:
                    indicator = "🔴"
                
                print(f"     {rank}. {token:<20} (score: {score:.3f}) {indicator}")
            
            # Calculate score distribution
            scores = [score for _, score in top_predictions]
            if scores:
                score_range = max(scores) - min(scores)
                print(f"   Score Range: {score_range:.3f} (higher = more diverse predictions)")
            
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

def conversation_context_demo(inference):
    """Demonstrate how conversation context affects predictions."""
    print("🔄 Conversation Context Demo")
    print("=" * 40)
    print("This demo shows how adding context changes model predictions.")
    print()
    
    # Start with a base question
    base_question = "What do you think?"
    
    contexts = [
        {
            "name": "No context",
            "context": [],
            "question": base_question
        },
        {
            "name": "Positive context",
            "context": ["I love this weather", "It's so beautiful outside"],
            "question": base_question
        },
        {
            "name": "Problem context",
            "context": ["I'm having trouble", "This is really difficult"],
            "question": base_question
        },
        {
            "name": "Technical context",
            "context": ["I'm working on programming", "The code isn't working"],
            "question": base_question
        },
        {
            "name": "Social context",
            "context": ["I met someone new today", "They seem really nice"],
            "question": base_question
        }
    ]
    
    print(f"Base question: '{base_question}'")
    print("Let's see how different contexts affect the response:")
    print()
    
    for i, ctx in enumerate(contexts, 1):
        try:
            # Build full dialogue
            full_dialogue = ctx["context"] + [ctx["question"]]
            
            print(f"{i}. {ctx['name']}:")
            if ctx["context"]:
                print(f"   Context: {' → '.join(ctx['context'])}")
            else:
                print("   Context: (none)")
            print(f"   Question: {ctx['question']}")
            
            # Get prediction with confidence
            prediction, confidence = inference.predict_with_confidence(full_dialogue)
            
            print(f"   Response: {prediction} (confidence: {confidence:.3f})")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()

def performance_demo(inference):
    """Demonstrate performance features and monitoring."""
    print("⚡ Performance Features Demo")
    print("=" * 40)
    print("This demo shows caching and performance optimization features.")
    print()
    
    # Test caching with repeated predictions
    test_dialogue = ["Hello", "How are you?", "I'm fine"]
    
    print("Testing caching performance:")
    print(f"Test dialogue: {' → '.join(test_dialogue)}")
    print()
    
    # First prediction (cache miss)
    print("1. First prediction (cache miss):")
    start_time = time.time()
    prediction1 = inference.predict_next_token(test_dialogue)
    time1 = time.time() - start_time
    print(f"   Result: {prediction1}")
    print(f"   Time: {time1*1000:.1f} ms")
    
    # Second prediction (cache hit)
    print("2. Second prediction (cache hit):")
    start_time = time.time()
    prediction2 = inference.predict_next_token(test_dialogue)
    time2 = time.time() - start_time
    print(f"   Result: {prediction2}")
    print(f"   Time: {time2*1000:.1f} ms")
    
    # Calculate speedup
    if time2 > 0:
        speedup = time1 / time2
        print(f"   Speedup: {speedup:.1f}x faster")
    
    print()
    
    # Show cache statistics
    print("Cache Statistics:")
    try:
        stats = inference.get_cache_stats()
        print(f"   Prediction cache size: {stats.get('prediction_cache_size', 0)}")
        print(f"   Embedding cache size: {stats.get('embedding_cache_size', 0)}")
        print(f"   Cache hit rate: {stats.get('hit_rate', 0):.2%}")
        print(f"   Total requests: {stats.get('total_requests', 0)}")
        
        if 'memory_mb' in stats:
            print(f"   Memory usage: {stats['memory_mb']:.1f} MB")
        
    except Exception as e:
        print(f"   ❌ Error getting cache stats: {e}")
    
    print()

def interactive_tips_demo():
    """Provide tips for effective interactive use."""
    print("💡 Interactive Usage Tips")
    print("=" * 40)
    
    tips = [
        "Start with simple greetings to warm up the conversation",
        "Build context gradually - each message affects the next prediction",
        "Try the same input with different contexts to see variations",
        "Use confidence scores to gauge prediction reliability",
        "Explore top-k predictions to see alternative responses",
        "Keep conversations focused for better coherence",
        "Use specific contexts to guide the model's responses",
        "Monitor cache hit rates for performance insights"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")
    
    print()

def main():
    """Run the interactive demonstration."""
    print("🚀 LSM Interactive Demo")
    print("=" * 40)
    print("This demo showcases the interactive capabilities of the LSM inference system.")
    print()
    
    # Find a model to use
    model_path = find_example_model()
    if not model_path:
        return
    
    try:
        # Initialize inference
        print("🔧 Initializing inference system...")
        inference = OptimizedLSMInference(
            model_path=model_path,
            lazy_load=True,
            cache_size=1000
        )
        print("✅ Inference system ready!")
        print()
        
        # Run demonstration modules
        guided_conversation_demo(inference)
        confidence_exploration_demo(inference)
        top_k_exploration_demo(inference)
        conversation_context_demo(inference)
        performance_demo(inference)
        interactive_tips_demo()
        
        print("🎉 Interactive demo completed!")
        print()
        print("Ready to try it yourself?")
        print("Run: python inference.py --interactive")
        print()
        print("Or explore other examples:")
        print("- python examples/basic_inference.py")
        print("- python examples/batch_processing.py")
        print("- python examples/model_management.py")
        
    except ModelLoadError as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure the model directory contains all required files.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check the troubleshooting guide in README.md for help.")

if __name__ == "__main__":
    main()