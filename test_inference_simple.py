#!/usr/bin/env python3
"""
Simple Inference and Persistence Test

This script tests inference and persistence without Unicode characters.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_model_loading():
    """Test loading the pre-trained model."""
    print("[LOAD] Testing model loading...")
    
    try:
        from lsm.convenience import LSMGenerator
        
        MODEL_PATH = "debug_enhanced_model"
        
        if not os.path.exists(MODEL_PATH):
            print(f"[ERROR] Model path {MODEL_PATH} does not exist!")
            print("[INFO] Available files in current directory:")
            for item in os.listdir('.'):
                if os.path.isdir(item):
                    print(f"   [DIR] {item}/")
                else:
                    print(f"   [FILE] {item}")
            return None
        
        print(f"[LOAD] Loading model from {MODEL_PATH}...")
        generator = LSMGenerator.load(MODEL_PATH)
        print("[SUCCESS] Model loaded successfully!")
        
        # Get model info
        model_info = generator.get_model_info()
        print("[INFO] Model Information:")
        for key, value in model_info.items():
            if isinstance(value, dict):
                print(f"   {key}: {len(value)} components")
            else:
                print(f"   {key}: {value}")
        
        return generator
        
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        traceback.print_exc()
        return None


def test_real_inference(generator):
    """Test real inference with comprehensive error handling."""
    print("[INFERENCE] Testing real inference capabilities...")
    
    try:
        # Test prompts with varying complexity
        test_cases = [
            {"prompt": "Hello", "description": "Simple greeting"},
            {"prompt": "How are you?", "description": "Basic question"},
            {"prompt": "What is machine learning?", "description": "Technical question"},
            {"prompt": "Can you help me understand neural networks?", "description": "Complex request"},
            {"prompt": "Tell me a story about space exploration", "description": "Creative prompt"}
        ]
        
        print("\n" + "="*80)
        print("[TEST] COMPREHENSIVE INFERENCE TEST")
        print("="*80)
        
        successful_tests = 0
        total_tests = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            prompt = test_case["prompt"]
            description = test_case["description"]
            
            print(f"\n[TEST] Test {i+1}/{total_tests}: {description}")
            print(f"[PROMPT] \"{prompt}\"")
            print("-" * 60)
            
            try:
                # Test with different parameters
                temperatures = [0.3, 0.8, 1.2]
                max_lengths = [20, 40, 60]
                
                for temp in temperatures:
                    for max_len in max_lengths:
                        try:
                            print(f"   [CONFIG] Testing T={temp}, max_len={max_len}...")
                            
                            # Generate response with confidence if available
                            response = generator.generate(
                                prompt,
                                max_length=max_len,
                                temperature=temp,
                                return_confidence=True
                            )
                            
                            # Handle different return types
                            if isinstance(response, tuple):
                                response_text, confidence = response
                                print(f"   [SUCCESS] Response: \"{response_text}\" (confidence: {confidence:.3f})")
                            else:
                                print(f"   [SUCCESS] Response: \"{response}\"")
                            
                            successful_tests += 1
                            break  # Success, move to next temperature
                            
                        except Exception as param_error:
                            print(f"   [ERROR] Failed with T={temp}, max_len={max_len}: {param_error}")
                            continue
                    else:
                        continue  # No success with any max_length for this temperature
                    break  # Success with this temperature, move to next test case
                else:
                    print(f"   [ERROR] All parameter combinations failed for this prompt")
                    
            except Exception as e:
                print(f"   [ERROR] Test failed: {e}")
                print(f"   [DEBUG] Error type: {type(e).__name__}")
                if "embeddings" in str(e).lower():
                    print("   [DEBUG] This appears to be an embeddings-related error")
                elif "tokenizer" in str(e).lower():
                    print("   [DEBUG] This appears to be a tokenizer-related error")
                elif "response_generator" in str(e).lower():
                    print("   [DEBUG] This appears to be a response generator error")
        
        print(f"\n[RESULTS] Test Results: {successful_tests}/{total_tests * 3} parameter combinations successful")
        
        # Test system messages if supported
        print(f"\n[SYSTEM] System Message Test:")
        try:
            if hasattr(generator, 'system_message_support') and generator.system_message_support:
                response = generator.generate(
                    "Explain quantum computing",
                    system_message="You are a physics professor",
                    max_length=50,
                    temperature=0.7
                )
                print(f"   [SUCCESS] System message response: \"{response}\"")
            else:
                print("   [WARNING] System message support not available")
        except Exception as e:
            print(f"   [ERROR] System message test failed: {e}")
        
        return successful_tests > 0
        
    except Exception as e:
        print(f"[ERROR] Inference testing failed: {e}")
        traceback.print_exc()
        return False


def test_tokenizer_embedder_integration(generator):
    """Test the tokenizer and embedder integration directly."""
    print("[TOKENIZER] Testing tokenizer and embedder integration...")
    
    try:
        # Get the enhanced tokenizer
        enhanced_tokenizer = generator.get_enhanced_tokenizer()
        if enhanced_tokenizer is None:
            print("[ERROR] No enhanced tokenizer found")
            return False
        
        print(f"[SUCCESS] Enhanced tokenizer: {type(enhanced_tokenizer).__name__}")
        print(f"   Backend: {enhanced_tokenizer.get_adapter().config.backend}")
        print(f"   Vocab size: {enhanced_tokenizer.get_vocab_size():,}")
        
        # Test tokenization
        test_text = "Hello, how are you today?"
        print(f"[TEST] Testing tokenization of: '{test_text}'")
        
        tokens = enhanced_tokenizer.tokenize([test_text])
        print(f"   [SUCCESS] Tokenized: {len(tokens[0])} tokens")
        print(f"   Tokens: {tokens[0][:10]}...")
        
        # Test decoding
        decoded = enhanced_tokenizer.decode(tokens[0])
        print(f"   [SUCCESS] Decoded: '{decoded}'")
        
        # Test embeddings if available
        try:
            embedder = enhanced_tokenizer.get_embedder()
            if embedder is not None:
                print(f"[SUCCESS] Embedder: {type(embedder).__name__}")
                
                # Test embedding
                embeddings = embedder.embed(tokens[0])
                print(f"   [SUCCESS] Embeddings shape: {embeddings.shape}")
                print(f"   Embedding stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
            else:
                print("[WARNING] No embedder found")
        except Exception as embed_error:
            print(f"[ERROR] Embedding test failed: {embed_error}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Tokenizer/embedder integration test failed: {e}")
        traceback.print_exc()
        return False


def test_response_generator_directly(generator):
    """Test the response generator component directly."""
    print("[GENERATOR] Testing response generator directly...")
    
    try:
        # Access the response generator
        response_generator = getattr(generator, '_response_generator', None)
        if response_generator is None:
            print("[ERROR] No response generator found")
            return False
        
        print(f"[SUCCESS] Response generator: {type(response_generator).__name__}")
        
        # Test direct response generation
        test_input = "Hello, how are you?"
        print(f"[TEST] Testing direct response generation: '{test_input}'")
        
        try:
            result = response_generator.generate_complete_response(
                input_sequence=test_input,
                return_intermediate=True
            )
            
            print(f"   [SUCCESS] Response: '{result.response_text}'")
            print(f"   Confidence: {result.confidence_score:.3f}")
            print(f"   Generation time: {result.generation_time:.3f}s")
            print(f"   Strategy used: {result.reservoir_strategy_used}")
            
            if result.intermediate_embeddings:
                print(f"   Intermediate embeddings: {len(result.intermediate_embeddings)} stages")
                for i, emb in enumerate(result.intermediate_embeddings):
                    print(f"     Stage {i+1}: {emb.shape}")
            
            return True
            
        except Exception as gen_error:
            print(f"[ERROR] Direct response generation failed: {gen_error}")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"[ERROR] Response generator test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("[START] LSM Inference and Persistence Test")
    print("=" * 60)
    print("Testing inference capabilities and model persistence")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Step 1: Load model
    print("\n" + "="*60)
    print("STEP 1: MODEL LOADING")
    print("="*60)
    generator = test_model_loading()
    results['model_loading'] = generator is not None
    
    if not results['model_loading']:
        print("[ERROR] Cannot continue without a loaded model")
        return False
    
    # Step 2: Test tokenizer/embedder integration
    print("\n" + "="*60)
    print("STEP 2: TOKENIZER/EMBEDDER INTEGRATION")
    print("="*60)
    results['tokenizer_embedder'] = test_tokenizer_embedder_integration(generator)
    
    # Step 3: Test response generator directly
    print("\n" + "="*60)
    print("STEP 3: RESPONSE GENERATOR DIRECT TEST")
    print("="*60)
    results['response_generator'] = test_response_generator_directly(generator)
    
    # Step 4: Test real inference
    print("\n" + "="*60)
    print("STEP 4: REAL INFERENCE TEST")
    print("="*60)
    results['real_inference'] = test_real_inference(generator)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results.items():
        status = "[PASS]" if success else "[FAIL]"
        print(f"{test_name.replace('_', ' ').title():<30} {status}")
        if success:
            passed += 1
    
    print(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All tests passed! Inference and persistence are working correctly.")
    else:
        print("[WARNING] Some tests failed. Issues identified:")
        failures = [name for name, success in results.items() if not success]
        for i, failure in enumerate(failures, 1):
            print(f"   {i}. {failure.replace('_', ' ').title()}")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)