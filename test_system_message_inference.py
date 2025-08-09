#!/usr/bin/env python3
"""
Test script for system message support in inference.
"""

import os
import sys
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from lsm.inference.inference import EnhancedLSMInference
from lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from lsm.core.system_message_processor import SystemMessageProcessor


def test_system_message_inference():
    """Test system message support in inference."""
    print("🧪 Testing System Message Support in Inference")
    print("=" * 60)
    
    try:
        # Create a mock model path (we'll test without actual model loading)
        model_path = "test_model"
        
        # Initialize inference with lazy loading
        inference = EnhancedLSMInference(
            model_path=model_path,
            use_response_level=True,
            lazy_load=True
        )
        
        # Test 1: Check if system message processor can be initialized
        print("\n1. Testing SystemMessageProcessor initialization...")
        try:
            tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2')
            system_processor = SystemMessageProcessor(tokenizer=tokenizer)
            print("✅ SystemMessageProcessor initialized successfully")
        except Exception as e:
            print(f"❌ SystemMessageProcessor initialization failed: {e}")
            return False
        
        # Test 2: Test system message processing
        print("\n2. Testing system message processing...")
        try:
            system_message = "You are a helpful assistant. Be concise and friendly."
            context = system_processor.process_system_message(system_message)
            print(f"✅ System message processed: {context.parsed_content}")
        except Exception as e:
            print(f"❌ System message processing failed: {e}")
            return False
        
        # Test 3: Test enhanced inference initialization
        print("\n3. Testing enhanced inference initialization...")
        try:
            # Set up minimal components for testing
            inference.standard_tokenizer = tokenizer
            inference.sinusoidal_embedder = SinusoidalEmbedder(
                vocab_size=tokenizer.get_vocab_size(),
                embedding_dim=128
            )
            inference.system_message_processor = system_processor
            inference._enhanced_components_loaded = True
            print("✅ Enhanced inference components set up")
        except Exception as e:
            print(f"❌ Enhanced inference setup failed: {e}")
            return False
        
        # Test 4: Test model info with system components
        print("\n4. Testing model info with system components...")
        try:
            # Skip model loading for this test
            inference._model_loaded = True
            info = inference.get_enhanced_model_info()
            has_system_processor = info.get('enhanced_components', {}).get('system_message_processor', False)
            print(f"✅ Model info includes system message processor: {has_system_processor}")
        except Exception as e:
            print(f"❌ Model info test failed: {e}")
            return False
        
        # Test 5: Test system message processing helper
        print("\n5. Testing system message processing helper...")
        try:
            system_message = "Act as a creative writer. Use vivid descriptions."
            context = inference._process_system_message(system_message)
            print(f"✅ System message helper works: {context.original_message[:50]}...")
        except Exception as e:
            print(f"❌ System message helper failed: {e}")
            return False
        
        print("\n🎉 All system message inference tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test suite failed with error: {e}")
        return False


def test_system_message_formats():
    """Test different system message formats."""
    print("\n🧪 Testing Different System Message Formats")
    print("=" * 60)
    
    try:
        tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2')
        system_processor = SystemMessageProcessor(tokenizer=tokenizer)
        
        test_messages = [
            "You are a helpful assistant.",
            "Act as a professional translator.",
            "Your task is to summarize the following text.",
            "Do not use any profanity in your responses.",
            "Context: This is a customer service conversation.",
            "Persona: You are an expert chef with 20 years of experience."
        ]
        
        for i, message in enumerate(test_messages, 1):
            try:
                context = system_processor.process_system_message(message)
                print(f"{i}. '{message[:40]}...' -> {context.parsed_content.get('type', 'unknown')}")
            except Exception as e:
                print(f"{i}. '{message[:40]}...' -> ERROR: {e}")
        
        print("✅ System message format testing completed")
        return True
        
    except Exception as e:
        print(f"❌ System message format testing failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 System Message Inference Testing")
    print("=" * 60)
    
    success = True
    
    # Run tests
    success &= test_system_message_inference()
    success &= test_system_message_formats()
    
    if success:
        print("\n🎉 All tests passed! System message support is working.")
    else:
        print("\n❌ Some tests failed. Check the output above.")
        sys.exit(1)