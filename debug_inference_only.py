#!/usr/bin/env python3
"""
Debug Inference Only Script

This script focuses only on the inference error to isolate the exact problem.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_inference_components():
    """Test individual inference components to isolate the error."""
    print("🔍 Testing individual inference components...")
    
    try:
        # Import required components
        from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
        from lsm.inference.response_generator import ResponseGenerator
        
        print("✅ Imports successful")
        
        # Create enhanced tokenizer
        print("🔤 Creating enhanced tokenizer...")
        tokenizer = EnhancedTokenizerWrapper(
            tokenizer='gpt2',
            embedding_dim=128,
            max_length=64,
            enable_caching=False
        )
        print(f"✅ Tokenizer created: {tokenizer}")
        
        # Create embedder
        print("🌊 Creating embedder...")
        embedder = tokenizer.create_configurable_sinusoidal_embedder(
            learnable_frequencies=True,
            base_frequency=10000.0
        )
        print(f"✅ Embedder created: {type(embedder).__name__}")
        
        # Test tokenization
        print("🧪 Testing tokenization...")
        test_text = ["Hello, how are you?"]
        tokens = tokenizer.tokenize(test_text)
        print(f"✅ Tokenization successful: {len(tokens[0])} tokens")
        
        # Test embedding
        print("🧪 Testing embedding...")
        try:
            # Test if embedder has the embed method
            if hasattr(embedder, 'embed'):
                print("✅ Embedder has embed method")
                # Try to embed the tokens
                embeddings = embedder.embed(tokens[0])
                print(f"✅ Embedding successful: shape {embeddings.shape}")
            else:
                print("❌ Embedder missing embed method")
                print(f"Available methods: {[m for m in dir(embedder) if not m.startswith('_')]}")
                return False
        except Exception as e:
            print(f"❌ Embedding failed: {e}")
            traceback.print_exc()
            return False
        
        # Create a minimal reservoir model (mock)
        print("🧠 Creating mock reservoir...")
        import numpy as np
        class MockReservoir:
            def predict(self, x):
                # Return same shape as input but with different values
                return np.random.random(x.shape)
        
        reservoir_model = MockReservoir()
        print("✅ Mock reservoir created")
        
        # Create ResponseGenerator
        print("🎭 Creating ResponseGenerator...")
        response_generator = ResponseGenerator(
            tokenizer=tokenizer,
            embedder=embedder,
            reservoir_model=reservoir_model,
            max_response_length=64
        )
        print("✅ ResponseGenerator created")
        
        # Test the problematic method directly
        print("🧪 Testing _process_text_to_embeddings...")
        try:
            token_embedding_seq = response_generator._process_text_to_embeddings(test_text)
            print(f"✅ _process_text_to_embeddings successful: {type(token_embedding_seq)}")
            print(f"   Embeddings shape: {token_embedding_seq.embeddings.shape}")
            print(f"   Tokens count: {len(token_embedding_seq.tokens)}")
        except Exception as e:
            print(f"❌ _process_text_to_embeddings failed: {e}")
            traceback.print_exc()
            return False
        
        # Test response generation
        print("🧪 Testing generate_complete_response...")
        try:
            result = response_generator.generate_complete_response(test_text)
            print(f"✅ Response generation successful!")
            print(f"   Response: {result.response_text}")
            print(f"   Confidence: {result.confidence_score}")
        except Exception as e:
            print(f"❌ Response generation failed: {e}")
            traceback.print_exc()
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Main execution function."""
    print("🔍 LSM Inference Debug Script")
    print("=" * 50)
    
    success = test_inference_components()
    
    if success:
        print("\n🎉 All inference components working correctly!")
        print("The issue might be in the integration with the convenience API.")
    else:
        print("\n❌ Inference components have issues.")
        print("This helps isolate the exact problem.")
    
    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️ Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)