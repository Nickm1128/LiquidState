#!/usr/bin/env python3
"""
Demo script for TiktokenAdapter.

This script demonstrates the usage of the OpenAI tiktoken adapter
with various OpenAI models and configurations.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from lsm.data.adapters.tiktoken_adapter import TiktokenAdapter
    from lsm.data.enhanced_tokenization import TokenizerConfig
    from lsm.utils.lsm_logging import get_logger
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

logger = get_logger(__name__)


def demo_basic_usage():
    """Demonstrate basic tiktoken adapter usage."""
    print("=== Basic TiktokenAdapter Usage ===")
    
    try:
        # Create configuration for GPT-3.5-turbo
        config = TokenizerConfig(
            backend='tiktoken',
            model_name='gpt-3.5-turbo',
            max_length=512
        )
        
        # Initialize adapter
        adapter = TiktokenAdapter(config)
        adapter.initialize()
        
        print(f"Initialized: {adapter}")
        print(f"Vocabulary size: {adapter.get_vocab_size():,}")
        print(f"Special tokens: {adapter.get_special_tokens()}")
        
        # Test tokenization
        texts = [
            "Hello, world!",
            "This is a test of the OpenAI tiktoken tokenizer.",
            "How are you doing today?"
        ]
        
        print(f"\nTokenizing {len(texts)} texts...")
        token_ids = adapter.tokenize(texts, padding=True)
        
        for i, (text, tokens) in enumerate(zip(texts, token_ids)):
            print(f"Text {i+1}: '{text}'")
            print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} (length: {len(tokens)})")
        
        # Test decoding
        print(f"\nDecoding tokens back to text...")
        decoded_texts = adapter.decode(token_ids, skip_special_tokens=True)
        
        for i, (original, decoded) in enumerate(zip(texts, decoded_texts)):
            print(f"Original {i+1}: '{original}'")
            print(f"Decoded {i+1}:  '{decoded.strip()}'")
            print(f"Match: {original.strip() == decoded.strip()}")
            print()
        
    except ImportError:
        print("tiktoken library not available. Install with: pip install tiktoken")
    except Exception as e:
        print(f"Error in basic usage demo: {e}")


def demo_different_models():
    """Demonstrate different OpenAI model tokenizers."""
    print("=== Different OpenAI Models ===")
    
    models_to_test = [
        'gpt-3.5-turbo',
        'gpt-4',
        'text-davinci-003',
        'text-davinci-001',
        'cl100k_base',
        'p50k_base',
        'r50k_base'
    ]
    
    test_text = "The quick brown fox jumps over the lazy dog."
    
    try:
        for model_name in models_to_test:
            try:
                config = TokenizerConfig(
                    backend='tiktoken',
                    model_name=model_name,
                    max_length=512
                )
                
                adapter = TiktokenAdapter(config)
                adapter.initialize()
                
                # Tokenize test text
                tokens = adapter.tokenize(test_text, add_special_tokens=False, padding=False)[0]
                
                print(f"Model: {model_name}")
                print(f"  Encoding: {adapter._encoding_name}")
                print(f"  Vocab size: {adapter.get_vocab_size():,}")
                print(f"  Test text tokens: {len(tokens)} tokens")
                print(f"  First 5 tokens: {tokens[:5]}")
                print()
                
            except Exception as e:
                print(f"Model {model_name}: Error - {e}")
                print()
                
    except ImportError:
        print("tiktoken library not available. Install with: pip install tiktoken")


def demo_special_tokens():
    """Demonstrate special token handling."""
    print("=== Special Token Handling ===")
    
    try:
        # Test with custom special tokens
        config = TokenizerConfig(
            backend='tiktoken',
            model_name='gpt-3.5-turbo',
            max_length=512,
            special_tokens={
                'pad_token': '<PAD>',
                'eos_token': '<END>'
            }
        )
        
        adapter = TiktokenAdapter(config)
        adapter.initialize()
        
        print(f"Special tokens: {adapter.get_special_tokens()}")
        
        # Test with and without special tokens
        text = "Hello, world!"
        
        tokens_with_special = adapter.tokenize(text, add_special_tokens=True, padding=False)[0]
        tokens_without_special = adapter.tokenize(text, add_special_tokens=False, padding=False)[0]
        
        print(f"\nText: '{text}'")
        print(f"With special tokens: {tokens_with_special} (length: {len(tokens_with_special)})")
        print(f"Without special tokens: {tokens_without_special} (length: {len(tokens_without_special)})")
        
        # Decode with and without special tokens
        decoded_with = adapter.decode(tokens_with_special, skip_special_tokens=False)
        decoded_without = adapter.decode(tokens_with_special, skip_special_tokens=True)
        
        print(f"Decoded with special: '{decoded_with}'")
        print(f"Decoded without special: '{decoded_without}'")
        
    except ImportError:
        print("tiktoken library not available. Install with: pip install tiktoken")
    except Exception as e:
        print(f"Error in special tokens demo: {e}")


def demo_truncation_and_padding():
    """Demonstrate truncation and padding features."""
    print("=== Truncation and Padding ===")
    
    try:
        # Use small max_length to demonstrate truncation
        config = TokenizerConfig(
            backend='tiktoken',
            model_name='gpt-3.5-turbo',
            max_length=10
        )
        
        adapter = TiktokenAdapter(config)
        adapter.initialize()
        
        texts = [
            "Short",
            "This is a medium length text that should be truncated.",
            "This is a very long text that definitely exceeds the maximum length and should be truncated to fit within the specified limits."
        ]
        
        print("Testing truncation and padding...")
        print(f"Max length: {config.max_length}")
        print()
        
        # Test with truncation and padding
        token_ids = adapter.tokenize(texts, truncation=True, padding=True)
        
        for i, (text, tokens) in enumerate(zip(texts, token_ids)):
            print(f"Text {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            print(f"Tokens: {tokens} (length: {len(tokens)})")
            
            # Decode to see result
            decoded = adapter.decode(tokens, skip_special_tokens=True)
            print(f"Decoded: '{decoded.strip()}'")
            print()
        
    except ImportError:
        print("tiktoken library not available. Install with: pip install tiktoken")
    except Exception as e:
        print(f"Error in truncation/padding demo: {e}")


def demo_tokenizer_info():
    """Demonstrate tokenizer information retrieval."""
    print("=== Tokenizer Information ===")
    
    try:
        config = TokenizerConfig(
            backend='tiktoken',
            model_name='gpt-4',
            max_length=1024
        )
        
        adapter = TiktokenAdapter(config)
        adapter.initialize()
        
        info = adapter.get_tokenizer_info()
        
        print("Tokenizer Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print(f"\nSupported models: {len(TiktokenAdapter.list_supported_models())} models")
        print("Sample supported models:")
        for model in TiktokenAdapter.list_supported_models()[:10]:
            print(f"  - {model}")
        print("  ...")
        
    except ImportError:
        print("tiktoken library not available. Install with: pip install tiktoken")
    except Exception as e:
        print(f"Error in tokenizer info demo: {e}")


def main():
    """Run all demo functions."""
    print("TiktokenAdapter Demo")
    print("=" * 50)
    
    demo_basic_usage()
    print("\n" + "=" * 50 + "\n")
    
    demo_different_models()
    print("\n" + "=" * 50 + "\n")
    
    demo_special_tokens()
    print("\n" + "=" * 50 + "\n")
    
    demo_truncation_and_padding()
    print("\n" + "=" * 50 + "\n")
    
    demo_tokenizer_info()
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()