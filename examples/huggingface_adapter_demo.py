#!/usr/bin/env python3
"""
Demo script for HuggingFace tokenizer adapter.

This script demonstrates the functionality of the HuggingFace adapter
including vocabulary extraction, token mapping, and special token handling.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper, TokenizerRegistry
from src.lsm.data.adapters.huggingface_adapter import HuggingFaceAdapter


def demo_basic_functionality():
    """Demonstrate basic HuggingFace adapter functionality."""
    print("=== HuggingFace Adapter Basic Functionality Demo ===\n")
    
    # Create enhanced tokenizer with HuggingFace backend
    print("1. Creating EnhancedTokenizerWrapper with GPT-2...")
    wrapper = EnhancedTokenizerWrapper('gpt2', embedding_dim=128, max_length=256)
    
    print(f"   Tokenizer: {wrapper.get_adapter()}")
    print(f"   Vocabulary size: {wrapper.get_vocab_size():,}")
    print(f"   Embedding dimension: {wrapper.embedding_dim}")
    print()
    
    # Test tokenization
    print("2. Testing tokenization...")
    test_texts = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating!"
    ]
    
    for i, text in enumerate(test_texts, 1):
        tokens = wrapper.encode_single(text)
        decoded = wrapper.decode_single(tokens)
        
        print(f"   Text {i}: {text}")
        print(f"   Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''} (length: {len(tokens)})")
        print(f"   Decoded: {decoded}")
        print()


def demo_special_tokens():
    """Demonstrate special token handling."""
    print("=== Special Token Handling Demo ===\n")
    
    wrapper = EnhancedTokenizerWrapper('gpt2')
    
    # Get special tokens
    special_tokens = wrapper.get_special_tokens()
    print("Special tokens:")
    for token_name, token_id in special_tokens.items():
        print(f"   {token_name}: {token_id}")
    print()
    
    # Test tokenization with and without special tokens
    text = "Hello world!"
    
    tokens_with_special = wrapper.get_adapter().tokenize(text, add_special_tokens=True)[0]
    tokens_without_special = wrapper.get_adapter().tokenize(text, add_special_tokens=False)[0]
    
    print(f"Text: {text}")
    print(f"With special tokens: {tokens_with_special}")
    print(f"Without special tokens: {tokens_without_special}")
    print()
    
    # Test decoding with and without special tokens
    decoded_with_special = wrapper.get_adapter().decode(tokens_with_special, skip_special_tokens=False)
    decoded_without_special = wrapper.get_adapter().decode(tokens_with_special, skip_special_tokens=True)
    
    print(f"Decoded with special tokens: '{decoded_with_special}'")
    print(f"Decoded without special tokens: '{decoded_without_special}'")
    print()


def demo_vocabulary_extraction():
    """Demonstrate vocabulary extraction functionality."""
    print("=== Vocabulary Extraction Demo ===\n")
    
    wrapper = EnhancedTokenizerWrapper('gpt2')
    
    # Get full vocabulary
    vocab = wrapper.get_vocab()
    print(f"Full vocabulary size: {len(vocab):,}")
    
    # Show some example tokens
    print("\nSample vocabulary entries:")
    sample_tokens = list(vocab.items())[:20]
    for token, token_id in sample_tokens:
        print(f"   '{token}' -> {token_id}")
    print("   ...")
    print()
    
    # Test specific token lookups
    test_words = ["hello", "world", "the", "and", "python"]
    print("Token ID lookups:")
    for word in test_words:
        if word in vocab:
            print(f"   '{word}' -> {vocab[word]}")
        else:
            # Try different cases
            found = False
            for case_word in [word.upper(), word.capitalize(), f" {word}", f"Ġ{word}"]:
                if case_word in vocab:
                    print(f"   '{word}' -> {vocab[case_word]} (as '{case_word}')")
                    found = True
                    break
            if not found:
                print(f"   '{word}' -> Not found in vocabulary")
    print()


def demo_different_models():
    """Demonstrate different HuggingFace models."""
    print("=== Different HuggingFace Models Demo ===\n")
    
    models_to_test = [
        ('gpt2', 'GPT-2 (small)'),
        ('bert-base-uncased', 'BERT Base Uncased'),
        ('distilbert-base-uncased', 'DistilBERT Base Uncased')
    ]
    
    test_text = "Hello world, this is a test sentence."
    
    for model_name, model_description in models_to_test:
        try:
            print(f"Testing {model_description} ({model_name})...")
            wrapper = EnhancedTokenizerWrapper(model_name, max_length=128)
            
            # Get basic info
            vocab_size = wrapper.get_vocab_size()
            special_tokens = wrapper.get_special_tokens()
            
            print(f"   Vocabulary size: {vocab_size:,}")
            print(f"   Special tokens: {list(special_tokens.keys())}")
            
            # Test tokenization
            tokens = wrapper.encode_single(test_text)
            decoded = wrapper.decode_single(tokens)
            
            print(f"   Tokenized length: {len(tokens)}")
            print(f"   First 10 tokens: {tokens[:10]}")
            print(f"   Decoded: {decoded}")
            print()
            
        except Exception as e:
            print(f"   Error with {model_name}: {e}")
            print()


def demo_tokenizer_registry():
    """Demonstrate tokenizer registry functionality."""
    print("=== Tokenizer Registry Demo ===\n")
    
    # List available backends
    backends = TokenizerRegistry.list_available_backends()
    print(f"Available backends: {backends}")
    
    # List supported models
    supported_models = TokenizerRegistry.list_supported_models()
    print("\nSupported models by backend:")
    for backend, models in supported_models.items():
        print(f"   {backend}: {models[:5]}{'...' if len(models) > 5 else ''}")
    print()
    
    # Test adapter creation through registry
    print("Creating adapter through registry...")
    adapter = TokenizerRegistry.create_adapter('gpt2', max_length=128)
    print(f"   Created: {adapter}")
    print(f"   Vocab size: {adapter.get_vocab_size():,}")
    print()


def demo_tokenizer_info():
    """Demonstrate detailed tokenizer information."""
    print("=== Detailed Tokenizer Information Demo ===\n")
    
    wrapper = EnhancedTokenizerWrapper('gpt2')
    adapter = wrapper.get_adapter()
    
    if hasattr(adapter, 'get_tokenizer_info'):
        info = adapter.get_tokenizer_info()
        
        print("Tokenizer Information:")
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        print()


def main():
    """Run all demos."""
    print("HuggingFace Tokenizer Adapter Demo")
    print("=" * 50)
    print()
    
    try:
        demo_basic_functionality()
        demo_special_tokens()
        demo_vocabulary_extraction()
        demo_different_models()
        demo_tokenizer_registry()
        demo_tokenizer_info()
        
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()