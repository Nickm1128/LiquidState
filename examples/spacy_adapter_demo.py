#!/usr/bin/env python3
"""
spaCy Tokenizer Adapter Demo

This script demonstrates the usage of the SpacyAdapter for tokenization
with linguistic features, language-specific processing, and Unicode handling.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.data.enhanced_tokenization import TokenizerConfig
from lsm.data.adapters.spacy_adapter import SpacyAdapter


def demo_basic_usage():
    """Demonstrate basic spaCy adapter usage."""
    print("=== Basic spaCy Adapter Usage ===")
    
    # Create configuration for English blank model
    config = TokenizerConfig(
        backend='spacy',
        model_name='en',  # Use blank English model
        max_length=128
    )
    
    try:
        # Initialize adapter
        adapter = SpacyAdapter(config)
        adapter.initialize()
        
        print(f"Initialized spaCy adapter: {adapter}")
        print(f"Vocabulary size: {adapter.get_vocab_size()}")
        print(f"Language: {adapter._language_code}")
        
        # Test tokenization
        texts = [
            "Hello, world! This is a test.",
            "spaCy is great for NLP tasks.",
            "Unicode text: café naïve résumé"
        ]
        
        print("\n--- Tokenization ---")
        for text in texts:
            token_ids = adapter.tokenize([text], add_special_tokens=True, padding=False)
            decoded = adapter.decode(token_ids[0])
            print(f"Text: {text}")
            print(f"Token IDs: {token_ids[0]}")
            print(f"Decoded: {decoded}")
            print()
        
    except Exception as e:
        print(f"Error in basic usage: {e}")
        print("Note: This demo requires spaCy to be installed: pip install spacy")


def demo_linguistic_features():
    """Demonstrate linguistic feature extraction."""
    print("=== Linguistic Features Demo ===")
    
    # Configuration with linguistic features enabled
    config = TokenizerConfig(
        backend='spacy',
        model_name='en',
        max_length=256,
        backend_specific_config={
            'use_linguistic_features': True,
            'unicode_normalization': 'NFC'
        }
    )
    
    try:
        adapter = SpacyAdapter(config)
        adapter.initialize()
        
        # Test text with various linguistic features
        test_text = "The quick brown fox jumps over the lazy dog. It's running fast!"
        
        print(f"Analyzing text: {test_text}")
        
        # Extract linguistic features
        features = adapter.get_linguistic_features(test_text)
        
        print("\n--- Linguistic Features ---")
        print(f"Tokens: {features.get('tokens', [])}")
        print(f"POS Tags: {features.get('pos_tags', [])}")
        print(f"Lemmas: {features.get('lemmas', [])}")
        print(f"Entities: {features.get('entities', [])}")
        print(f"Sentences: {features.get('sentences', [])}")
        
        # Show dependencies
        dependencies = features.get('dependencies', [])
        if dependencies:
            print("\n--- Dependencies ---")
            for dep in dependencies[:5]:  # Show first 5
                print(f"  {dep['text']} -> {dep['head']} ({dep['dep']})")
        
    except Exception as e:
        print(f"Error in linguistic features demo: {e}")


def demo_multilingual_support():
    """Demonstrate multilingual tokenization."""
    print("=== Multilingual Support Demo ===")
    
    # Test different languages
    languages = [
        ('en', "Hello world! How are you today?"),
        ('de', "Hallo Welt! Wie geht es dir heute?"),
        ('fr', "Bonjour le monde! Comment allez-vous aujourd'hui?"),
        ('es', "¡Hola mundo! ¿Cómo estás hoy?")
    ]
    
    for lang_code, text in languages:
        print(f"\n--- Language: {lang_code} ---")
        
        config = TokenizerConfig(
            backend='spacy',
            model_name=lang_code,
            max_length=128,
            backend_specific_config={
                'unicode_normalization': 'NFC'
            }
        )
        
        try:
            adapter = SpacyAdapter(config)
            adapter.initialize()
            
            print(f"Text: {text}")
            
            # Tokenize
            token_ids = adapter.tokenize([text], add_special_tokens=True)
            decoded = adapter.decode(token_ids[0])
            
            print(f"Tokens: {len(token_ids[0])}")
            print(f"Decoded: {decoded}")
            
            # Get tokenizer info
            info = adapter.get_tokenizer_info()
            print(f"Language info: {info.get('language_info', {})}")
            
        except Exception as e:
            print(f"Error with language {lang_code}: {e}")


def demo_custom_configuration():
    """Demonstrate custom configuration options."""
    print("=== Custom Configuration Demo ===")
    
    # Advanced configuration
    config = TokenizerConfig(
        backend='spacy',
        model_name='en',
        max_length=64,
        special_tokens={
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'bos_token': '[START]',
            'eos_token': '[END]'
        },
        backend_specific_config={
            'use_linguistic_features': True,
            'unicode_normalization': 'NFKC',  # Different normalization
            'custom_components': []  # Could add custom pipeline components
        }
    )
    
    try:
        adapter = SpacyAdapter(config)
        adapter.initialize()
        
        print("Custom configuration loaded successfully!")
        
        # Show special tokens
        special_tokens = adapter.get_special_tokens()
        print(f"Special tokens: {special_tokens}")
        
        # Test with custom special tokens
        text = "This is a test with custom tokens."
        token_ids = adapter.tokenize([text], add_special_tokens=True)
        decoded = adapter.decode(token_ids[0], skip_special_tokens=False)
        
        print(f"\nText: {text}")
        print(f"With special tokens: {decoded}")
        
        # Test truncation
        long_text = "This is a very long text that should be truncated because it exceeds the maximum length limit set in the configuration."
        token_ids = adapter.tokenize([long_text], truncation=True, add_special_tokens=True)
        
        print(f"\nLong text truncated to {len(token_ids[0])} tokens (max: {config.max_length})")
        
    except Exception as e:
        print(f"Error in custom configuration demo: {e}")


def demo_batch_processing():
    """Demonstrate batch processing with padding."""
    print("=== Batch Processing Demo ===")
    
    config = TokenizerConfig(
        backend='spacy',
        model_name='en',
        max_length=32
    )
    
    try:
        adapter = SpacyAdapter(config)
        adapter.initialize()
        
        # Texts of different lengths
        texts = [
            "Short text.",
            "This is a medium length text for testing.",
            "This is a much longer text that contains more words and should demonstrate the padding functionality of the tokenizer.",
            "Another short one."
        ]
        
        print("--- Without Padding ---")
        token_ids_no_pad = adapter.tokenize(texts, padding=False, truncation=True)
        for i, (text, tokens) in enumerate(zip(texts, token_ids_no_pad)):
            print(f"Text {i+1} ({len(tokens)} tokens): {text[:50]}...")
        
        print("\n--- With Padding ---")
        token_ids_padded = adapter.tokenize(texts, padding=True, truncation=True)
        for i, (text, tokens) in enumerate(zip(texts, token_ids_padded)):
            print(f"Text {i+1} ({len(tokens)} tokens): {text[:50]}...")
        
        # Verify all sequences have same length when padded
        lengths = [len(seq) for seq in token_ids_padded]
        print(f"\nAll padded sequences have same length: {len(set(lengths)) == 1}")
        print(f"Sequence lengths: {lengths}")
        
    except Exception as e:
        print(f"Error in batch processing demo: {e}")


def demo_supported_models():
    """Show supported models and languages."""
    print("=== Supported Models and Languages ===")
    
    try:
        # List supported models
        models = SpacyAdapter.list_supported_models()
        print(f"Supported models ({len(models)}):")
        for model in sorted(models)[:10]:  # Show first 10
            print(f"  - {model}")
        if len(models) > 10:
            print(f"  ... and {len(models) - 10} more")
        
        # List supported languages
        languages = SpacyAdapter.list_supported_languages()
        print(f"\nSupported languages ({len(languages)}):")
        for lang in sorted(languages)[:15]:  # Show first 15
            print(f"  - {lang}")
        if len(languages) > 15:
            print(f"  ... and {len(languages) - 15} more")
            
    except Exception as e:
        print(f"Error listing supported models: {e}")


def main():
    """Run all demos."""
    print("spaCy Tokenizer Adapter Demo")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        print("\n" + "=" * 50)
        
        demo_linguistic_features()
        print("\n" + "=" * 50)
        
        demo_multilingual_support()
        print("\n" + "=" * 50)
        
        demo_custom_configuration()
        print("\n" + "=" * 50)
        
        demo_batch_processing()
        print("\n" + "=" * 50)
        
        demo_supported_models()
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    print("\nDemo completed!")


if __name__ == '__main__':
    main()