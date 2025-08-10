#!/usr/bin/env python3
"""
Custom Tokenizer Adapter Demo

This example demonstrates how to create and use custom tokenizer adapters
with the enhanced tokenization system. It shows different ways to integrate
custom tokenizers and validates their functionality.
"""

import numpy as np
from typing import List, Dict
import re

# Import the custom adapter functionality
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lsm.data.adapters.custom_adapter import (
    create_custom_tokenizer, 
    create_custom_tokenizer_from_object,
    CustomTokenizerProtocol
)
from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper


class WordTokenizer:
    """
    Simple word-based tokenizer that splits on whitespace and punctuation.
    
    This demonstrates how to create a custom tokenizer class that implements
    the required interface for the enhanced tokenization system.
    """
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.next_id = 0
        
        # Add special tokens
        self._add_special_token('<PAD>', 0)
        self._add_special_token('<UNK>', 1)
        self._add_special_token('<BOS>', 2)
        self._add_special_token('<EOS>', 3)
        self.next_id = 4
    
    def _add_special_token(self, token: str, token_id: int):
        """Add a special token to the vocabulary."""
        self.word_to_id[token] = token_id
        self.id_to_word[token_id] = token
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Split text into words."""
        # Simple tokenization: split on whitespace and punctuation
        words = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        return words
    
    def _get_or_add_word_id(self, word: str) -> int:
        """Get word ID, adding to vocabulary if needed."""
        if word in self.word_to_id:
            return self.word_to_id[word]
        
        if self.next_id >= self.vocab_size:
            return self.word_to_id['<UNK>']  # Return UNK for out-of-vocab words
        
        word_id = self.next_id
        self.word_to_id[word] = word_id
        self.id_to_word[word_id] = word
        self.next_id += 1
        
        return word_id
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        words = self._tokenize_text(text)
        return [self._get_or_add_word_id(word) for word in words]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        words = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                word = self.id_to_word[token_id]
                if word not in ['<PAD>', '<BOS>', '<EOS>']:
                    words.append(word)
            else:
                words.append('<UNK>')
        
        return ' '.join(words)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self.word_to_id.copy()
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad_token_id': 0,
            'unk_token_id': 1,
            'bos_token_id': 2,
            'eos_token_id': 3
        }


class BytePairTokenizer:
    """
    Simplified Byte Pair Encoding (BPE) tokenizer.
    
    This demonstrates a more sophisticated custom tokenizer that learns
    subword units from training data.
    """
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []
        self.trained = False
    
    def train(self, texts: List[str]):
        """Train the BPE tokenizer on a corpus."""
        print("Training BPE tokenizer...")
        
        # Initialize vocabulary with characters
        char_freq = {}
        for text in texts:
            for char in text:
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Add most frequent characters to vocabulary
        sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        
        for i, (char, _) in enumerate(sorted_chars[:min(256, self.vocab_size - 4)]):
            self.vocab[char] = len(self.vocab)
        
        # Simple BPE: find most frequent bigrams and merge them
        word_freqs = {}
        for text in texts:
            words = text.split()
            for word in words:
                word_chars = ' '.join(word) + ' </w>'
                word_freqs[word_chars] = word_freqs.get(word_chars, 0) + 1
        
        # Perform merges until we reach vocab size
        while len(self.vocab) < self.vocab_size:
            pairs = {}
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            
            if not pairs:
                break
            
            best_pair = max(pairs, key=pairs.get)
            new_symbol = best_pair[0] + best_pair[1]
            
            if new_symbol not in self.vocab:
                self.vocab[new_symbol] = len(self.vocab)
                self.merges.append(best_pair)
            
            # Update word frequencies with merged symbols
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(f"{best_pair[0]} {best_pair[1]}", new_symbol)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs
        
        self.trained = True
        print(f"BPE training complete. Vocabulary size: {len(self.vocab)}")
    
    def _apply_bpe(self, word: str) -> List[str]:
        """Apply BPE merges to a word."""
        if not self.trained:
            return list(word)  # Fallback to character-level
        
        symbols = list(word)
        
        for merge in self.merges:
            i = 0
            while i < len(symbols) - 1:
                if symbols[i] == merge[0] and symbols[i + 1] == merge[1]:
                    symbols = symbols[:i] + [merge[0] + merge[1]] + symbols[i + 2:]
                else:
                    i += 1
        
        return symbols
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not self.trained:
            # Fallback to character-level encoding
            return [self.vocab.get(char, self.vocab.get('<UNK>', 1)) for char in text]
        
        token_ids = []
        words = text.split()
        
        for word in words:
            symbols = self._apply_bpe(word)
            for symbol in symbols:
                token_ids.append(self.vocab.get(symbol, self.vocab.get('<UNK>', 1)))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        id_to_token = {v: k for k, v in self.vocab.items()}
        tokens = []
        
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                if token not in ['<PAD>', '<UNK>', '<BOS>', '<EOS>']:
                    tokens.append(token)
        
        # Simple reconstruction (not perfect for BPE)
        return ''.join(tokens).replace('</w>', ' ').strip()
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab) if self.trained else self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self.vocab.copy()


def demo_simple_function_tokenizer():
    """Demonstrate creating a custom tokenizer with simple functions."""
    print("=== Simple Function Tokenizer Demo ===")
    
    def char_encode(text: str) -> List[int]:
        """Encode text as character codes."""
        return [ord(c) for c in text]
    
    def char_decode(token_ids: List[int]) -> str:
        """Decode character codes to text."""
        return ''.join(chr(tid) for tid in token_ids if 0 <= tid <= 1114111)
    
    # Create custom tokenizer adapter
    adapter = create_custom_tokenizer(
        encode_fn=char_encode,
        decode_fn=char_decode,
        vocab_size=1114112,  # Full Unicode range
        model_name="character_tokenizer"
    )
    
    print(f"Tokenizer: {adapter}")
    print(f"Vocabulary size: {adapter.get_vocab_size()}")
    
    # Test tokenization
    test_text = "Hello, world! 🌍"
    print(f"Original text: '{test_text}'")
    
    token_ids = adapter.encode_single(test_text)
    print(f"Token IDs: {token_ids}")
    
    decoded_text = adapter.decode_single(token_ids)
    print(f"Decoded text: '{decoded_text}'")
    
    # Test batch processing
    texts = ["Hello", "World", "Test"]
    batch_tokens = adapter.tokenize(texts, padding=True)
    print(f"Batch tokens shape: {len(batch_tokens)}x{len(batch_tokens[0])}")
    
    batch_decoded = adapter.decode(batch_tokens)
    print(f"Batch decoded: {batch_decoded}")
    
    print()


def demo_word_tokenizer():
    """Demonstrate creating a custom word-based tokenizer."""
    print("=== Word Tokenizer Demo ===")
    
    # Create word tokenizer
    word_tokenizer = WordTokenizer(vocab_size=1000)
    
    # Create adapter from tokenizer object
    adapter = create_custom_tokenizer_from_object(
        word_tokenizer,
        model_name="word_tokenizer"
    )
    
    print(f"Tokenizer: {adapter}")
    print(f"Vocabulary size: {adapter.get_vocab_size()}")
    
    # Test with sample texts
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello, how are you today?",
        "Machine learning is fascinating!",
        "Natural language processing with custom tokenizers."
    ]
    
    print("Testing word tokenization:")
    for text in sample_texts:
        token_ids = adapter.encode_single(text)
        decoded = adapter.decode_single(token_ids)
        print(f"  '{text}' -> {len(token_ids)} tokens -> '{decoded}'")
    
    # Show vocabulary growth
    print(f"Vocabulary after processing: {len(word_tokenizer.word_to_id)} words")
    print(f"Sample vocabulary: {list(word_tokenizer.word_to_id.items())[:10]}")
    
    # Test special tokens
    special_tokens = adapter.get_special_tokens()
    print(f"Special tokens: {special_tokens}")
    
    print()


def demo_bpe_tokenizer():
    """Demonstrate creating a BPE-style tokenizer."""
    print("=== BPE Tokenizer Demo ===")
    
    # Training corpus
    training_texts = [
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test",
        "machine learning natural language processing",
        "tokenization is important for nlp",
        "custom tokenizers can be very useful",
        "byte pair encoding learns subword units",
        "the the the quick quick brown fox fox"
    ]
    
    # Create and train BPE tokenizer
    bpe_tokenizer = BytePairTokenizer(vocab_size=200)
    bpe_tokenizer.train(training_texts)
    
    # Create adapter
    adapter = create_custom_tokenizer_from_object(
        bpe_tokenizer,
        model_name="bpe_tokenizer"
    )
    
    print(f"Tokenizer: {adapter}")
    print(f"Vocabulary size: {adapter.get_vocab_size()}")
    
    # Test tokenization
    test_texts = [
        "the quick brown fox",
        "hello world",
        "machine learning",
        "unseen words here"
    ]
    
    print("Testing BPE tokenization:")
    for text in test_texts:
        token_ids = adapter.encode_single(text)
        decoded = adapter.decode_single(token_ids)
        print(f"  '{text}' -> {len(token_ids)} tokens -> '{decoded}'")
    
    # Show some learned merges
    print(f"Learned merges (first 10): {bpe_tokenizer.merges[:10]}")
    
    print()


def demo_enhanced_wrapper_integration():
    """Demonstrate integration with EnhancedTokenizerWrapper."""
    print("=== Enhanced Wrapper Integration Demo ===")
    
    # Create a simple custom tokenizer
    def simple_encode(text: str) -> List[int]:
        # Simple word-based encoding with hash
        words = text.lower().split()
        return [hash(word) % 1000 for word in words]
    
    def simple_decode(token_ids: List[int]) -> str:
        # Can't perfectly decode hash-based encoding
        return f"[{len(token_ids)} tokens]"
    
    # Create custom adapter
    custom_adapter = create_custom_tokenizer(
        encode_fn=simple_encode,
        decode_fn=simple_decode,
        vocab_size=1000,
        model_name="hash_tokenizer"
    )
    
    # Create enhanced wrapper
    enhanced_wrapper = EnhancedTokenizerWrapper(
        tokenizer=custom_adapter,
        embedding_dim=64,
        max_length=20
    )
    
    print(f"Enhanced wrapper: {enhanced_wrapper}")
    print(f"Adapter: {enhanced_wrapper.get_adapter()}")
    
    # Test tokenization through wrapper
    test_texts = [
        "This is a test sentence.",
        "Another example text.",
        "Short text."
    ]
    
    print("Testing through enhanced wrapper:")
    for text in test_texts:
        tokens = enhanced_wrapper.encode_single(text)
        print(f"  '{text}' -> {tokens}")
    
    # Test batch processing
    batch_tokens = enhanced_wrapper.tokenize(test_texts, padding=True)
    print(f"Batch processing: {len(batch_tokens)} sequences of length {len(batch_tokens[0])}")
    
    # Get embedding shape
    embedding_shape = enhanced_wrapper.get_token_embeddings_shape()
    print(f"Embedding shape: {embedding_shape}")
    
    print()


def demo_error_handling():
    """Demonstrate error handling and validation."""
    print("=== Error Handling Demo ===")
    
    # Test with broken encode function
    def broken_encode(text: str) -> List[int]:
        raise ValueError("Broken encode function")
    
    def working_decode(token_ids: List[int]) -> str:
        return "decoded"
    
    print("Testing with broken encode function:")
    try:
        adapter = create_custom_tokenizer(
            encode_fn=broken_encode,
            decode_fn=working_decode,
            vocab_size=100
        )
        print("  ERROR: Should have failed!")
    except Exception as e:
        print(f"  Expected error caught: {e}")
    
    # Test with invalid return types
    def invalid_encode(text: str) -> str:  # Should return List[int]
        return "invalid"
    
    print("Testing with invalid return type:")
    try:
        adapter = create_custom_tokenizer(
            encode_fn=invalid_encode,
            decode_fn=working_decode,
            vocab_size=100
        )
        print("  ERROR: Should have failed!")
    except Exception as e:
        print(f"  Expected error caught: {e}")
    
    print()


def main():
    """Run all demos."""
    print("Custom Tokenizer Adapter Demo")
    print("=" * 50)
    
    demo_simple_function_tokenizer()
    demo_word_tokenizer()
    demo_bpe_tokenizer()
    demo_enhanced_wrapper_integration()
    demo_error_handling()
    
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()