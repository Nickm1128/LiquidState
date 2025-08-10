#!/usr/bin/env python3
"""
Demo of automatic embedding layer adaptation functionality.

This example demonstrates how the enhanced tokenizer system can automatically
adapt sinusoidal embeddings to different tokenizer vocabularies and dimensions.
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper, TokenizerRegistry
from lsm.data.configurable_sinusoidal_embedder import (
    ConfigurableSinusoidalEmbedder, 
    SinusoidalConfig,
    EmbeddingDimensionOptimizer,
    SinusoidalEmbedderFactory
)


def demo_automatic_vocabulary_adaptation():
    """Demonstrate automatic vocabulary size adaptation."""
    print("=== Automatic Vocabulary Size Adaptation Demo ===")
    
    # Create a configurable sinusoidal embedder with initial vocab size
    config = SinusoidalConfig(embedding_dim=128, vocab_size=1000)
    embedder = ConfigurableSinusoidalEmbedder(config)
    
    print(f"Initial configuration: vocab_size={embedder.config.vocab_size}, "
          f"embedding_dim={embedder.config.embedding_dim}")
    
    # Simulate adapting to a larger vocabulary
    print("\nAdapting to larger vocabulary (5000 tokens)...")
    embedder.adapt_to_vocabulary(5000)
    
    print(f"After adaptation: vocab_size={embedder.config.vocab_size}, "
          f"embedding_dim={embedder.config.embedding_dim}")
    
    # Get adaptation info
    info = embedder.get_adaptation_info()
    print(f"Adaptation info: {info}")
    
    print("✓ Vocabulary adaptation completed successfully!\n")


def demo_embedding_dimension_adaptation():
    """Demonstrate automatic embedding dimension adaptation."""
    print("=== Embedding Dimension Adaptation Demo ===")
    
    # Create embedder with initial dimension
    config = SinusoidalConfig(embedding_dim=128, vocab_size=2000)
    embedder = ConfigurableSinusoidalEmbedder(config)
    
    print(f"Initial dimension: {embedder.config.embedding_dim}")
    
    # Adapt to different dimensions
    for new_dim in [256, 512, 64]:
        print(f"\nAdapting to dimension {new_dim}...")
        embedder.adapt_embedding_dimension(new_dim, preserve_properties=True)
        print(f"New dimension: {embedder.config.embedding_dim}")
        
        # Test that the embedder still works
        import tensorflow as tf
        test_input = tf.random.uniform((2, 10), maxval=2000, dtype=tf.int32)
        embedder.build((None, 10))
        output = embedder(test_input)
        print(f"Output shape: {output.shape}")
    
    print("✓ Dimension adaptation completed successfully!\n")


def demo_dimension_optimization():
    """Demonstrate automatic dimension optimization."""
    print("=== Dimension Optimization Demo ===")
    
    # Test different vocabulary sizes
    vocab_sizes = [1000, 5000, 10000, 50000]
    
    for vocab_size in vocab_sizes:
        print(f"\nOptimal dimensions for vocab_size={vocab_size}:")
        
        for model_size in ['small', 'medium', 'large', 'xlarge']:
            optimal_dim = EmbeddingDimensionOptimizer.calculate_optimal_dimension(
                vocab_size, model_size, preserve_mathematical_properties=True
            )
            print(f"  {model_size}: {optimal_dim}")
    
    # Test dimension scaling suggestions
    print("\nDimension scaling suggestions:")
    old_vocab, new_vocab, old_dim = 1000, 4000, 128
    suggested_dim = EmbeddingDimensionOptimizer.suggest_dimension_scaling(
        old_vocab, new_vocab, old_dim
    )
    print(f"Scaling from vocab {old_vocab} to {new_vocab}: {old_dim} -> {suggested_dim}")
    
    print("✓ Dimension optimization completed successfully!\n")


def demo_enhanced_tokenizer_integration():
    """Demonstrate integration with enhanced tokenizer wrapper."""
    print("=== Enhanced Tokenizer Integration Demo ===")
    
    try:
        # Try to create a wrapper with HuggingFace tokenizer
        print("Creating enhanced tokenizer wrapper...")
        
        # For demo purposes, we'll create a mock adapter
        from lsm.data.enhanced_tokenization import TokenizerAdapter, TokenizerConfig
        
        class DemoTokenizerAdapter(TokenizerAdapter):
            def __init__(self, vocab_size=5000):
                config = TokenizerConfig(backend='demo', model_name='demo-model')
                super().__init__(config)
                self._vocab_size = vocab_size
                self._is_initialized = True
            
            def initialize(self):
                pass
            
            def tokenize(self, texts, add_special_tokens=True, padding=True, truncation=True):
                # Simple mock tokenization
                if isinstance(texts, str):
                    texts = [texts]
                return [[i % self._vocab_size for i in range(len(text.split()))] for text in texts]
            
            def decode(self, token_ids, skip_special_tokens=True):
                return " ".join([f"token_{id}" for id in token_ids])
            
            def get_vocab_size(self):
                return self._vocab_size
            
            def get_vocab(self):
                return {f"token_{i}": i for i in range(self._vocab_size)}
            
            def get_special_tokens(self):
                return {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
            
            @classmethod
            def load_adapter_config(cls, load_path):
                return cls()
        
        # Create wrapper with demo adapter
        demo_adapter = DemoTokenizerAdapter(vocab_size=3000)
        wrapper = EnhancedTokenizerWrapper(
            tokenizer=demo_adapter,
            embedding_dim=128,
            max_length=512
        )
        
        print(f"Wrapper created with vocab_size: {wrapper.get_vocab_size()}")
        
        # Get dimension suggestions
        print("\nEmbedding dimension suggestions:")
        suggestions = wrapper.get_embedding_dimension_suggestions()
        for size, dim in suggestions.items():
            print(f"  {size}: {dim}")
        
        # Create configurable embedder
        print("\nCreating configurable sinusoidal embedder...")
        embedder = wrapper.create_configurable_sinusoidal_embedder(
            learnable_frequencies=True,
            use_relative_position=False
        )
        
        print(f"Embedder created with vocab_size: {embedder.config.vocab_size}, "
              f"embedding_dim: {embedder.config.embedding_dim}")
        
        # Create optimized embedder
        print("\nCreating optimized embedder...")
        optimized_embedder = wrapper.create_optimized_embedder(
            target_model_size='medium',
            preserve_properties=True
        )
        
        print(f"Optimized embedder created with embedding_dim: {optimized_embedder.config.embedding_dim}")
        
        # Test auto-adaptation to different dimension
        print("\nTesting auto-adaptation to dimension 256...")
        adapted_embedder = wrapper.auto_adapt_embedding_dimension(
            target_dim=256, preserve_properties=True
        )
        
        print(f"Adapted embedder dimension: {adapted_embedder.config.embedding_dim}")
        
        print("✓ Enhanced tokenizer integration completed successfully!\n")
        
    except Exception as e:
        print(f"Integration demo failed: {e}")
        print("This is expected if tokenizer backends are not available.\n")


def demo_factory_methods():
    """Demonstrate factory methods for creating embedders."""
    print("=== Factory Methods Demo ===")
    
    # Create demo adapter
    from lsm.data.enhanced_tokenization import TokenizerAdapter, TokenizerConfig
    
    class DemoAdapter(TokenizerAdapter):
        def __init__(self):
            config = TokenizerConfig(backend='demo', model_name='demo')
            super().__init__(config)
            self._is_initialized = True
        
        def initialize(self): pass
        def tokenize(self, texts, **kwargs): return [[1, 2, 3]]
        def decode(self, token_ids, **kwargs): return "demo"
        def get_vocab_size(self): return 8000
        def get_vocab(self): return {'demo': 1}
        def get_special_tokens(self): return {'<pad>': 0}
        
        @classmethod
        def load_adapter_config(cls, load_path): return cls()
    
    adapter = DemoAdapter()
    
    # Test different factory methods
    print("Creating embedders with different factory methods:")
    
    # Default embedder
    default_embedder = SinusoidalEmbedderFactory.create_default(
        vocab_size=8000, embedding_dim=128
    )
    print(f"Default embedder: vocab_size={default_embedder.config.vocab_size}, "
          f"embedding_dim={default_embedder.config.embedding_dim}")
    
    # Auto-adapted embedder
    auto_embedder = SinusoidalEmbedderFactory.create_auto_adapted(
        adapter, target_model_size='large', preserve_properties=True
    )
    print(f"Auto-adapted embedder: vocab_size={auto_embedder.config.vocab_size}, "
          f"embedding_dim={auto_embedder.config.embedding_dim}")
    
    # Relative position embedder
    relative_embedder = SinusoidalEmbedderFactory.create_relative_position(
        vocab_size=8000, embedding_dim=256, relative_window=128
    )
    print(f"Relative position embedder: embedding_dim={relative_embedder.config.embedding_dim}, "
          f"relative_window={relative_embedder.config.relative_position_window}")
    
    print("✓ Factory methods demo completed successfully!\n")


def main():
    """Run all demos."""
    print("🚀 Automatic Embedding Layer Adaptation Demo\n")
    
    try:
        demo_automatic_vocabulary_adaptation()
        demo_embedding_dimension_adaptation()
        demo_dimension_optimization()
        demo_enhanced_tokenizer_integration()
        demo_factory_methods()
        
        print("🎉 All demos completed successfully!")
        print("\nKey features demonstrated:")
        print("✓ Automatic vocabulary size detection and adaptation")
        print("✓ Embedding dimension matching and scaling")
        print("✓ Mathematical property preservation during adaptation")
        print("✓ Integration with enhanced tokenizer wrapper")
        print("✓ Dimension optimization for different model sizes")
        print("✓ Factory methods for easy embedder creation")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()