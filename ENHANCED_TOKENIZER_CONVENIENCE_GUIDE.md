# Enhanced Tokenizer Convenience Functions Guide

This guide covers the new enhanced tokenizer convenience functions available in the LSM project, making it easy to work with different tokenizer backends and advanced embedding techniques.

## 🚀 Quick Start

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from lsm.convenience import LSMGenerator
from lsm.convenience.utils import preprocess_conversation_data

# Create enhanced tokenizer with automatic backend detection
tokenizer = EnhancedTokenizerWrapper(
    tokenizer='gpt2',  # or 'bert-base-uncased', 'distilbert-base-uncased', etc.
    embedding_dim=256,
    max_length=128,
    enable_caching=True
)

# Create configurable sinusoidal embedder
embedder = tokenizer.create_configurable_sinusoidal_embedder(
    learnable_frequencies=True,
    base_frequency=10000.0
)

# Use with LSM Generator
generator = LSMGenerator(
    tokenizer='gpt2',
    embedding_type='configurable_sinusoidal',
    embedding_dim=256,
    reservoir_type='attentive'
)
```

## 🔤 Enhanced Tokenizer Features

### Automatic Backend Detection

The enhanced tokenizer automatically detects and uses the appropriate backend:

```python
# These all work automatically
tokenizer_gpt2 = EnhancedTokenizerWrapper('gpt2')
tokenizer_bert = EnhancedTokenizerWrapper('bert-base-uncased')
tokenizer_distilbert = EnhancedTokenizerWrapper('distilbert-base-uncased')
```

### Configurable Sinusoidal Embeddings

Create advanced sinusoidal embeddings with learnable parameters:

```python
# Basic sinusoidal embedder
embedder = tokenizer.create_sinusoidal_embedder(
    max_position=10000,
    temperature=1.0
)

# Advanced configurable embedder
embedder = tokenizer.create_configurable_sinusoidal_embedder(
    learnable_frequencies=True,
    use_relative_position=False,
    base_frequency=10000.0,
    frequency_scaling=1.0
)

# Auto-adapt embedding dimensions
embedder = tokenizer.auto_adapt_embedding_dimension(
    target_dim=512,
    preserve_properties=True
)
```

### Intelligent Caching

Enable caching for improved performance:

```python
from lsm.data.intelligent_caching import CacheConfig

cache_config = CacheConfig(
    max_cache_size=10000,
    enable_batch_caching=True,
    enable_cache_warming=True
)

tokenizer = EnhancedTokenizerWrapper(
    tokenizer='gpt2',
    enable_caching=True,
    cache_config=cache_config
)
```

## 🧠 LSM Generator Integration

### Basic Usage with Enhanced Tokenizer

```python
from lsm.convenience import LSMGenerator

generator = LSMGenerator(
    # Enhanced tokenizer settings
    tokenizer='gpt2',
    embedding_dim=256,
    embedding_type='configurable_sinusoidal',
    enable_caching=True,
    
    # Sinusoidal configuration
    sinusoidal_config={
        'learnable_frequencies': True,
        'base_frequency': 10000.0,
        'use_relative_position': False
    },
    
    # Model architecture
    reservoir_type='attentive',
    window_size=8,
    system_message_support=True
)
```

### Advanced Configuration

```python
# Get preset and customize
config = ConvenienceConfig.get_preset('balanced')
config.update({
    'tokenizer': 'bert-base-uncased',
    'embedding_dim': 512,
    'embedding_type': 'configurable_sinusoidal',
    'tokenizer_backend_config': {
        'trust_remote_code': False,
        'use_fast': True,
        'do_lower_case': True
    },
    'sinusoidal_config': {
        'learnable_frequencies': True,
        'base_frequency': 10000.0,
        'frequency_scaling': 1.2,
        'use_relative_position': True
    }
})

generator = LSMGenerator(**config)
```

## 📊 Data Processing Convenience Functions

### Conversation Data Preprocessing

```python
from lsm.convenience.utils import (
    preprocess_conversation_data,
    detect_conversation_format,
    convert_conversation_format,
    validate_conversation_data
)

# Detect format automatically
format_type = detect_conversation_format(raw_data)
print(f"Detected format: {format_type}")

# Preprocess with custom settings
processed_data = preprocess_conversation_data(
    raw_data,
    min_message_length=5,
    max_message_length=500,
    min_conversation_length=2,
    normalize_whitespace=True,
    return_format="simple_list"
)

# Convert between formats
converted_data = convert_conversation_format(
    data=raw_data,
    target_format="structured",
    source_format="simple_list"
)

# Validate data
validated_data = validate_conversation_data(raw_data)
```

### Structured Conversation Support

```python
from lsm.convenience.utils import validate_structured_conversation_data

# Validate structured conversations with system messages
structured_data = validate_structured_conversation_data(
    data=conversations,
    require_system_messages=True,
    require_roles=True
)

# Each conversation will have:
# - messages: List of message texts
# - system_message: System prompt
# - roles: List of speaker roles
# - conversation_id: Unique identifier
# - metadata: Additional information
```

## 🎯 Training with Enhanced Features

### Standard Training

```python
# Prepare data
conversations = preprocess_conversation_data(raw_conversations)

# Train with enhanced tokenizer
generator.fit(
    conversations,
    epochs=20,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)
```

### Streaming Training for Large Datasets

```python
# For large datasets that don't fit in memory
generator.fit_streaming(
    data_source="large_dataset.txt",
    batch_size=1000,
    epochs=10,
    auto_adjust_batch_size=True,
    memory_threshold_mb=500.0
)
```

## 🎭 Inference and Generation

### Basic Generation

```python
# Simple generation
response = generator.generate(
    "Hello, how are you?",
    max_length=50,
    temperature=0.8
)

# With system message
response = generator.generate(
    "Explain machine learning",
    system_message="You are a helpful AI teacher",
    max_length=100,
    temperature=0.7
)
```

### Batch Generation

```python
prompts = [
    "What is AI?",
    "How do neural networks work?",
    "Explain deep learning"
]

responses = generator.batch_generate(
    prompts,
    max_length=60,
    temperature=0.8
)
```

### Advanced Generation Options

```python
# Create inference-optimized tokenizer
inference_tokenizer = generator.create_tokenizer_for_inference()

# Get enhanced tokenizer for custom processing
enhanced_tokenizer = generator.get_enhanced_tokenizer()
if enhanced_tokenizer:
    # Custom tokenization
    tokens = enhanced_tokenizer.tokenize(["Custom text"])
    embeddings = enhanced_tokenizer.create_configurable_sinusoidal_embedder()
```

## 🔧 Utility Functions

### System Resource Management

```python
from lsm.convenience.utils import (
    check_system_resources,
    estimate_training_time,
    get_optimal_batch_size
)

# Check available resources
resources = check_system_resources(
    operation="training",
    estimated_memory_mb=1000,
    estimated_disk_mb=500
)

# Estimate training time
time_estimate = estimate_training_time(
    data_size=10000,
    config={'reservoir_type': 'attentive', 'epochs': 20}
)

# Get optimal batch size
batch_size = get_optimal_batch_size(
    data_size=10000,
    available_memory_mb=8000,
    model_memory_mb=2000
)
```

### Model Information and Analysis

```python
# Get comprehensive model info
model_info = generator.get_model_info()

# Get tokenizer-specific information
tokenizer_info = generator.get_tokenizer_info()

# Get enhanced tokenizer details
enhanced_tokenizer = generator.get_enhanced_tokenizer()
if enhanced_tokenizer:
    vocab_size = enhanced_tokenizer.get_vocab_size()
    backend = enhanced_tokenizer.get_adapter().config.backend
    embedding_shape = enhanced_tokenizer.get_token_embeddings_shape()
```

## 💾 Model Persistence

### Save and Load Models

```python
# Save complete model with enhanced tokenizer
generator.save("my_enhanced_model")

# Load model
loaded_generator = LSMGenerator.load("my_enhanced_model")

# Verify loaded model
test_response = loaded_generator.generate("Test prompt")
```

## 🎛️ Configuration Presets

### Available Presets

```python
from lsm.convenience.config import ConvenienceConfig

# Fast training (lower quality, faster)
fast_config = ConvenienceConfig.get_preset('fast')

# Balanced (good quality/speed tradeoff)
balanced_config = ConvenienceConfig.get_preset('balanced')

# Quality (best quality, slower)
quality_config = ConvenienceConfig.get_preset('quality')

# Customize any preset
custom_config = ConvenienceConfig.get_preset('balanced')
custom_config.update({
    'tokenizer': 'bert-base-uncased',
    'embedding_dim': 512,
    'reservoir_type': 'hierarchical'
})
```

## 🚀 Best Practices

### 1. Tokenizer Selection

- **GPT-2**: Good for general text generation
- **BERT**: Better for understanding tasks
- **DistilBERT**: Faster, smaller version of BERT

### 2. Embedding Configuration

- Use `learnable_frequencies=True` for better adaptation
- Set `base_frequency=10000.0` for most tasks
- Enable caching for repeated tokenization

### 3. Training Optimization

- Start with 'balanced' preset and customize
- Use streaming for datasets > 1GB
- Enable validation split for monitoring

### 4. Memory Management

- Use batch processing for multiple prompts
- Enable intelligent caching
- Monitor system resources during training

## 📚 Examples

See the complete examples in:
- `LSM_Enhanced_Pipeline_Demo.ipynb` - Full pipeline demonstration
- `examples/enhanced_tokenizer_examples.py` - Code examples
- `tests/test_convenience_integration.py` - Integration tests

## 🔗 Related Documentation

- [Enhanced Tokenization API](src/lsm/data/enhanced_tokenization.py)
- [Convenience API Reference](src/lsm/convenience/)
- [Configuration Guide](src/lsm/convenience/config.py)
- [Data Format Handling](src/lsm/convenience/data_formats.py)