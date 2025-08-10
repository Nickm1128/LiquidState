# Enhanced Tokenizer API Documentation

This document provides comprehensive API documentation for the enhanced tokenizer system in LSM, including all classes, methods, parameters, and usage examples.

## Table of Contents

1. [Overview](#overview)
2. [Core Classes](#core-classes)
3. [Tokenizer Adapters](#tokenizer-adapters)
4. [Sinusoidal Embeddings](#sinusoidal-embeddings)
5. [Streaming Data Processing](#streaming-data-processing)
6. [Performance Optimization](#performance-optimization)
7. [Usage Examples](#usage-examples)
8. [Error Handling](#error-handling)
9. [Best Practices](#best-practices)

## Overview

The enhanced tokenizer system provides a flexible, high-performance tokenization and embedding framework that supports:

- **Multiple Tokenizer Backends**: HuggingFace, OpenAI tiktoken, spaCy, and custom tokenizers
- **Sinusoidal Embeddings**: Configurable sinusoidal embeddings with learnable parameters
- **Streaming Data Processing**: Memory-efficient processing of large datasets
- **Performance Optimization**: GPU acceleration, intelligent caching, and memory-efficient storage
- **Seamless Integration**: Drop-in compatibility with existing LSM convenience API

## Core Classes

### EnhancedTokenizerWrapper

The main wrapper class that provides a unified interface for different tokenizer backends.

```python
class EnhancedTokenizerWrapper:
    def __init__(self, 
                 tokenizer: Union[str, TokenizerAdapter], 
                 embedding_dim: int = 128,
                 max_length: int = 512,
                 special_tokens: Optional[Dict[str, str]] = None,
                 backend_specific_config: Optional[Dict[str, Any]] = None,
                 enable_caching: bool = True,
                 cache_config: Optional[CacheConfig] = None)
```

**Parameters:**
- `tokenizer` (Union[str, TokenizerAdapter]): Tokenizer backend name, model name, or adapter instance
- `embedding_dim` (int, optional): Dimension for sinusoidal embeddings. Defaults to 128.
- `max_length` (int, optional): Maximum sequence length. Defaults to 512.
- `special_tokens` (Optional[Dict[str, str]], optional): Special token configuration
- `backend_specific_config` (Optional[Dict[str, Any]], optional): Backend-specific configuration
- `enable_caching` (bool, optional): Whether to enable intelligent caching. Defaults to True.
- `cache_config` (Optional[CacheConfig], optional): Configuration for caching system

**Key Methods:**

#### tokenize()
```python
def tokenize(self, texts: Union[str, List[str]], 
             add_special_tokens: bool = True,
             padding: bool = True, 
             truncation: bool = True) -> List[List[int]]
```

Tokenize texts to token IDs.

**Parameters:**
- `texts` (Union[str, List[str]]): Single text or list of texts to tokenize
- `add_special_tokens` (bool, optional): Whether to add special tokens. Defaults to True.
- `padding` (bool, optional): Whether to pad sequences. Defaults to True.
- `truncation` (bool, optional): Whether to truncate sequences. Defaults to True.

**Returns:**
- `List[List[int]]`: List of token ID sequences

**Example:**
```python
tokenizer = EnhancedTokenizerWrapper('gpt2')
tokens = tokenizer.tokenize(['Hello world', 'How are you?'])
print(tokens)  # [[15496, 995], [2437, 389, 345, 30]]
```

#### create_configurable_sinusoidal_embedder()
```python
def create_configurable_sinusoidal_embedder(self, 
                                           learnable_frequencies: bool = True,
                                           use_relative_position: bool = False,
                                           base_frequency: float = 10000.0,
                                           frequency_scaling: float = 1.0,
                                           **kwargs) -> ConfigurableSinusoidalEmbedder
```

Create a configurable sinusoidal embedder automatically adapted to this tokenizer.

**Parameters:**
- `learnable_frequencies` (bool, optional): Whether to use learnable frequency parameters. Defaults to True.
- `use_relative_position` (bool, optional): Whether to use relative positional encoding. Defaults to False.
- `base_frequency` (float, optional): Base frequency for sinusoidal patterns. Defaults to 10000.0.
- `frequency_scaling` (float, optional): Scaling factor for frequencies. Defaults to 1.0.
- `**kwargs`: Additional configuration parameters

**Returns:**
- `ConfigurableSinusoidalEmbedder`: Configured embedder instance

**Example:**
```python
tokenizer = EnhancedTokenizerWrapper('bert-base-uncased', embedding_dim=512)
embedder = tokenizer.create_configurable_sinusoidal_embedder(
    learnable_frequencies=True,
    base_frequency=5000.0,
    use_relative_position=True
)
```

#### fit_streaming()
```python
def fit_streaming(self, data_source: Union[str, List[str], StreamingDataIterator],
                 batch_size: int = 1000,
                 epochs: int = 100,
                 max_position: int = 10000,
                 temperature: float = 1.0,
                 memory_threshold_mb: float = 1000.0,
                 progress_callback: Optional[callable] = None,
                 auto_adjust_batch_size: bool = True,
                 min_batch_size: int = 100,
                 max_batch_size: int = 10000) -> SinusoidalEmbedder
```

Fit a sinusoidal embedder on streaming data for memory-efficient training.

**Parameters:**
- `data_source` (Union[str, List[str], StreamingDataIterator]): Data source specification
- `batch_size` (int, optional): Initial batch size for processing. Defaults to 1000.
- `epochs` (int, optional): Number of training epochs. Defaults to 100.
- `max_position` (int, optional): Maximum position for positional encoding. Defaults to 10000.
- `temperature` (float, optional): Temperature parameter. Defaults to 1.0.
- `memory_threshold_mb` (float, optional): Memory threshold in MB. Defaults to 1000.0.
- `progress_callback` (Optional[callable], optional): Progress callback function
- `auto_adjust_batch_size` (bool, optional): Whether to auto-adjust batch size. Defaults to True.
- `min_batch_size` (int, optional): Minimum batch size. Defaults to 100.
- `max_batch_size` (int, optional): Maximum batch size. Defaults to 10000.

**Returns:**
- `SinusoidalEmbedder`: Fitted embedder instance

**Example:**
```python
tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=256)

def progress_callback(processed, total):
    print(f"Processed {processed}/{total} items")

embedder = tokenizer.fit_streaming(
    data_source='large_dataset.txt',
    batch_size=2000,
    memory_threshold_mb=500.0,
    progress_callback=progress_callback,
    auto_adjust_batch_size=True
)
```

### TokenizerConfig

Configuration dataclass for tokenizer adapters.

```python
@dataclass
class TokenizerConfig:
    backend: str
    model_name: str
    max_length: int = 512
    special_tokens: Optional[Dict[str, str]] = None
    backend_specific_config: Optional[Dict[str, Any]] = None
```

**Attributes:**
- `backend` (str): Backend name (e.g., 'huggingface', 'openai', 'spacy')
- `model_name` (str): Model name or identifier
- `max_length` (int, optional): Maximum sequence length. Defaults to 512.
- `special_tokens` (Optional[Dict[str, str]], optional): Special token mapping
- `backend_specific_config` (Optional[Dict[str, Any]], optional): Backend-specific options

### TokenizerRegistry

Registry system for automatic tokenizer backend detection and loading.

```python
class TokenizerRegistry:
    @classmethod
    def register_adapter(cls, backend_name: str, adapter_class: Type[TokenizerAdapter],
                        model_patterns: Optional[List[str]] = None) -> None
    
    @classmethod
    def create_adapter(cls, backend_or_model: str, max_length: int = 512,
                      special_tokens: Optional[Dict[str, str]] = None,
                      backend_specific_config: Optional[Dict[str, Any]] = None) -> TokenizerAdapter
    
    @classmethod
    def list_available_backends(cls) -> List[str]
```

**Key Methods:**

#### register_adapter()
Register a new tokenizer adapter with the registry.

**Parameters:**
- `backend_name` (str): Name of the backend
- `adapter_class` (Type[TokenizerAdapter]): Adapter class to register
- `model_patterns` (Optional[List[str]], optional): Model name patterns

#### create_adapter()
Create a tokenizer adapter instance.

**Parameters:**
- `backend_or_model` (str): Backend name or model name
- `max_length` (int, optional): Maximum sequence length. Defaults to 512.
- `special_tokens` (Optional[Dict[str, str]], optional): Special token configuration
- `backend_specific_config` (Optional[Dict[str, Any]], optional): Backend-specific config

**Returns:**
- `TokenizerAdapter`: Initialized adapter instance

## Tokenizer Adapters

### TokenizerAdapter (Abstract Base Class)

Abstract base class defining the interface for tokenizer adapters.

```python
class TokenizerAdapter(ABC):
    def __init__(self, config: TokenizerConfig)
    
    @abstractmethod
    def initialize(self) -> None
    
    @abstractmethod
    def tokenize(self, texts: Union[str, List[str]], 
                 add_special_tokens: bool = True,
                 padding: bool = True, 
                 truncation: bool = True) -> List[List[int]]
    
    @abstractmethod
    def decode(self, token_ids: Union[List[int], List[List[int]]], 
               skip_special_tokens: bool = True) -> Union[str, List[str]]
    
    @abstractmethod
    def get_vocab_size(self) -> int
    
    @abstractmethod
    def get_vocab(self) -> Dict[str, int]
    
    @abstractmethod
    def get_special_tokens(self) -> Dict[str, int]
```

### HuggingFaceAdapter

Adapter for HuggingFace transformers tokenizers.

```python
class HuggingFaceAdapter(TokenizerAdapter):
    SUPPORTED_MODELS = {
        'gpt2': 'gpt2',
        'bert-base-uncased': 'bert-base-uncased',
        'roberta-base': 'roberta-base',
        # ... more models
    }
```

**Supported Models:**
- GPT-2 variants: `gpt2`, `gpt2-medium`, `gpt2-large`, `gpt2-xl`
- BERT variants: `bert-base-uncased`, `bert-base-cased`, `bert-large-uncased`, `bert-large-cased`
- RoBERTa variants: `roberta-base`, `roberta-large`
- DistilBERT variants: `distilbert-base-uncased`, `distilbert-base-cased`
- T5 variants: `t5-small`, `t5-base`, `t5-large`
- ALBERT variants: `albert-base-v2`, `albert-large-v2`

**Example:**
```python
config = TokenizerConfig(
    backend='huggingface',
    model_name='bert-base-uncased',
    max_length=256,
    backend_specific_config={'do_lower_case': True}
)
adapter = HuggingFaceAdapter(config)
adapter.initialize()
```

### TiktokenAdapter

Adapter for OpenAI's tiktoken tokenizers.

```python
class TiktokenAdapter(TokenizerAdapter):
    SUPPORTED_MODELS = {
        'gpt-3.5-turbo': 'cl100k_base',
        'gpt-4': 'cl100k_base',
        'text-davinci-003': 'p50k_base',
        # ... more models
    }
```

**Supported Models:**
- GPT-3.5 models: `gpt-3.5-turbo`, `gpt-3.5-turbo-16k`, `gpt-3.5-turbo-instruct`
- GPT-4 models: `gpt-4`, `gpt-4-32k`, `gpt-4-turbo-preview`
- Text models: `text-davinci-003`, `text-davinci-002`, `text-curie-001`
- Code models: `code-davinci-002`, `code-cushman-002`
- Direct encodings: `cl100k_base`, `p50k_base`, `r50k_base`, `gpt2`

**Example:**
```python
config = TokenizerConfig(
    backend='openai',
    model_name='gpt-4',
    max_length=8192
)
adapter = TiktokenAdapter(config)
adapter.initialize()
```

### SpacyAdapter

Adapter for spaCy tokenizers with linguistic features.

```python
class SpacyAdapter(TokenizerAdapter):
    SUPPORTED_MODELS = {
        'en_core_web_sm': 'en',
        'de_core_news_sm': 'de',
        'fr_core_news_sm': 'fr',
        # ... more models
    }
```

**Supported Models:**
- English: `en_core_web_sm`, `en_core_web_md`, `en_core_web_lg`, `en_core_web_trf`
- German: `de_core_news_sm`, `de_core_news_md`, `de_core_news_lg`
- French: `fr_core_news_sm`, `fr_core_news_md`, `fr_core_news_lg`
- Spanish: `es_core_news_sm`, `es_core_news_md`, `es_core_news_lg`
- And many more languages...

**Example:**
```python
config = TokenizerConfig(
    backend='spacy',
    model_name='en_core_web_sm',
    backend_specific_config={
        'disable_components': ['ner', 'parser'],
        'normalize_unicode': True
    }
)
adapter = SpacyAdapter(config)
adapter.initialize()
```

## Sinusoidal Embeddings

### SinusoidalConfig

Configuration for sinusoidal embeddings.

```python
@dataclass
class SinusoidalConfig:
    # Core embedding parameters
    embedding_dim: int = 128
    vocab_size: int = 10000
    max_sequence_length: int = 512
    
    # Frequency parameters
    base_frequency: float = 10000.0
    frequency_scaling: float = 1.0
    learnable_frequencies: bool = True
    
    # Positional encoding options
    use_absolute_position: bool = True
    use_relative_position: bool = False
    relative_position_window: int = 64
    
    # Advanced configuration
    frequency_init_std: float = 0.02
    phase_shift: float = 0.0
    temperature: float = 1.0
    
    # Performance options
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = False
    enable_gpu_acceleration: bool = True
    use_vectorized_operations: bool = True
    enable_xla_compilation: bool = True
```

### ConfigurableSinusoidalEmbedder

Main sinusoidal embedding layer with configurable parameters.

```python
class ConfigurableSinusoidalEmbedder(keras.layers.Layer):
    def __init__(self, config: SinusoidalConfig, **kwargs)
    
    def call(self, inputs, training=None, mask=None)
    
    def adapt_to_vocabulary(self, vocab_size: int)
    
    def adapt_to_tokenizer(self, tokenizer_adapter)
    
    def adapt_embedding_dimension(self, new_embedding_dim: int, preserve_properties: bool = True)
```

**Key Methods:**

#### adapt_to_vocabulary()
```python
def adapt_to_vocabulary(self, vocab_size: int)
```

Adapt the embedder to a new vocabulary size.

**Parameters:**
- `vocab_size` (int): New vocabulary size to adapt to

**Example:**
```python
config = SinusoidalConfig(embedding_dim=256, vocab_size=50000)
embedder = ConfigurableSinusoidalEmbedder(config)
embedder.adapt_to_vocabulary(30000)  # Adapt to smaller vocabulary
```

#### adapt_to_tokenizer()
```python
def adapt_to_tokenizer(self, tokenizer_adapter)
```

Automatically adapt the embedder to a tokenizer's vocabulary size.

**Parameters:**
- `tokenizer_adapter` (TokenizerAdapter): Tokenizer adapter instance

**Example:**
```python
tokenizer = EnhancedTokenizerWrapper('gpt2')
embedder = ConfigurableSinusoidalEmbedder(config)
embedder.adapt_to_tokenizer(tokenizer.get_adapter())
```

#### adapt_embedding_dimension()
```python
def adapt_embedding_dimension(self, new_embedding_dim: int, preserve_properties: bool = True)
```

Adapt the embedder to a new embedding dimension.

**Parameters:**
- `new_embedding_dim` (int): New embedding dimension
- `preserve_properties` (bool, optional): Whether to preserve mathematical properties. Defaults to True.

## Streaming Data Processing

### StreamingDataIterator

Memory-efficient data iterator for processing large datasets.

```python
class StreamingDataIterator:
    def __init__(self, 
                 data_source: Union[str, List[str]], 
                 batch_size: int = 1000,
                 memory_threshold_mb: float = 1000.0,
                 auto_adjust_batch_size: bool = False,
                 progress_callback: Optional[Callable] = None,
                 text_field: Optional[str] = None,
                 min_batch_size: int = 10,
                 max_batch_size: int = 50000,
                 memory_check_interval: int = 10,
                 emergency_threshold_mb: Optional[float] = None,
                 extract_text: bool = False,
                 **kwargs)
```

**Parameters:**
- `data_source` (Union[str, List[str]]): Path to file/directory or list of file paths
- `batch_size` (int, optional): Number of items per batch. Defaults to 1000.
- `memory_threshold_mb` (float, optional): Memory threshold for adjustment. Defaults to 1000.0.
- `auto_adjust_batch_size` (bool, optional): Whether to auto-adjust batch size. Defaults to False.
- `progress_callback` (Optional[Callable], optional): Progress callback function
- `text_field` (Optional[str], optional): Field name for text data in structured formats
- `min_batch_size` (int, optional): Minimum allowed batch size. Defaults to 10.
- `max_batch_size` (int, optional): Maximum allowed batch size. Defaults to 50000.
- `memory_check_interval` (int, optional): Batches between memory checks. Defaults to 10.
- `emergency_threshold_mb` (Optional[float], optional): Emergency memory threshold
- `extract_text` (bool, optional): Whether to extract text from data items. Defaults to False.

**Key Methods:**

#### __iter__() and __next__()
Standard iterator protocol for batch processing.

**Example:**
```python
iterator = StreamingDataIterator(
    data_source='large_dataset.jsonl',
    batch_size=2000,
    auto_adjust_batch_size=True,
    extract_text=True,
    text_field='content'
)

for batch in iterator:
    # Process batch of text data
    process_batch(batch)
```

#### get_adaptive_stats()
```python
def get_adaptive_stats(self) -> Dict[str, Any]
```

Get detailed statistics about adaptive batch size management.

**Returns:**
- `Dict[str, Any]`: Statistics including memory history, adjustments, and current state

#### configure_adaptive_settings()
```python
def configure_adaptive_settings(self, 
                              memory_threshold_mb: Optional[float] = None,
                              emergency_threshold_mb: Optional[float] = None,
                              min_batch_size: Optional[int] = None,
                              max_batch_size: Optional[int] = None,
                              memory_check_interval: Optional[int] = None,
                              auto_adjust_batch_size: Optional[bool] = None)
```

Update adaptive settings during runtime.

## Performance Optimization

### CacheConfig

Configuration for intelligent caching system.

```python
@dataclass
class CacheConfig:
    max_cache_size: int = 10000
    cache_hit_threshold: float = 0.7
    enable_batch_caching: bool = True
    batch_cache_size: int = 5000
    enable_cache_warming: bool = True
    warmup_strategy: str = "frequency"
    warmup_size: int = 1000
    enable_metrics: bool = True
```

### GPUConfig

Configuration for GPU acceleration.

```python
@dataclass
class GPUConfig:
    enable_gpu: bool = True
    gpu_device: Optional[str] = None
    allow_memory_growth: bool = True
    memory_limit: Optional[int] = None
    enable_mixed_precision: bool = True
    mixed_precision_policy: str = "mixed_float16"
    enable_vectorization: bool = True
    enable_xla: bool = True
```

### MemoryStorageConfig

Configuration for memory-efficient storage.

```python
@dataclass
class MemoryStorageConfig:
    use_memory_mapping: bool = True
    memory_map_threshold: int = 100000
    use_compression: bool = True
    compression_level: int = 6
    use_gradient_checkpointing: bool = False
    enable_embedding_cache: bool = True
    cache_size: int = 10000
```

## Usage Examples

### Basic Tokenization

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper

# Simple usage with GPT-2
tokenizer = EnhancedTokenizerWrapper('gpt2')
tokens = tokenizer.tokenize(['Hello world', 'How are you?'])
text = tokenizer.decode(tokens[0])
print(f"Tokens: {tokens}")
print(f"Decoded: {text}")
```

### Advanced Sinusoidal Embeddings

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from lsm.data.configurable_sinusoidal_embedder import SinusoidalConfig

# Create tokenizer with custom embedding configuration
tokenizer = EnhancedTokenizerWrapper(
    'bert-base-uncased',
    embedding_dim=512,
    max_length=256
)

# Create advanced sinusoidal embedder
embedder = tokenizer.create_configurable_sinusoidal_embedder(
    learnable_frequencies=True,
    base_frequency=5000.0,
    use_relative_position=True,
    frequency_scaling=1.5
)

# Use in a Keras model
import tensorflow as tf

inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
embeddings = embedder(inputs)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(embeddings)
model = tf.keras.Model(inputs, outputs)
```

### Streaming Data Processing

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from lsm.data.streaming_data_iterator import StreamingDataIterator

# Setup tokenizer for streaming
tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=256)

# Define progress callback
def progress_callback(processed, total):
    percent = (processed / total) * 100
    print(f"Progress: {processed}/{total} ({percent:.1f}%)")

# Fit embedder on streaming data
embedder = tokenizer.fit_streaming(
    data_source=['data1.txt', 'data2.jsonl', 'data3.csv'],
    batch_size=2000,
    memory_threshold_mb=500.0,
    auto_adjust_batch_size=True,
    progress_callback=progress_callback,
    epochs=50
)

# Use the fitted embedder
tokens = tokenizer.tokenize(['Test sentence'])
embeddings = embedder.embed(tokens)
```

### Memory-Efficient Large Vocabulary

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from lsm.data.configurable_sinusoidal_embedder import SinusoidalConfig
from lsm.data.memory_efficient_storage import MemoryStorageConfig

# Configure memory-efficient storage
memory_config = MemoryStorageConfig(
    use_memory_mapping=True,
    use_compression=True,
    compression_level=6,
    use_gradient_checkpointing=True
)

# Configure sinusoidal embedder for large vocabulary
sinusoidal_config = SinusoidalConfig(
    embedding_dim=1024,
    vocab_size=100000,
    learnable_frequencies=True,
    use_memory_efficient_storage=True,
    memory_storage_config=memory_config,
    enable_gpu_acceleration=True
)

# Create tokenizer and embedder
tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=1024)
embedder = ConfigurableSinusoidalEmbedder(sinusoidal_config)
embedder.adapt_to_tokenizer(tokenizer.get_adapter())
```

### Multi-Backend Comparison

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper

# Compare different tokenizer backends
backends = ['gpt2', 'bert-base-uncased', 'en_core_web_sm']
text = "This is a sample sentence for tokenization comparison."

for backend in backends:
    try:
        tokenizer = EnhancedTokenizerWrapper(backend)
        tokens = tokenizer.tokenize([text])
        vocab_size = tokenizer.get_vocab_size()
        
        print(f"\nBackend: {backend}")
        print(f"Vocab size: {vocab_size}")
        print(f"Tokens: {tokens[0][:10]}...")  # First 10 tokens
        print(f"Token count: {len(tokens[0])}")
        
    except Exception as e:
        print(f"Backend {backend} failed: {e}")
```

## Error Handling

The enhanced tokenizer system provides comprehensive error handling with specific exception types:

### Exception Types

- `TokenizerError`: General tokenizer-related errors
- `TokenizerNotFittedError`: Attempting to use unfitted tokenizer
- `TokenizerLoadError`: Errors loading tokenizer models or configurations
- `TokenizerSaveError`: Errors saving tokenizer state
- `InvalidInputError`: Invalid input parameters or data
- `DataLoadError`: Errors loading or processing data files

### Error Handling Examples

```python
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
from lsm.utils.lsm_exceptions import TokenizerError, InvalidInputError

try:
    # This might fail if the model is not available
    tokenizer = EnhancedTokenizerWrapper('nonexistent-model')
    
except TokenizerError as e:
    print(f"Tokenizer error: {e}")
    # Fallback to a known working model
    tokenizer = EnhancedTokenizerWrapper('gpt2')

try:
    # This might fail with invalid parameters
    embedder = tokenizer.create_configurable_sinusoidal_embedder(
        embedding_dim=-1  # Invalid dimension
    )
    
except InvalidInputError as e:
    print(f"Invalid input: {e}")
    # Use valid parameters
    embedder = tokenizer.create_configurable_sinusoidal_embedder(
        embedding_dim=256
    )
```

## Best Practices

### 1. Tokenizer Selection

Choose the appropriate tokenizer based on your use case:

- **GPT-2/GPT-4**: For generative tasks and modern language modeling
- **BERT**: For classification and understanding tasks
- **spaCy**: For linguistic analysis and multilingual support
- **Custom**: For domain-specific vocabularies

### 2. Memory Management

For large datasets and vocabularies:

```python
# Enable memory-efficient features
tokenizer = EnhancedTokenizerWrapper(
    'your-model',
    enable_caching=True,
    cache_config=CacheConfig(
        max_cache_size=20000,
        enable_batch_caching=True,
        enable_cache_warming=True
    )
)

# Use streaming for large datasets
embedder = tokenizer.fit_streaming(
    data_source='large_dataset.txt',
    batch_size=1000,
    auto_adjust_batch_size=True,
    memory_threshold_mb=500.0
)
```

### 3. Performance Optimization

Enable GPU acceleration and mixed precision:

```python
from lsm.data.configurable_sinusoidal_embedder import SinusoidalConfig
from lsm.data.gpu_acceleration import GPUConfig

gpu_config = GPUConfig(
    enable_gpu=True,
    enable_mixed_precision=True,
    enable_vectorization=True,
    enable_xla=True
)

config = SinusoidalConfig(
    embedding_dim=512,
    learnable_frequencies=True,
    enable_gpu_acceleration=True,
    gpu_config=gpu_config,
    use_mixed_precision=True
)

embedder = ConfigurableSinusoidalEmbedder(config)
```

### 4. Monitoring and Debugging

Use progress callbacks and metrics:

```python
def detailed_progress_callback(processed, total):
    percent = (processed / total) * 100
    print(f"Processed: {processed:,} / {total:,} ({percent:.2f}%)")

# Enable detailed monitoring
iterator = StreamingDataIterator(
    data_source='dataset.txt',
    progress_callback=detailed_progress_callback,
    auto_adjust_batch_size=True
)

# Check adaptive statistics
for batch in iterator:
    process_batch(batch)
    
    # Periodically check stats
    if iterator._batch_count % 100 == 0:
        stats = iterator.get_adaptive_stats()
        print(f"Current batch size: {stats['current_config']['batch_size']}")
        print(f"Memory usage: {stats['memory_history'][-1]['memory_usage']:.1f}MB")
```

### 5. Model Integration

Integrate with existing LSM convenience API:

```python
from lsm.convenience import LSMGenerator

# Create enhanced tokenizer
tokenizer = EnhancedTokenizerWrapper('gpt2', embedding_dim=256)
embedder = tokenizer.create_configurable_sinusoidal_embedder()

# Use with LSMGenerator
generator = LSMGenerator(
    tokenizer=tokenizer,
    embedder=embedder,
    # ... other parameters
)

# Train and use as normal
generator.fit(training_data)
predictions = generator.predict(test_data)
```

This comprehensive API documentation covers all the key components, methods, and usage patterns for the enhanced tokenizer system. For additional examples and advanced usage patterns, refer to the example files in the `examples/` directory.