# LSM Convenience API Troubleshooting Guide

This guide helps you diagnose and fix common issues when using the LSM Convenience API. Issues are organized by category with clear symptoms, causes, and solutions.

## Quick Diagnostic Checklist

Before diving into specific issues, run this quick diagnostic:

```python
# 1. Check installation
try:
    from lsm import LSMGenerator, LSMClassifier, LSMRegressor
    print("✅ Convenience API imports working")
except ImportError as e:
    print(f"❌ Import error: {e}")

# 2. Check basic functionality
try:
    generator = LSMGenerator(preset='fast')
    print("✅ Generator creation working")
except Exception as e:
    print(f"❌ Generator creation failed: {e}")

# 3. Check training components
try:
    generator.fit(["Hello", "Hi there"], epochs=1)
    print("✅ Basic training working")
except Exception as e:
    print(f"❌ Training failed: {e}")

# 4. Check generation
try:
    response = generator.generate("Test")
    print(f"✅ Generation working: {response}")
except Exception as e:
    print(f"❌ Generation failed: {e}")
```

## Installation and Import Issues

### Issue: ImportError when importing convenience classes

**Symptoms:**
```python
from lsm import LSMGenerator
# ImportError: No module named 'lsm'
```

**Causes:**
1. Package not installed correctly
2. Python path doesn't include src/ directory
3. Missing dependencies

**Solutions:**

**Solution 1: Fix Python path**
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from lsm import LSMGenerator
```

**Solution 2: Install in development mode**
```bash
pip install -e .
```

**Solution 3: Check dependencies**
```bash
pip install -r requirements.txt
```

### Issue: Enhanced tokenizer dependencies missing

**Symptoms:**
```python
generator = LSMGenerator(tokenizer='bert-base-uncased')
# ImportError: No module named 'transformers'
```

**Causes:**
1. Missing tokenizer-specific dependencies
2. Outdated package versions
3. Conflicting package versions

**Solutions:**

**Solution 1: Install tokenizer dependencies**
```bash
# For HuggingFace tokenizers
pip install transformers>=4.20.0

# For OpenAI tokenizers
pip install tiktoken>=0.4.0

# For spaCy tokenizers
pip install spacy>=3.4.0
python -m spacy download en_core_web_sm
```

**Solution 2: Install all tokenizer dependencies**
```bash
pip install transformers tiktoken spacy
python -m spacy download en_core_web_sm de_core_news_sm fr_core_news_sm
```

**Solution 3: Check installed versions**
```python
import transformers, tiktoken, spacy
print(f"Transformers: {transformers.__version__}")
print(f"Tiktoken: {tiktoken.__version__}")
print(f"spaCy: {spacy.__version__}")
```

### Issue: ConvenienceValidationError on import

**Symptoms:**
```python
from lsm.convenience import LSMGenerator
# ConvenienceValidationError: Training components not available
```

**Causes:**
1. Missing TensorFlow or PyTorch
2. Incomplete installation
3. Version conflicts

**Solutions:**

**Solution 1: Install missing dependencies**
```bash
pip install tensorflow>=2.8.0
# or
pip install torch>=1.12.0
```

**Solution 2: Check component availability**
```python
from lsm.convenience.generator import _check_training_components
available = _check_training_components()
print(f"Training components available: {available}")
```

## Configuration and Parameter Issues

### Issue: Invalid parameter values

**Symptoms:**
```python
generator = LSMGenerator(window_size=-1)
# ConvenienceValidationError: window_size must be positive, got -1
```

**Causes:**
1. Invalid parameter ranges
2. Incompatible parameter combinations
3. Typos in parameter names

**Solutions:**

**Solution 1: Check parameter documentation**
```python
from lsm.convenience import ConvenienceConfig
help(ConvenienceConfig.validate_params)
```

**Solution 2: Use presets for safe defaults**
```python
generator = LSMGenerator(preset='balanced')
# Then customize specific parameters
generator.set_params(window_size=15)
```

**Solution 3: Validate parameters before use**
```python
from lsm.convenience import ConvenienceValidationError

try:
    params = {'window_size': 10, 'embedding_dim': 128}
    ConvenienceConfig.validate_params(params)
    generator = LSMGenerator(**params)
except ConvenienceValidationError as e:
    print(f"Invalid parameters: {e}")
    print(f"Suggestion: {e.suggestion}")
```

### Issue: Unsupported tokenizer backend

**Symptoms:**
```python
generator = LSMGenerator(tokenizer='unsupported-model')
# TokenizerError: Unsupported tokenizer backend 'unsupported-model'
```

**Causes:**
1. Typo in tokenizer name
2. Model not available in the backend
3. Missing model dependencies

**Solutions:**

**Solution 1: Check supported tokenizers**
```python
from lsm.data.enhanced_tokenization import TokenizerRegistry
backends = TokenizerRegistry.list_available_backends()
print("Supported backends:", backends)

# Check specific models
from lsm.data.adapters.huggingface_adapter import HuggingFaceAdapter
print("HuggingFace models:", list(HuggingFaceAdapter.SUPPORTED_MODELS.keys()))
```

**Solution 2: Use correct tokenizer names**
```python
# Correct HuggingFace model names
generator = LSMGenerator(tokenizer='gpt2')
generator = LSMGenerator(tokenizer='bert-base-uncased')
generator = LSMGenerator(tokenizer='roberta-base')

# Correct OpenAI model names
generator = LSMGenerator(tokenizer='gpt-4')
generator = LSMGenerator(tokenizer='gpt-3.5-turbo')

# Correct spaCy model names
generator = LSMGenerator(tokenizer='en_core_web_sm')
```

**Solution 3: Install missing models**
```bash
# For spaCy models
python -m spacy download en_core_web_sm

# For HuggingFace models (downloaded automatically)
# Just ensure transformers is installed
pip install transformers
```

### Issue: Incompatible embedding and tokenizer configuration

**Symptoms:**
```python
generator = LSMGenerator(
    tokenizer='gpt2',
    embedding_type='configurable_sinusoidal',
    embedding_dim=1000000  # Too large
)
# MemoryError: Cannot allocate memory for embedding matrix
```

**Causes:**
1. Embedding dimension too large for vocabulary
2. Incompatible tokenizer and embedding settings
3. Insufficient memory for configuration

**Solutions:**

**Solution 1: Use appropriate embedding dimensions**
```python
# Check tokenizer vocabulary size first
from lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper
tokenizer = EnhancedTokenizerWrapper('gpt2')
vocab_size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {vocab_size}")

# Use reasonable embedding dimension
generator = LSMGenerator(
    tokenizer='gpt2',
    embedding_type='configurable_sinusoidal',
    embedding_dim=min(512, vocab_size // 10)  # Reasonable size
)
```

**Solution 2: Enable memory-efficient storage**
```python
generator = LSMGenerator(
    tokenizer='gpt2',
    embedding_type='configurable_sinusoidal',
    embedding_dim=512,
    embedding_config={
        'use_memory_efficient_storage': True,
        'use_compression': True
    }
)
```

### Issue: Preset not found

**Symptoms:**
```python
generator = LSMGenerator(preset='ultra_fast')
# ConvenienceValidationError: Unknown preset 'ultra_fast'
```

**Causes:**
1. Typo in preset name
2. Using non-existent preset

**Solutions:**

**Solution 1: List available presets**
```python
from lsm.convenience import ConvenienceConfig
presets = ConvenienceConfig.list_presets()
print("Available presets:")
for name, description in presets.items():
    print(f"  {name}: {description}")
```

**Solution 2: Use correct preset names**
```python
# Correct preset names
generator = LSMGenerator(preset='fast')     # For experimentation
generator = LSMGenerator(preset='balanced') # For general use
generator = LSMGenerator(preset='quality')  # For production
```

## Enhanced Tokenizer and Streaming Issues

### Issue: Streaming data processing fails

**Symptoms:**
```python
generator = LSMGenerator(streaming=True)
generator.fit('large_dataset.txt')
# MemoryError: Out of memory during streaming
```

**Causes:**
1. Batch size too large for available memory
2. Memory threshold set too high
3. Data format causing memory spikes
4. Insufficient disk space for temporary files

**Solutions:**

**Solution 1: Configure streaming parameters**
```python
generator = LSMGenerator(
    streaming=True,
    embedding_type='sinusoidal'
)
generator.fit(
    'large_dataset.txt',
    streaming_config={
        'batch_size': 500,  # Smaller initial batch size
        'memory_threshold_mb': 200.0,  # Lower memory threshold
        'auto_adjust_batch_size': True,
        'min_batch_size': 50,
        'max_batch_size': 2000
    }
)
```

**Solution 2: Monitor memory usage**
```python
def memory_callback(processed, total):
    import psutil
    memory_percent = psutil.virtual_memory().percent
    print(f"Progress: {processed}/{total}, Memory: {memory_percent:.1f}%")

generator.fit(
    'large_dataset.txt',
    streaming_config={
        'progress_callback': memory_callback,
        'auto_adjust_batch_size': True
    }
)
```

**Solution 3: Use memory-efficient data formats**
```python
# For JSON Lines format
generator.fit(
    'data.jsonl',
    streaming_config={
        'text_field': 'content',  # Specify text field
        'extract_text': True      # Extract text automatically
    }
)
```

### Issue: Tokenizer loading is slow

**Symptoms:**
- First tokenization takes a long time
- Model loading seems to hang
- High disk I/O during initialization

**Causes:**
1. Large tokenizer models being downloaded
2. No caching enabled
3. Network issues during model download
4. Insufficient disk space

**Solutions:**

**Solution 1: Enable caching and preloading**
```python
generator = LSMGenerator(
    tokenizer='bert-base-uncased',
    tokenizer_config={
        'enable_caching': True,
        'cache_config': {
            'enable_cache_warming': True,
            'warmup_strategy': 'frequency',
            'warmup_size': 1000
        }
    }
)
```

**Solution 2: Use local model cache**
```python
import os
# Set HuggingFace cache directory
os.environ['TRANSFORMERS_CACHE'] = '/path/to/local/cache'

generator = LSMGenerator(tokenizer='bert-base-uncased')
```

**Solution 3: Pre-download models**
```python
# Pre-download models
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('bert-base-uncased')

# Then use in generator
generator = LSMGenerator(tokenizer='bert-base-uncased')
```

### Issue: Sinusoidal embedding training is unstable

**Symptoms:**
```python
generator = LSMGenerator(embedding_type='configurable_sinusoidal')
generator.fit(data)
# Training loss oscillates or doesn't converge
```

**Causes:**
1. Learning rate too high for frequency parameters
2. Incompatible frequency initialization
3. Gradient scaling issues
4. Insufficient training data

**Solutions:**

**Solution 1: Adjust sinusoidal configuration**
```python
generator = LSMGenerator(
    embedding_type='configurable_sinusoidal',
    embedding_config={
        'learnable_frequencies': True,
        'frequency_init_std': 0.01,  # Smaller initialization
        'base_frequency': 10000.0,   # Stable base frequency
        'temperature': 0.5           # Lower temperature
    }
)
```

**Solution 2: Use mixed precision training**
```python
generator = LSMGenerator(
    embedding_type='configurable_sinusoidal',
    embedding_config={
        'use_mixed_precision': True,
        'gradient_checkpointing': True
    }
)
```

**Solution 3: Start with fixed frequencies**
```python
# Train with fixed frequencies first
generator = LSMGenerator(
    embedding_type='sinusoidal',  # Fixed frequencies
    embedding_dim=256
)
generator.fit(data, epochs=20)

# Then switch to learnable frequencies
generator = LSMGenerator(
    embedding_type='configurable_sinusoidal',
    embedding_config={'learnable_frequencies': True}
)
generator.fit(data, epochs=30)
```

### Issue: Multi-language tokenization problems

**Symptoms:**
```python
generator = LSMGenerator(tokenizer='en_core_web_sm')
generator.fit(['Hello', 'Bonjour', '你好'])  # Mixed languages
# TokenizerError: Unsupported characters in text
```

**Causes:**
1. Language-specific tokenizer used for multi-language text
2. Unicode normalization issues
3. Missing language models
4. Encoding problems

**Solutions:**

**Solution 1: Use multilingual tokenizers**
```python
# Use multilingual BERT
generator = LSMGenerator(tokenizer='bert-base-multilingual-cased')

# Or use GPT-2 which handles multiple languages
generator = LSMGenerator(tokenizer='gpt2')
```

**Solution 2: Configure Unicode handling**
```python
generator = LSMGenerator(
    tokenizer='en_core_web_sm',
    tokenizer_config={
        'backend_specific_config': {
            'normalize_unicode': True,
            'handle_mixed_languages': True
        }
    }
)
```

**Solution 3: Preprocess text for language consistency**
```python
from langdetect import detect

def preprocess_multilingual(texts):
    # Group by language or use language-specific processing
    processed = []
    for text in texts:
        try:
            lang = detect(text)
            # Apply language-specific preprocessing
            processed.append(text)
        except:
            processed.append(text)  # Fallback
    return processed

texts = preprocess_multilingual(['Hello', 'Bonjour', '你好'])
generator.fit(texts)
```

## Training Issues

### Issue: Training fails with empty data

**Symptoms:**
```python
generator.fit([])
# ConvenienceValidationError: Training data cannot be empty
```

**Causes:**
1. Empty training data
2. All data filtered out during preprocessing
3. Invalid data format

**Solutions:**

**Solution 1: Check data before training**
```python
conversations = load_your_data()
print(f"Data size: {len(conversations)}")
print(f"Sample data: {conversations[:2]}")

if len(conversations) == 0:
    print("❌ No training data available")
else:
    generator.fit(conversations)
```

**Solution 2: Use minimum viable data for testing**
```python
# Minimum test data
test_conversations = [
    "User: Hello\nAssistant: Hi there!",
    "User: How are you?\nAssistant: I'm doing well!",
    "User: Goodbye\nAssistant: See you later!"
]
generator.fit(test_conversations, epochs=5)
```

### Issue: Out of memory during training

**Symptoms:**
```python
generator.fit(large_dataset)
# tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM
```

**Causes:**
1. Dataset too large for available memory
2. Batch size too large
3. Model parameters too large

**Solutions:**

**Solution 1: Enable automatic memory management**
```python
generator = LSMGenerator(
    auto_memory_management=True,
    preset='fast'  # Uses smaller model
)
```

**Solution 2: Reduce batch size**
```python
generator.fit(
    data,
    batch_size=8,  # Smaller batches
    gradient_accumulation_steps=4  # Maintain effective batch size
)
```

**Solution 3: Use smaller model configuration**
```python
generator = LSMGenerator(
    window_size=5,      # Smaller window
    embedding_dim=64,   # Smaller embeddings
    reservoir_config={'n_reservoir': 200}  # Smaller reservoir
)
```

**Solution 4: Use streaming data processing**
```python
# For very large datasets, use streaming
generator = LSMGenerator(
    streaming=True,
    embedding_type='sinusoidal',
    tokenizer_config={'enable_caching': True}
)

generator.fit(
    'large_dataset.txt',  # File path for streaming
    streaming_config={
        'batch_size': 1000,
        'auto_adjust_batch_size': True,
        'memory_threshold_mb': 500.0
    },
    epochs=50
)
```

**Solution 5: Enable memory-efficient embeddings**
```python
generator = LSMGenerator(
    embedding_type='configurable_sinusoidal',
    embedding_config={
        'use_memory_efficient_storage': True,
        'use_compression': True,
        'gradient_checkpointing': True
    }
)
```

### Issue: Training is very slow

**Symptoms:**
- Training takes hours for small datasets
- Progress bars move very slowly
- High CPU/GPU usage but slow progress

**Causes:**
1. Using 'quality' preset for experimentation
2. Too many epochs
3. Large batch sizes on limited hardware
4. Inefficient data preprocessing

**Solutions:**

**Solution 1: Use appropriate preset for your use case**
```python
# For experimentation
generator = LSMGenerator(preset='fast')
generator.fit(data, epochs=10)

# For production (after experimentation)
generator = LSMGenerator(preset='quality')
generator.fit(data, epochs=100)
```

**Solution 2: Optimize training parameters**
```python
generator.fit(
    data,
    epochs=20,           # Fewer epochs initially
    batch_size=16,       # Smaller batches
    validation_split=0.1 # Less validation data
)
```

**Solution 3: Enable performance monitoring**
```python
from lsm.convenience import monitor_performance

with monitor_performance():
    generator.fit(data)
# Automatically logs performance metrics
```

### Issue: Poor training results

**Symptoms:**
- Generated text is nonsensical
- Classification accuracy is very low
- Model seems to not learn anything

**Causes:**
1. Insufficient training data
2. Poor data quality
3. Inappropriate model configuration
4. Too few training epochs

**Solutions:**

**Solution 1: Check data quality**
```python
from lsm.convenience.utils import get_conversation_statistics

stats = get_conversation_statistics(conversations)
print(f"Data statistics: {stats}")

# Look for:
# - Sufficient data volume (100+ conversations for generation)
# - Reasonable conversation lengths
# - Diverse vocabulary
```

**Solution 2: Increase training data and epochs**
```python
# Ensure sufficient training
generator = LSMGenerator(preset='balanced')
generator.fit(
    conversations,
    epochs=50,  # More epochs
    validation_split=0.2,
    verbose=True  # Monitor progress
)
```

**Solution 3: Try different model configurations**
```python
# For text generation
generator = LSMGenerator(
    reservoir_type='hierarchical',  # Better for text
    system_message_support=True,    # Better context handling
    response_level=True             # Response-level generation
)

# For classification
classifier = LSMClassifier(
    classifier_type='random_forest',  # Often better than logistic
    window_size=8                     # Appropriate for text
)
```

## Generation and Inference Issues

### Issue: Generated text is repetitive or nonsensical

**Symptoms:**
```python
response = generator.generate("Hello")
# Output: "the the the the the..."
```

**Causes:**
1. Insufficient training
2. Poor temperature setting
3. Model overfitting
4. Data quality issues

**Solutions:**

**Solution 1: Adjust generation parameters**
```python
response = generator.generate(
    "Hello",
    temperature=0.8,    # Add randomness (0.1-2.0)
    max_length=50,      # Limit length
    top_p=0.9          # Nucleus sampling (if supported)
)
```

**Solution 2: Improve training**
```python
# More diverse training data
generator.fit(
    diverse_conversations,
    epochs=100,
    validation_split=0.2
)
```

**Solution 3: Use system messages for better control**
```python
response = generator.generate(
    "Hello",
    system_message="You are a helpful assistant. Provide natural, varied responses.",
    temperature=0.7
)
```

### Issue: Generation is too slow

**Symptoms:**
- Single generation takes several seconds
- Batch generation is much slower than expected

**Causes:**
1. Large model configuration
2. Inefficient inference setup
3. No GPU acceleration

**Solutions:**

**Solution 1: Optimize for inference**
```python
# Use smaller model for faster inference
generator = LSMGenerator(
    preset='fast',
    inference_optimization=True  # If available
)
```

**Solution 2: Use batch generation for multiple prompts**
```python
# More efficient than individual calls
prompts = ["Hello", "How are you?", "Goodbye"]
responses = generator.batch_generate(prompts)
```

**Solution 3: Check hardware utilization**
```python
import tensorflow as tf
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Enable GPU memory growth if needed
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
```

## Model Persistence Issues

### Issue: Cannot save model

**Symptoms:**
```python
generator.save("my_model")
# PermissionError: [Errno 13] Permission denied
```

**Causes:**
1. Insufficient file permissions
2. Path doesn't exist
3. Disk space issues

**Solutions:**

**Solution 1: Check and create directory**
```python
import os
from pathlib import Path

model_path = Path("my_model")
model_path.mkdir(parents=True, exist_ok=True)
generator.save(model_path)
```

**Solution 2: Use absolute path**
```python
import os
model_path = os.path.abspath("my_model")
generator.save(model_path)
```

**Solution 3: Check disk space and permissions**
```bash
# Check disk space
df -h

# Check permissions
ls -la
```

### Issue: Cannot load saved model

**Symptoms:**
```python
generator = LSMGenerator.load("my_model")
# FileNotFoundError: Model directory not found
```

**Causes:**
1. Model path incorrect
2. Model files corrupted
3. Version incompatibility

**Solutions:**

**Solution 1: Verify model directory structure**
```python
import os
model_path = "my_model"
if os.path.exists(model_path):
    print("Model directory contents:")
    for root, dirs, files in os.walk(model_path):
        for file in files:
            print(os.path.join(root, file))
else:
    print(f"Model directory {model_path} does not exist")
```

**Solution 2: Check model integrity**
```python
from lsm.convenience.base import LSMBase

try:
    # Validate model before loading
    if LSMBase.validate_model_directory("my_model"):
        generator = LSMGenerator.load("my_model")
    else:
        print("Model directory is invalid")
except Exception as e:
    print(f"Model validation failed: {e}")
```

## Data Format Issues

### Issue: Unsupported data format

**Symptoms:**
```python
generator.fit(my_data)
# ConvenienceValidationError: Unsupported data format
```

**Causes:**
1. Data in unexpected format
2. Missing required fields
3. Encoding issues

**Solutions:**

**Solution 1: Check supported formats**
```python
from lsm.convenience.utils import detect_conversation_format

format_info = detect_conversation_format(my_data)
print(f"Detected format: {format_info}")
```

**Solution 2: Convert to supported format**
```python
# Convert to simple string list
if isinstance(my_data, dict):
    conversations = [str(item) for item in my_data.values()]
elif isinstance(my_data, pd.DataFrame):
    conversations = my_data['text'].tolist()
else:
    conversations = [str(item) for item in my_data]

generator.fit(conversations)
```

**Solution 3: Use structured format**
```python
# Convert to structured format
structured_data = []
for item in my_data:
    structured_data.append({
        "messages": [item['input'], item['output']],
        "system": item.get('system', None)
    })

generator.fit(structured_data)
```

## Performance Issues

### Issue: High memory usage

**Symptoms:**
- System runs out of RAM
- Swap usage increases dramatically
- Other applications become slow

**Causes:**
1. Large model parameters
2. Large batch sizes
3. Memory leaks
4. Inefficient data loading

**Solutions:**

**Solution 1: Enable automatic memory management**
```python
from lsm.convenience import LSMGenerator, manage_memory

with manage_memory():
    generator = LSMGenerator(auto_memory_management=True)
    generator.fit(data)
```

**Solution 2: Monitor memory usage**
```python
from lsm.convenience import MemoryMonitor

monitor = MemoryMonitor()
monitor.start()

generator = LSMGenerator()
generator.fit(data)

memory_stats = monitor.get_stats()
print(f"Peak memory usage: {memory_stats['peak_memory_mb']} MB")
```

**Solution 3: Use memory-efficient configurations**
```python
generator = LSMGenerator(
    preset='fast',              # Smaller model
    batch_size=8,              # Smaller batches
    gradient_accumulation_steps=4,  # Maintain effective batch size
    mixed_precision=True,       # If supported
    embedding_type='sinusoidal',
    embedding_config={
        'use_memory_efficient_storage': True,
        'use_compression': True
    },
    tokenizer_config={
        'enable_caching': True,
        'cache_config': {'max_cache_size': 5000}
    }
)
```

### Issue: Slow performance on CPU

**Symptoms:**
- Training/inference much slower than expected
- High CPU usage but slow progress
- No GPU utilization

**Causes:**
1. No GPU available or not configured
2. TensorFlow not using GPU
3. Inefficient CPU operations

**Solutions:**

**Solution 1: Check GPU availability**
```python
import tensorflow as tf
print("GPU devices:", tf.config.list_physical_devices('GPU'))

# Enable GPU if available
if tf.config.list_physical_devices('GPU'):
    print("GPU is available and will be used")
else:
    print("No GPU available, using CPU")
```

**Solution 2: Optimize for CPU**
```python
# CPU-optimized configuration
generator = LSMGenerator(
    preset='fast',
    batch_size=4,      # Smaller batches for CPU
    num_threads=4      # Match your CPU cores
)
```

**Solution 3: Use CPU-specific optimizations**
```python
import os
# Enable CPU optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'

generator = LSMGenerator()
```

## Integration Issues

### Issue: Sklearn compatibility problems

**Symptoms:**
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X, y)
# AttributeError: 'LSMClassifier' object has no attribute 'fit'
```

**Causes:**
1. Missing sklearn methods
2. Incorrect parameter passing
3. Data format incompatibility

**Solutions:**

**Solution 1: Verify sklearn compatibility**
```python
from sklearn.utils.estimator_checks import check_estimator
from lsm import LSMClassifier

try:
    check_estimator(LSMClassifier())
    print("✅ Sklearn compatibility verified")
except Exception as e:
    print(f"❌ Sklearn compatibility issue: {e}")
```

**Solution 2: Use proper sklearn integration**
```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from lsm import LSMClassifier

# Proper usage
classifier = LSMClassifier()
scores = cross_val_score(classifier, texts, labels, cv=3)

# Pipeline usage
pipeline = Pipeline([
    ('lsm', LSMClassifier())
])
pipeline.fit(texts, labels)
```

## Debugging Tips

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or specific to LSM
from lsm.utils.lsm_logging import get_logger
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
```

### Use Performance Profiling

```python
from lsm.convenience import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start()

# Your code here
generator = LSMGenerator()
generator.fit(data)

stats = profiler.get_stats()
print(f"Training time: {stats['training_time']:.2f}s")
print(f"Memory usage: {stats['peak_memory_mb']:.1f}MB")
```

### Check Component Status

```python
from lsm.convenience.generator import _check_training_components
from lsm.convenience.utils import check_system_resources

# Check if training components are available
training_available = _check_training_components()
print(f"Training components: {training_available}")

# Check system resources
resources = check_system_resources()
print(f"System resources: {resources}")
```

## Getting Help

### Before Asking for Help

1. **Run the diagnostic checklist** at the top of this guide
2. **Check the error message** for specific details
3. **Enable debug logging** to get more information
4. **Try with minimal example** to isolate the issue
5. **Check system resources** (memory, disk space, GPU)

### Information to Include

When reporting issues, include:

1. **Full error message** and stack trace
2. **Minimal code example** that reproduces the issue
3. **System information** (OS, Python version, GPU)
4. **Data information** (size, format, sample)
5. **Configuration used** (parameters, presets)

### Common Solutions Summary

| Issue Type | Quick Fix | Long-term Solution |
|------------|-----------|-------------------|
| Import errors | Check Python path | Proper installation |
| Memory issues | Smaller batch size | Auto memory management |
| Slow training | Use 'fast' preset | Optimize configuration |
| Poor results | More data/epochs | Better data quality |
| Save/load issues | Check permissions | Use absolute paths |
| Format issues | Convert data format | Use structured data |

## Enhanced Tokenizer Quick Reference

### Supported Tokenizer Backends

| Backend | Example Models | Installation |
|---------|----------------|--------------|
| HuggingFace | `gpt2`, `bert-base-uncased`, `roberta-base` | `pip install transformers` |
| OpenAI | `gpt-4`, `gpt-3.5-turbo`, `text-davinci-003` | `pip install tiktoken` |
| spaCy | `en_core_web_sm`, `de_core_news_sm`, `fr_core_news_sm` | `pip install spacy && python -m spacy download en_core_web_sm` |

### Common Enhanced Tokenizer Configurations

```python
# Basic sinusoidal embeddings
generator = LSMGenerator(
    tokenizer='gpt2',
    embedding_type='sinusoidal',
    embedding_dim=256
)

# Configurable sinusoidal with streaming
generator = LSMGenerator(
    tokenizer='bert-base-uncased',
    embedding_type='configurable_sinusoidal',
    streaming=True,
    sinusoidal_config={
        'learnable_frequencies': True,
        'base_frequency': 5000.0
    },
    streaming_config={
        'auto_adjust_batch_size': True,
        'memory_threshold_mb': 500.0
    }
)

# Performance optimized
generator = LSMGenerator(
    tokenizer='gpt2',
    embedding_type='sinusoidal',
    tokenizer_config={
        'enable_caching': True,
        'cache_config': {'enable_batch_caching': True}
    },
    embedding_config={
        'enable_gpu_acceleration': True,
        'use_mixed_precision': True
    }
)
```

### Troubleshooting Decision Tree

1. **Import/Installation Issues**
   - Missing dependencies → Install tokenizer libraries
   - Module not found → Check Python path and installation

2. **Configuration Issues**
   - Unsupported tokenizer → Check supported models list
   - Memory errors → Enable memory-efficient storage
   - Slow performance → Enable caching and GPU acceleration

3. **Training Issues**
   - Out of memory → Use streaming or smaller batch sizes
   - Slow training → Use performance optimizations
   - Poor convergence → Adjust sinusoidal configuration

4. **Streaming Issues**
   - Memory spikes → Lower memory threshold
   - Slow processing → Enable auto batch size adjustment
   - Data format errors → Specify text field and extraction options

### Resources

- **API Documentation**: `docs/CONVENIENCE_API_DOCUMENTATION.md`
- **Enhanced Tokenizer Documentation**: `docs/ENHANCED_TOKENIZER_API_DOCUMENTATION.md`
- **Getting Started**: `docs/GETTING_STARTED_TUTORIAL.md`
- **Migration Guide**: `MIGRATION_GUIDE.md` (includes enhanced tokenizer migration)
- **Examples**: `examples/` directory (includes enhanced tokenizer examples)
- **Advanced Troubleshooting**: `docs/TROUBLESHOOTING_GUIDE.md`

### Enhanced Tokenizer Examples

- `examples/enhanced_tokenizer_api_examples.py` - Basic usage patterns
- `examples/configurable_sinusoidal_embedder_demo.py` - Advanced embedding configuration
- `examples/streaming_tokenizer_fitting_demo.py` - Streaming data processing
- `examples/huggingface_adapter_demo.py` - HuggingFace tokenizer usage
- `examples/tiktoken_adapter_demo.py` - OpenAI tokenizer usage
- `examples/spacy_adapter_demo.py` - spaCy tokenizer usage

Remember: The enhanced tokenizer system is designed to be backward compatible and self-configuring. Most issues can be resolved by enabling automatic features (streaming, caching, memory management) or using appropriate presets. When in doubt, start with basic configuration and gradually add advanced features.