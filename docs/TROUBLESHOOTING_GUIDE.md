# LSM Inference System Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Sparse Sine-Activated Liquid State Machine (LSM) inference system.

## Table of Contents

1. [Model Loading Issues](#model-loading-issues)
2. [Inference Problems](#inference-problems)
3. [Performance Issues](#performance-issues)
4. [Memory Problems](#memory-problems)
5. [Input Validation Errors](#input-validation-errors)
6. [Compatibility Issues](#compatibility-issues)
7. [Installation Problems](#installation-problems)
8. [Debug Mode and Logging](#debug-mode-and-logging)
9. [Getting Help](#getting-help)

## Model Loading Issues

### Problem: ModelLoadError - Missing required files

**Symptoms:**
```
ModelLoadError: Missing required files: ['tokenizer/vectorizer.pkl', 'config.json']
```

**Causes:**
- Model was trained with an older version of the system
- Model directory is incomplete or corrupted
- Files were accidentally deleted

**Solutions:**

1. **Check model directory structure:**
   ```bash
   python manage_models.py validate --model-path ./your_model_directory
   ```

2. **Use ModelManager to identify missing components:**
   ```python
   from src.lsm.management.model_manager import ModelManager
   manager = ModelManager()
   is_valid, errors = manager.validate_model("./your_model_directory")
   print("Missing components:", errors)
   ```

3. **For old models, use legacy inference:**
   ```python
   from inference import LSMInference  # Legacy class
   inference = LSMInference("./old_model_directory")
   ```

4. **Retrain the model with the enhanced system:**
   ```bash
   python ../main.py train --window-size 10 --epochs 20
   ```

### Problem: TokenizerNotFittedError

**Symptoms:**
```
TokenizerNotFittedError: Tokenizer not fitted or vocabulary not loaded
```

**Causes:**
- Model was saved without tokenizer persistence
- Tokenizer files are corrupted
- Version mismatch between training and inference

**Solutions:**

1. **Check tokenizer directory:**
   ```bash
   ls -la ./your_model_directory/tokenizer/
   ```
   Should contain: `vectorizer.pkl`, `vocab_mapping.json`, `config.json`

2. **Validate tokenizer files:**
   ```python
   import pickle
   import json
   
   # Check if files can be loaded
   try:
       with open("./model/tokenizer/vectorizer.pkl", "rb") as f:
           vectorizer = pickle.load(f)
       print("Vectorizer loaded successfully")
       
       with open("./model/tokenizer/vocab_mapping.json", "r") as f:
           vocab = json.load(f)
       print("Vocabulary loaded successfully")
   except Exception as e:
       print(f"File loading error: {e}")
   ```

3. **Use backward compatibility mode:**
   ```python
   from inference import LSMInference
   inference = LSMInference("./model_directory")
   ```

### Problem: Configuration file errors

**Symptoms:**
```
JSONDecodeError: Expecting property name enclosed in double quotes
```

**Causes:**
- Corrupted configuration file
- Invalid JSON format
- Encoding issues

**Solutions:**

1. **Validate JSON format:**
   ```bash
   python -m json.tool ./your_model_directory/config.json
   ```

2. **Check file encoding:**
   ```python
   with open("./model/config.json", "r", encoding="utf-8") as f:
       content = f.read()
       print("File readable:", len(content) > 0)
   ```

3. **Recreate configuration from metadata:**
   ```python
   from model_config import ModelConfiguration
   
   # Create default configuration
   config = ModelConfiguration(
       window_size=10,
       embedding_dim=128,
       reservoir_type="standard",
       reservoir_config={},
       reservoir_units=[256, 128, 64],
       sparsity=0.1,
       use_multichannel=True
   )
   config.save("./model/config.json")
   ```

## Inference Problems

### Problem: Poor prediction quality

**Symptoms:**
- Nonsensical or repetitive predictions
- Low confidence scores consistently
- Predictions don't match expected patterns

**Diagnosis:**

1. **Check model performance metrics:**
   ```python
   from src.lsm.management.model_manager import ModelManager
   manager = ModelManager()
   info = manager.get_model_info("./your_model")
   metrics = info.get('metadata', {}).get('performance_metrics', {})
   print(f"Test MSE: {metrics.get('final_test_mse', 'N/A')}")
   print(f"Test MAE: {metrics.get('final_test_mae', 'N/A')}")
   ```

2. **Validate input format:**
   ```python
   from input_validation import validate_dialogue_sequence
   
   dialogue = ["Your", "input", "sequence"]
   is_valid, error = validate_dialogue_sequence(dialogue)
   if not is_valid:
       print(f"Input validation error: {error}")
   ```

3. **Check tokenizer vocabulary:**
   ```python
   inference = OptimizedLSMInference("./model")
   info = inference.get_model_info()
   tokenizer_config = info.get('configuration', {}).get('tokenizer_config', {})
   print(f"Vocabulary size: {tokenizer_config.get('vocabulary_size', 'N/A')}")
   ```

**Solutions:**

1. **Try different input formats:**
   ```python
   # Test with various dialogue lengths
   short_dialogue = ["Hello"]
   medium_dialogue = ["Hello", "How are you?", "I'm fine"]
   long_dialogue = ["Hello", "How are you?", "I'm fine", "What about you?", "Good"]
   
   for dialogue in [short_dialogue, medium_dialogue, long_dialogue]:
       try:
           prediction, confidence = inference.predict_with_confidence(dialogue)
           print(f"Input length {len(dialogue)}: {prediction} (conf: {confidence:.3f})")
       except Exception as e:
           print(f"Error with length {len(dialogue)}: {e}")
   ```

2. **Use top-k predictions to see alternatives:**
   ```python
   top_predictions = inference.predict_top_k(dialogue, k=5)
   print("Alternative predictions:")
   for rank, (token, score) in enumerate(top_predictions, 1):
       print(f"{rank}. {token}: {score:.3f}")
   ```

3. **Retrain with better data or parameters:**
   ```bash
   python ../main.py train --epochs 30 --batch-size 64 --window-size 12
   ```

### Problem: InferenceError during prediction

**Symptoms:**
```
InferenceError: Prediction failed during token generation
```

**Causes:**
- Model architecture mismatch
- Input preprocessing errors
- Memory issues during inference

**Solutions:**

1. **Enable debug logging:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   # Now run inference with detailed logs
   inference = OptimizedLSMInference("./model")
   ```

2. **Test with minimal input:**
   ```python
   try:
       # Test with single token
       result = inference.predict_next_token(["Hello"])
       print("Basic inference works")
   except Exception as e:
       print(f"Basic inference failed: {e}")
   ```

3. **Check model components individually:**
   ```python
   from train import LSMTrainer
   
   try:
       trainer = LSMTrainer()
       trainer, tokenizer = trainer.load_complete_model("./model")
       print("Model loading successful")
   except Exception as e:
       print(f"Model component loading failed: {e}")
   ```

### Problem: Inconsistent predictions

**Symptoms:**
- Same input gives different outputs
- Predictions vary significantly between runs

**Causes:**
- Non-deterministic model behavior
- Caching issues
- Random seed variations

**Solutions:**

1. **Clear caches and test:**
   ```python
   inference.clear_caches()
   
   # Test multiple times
   dialogue = ["Hello", "How are you?"]
   for i in range(5):
       prediction = inference.predict_next_token(dialogue)
       print(f"Run {i+1}: {prediction}")
   ```

2. **Check for randomness in model:**
   ```python
   # Set random seeds if available
   import numpy as np
   import tensorflow as tf
   
   np.random.seed(42)
   tf.random.set_seed(42)
   ```

3. **Use confidence scores to assess stability:**
   ```python
   confidences = []
   for i in range(10):
       _, confidence = inference.predict_with_confidence(dialogue)
       confidences.append(confidence)
   
   print(f"Confidence std dev: {np.std(confidences):.4f}")
   ```

## Performance Issues

### Problem: Slow inference speed

**Symptoms:**
- High latency for single predictions
- Poor throughput for batch processing
- Long model loading times

**Diagnosis:**

1. **Measure performance:**
   ```python
   import time
   
   dialogue = ["Hello", "How are you?"]
   
   # Single prediction timing
   start_time = time.time()
   prediction = inference.predict_next_token(dialogue)
   single_time = time.time() - start_time
   print(f"Single prediction: {single_time*1000:.1f} ms")
   
   # Batch timing
   dialogues = [dialogue] * 100
   start_time = time.time()
   predictions = inference.batch_predict(dialogues)
   batch_time = time.time() - start_time
   print(f"Batch (100): {batch_time:.3f} s ({batch_time/100*1000:.1f} ms per item)")
   ```

2. **Check cache performance:**
   ```python
   stats = inference.get_cache_stats()
   print(f"Cache hit rate: {stats['hit_rate']:.2%}")
   print(f"Total requests: {stats['total_requests']}")
   ```

**Solutions:**

1. **Enable optimizations:**
   ```python
   # Use optimized inference with larger caches
   inference = OptimizedLSMInference(
       model_path="./model",
       lazy_load=True,
       cache_size=2000,
       max_batch_size=64
   )
   ```

2. **Use appropriate batch sizes:**
   ```python
   # Test different batch sizes
   dialogues = [["Hello", "World"]] * 200
   
   for batch_size in [8, 16, 32, 64]:
       start_time = time.time()
       predictions = inference.batch_predict(dialogues, batch_size=batch_size)
       elapsed = time.time() - start_time
       print(f"Batch size {batch_size}: {elapsed:.3f}s ({len(dialogues)/elapsed:.1f} seq/s)")
   ```

3. **Optimize for your use case:**
   ```python
   # For repeated similar inputs, increase cache size
   inference = OptimizedLSMInference("./model", cache_size=5000)
   
   # For memory-constrained environments, use smaller caches
   inference = OptimizedLSMInference("./model", cache_size=100)
   ```

### Problem: High memory usage

**Symptoms:**
- System runs out of memory
- Gradual memory increase over time
- Poor performance due to swapping

**Diagnosis:**

1. **Monitor memory usage:**
   ```python
   import psutil
   import os
   
   process = psutil.Process(os.getpid())
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB")
   
   # Check cache stats
   stats = inference.get_cache_stats()
   if 'memory_mb' in stats:
       print(f"Inference memory: {stats['memory_mb']:.1f} MB")
   ```

**Solutions:**

1. **Reduce cache sizes:**
   ```python
   inference = OptimizedLSMInference(
       model_path="./model",
       cache_size=500,  # Smaller cache
       max_batch_size=16  # Smaller batches
   )
   ```

2. **Clear caches periodically:**
   ```python
   # For long-running processes
   request_count = 0
   for dialogue in dialogues:
       prediction = inference.predict_next_token(dialogue)
       request_count += 1
       
       # Clear cache every 1000 requests
       if request_count % 1000 == 0:
           inference.clear_caches()
   ```

3. **Use lazy loading:**
   ```python
   inference = OptimizedLSMInference("./model", lazy_load=True)
   ```

4. **Process in chunks for large datasets:**
   ```python
   def process_large_dataset(dialogues, chunk_size=100):
       results = []
       for i in range(0, len(dialogues), chunk_size):
           chunk = dialogues[i:i + chunk_size]
           chunk_results = inference.batch_predict(chunk)
           results.extend(chunk_results)
           
           # Clear caches between chunks
           if i % (chunk_size * 5) == 0:
               inference.clear_caches()
       
       return results
   ```

## Memory Problems

### Problem: Out of memory errors

**Symptoms:**
```
MemoryError: Unable to allocate array
```
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   # Use smaller batches
   predictions = inference.batch_predict(dialogues, batch_size=8)
   ```

2. **Enable memory management:**
   ```python
   import gc
   
   # Force garbage collection
   gc.collect()
   
   # Clear TensorFlow memory (if using GPU)
   import tensorflow as tf
   tf.keras.backend.clear_session()
   ```

3. **Use streaming processing:**
   ```python
   def stream_predictions(dialogues, batch_size=16):
       for i in range(0, len(dialogues), batch_size):
           batch = dialogues[i:i + batch_size]
           batch_predictions = inference.batch_predict(batch)
           
           for prediction in batch_predictions:
               yield prediction
           
           # Clean up after each batch
           inference.clear_caches()
           gc.collect()
   ```

### Problem: Memory leaks

**Symptoms:**
- Memory usage increases over time
- Performance degrades with usage
- System becomes unresponsive

**Solutions:**

1. **Monitor memory growth:**
   ```python
   import psutil
   import time
   
   process = psutil.Process()
   initial_memory = process.memory_info().rss
   
   # Run inference
   for i in range(1000):
       prediction = inference.predict_next_token(["Hello", "World"])
       
       if i % 100 == 0:
           current_memory = process.memory_info().rss
           growth = (current_memory - initial_memory) / 1024 / 1024
           print(f"Iteration {i}: Memory growth: {growth:.1f} MB")
   ```

2. **Implement periodic cleanup:**
   ```python
   class ManagedInference:
       def __init__(self, model_path):
           self.inference = OptimizedLSMInference(model_path)
           self.request_count = 0
       
       def predict(self, dialogue):
           result = self.inference.predict_next_token(dialogue)
           self.request_count += 1
           
           # Cleanup every 500 requests
           if self.request_count % 500 == 0:
               self.inference.clear_caches()
               gc.collect()
           
           return result
   ```

## Input Validation Errors

### Problem: InvalidInputError - Empty sequence

**Symptoms:**
```
InvalidInputError: Dialogue sequence cannot be empty
```

**Solutions:**

1. **Check input before processing:**
   ```python
   from input_validation import validate_dialogue_sequence
   
   dialogue = ["Hello", "World"]
   is_valid, error = validate_dialogue_sequence(dialogue)
   
   if is_valid:
       prediction = inference.predict_next_token(dialogue)
   else:
       print(f"Input validation failed: {error}")
   ```

2. **Handle edge cases:**
   ```python
   def safe_predict(inference, dialogue):
       if not dialogue:
           return "[EMPTY_INPUT]"
       
       # Remove empty strings
       cleaned_dialogue = [turn for turn in dialogue if turn.strip()]
       
       if not cleaned_dialogue:
           return "[NO_VALID_INPUT]"
       
       try:
           return inference.predict_next_token(cleaned_dialogue)
       except Exception as e:
           return f"[ERROR: {e}]"
   ```

### Problem: InvalidInputError - Sequence too long

**Symptoms:**
```
InvalidInputError: Dialogue sequence too long (max: 50, got: 75)
```

**Solutions:**

1. **Truncate long sequences:**
   ```python
   def truncate_dialogue(dialogue, max_length=50):
       if len(dialogue) <= max_length:
           return dialogue
       
       # Keep the most recent turns
       return dialogue[-max_length:]
   
   # Use truncated dialogue
   truncated = truncate_dialogue(long_dialogue)
   prediction = inference.predict_next_token(truncated)
   ```

2. **Use sliding window approach:**
   ```python
   def sliding_window_predict(inference, dialogue, window_size=10):
       if len(dialogue) <= window_size:
           return inference.predict_next_token(dialogue)
       
       # Use the last window_size turns
       window = dialogue[-window_size:]
       return inference.predict_next_token(window)
   ```

## Compatibility Issues

### Problem: Version mismatch errors

**Symptoms:**
```
AttributeError: 'LSMTrainer' object has no attribute 'save_complete_model'
```

**Causes:**
- Using old model with new inference code
- Missing dependencies
- Version conflicts

**Solutions:**

1. **Check model version:**
   ```python
   import json
   
   try:
       with open("./model/config.json", "r") as f:
           config = json.load(f)
       print(f"Model version: {config.get('model_version', 'Unknown')}")
   except:
       print("No version information available - likely old model")
   ```

2. **Use appropriate inference class:**
   ```python
   # For new models
   try:
       inference = OptimizedLSMInference("./model")
       print("Using optimized inference")
   except:
       # Fallback to legacy
       inference = LSMInference("./model")
       print("Using legacy inference")
   ```

3. **Update dependencies:**
   ```bash
   pip install --upgrade tensorflow numpy scikit-learn
   ```

### Problem: Platform compatibility issues

**Symptoms:**
- Different results on different operating systems
- File path errors
- Encoding issues

**Solutions:**

1. **Use cross-platform paths:**
   ```python
   import os
   
   model_path = os.path.join("models", "model_20250107_143022")
   inference = OptimizedLSMInference(model_path)
   ```

2. **Handle encoding explicitly:**
   ```python
   import json
   
   with open("config.json", "r", encoding="utf-8") as f:
       config = json.load(f)
   ```

3. **Check platform-specific issues:**
   ```python
   import platform
   print(f"Platform: {platform.system()}")
   print(f"Python version: {platform.python_version()}")
   
   # Platform-specific handling
   if platform.system() == "Windows":
       # Windows-specific code
       pass
   ```

## Installation Problems

### Problem: Missing dependencies

**Symptoms:**
```
ModuleNotFoundError: No module named 'tensorflow'
```

**Solutions:**

1. **Install all requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Check specific dependencies:**
   ```python
   try:
       import tensorflow as tf
       print(f"TensorFlow version: {tf.__version__}")
   except ImportError:
       print("TensorFlow not installed")
   
   try:
       import numpy as np
       print(f"NumPy version: {np.__version__}")
   except ImportError:
       print("NumPy not installed")
   ```

3. **Create virtual environment:**
   ```bash
   python -m venv lsm_env
   source lsm_env/bin/activate  # On Windows: lsm_env\Scripts\activate
   pip install -r requirements.txt
   ```

### Problem: CUDA/GPU issues

**Symptoms:**
```
Could not load dynamic library 'cudart64_110.dll'
```

**Solutions:**

1. **Check GPU availability:**
   ```python
   import tensorflow as tf
   
   print("GPU Available: ", tf.config.list_physical_devices('GPU'))
   print("Built with CUDA: ", tf.test.is_built_with_cuda())
   ```

2. **Force CPU usage if needed:**
   ```python
   import os
   os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
   
   # Then import and use inference
   from inference import OptimizedLSMInference
   ```

3. **Install CPU-only TensorFlow:**
   ```bash
   pip uninstall tensorflow
   pip install tensorflow-cpu
   ```

## Debug Mode and Logging

### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now run your inference code
inference = OptimizedLSMInference("./model")
```

### Custom Logging Configuration

```python
from lsm_logging import get_logger

# Get logger for your module
logger = get_logger(__name__)

# Log important events
logger.info("Starting inference")
logger.debug("Processing dialogue: %s", dialogue)
logger.warning("Low confidence prediction: %f", confidence)
```

### Performance Profiling

```python
import cProfile
import pstats

def profile_inference():
    inference = OptimizedLSMInference("./model")
    dialogue = ["Hello", "How are you?"]
    
    for _ in range(100):
        prediction = inference.predict_next_token(dialogue)

# Profile the function
cProfile.run('profile_inference()', 'inference_profile.stats')

# Analyze results
stats = pstats.Stats('inference_profile.stats')
stats.sort_stats('cumulative').print_stats(10)
```

## Getting Help

### Diagnostic Information

When reporting issues, include this diagnostic information:

```python
import sys
import platform
import tensorflow as tf
import numpy as np
from src.lsm.management.model_manager import ModelManager

print("=== System Information ===")
print(f"Platform: {platform.system()} {platform.release()}")
print(f"Python version: {sys.version}")
print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")

print("\n=== Model Information ===")
try:
    manager = ModelManager()
    models = manager.list_available_models()
    print(f"Available models: {len(models)}")
    
    if models:
        model = models[0]
        print(f"Latest model: {model['path']}")
        print(f"Model status: {model.get('status', 'Unknown')}")
        
        # Validate model
        is_valid, errors = manager.validate_model(model['path'])
        print(f"Model valid: {is_valid}")
        if errors:
            print(f"Validation errors: {errors}")
            
except Exception as e:
    print(f"Model check failed: {e}")

print("\n=== GPU Information ===")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
```

### Common Solutions Checklist

Before reporting issues, try these common solutions:

- [ ] Update all dependencies: `pip install --upgrade -r requirements.txt`
- [ ] Clear Python cache: `find . -name "*.pyc" -delete`
- [ ] Restart Python interpreter
- [ ] Check disk space and permissions
- [ ] Validate model integrity: `python manage_models.py validate`
- [ ] Try with a fresh model: `python ../main.py train`
- [ ] Enable debug logging
- [ ] Test with minimal example
- [ ] Check system resources (RAM, CPU, GPU)

### Where to Get Help

1. **Check the examples:** Look at `../examples/` directory for working code
2. **Read the API documentation:** See `API_DOCUMENTATION.md`
3. **Enable debug logging:** Use debug mode to get detailed error information
4. **Test with minimal examples:** Isolate the problem with simple test cases
5. **Check system requirements:** Ensure your system meets all requirements

### Creating Minimal Reproducible Examples

When reporting issues, create a minimal example:

```python
# Minimal example template
from inference import OptimizedLSMInference

try:
    # Initialize inference
    inference = OptimizedLSMInference("./your_model_path")
    
    # Simple test
    dialogue = ["Hello", "World"]
    prediction = inference.predict_next_token(dialogue)
    print(f"Success: {prediction}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

This troubleshooting guide covers the most common issues you might encounter. For additional help, refer to the API documentation and example scripts.