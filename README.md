# Sparse Sine-Activated Liquid State Machine for Next-Token Prediction

This project implements a novel neural architecture that combines **Liquid State Machines (LSM)** with **Convolutional Neural Networks (CNN)** for next-token prediction on dialogue data. The system uses sparse connectivity patterns and parametric sine activation functions to create complex temporal dynamics, which are then encoded as 2D "waveforms" for CNN processing.

## Overview

### What is a Liquid State Machine?
A Liquid State Machine is a type of recurrent neural network inspired by biological neural circuits. Unlike traditional RNNs, LSMs maintain a "reservoir" of randomly connected neurons that create rich temporal dynamics. The key innovation in this project is:

1. **Sparse Connectivity**: Only a fraction of connections exist between neurons, making the network more efficient and biologically plausible
2. **Parametric Sine Activation**: Instead of standard activations, we use learnable sine functions: `A * exp(-α * |x|) * sin(ω * x)`
3. **Rolling Wave Encoding**: LSM outputs are encoded as 2D spatial-temporal patterns
4. **CNN Processing**: A CNN learns to interpret these 2D patterns for next-token prediction

### Architecture Components

1. **Data Loader**: Downloads and processes HuggingFace Synthetic-Persona-Chat dataset
2. **Sparse Reservoir**: Custom Keras layers with learnable sparse connectivity
3. **Rolling Wave Buffer**: Converts temporal dynamics to 2D waveforms
4. **CNN Model**: Processes waveforms to predict next token embeddings
5. **Training Pipeline**: Integrates all components for end-to-end learning
6. **Enhanced Inference System**: Complete text-to-text prediction with tokenizer persistence
7. **Model Management**: Comprehensive model storage, validation, and discovery utilities

## Installation

### Requirements
- Python 3.9+
- TensorFlow 2.10+
- CUDA support recommended for GPU acceleration

### Setup
1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start Guide

### Training a New Model
```bash
# Train with default settings
python main.py train

# Train with specific parameters
python main.py train --window-size 8 --batch-size 16 --epochs 10

# Train with custom reservoir configuration
python main.py train --reservoir-type advanced --sparsity 0.15 --multichannel
```

### Using Trained Models for Inference

The enhanced inference system provides multiple ways to interact with trained models:

#### Interactive Mode
Start an interactive session for continuous dialogue:
```bash
python inference.py --model-path ./models_20250107_143022 --interactive
```

#### Single Prediction
Get a single next-token prediction:
```bash
python inference.py --model-path ./models_20250107_143022 --input-text "Hello" "How are you?" "I'm fine"
```

#### Batch Processing
Process multiple dialogue sequences at once:
```bash
python inference.py --model-path ./models_20250107_143022 --batch-file dialogues.txt
```

#### Top-K Predictions
Get multiple prediction candidates with confidence scores:
```bash
python inference.py --model-path ./models_20250107_143022 --input-text "Hello" "How are you?" --top-k 5
```

#### Performance Optimized Mode
Use optimized inference for production environments:
```bash
python inference.py --model-path ./models_20250107_143022 --optimized --cache-size 2000 --batch-size 64
```

### Model Management

#### List Available Models
```bash
python manage_models.py list
```

#### Get Model Information
```bash
python manage_models.py info --model-path ./models_20250107_143022
```

#### Validate Model Integrity
```bash
python manage_models.py validate --model-path ./models_20250107_143022
```

#### Clean Up Incomplete Models
```bash
python manage_models.py cleanup --dry-run
```

## Enhanced Inference Capabilities

### Complete Text Processing Pipeline

The enhanced inference system provides a complete text-to-text prediction pipeline with the following features:

- **Automatic Tokenizer Persistence**: Tokenizers are automatically saved and loaded with models
- **Text Decoding**: Model embeddings are converted back to human-readable text
- **Confidence Scoring**: Predictions include confidence scores for reliability assessment
- **Batch Processing**: Efficient processing of multiple dialogue sequences
- **Caching**: Intelligent caching for improved performance
- **Memory Optimization**: Lazy loading and memory management for production use

### Inference Modes

#### 1. Interactive Session
Perfect for testing and experimentation:
```python
from inference import OptimizedLSMInference

# Initialize inference
inference = OptimizedLSMInference("./models_20250107_143022")

# Start interactive session
inference.interactive_session()
```

#### 2. Programmatic API
For integration into applications:
```python
from inference import OptimizedLSMInference

# Initialize inference
inference = OptimizedLSMInference("./models_20250107_143022")

# Single prediction
dialogue = ["Hello", "How are you?", "I'm fine"]
next_token = inference.predict_next_token(dialogue)
print(f"Next token: {next_token}")

# Prediction with confidence
next_token, confidence = inference.predict_with_confidence(dialogue)
print(f"Next token: {next_token} (confidence: {confidence:.3f})")

# Top-K predictions
top_predictions = inference.predict_top_k(dialogue, k=5)
for token, score in top_predictions:
    print(f"{token}: {score:.3f}")

# Batch processing
dialogues = [
    ["Hello", "How are you?"],
    ["Good morning", "Nice weather"],
    ["What's your name?", "I'm Alice"]
]
predictions = inference.batch_predict(dialogues)
```

#### 3. Legacy Compatibility
For backward compatibility with older models:
```python
from inference import LSMInference

# Use legacy interface
inference = LSMInference("./old_model_directory")
inference.interactive_session()
```

### Model Storage Structure

Enhanced models are stored with the following structure:
```
models_YYYYMMDD_HHMMSS/
├── reservoir_model/           # Keras reservoir model
├── cnn_model/                # Keras CNN model  
├── tokenizer/                # Serialized tokenizer
│   ├── vectorizer.pkl        # TF-IDF vectorizer
│   ├── vocab_mapping.json    # Vocabulary mappings
│   └── config.json          # Tokenizer configuration
├── config.json              # Complete model configuration
├── metadata.json            # Training metadata & performance
├── training_history.csv     # Training metrics history
└── inference_cache/         # Optional: cached embeddings
```

### Performance Features

#### Lazy Loading
Models are loaded on-demand to reduce memory usage:
```python
# Model components loaded only when needed
inference = OptimizedLSMInference("./model", lazy_load=True)
```

#### Caching
Intelligent caching improves performance for repeated predictions:
```python
# Configure cache size
inference = OptimizedLSMInference("./model", cache_size=2000)

# Check cache statistics
stats = inference.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
```

#### Memory Management
Automatic memory management for long-running applications:
```python
# Memory is automatically managed
# Manual cache clearing if needed
inference.clear_caches()
```

## API Reference

### OptimizedLSMInference Class

The main inference class with performance optimizations.

#### Constructor
```python
OptimizedLSMInference(
    model_path: str,
    lazy_load: bool = True,
    cache_size: int = 1000,
    max_batch_size: int = 32
)
```

#### Methods

**predict_next_token(dialogue_sequence: List[str]) -> str**
- Predict the next token for a dialogue sequence
- Returns the most likely next token as text

**predict_with_confidence(dialogue_sequence: List[str]) -> Tuple[str, float]**
- Predict with confidence score
- Returns tuple of (token, confidence_score)

**predict_top_k(dialogue_sequence: List[str], k: int = 5) -> List[Tuple[str, float]]**
- Get top-k predictions with scores
- Returns list of (token, score) tuples

**batch_predict(dialogue_sequences: List[List[str]], batch_size: Optional[int] = None) -> List[str]**
- Process multiple sequences efficiently
- Returns list of predicted tokens

**interactive_session()**
- Start interactive dialogue session
- Includes performance monitoring and help commands

**get_model_info() -> Dict[str, Any]**
- Get comprehensive model information
- Includes configuration, metadata, and performance stats

**get_cache_stats() -> Dict[str, Any]**
- Get cache performance statistics
- Useful for monitoring and optimization

### ModelManager Class

Utility class for model discovery and management.

#### Constructor
```python
ModelManager(models_root_dir: str = ".")
```

#### Methods

**list_available_models() -> List[Dict[str, Any]]**
- Scan for valid model directories
- Returns list of model information dictionaries

**get_model_info(model_path: str) -> Dict[str, Any]**
- Get detailed information about a specific model
- Includes configuration, metadata, and file information

**validate_model(model_path: str) -> Tuple[bool, List[str]]**
- Check model integrity and completeness
- Returns (is_valid, error_list)

**cleanup_incomplete_models(dry_run: bool = True) -> List[str]**
- Find and optionally remove incomplete models
- Returns list of cleanup candidates

**get_model_summary(model_path: str) -> str**
- Get human-readable model summary
- Formatted for display purposes

## Configuration Management

### ModelConfiguration Class

Centralized configuration management for all model parameters.

```python
from model_config import ModelConfiguration

# Load configuration from saved model
config = ModelConfiguration.load("./models_20250107_143022/config.json")

# Access configuration parameters
print(f"Window size: {config.window_size}")
print(f"Embedding dimension: {config.embedding_dim}")
print(f"Reservoir type: {config.reservoir_type}")

# Create new configuration
config = ModelConfiguration(
    window_size=10,
    embedding_dim=128,
    reservoir_type="standard",
    reservoir_config={},
    reservoir_units=[256, 128, 64],
    sparsity=0.1,
    use_multichannel=True
)

# Save configuration
config.save("./new_model/config.json")
```

## Error Handling and Validation

The system includes comprehensive error handling with helpful error messages:

### Common Error Types

**ModelLoadError**: Issues loading model components
```python
try:
    inference = OptimizedLSMInference("./invalid_model")
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
    print(f"Missing files: {e.missing_files}")
```

**InferenceError**: Problems during prediction
```python
try:
    result = inference.predict_next_token(dialogue)
except InferenceError as e:
    print(f"Prediction failed: {e}")
```

**InvalidInputError**: Input validation failures
```python
try:
    result = inference.predict_next_token([])  # Empty sequence
except InvalidInputError as e:
    print(f"Invalid input: {e}")
    print(f"Suggestion: {e.suggestion}")
```

### Input Validation

The system validates inputs and provides helpful suggestions:

```python
from input_validation import validate_dialogue_sequence

# Validate dialogue sequence
is_valid, error_msg = validate_dialogue_sequence(["Hello", "", "World"])
if not is_valid:
    print(f"Validation error: {error_msg}")
```

## Logging and Monitoring

### Performance Logging

The system includes comprehensive logging for monitoring:

```python
from lsm_logging import get_logger, log_performance

# Get logger for your module
logger = get_logger(__name__)

# Performance logging is automatic for key operations
# Check logs for timing information
```

### Memory Monitoring

Monitor memory usage during inference:

```python
# Memory monitoring (requires psutil)
inference = OptimizedLSMInference("./model")
stats = inference.get_cache_stats()
print(f"Memory usage: {stats.get('memory_mb', 'N/A')} MB")
```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Model Loading Failures

**Problem**: `ModelLoadError: Missing required files`
**Solution**: 
- Check that all model components exist
- Use `ModelManager.validate_model()` to identify missing files
- Ensure model was saved with the enhanced training pipeline

**Problem**: `TokenizerNotFittedError: Tokenizer not fitted`
**Solution**:
- Retrain the model with the enhanced pipeline
- Use backward compatibility mode for old models
- Check tokenizer directory for required files

#### 2. Memory Issues

**Problem**: High memory usage during inference
**Solution**:
- Enable lazy loading: `OptimizedLSMInference(model_path, lazy_load=True)`
- Reduce cache size: `cache_size=500`
- Use smaller batch sizes for batch processing
- Clear caches periodically: `inference.clear_caches()`

**Problem**: Out of memory during batch processing
**Solution**:
- Reduce `max_batch_size` parameter
- Process sequences in smaller chunks
- Enable memory management: automatic garbage collection is included

#### 3. Performance Issues

**Problem**: Slow inference speed
**Solution**:
- Enable caching for repeated predictions
- Use batch processing for multiple sequences
- Ensure GPU acceleration is available
- Use optimized inference class instead of legacy

**Problem**: Low cache hit rate
**Solution**:
- Increase cache size if memory allows
- Check for consistent input formatting
- Monitor cache statistics: `inference.get_cache_stats()`

#### 4. Prediction Quality Issues

**Problem**: Poor prediction quality
**Solution**:
- Check model training metrics in metadata.json
- Validate input sequence format and length
- Ensure tokenizer vocabulary matches training data
- Try top-k predictions to see alternative candidates

**Problem**: Inconsistent predictions
**Solution**:
- Check for tokenizer consistency between training and inference
- Validate model integrity: `ModelManager.validate_model()`
- Ensure proper sequence preprocessing

#### 5. Compatibility Issues

**Problem**: Cannot load old models
**Solution**:
- Use `LSMInference` class for backward compatibility
- Check model directory structure
- Consider retraining with enhanced pipeline
- Use migration utilities (when available)

### Getting Help

1. **Check Logs**: Enable debug logging to see detailed error information
2. **Validate Models**: Use `ModelManager.validate_model()` to check integrity
3. **Test with Examples**: Use provided example scripts to verify setup
4. **Monitor Performance**: Use cache statistics and performance logs
5. **Check Requirements**: Ensure all dependencies are installed correctly

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now all operations will include detailed debug information
inference = OptimizedLSMInference("./model")
```

## Examples and Tutorials

See the `examples/` directory for complete usage examples:

- `basic_inference.py`: Simple prediction examples
- `batch_processing.py`: Efficient batch processing
- `interactive_demo.py`: Interactive session demonstration
- `model_management.py`: Model discovery and validation

## Documentation

- **[API Documentation](docs/API_DOCUMENTATION.md)**: Comprehensive API reference for all classes and methods
- **[Troubleshooting Guide](docs/TROUBLESHOOTING_GUIDE.md)**: Solutions for common issues and problems
- **[Examples Directory](examples/)**: Complete working examples for different use cases

## Contributing

When contributing to the inference system:

1. Add comprehensive error handling with helpful messages
2. Include input validation for all public methods
3. Add logging for performance monitoring
4. Write tests for new functionality
5. Update documentation for API changes
6. Consider backward compatibility impact

## License

This project is licensed under the MIT License - see the LICENSE file for details.