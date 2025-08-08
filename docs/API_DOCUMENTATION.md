# LSM Inference System API Documentation

This document provides comprehensive API documentation for the enhanced Sparse Sine-Activated Liquid State Machine (LSM) inference system.

## Table of Contents

1. [Core Inference Classes](#core-inference-classes)
2. [Model Management](#model-management)
3. [Configuration Management](#configuration-management)
4. [Error Handling](#error-handling)
5. [Input Validation](#input-validation)
6. [Logging and Monitoring](#logging-and-monitoring)
7. [Usage Examples](#usage-examples)

## Core Inference Classes

### OptimizedLSMInference

The main inference class with performance optimizations, caching, and memory management.

#### Constructor

```python
OptimizedLSMInference(
    model_path: str,
    lazy_load: bool = True,
    cache_size: int = 1000,
    max_batch_size: int = 32
)
```

**Parameters:**
- `model_path` (str): Path to the trained model directory
- `lazy_load` (bool, optional): Enable lazy loading of model components. Default: True
- `cache_size` (int, optional): Maximum number of cached predictions. Default: 1000
- `max_batch_size` (int, optional): Maximum batch size for processing. Default: 32

**Raises:**
- `ModelLoadError`: If model cannot be loaded or is invalid
- `FileNotFoundError`: If model path does not exist

#### Methods

##### predict_next_token

```python
predict_next_token(dialogue_sequence: List[str]) -> str
```

Predict the next token for a dialogue sequence.

**Parameters:**
- `dialogue_sequence` (List[str]): List of dialogue turns

**Returns:**
- `str`: The predicted next token

**Raises:**
- `InvalidInputError`: If dialogue sequence is invalid
- `InferenceError`: If prediction fails
- `TokenizerNotFittedError`: If tokenizer is not properly loaded

**Example:**
```python
inference = OptimizedLSMInference("./models_20250107_143022")
dialogue = ["Hello", "How are you?", "I'm fine"]
next_token = inference.predict_next_token(dialogue)
print(f"Next token: {next_token}")
```

##### predict_with_confidence

```python
predict_with_confidence(dialogue_sequence: List[str]) -> Tuple[str, float]
```

Predict the next token with confidence score.

**Parameters:**
- `dialogue_sequence` (List[str]): List of dialogue turns

**Returns:**
- `Tuple[str, float]`: Tuple of (predicted_token, confidence_score)
  - `predicted_token` (str): The predicted next token
  - `confidence_score` (float): Confidence score between 0.0 and 1.0

**Raises:**
- `InvalidInputError`: If dialogue sequence is invalid
- `InferenceError`: If prediction fails

**Example:**
```python
dialogue = ["Hello", "How are you?"]
token, confidence = inference.predict_with_confidence(dialogue)
print(f"Prediction: {token} (confidence: {confidence:.3f})")
```

##### predict_top_k

```python
predict_top_k(dialogue_sequence: List[str], k: int = 5) -> List[Tuple[str, float]]
```

Get top-k most likely next tokens with scores.

**Parameters:**
- `dialogue_sequence` (List[str]): List of dialogue turns
- `k` (int, optional): Number of top predictions to return. Default: 5

**Returns:**
- `List[Tuple[str, float]]`: List of (token, score) tuples, sorted by score descending

**Raises:**
- `InvalidInputError`: If dialogue sequence is invalid or k is not positive
- `InferenceError`: If prediction fails

**Example:**
```python
dialogue = ["What's your favorite", "color?"]
top_predictions = inference.predict_top_k(dialogue, k=3)
for rank, (token, score) in enumerate(top_predictions, 1):
    print(f"{rank}. {token}: {score:.3f}")
```

##### batch_predict

```python
batch_predict(
    dialogue_sequences: List[List[str]], 
    batch_size: Optional[int] = None
) -> List[str]
```

Process multiple dialogue sequences efficiently.

**Parameters:**
- `dialogue_sequences` (List[List[str]]): List of dialogue sequences
- `batch_size` (int, optional): Batch size for processing. Uses instance default if None

**Returns:**
- `List[str]`: List of predicted tokens, one for each input sequence

**Raises:**
- `InvalidInputError`: If any dialogue sequence is invalid
- `InferenceError`: If batch processing fails

**Example:**
```python
dialogues = [
    ["Hello", "How are you?"],
    ["Good morning", "Nice weather"],
    ["What's your name?", "I'm Alice"]
]
predictions = inference.batch_predict(dialogues)
for dialogue, prediction in zip(dialogues, predictions):
    print(f"{' → '.join(dialogue)} → {prediction}")
```

##### interactive_session

```python
interactive_session() -> None
```

Start an interactive dialogue session with performance monitoring.

**Features:**
- Continuous conversation with context preservation
- Performance monitoring and statistics
- Help commands and usage tips
- Graceful error handling
- Cache statistics display

**Commands available in interactive mode:**
- `help`: Show available commands
- `stats`: Display cache and performance statistics
- `clear`: Clear conversation history
- `info`: Show model information
- `quit` or `exit`: Exit the session

**Example:**
```python
inference = OptimizedLSMInference("./models_20250107_143022")
inference.interactive_session()
```

##### get_model_info

```python
get_model_info() -> Dict[str, Any]
```

Get comprehensive information about the loaded model.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `configuration`: Model configuration parameters
  - `metadata`: Training metadata and performance metrics
  - `file_info`: Information about model files
  - `status`: Model loading status

**Example:**
```python
info = inference.get_model_info()
print(f"Window size: {info['configuration']['window_size']}")
print(f"Test MSE: {info['metadata']['performance_metrics']['final_test_mse']}")
```

##### get_cache_stats

```python
get_cache_stats() -> Dict[str, Any]
```

Get cache performance statistics.

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `prediction_cache_size`: Number of cached predictions
  - `embedding_cache_size`: Number of cached embeddings
  - `hit_rate`: Cache hit rate (0.0 to 1.0)
  - `total_requests`: Total number of requests
  - `cache_hits`: Number of cache hits
  - `memory_mb`: Memory usage in MB (if available)

**Example:**
```python
stats = inference.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Memory usage: {stats.get('memory_mb', 'N/A')} MB")
```

##### clear_caches

```python
clear_caches() -> None
```

Manually clear all caches to free memory.

**Example:**
```python
inference.clear_caches()
print("Caches cleared")
```

##### validate_input

```python
validate_input(dialogue_sequence: List[str]) -> Tuple[bool, str]
```

Validate input dialogue sequence.

**Parameters:**
- `dialogue_sequence` (List[str]): Dialogue sequence to validate

**Returns:**
- `Tuple[bool, str]`: Tuple of (is_valid, error_message)

**Example:**
```python
is_valid, error_msg = inference.validate_input(["Hello", "", "World"])
if not is_valid:
    print(f"Validation error: {error_msg}")
```

### LSMInference (Legacy)

Backward-compatible inference class for older models.

#### Constructor

```python
LSMInference(model_path: str)
```

**Parameters:**
- `model_path` (str): Path to the trained model directory

**Note:** This class provides the same interface as `OptimizedLSMInference` but without performance optimizations. Use for backward compatibility with older models.

## Model Management

### ModelManager

Utility class for model discovery, validation, and management.

#### Constructor

```python
ModelManager(models_root_dir: str = ".")
```

**Parameters:**
- `models_root_dir` (str, optional): Root directory to search for models. Default: current directory

#### Methods

##### list_available_models

```python
list_available_models() -> List[Dict[str, Any]]
```

Scan for valid model directories and return their information.

**Returns:**
- `List[Dict[str, Any]]`: List of model information dictionaries, each containing:
  - `path`: Model directory path
  - `created_at`: Creation timestamp
  - `status`: Model status (complete/incomplete)
  - `configuration`: Model configuration (if available)
  - `test_mse`: Test MSE score (if available)
  - `test_mae`: Test MAE score (if available)

**Example:**
```python
manager = ModelManager()
models = manager.list_available_models()
for model in models:
    print(f"Model: {model['path']}")
    print(f"Created: {model.get('created_at', 'Unknown')}")
    print(f"Test MSE: {model.get('test_mse', 'N/A')}")
```

##### get_model_info

```python
get_model_info(model_path: str) -> Dict[str, Any]
```

Extract comprehensive metadata and configuration details from a model directory.

**Parameters:**
- `model_path` (str): Path to model directory

**Returns:**
- `Dict[str, Any]`: Comprehensive model information including:
  - `configuration`: Complete model configuration
  - `metadata`: Training metadata and performance metrics
  - `file_info`: File sizes and information
  - `components`: Available model components

**Raises:**
- `ModelLoadError`: If model directory is invalid or inaccessible

**Example:**
```python
info = manager.get_model_info("./models_20250107_143022")
config = info['configuration']
print(f"Reservoir type: {config['reservoir_type']}")
print(f"Window size: {config['window_size']}")
```

##### validate_model

```python
validate_model(model_path: str) -> Tuple[bool, List[str]]
```

Check model integrity and completeness.

**Parameters:**
- `model_path` (str): Path to model directory

**Returns:**
- `Tuple[bool, List[str]]`: Tuple of (is_valid, error_list)
  - `is_valid` (bool): True if model is valid and complete
  - `error_list` (List[str]): List of validation errors (empty if valid)

**Example:**
```python
is_valid, errors = manager.validate_model("./models_20250107_143022")
if not is_valid:
    print("Model validation failed:")
    for error in errors:
        print(f"  - {error}")
```

##### cleanup_incomplete_models

```python
cleanup_incomplete_models(dry_run: bool = True) -> List[str]
```

Find and optionally remove incomplete or corrupted model directories.

**Parameters:**
- `dry_run` (bool, optional): If True, only identify candidates without removing. Default: True

**Returns:**
- `List[str]`: List of model paths that are candidates for cleanup

**Example:**
```python
# Find cleanup candidates
candidates = manager.cleanup_incomplete_models(dry_run=True)
print(f"Found {len(candidates)} incomplete models")

# Actually clean up (use with caution)
# removed = manager.cleanup_incomplete_models(dry_run=False)
```

##### get_model_summary

```python
get_model_summary(model_path: str) -> str
```

Get a human-readable summary of a model.

**Parameters:**
- `model_path` (str): Path to model directory

**Returns:**
- `str`: Formatted summary string

**Example:**
```python
summary = manager.get_model_summary("./models_20250107_143022")
print(summary)
```

##### list_models_summary

```python
list_models_summary() -> str
```

Get a formatted summary of all available models.

**Returns:**
- `str`: Formatted summary of all models

**Example:**
```python
summary = manager.list_models_summary()
print(summary)
```

## Configuration Management

### ModelConfiguration

Centralized configuration management for all model parameters.

#### Constructor

```python
ModelConfiguration(
    window_size: int,
    embedding_dim: int,
    reservoir_type: str,
    reservoir_config: Dict,
    reservoir_units: List[int],
    sparsity: float,
    use_multichannel: bool,
    training_params: Dict = None
)
```

**Parameters:**
- `window_size` (int): Sequence window size
- `embedding_dim` (int): Embedding dimension
- `reservoir_type` (str): Type of reservoir ("standard", "advanced", etc.)
- `reservoir_config` (Dict): Reservoir-specific configuration
- `reservoir_units` (List[int]): List of reservoir layer sizes
- `sparsity` (float): Sparsity level (0.0 to 1.0)
- `use_multichannel` (bool): Whether to use multichannel processing
- `training_params` (Dict, optional): Training parameters

#### Class Methods

##### load

```python
@classmethod
load(cls, path: str) -> 'ModelConfiguration'
```

Load configuration from a JSON file.

**Parameters:**
- `path` (str): Path to configuration file

**Returns:**
- `ModelConfiguration`: Loaded configuration instance

**Example:**
```python
config = ModelConfiguration.load("./models_20250107_143022/config.json")
print(f"Window size: {config.window_size}")
```

##### from_dict

```python
@classmethod
from_dict(cls, data: Dict) -> 'ModelConfiguration'
```

Create configuration from dictionary.

**Parameters:**
- `data` (Dict): Configuration dictionary

**Returns:**
- `ModelConfiguration`: Configuration instance

#### Instance Methods

##### save

```python
save(self, path: str) -> None
```

Save configuration to a JSON file.

**Parameters:**
- `path` (str): Path where to save the configuration

**Example:**
```python
config.save("./new_model/config.json")
```

##### to_dict

```python
to_dict(self) -> Dict
```

Convert configuration to dictionary.

**Returns:**
- `Dict`: Configuration as dictionary

**Example:**
```python
config_dict = config.to_dict()
print(config_dict)
```

## Error Handling

The system includes comprehensive error handling with custom exception classes.

### Exception Classes

#### ModelLoadError

Raised when model loading fails.

**Attributes:**
- `model_path` (str): Path to the model that failed to load
- `missing_files` (List[str]): List of missing required files
- `message` (str): Error message

**Example:**
```python
try:
    inference = OptimizedLSMInference("./invalid_model")
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
    print(f"Missing files: {e.missing_files}")
```

#### InferenceError

Raised when inference operations fail.

**Attributes:**
- `operation` (str): The operation that failed
- `input_data` (Any): The input that caused the error
- `message` (str): Error message

**Example:**
```python
try:
    result = inference.predict_next_token(dialogue)
except InferenceError as e:
    print(f"Inference failed during {e.operation}: {e}")
```

#### InvalidInputError

Raised when input validation fails.

**Attributes:**
- `input_value` (Any): The invalid input
- `suggestion` (str): Suggestion for fixing the input
- `message` (str): Error message

**Example:**
```python
try:
    result = inference.predict_next_token([])  # Empty sequence
except InvalidInputError as e:
    print(f"Invalid input: {e}")
    print(f"Suggestion: {e.suggestion}")
```

#### TokenizerNotFittedError

Raised when tokenizer is not properly fitted or loaded.

**Example:**
```python
try:
    result = inference.predict_next_token(dialogue)
except TokenizerNotFittedError as e:
    print(f"Tokenizer error: {e}")
    print("Try retraining the model or check tokenizer files")
```

#### PredictionError

Raised when prediction generation fails.

**Attributes:**
- `prediction_type` (str): Type of prediction that failed
- `context` (Dict): Additional context about the error

## Input Validation

### Validation Functions

#### validate_dialogue_sequence

```python
validate_dialogue_sequence(sequence: List[str]) -> Tuple[bool, str]
```

Validate a dialogue sequence.

**Parameters:**
- `sequence` (List[str]): Dialogue sequence to validate

**Returns:**
- `Tuple[bool, str]`: (is_valid, error_message)

**Example:**
```python
from input_validation import validate_dialogue_sequence

is_valid, error = validate_dialogue_sequence(["Hello", "", "World"])
if not is_valid:
    print(f"Validation error: {error}")
```

#### validate_file_path

```python
validate_file_path(path: str, must_exist: bool = True) -> Tuple[bool, str]
```

Validate a file path.

**Parameters:**
- `path` (str): File path to validate
- `must_exist` (bool, optional): Whether file must exist. Default: True

**Returns:**
- `Tuple[bool, str]`: (is_valid, error_message)

#### validate_positive_integer

```python
validate_positive_integer(value: Any, name: str = "value") -> Tuple[bool, str]
```

Validate that a value is a positive integer.

**Parameters:**
- `value` (Any): Value to validate
- `name` (str, optional): Name of the parameter for error messages

**Returns:**
- `Tuple[bool, str]`: (is_valid, error_message)

#### create_helpful_error_message

```python
create_helpful_error_message(error_type: str, details: Dict) -> str
```

Create a helpful error message with suggestions.

**Parameters:**
- `error_type` (str): Type of error
- `details` (Dict): Error details and context

**Returns:**
- `str`: Formatted error message with suggestions

## Logging and Monitoring

### Logging Functions

#### get_logger

```python
get_logger(name: str) -> logging.Logger
```

Get a configured logger for a module.

**Parameters:**
- `name` (str): Logger name (usually `__name__`)

**Returns:**
- `logging.Logger`: Configured logger instance

**Example:**
```python
from lsm_logging import get_logger

logger = get_logger(__name__)
logger.info("Starting inference")
```

#### log_performance

```python
@log_performance(operation_name: str)
def your_function():
    # Function implementation
    pass
```

Decorator for automatic performance logging.

**Parameters:**
- `operation_name` (str): Name of the operation for logging

**Example:**
```python
from lsm_logging import log_performance

@log_performance("model prediction")
def predict_token(self, sequence):
    # Prediction logic
    return result
```

#### create_operation_logger

```python
create_operation_logger(operation: str, **kwargs) -> ContextManager
```

Create a context manager for operation logging.

**Parameters:**
- `operation` (str): Operation name
- `**kwargs`: Additional context to log

**Returns:**
- `ContextManager`: Context manager for logging

**Example:**
```python
from lsm_logging import create_operation_logger

with create_operation_logger("batch_processing", batch_size=32):
    # Processing logic
    results = process_batch(data)
```

## Usage Examples

### Basic Usage

```python
from inference import OptimizedLSMInference

# Initialize inference
inference = OptimizedLSMInference("./models_20250107_143022")

# Simple prediction
dialogue = ["Hello", "How are you?"]
next_token = inference.predict_next_token(dialogue)
print(f"Next token: {next_token}")

# Prediction with confidence
token, confidence = inference.predict_with_confidence(dialogue)
print(f"Prediction: {token} (confidence: {confidence:.3f})")

# Top-k predictions
top_predictions = inference.predict_top_k(dialogue, k=3)
for rank, (token, score) in enumerate(top_predictions, 1):
    print(f"{rank}. {token}: {score:.3f}")
```

### Batch Processing

```python
# Process multiple sequences
dialogues = [
    ["Hello", "How are you?"],
    ["Good morning", "Nice weather"],
    ["What's your name?", "I'm Alice"]
]

predictions = inference.batch_predict(dialogues, batch_size=16)
for dialogue, prediction in zip(dialogues, predictions):
    print(f"{' → '.join(dialogue)} → {prediction}")
```

### Model Management

```python
from src.lsm.management.model_manager import ModelManager

# Initialize manager
manager = ModelManager()

# List available models
models = manager.list_available_models()
for model in models:
    print(f"Model: {model['path']}")
    print(f"Created: {model.get('created_at', 'Unknown')}")

# Validate a model
is_valid, errors = manager.validate_model("./models_20250107_143022")
if not is_valid:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

### Error Handling

```python
from lsm_exceptions import ModelLoadError, InferenceError, InvalidInputError

try:
    inference = OptimizedLSMInference("./model_path")
    result = inference.predict_next_token(["Hello", "World"])
    
except ModelLoadError as e:
    print(f"Model loading failed: {e}")
    print(f"Missing files: {e.missing_files}")
    
except InvalidInputError as e:
    print(f"Invalid input: {e}")
    print(f"Suggestion: {e.suggestion}")
    
except InferenceError as e:
    print(f"Inference failed: {e}")
    
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Performance Monitoring

```python
# Monitor cache performance
stats = inference.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate']:.2%}")
print(f"Memory usage: {stats.get('memory_mb', 'N/A')} MB")

# Clear caches if needed
if stats['hit_rate'] < 0.1:
    inference.clear_caches()
    print("Caches cleared due to low hit rate")
```

### Configuration Management

```python
from model_config import ModelConfiguration

# Load existing configuration
config = ModelConfiguration.load("./models_20250107_143022/config.json")
print(f"Window size: {config.window_size}")
print(f"Reservoir type: {config.reservoir_type}")

# Create new configuration
new_config = ModelConfiguration(
    window_size=12,
    embedding_dim=256,
    reservoir_type="advanced",
    reservoir_config={},
    reservoir_units=[512, 256, 128],
    sparsity=0.15,
    use_multichannel=True
)

# Save configuration
new_config.save("./new_model/config.json")
```

This API documentation provides comprehensive coverage of all classes, methods, and functions in the enhanced LSM inference system. For additional examples and tutorials, see the `../examples/` directory and the main ../README.md file.