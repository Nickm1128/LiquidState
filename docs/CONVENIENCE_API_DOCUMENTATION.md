# LSM Convenience API Documentation

## Overview

The LSM Convenience API provides a scikit-learn-compatible interface for Liquid State Machine models, making it easy to train and use LSM models without dealing with the complexity of the underlying multi-component architecture.

## Quick Start

```python
from lsm import LSMGenerator, LSMClassifier, LSMRegressor

# Text generation
generator = LSMGenerator()
generator.fit(conversations)
response = generator.generate("Hello, how are you?")

# Classification
classifier = LSMClassifier()
classifier.fit(texts, labels)
predictions = classifier.predict(new_texts)

# Regression
regressor = LSMRegressor()
regressor.fit(sequences, targets)
predictions = regressor.predict(new_sequences)
```

## Core Classes

### LSMBase

Base class for all convenience API classes, providing common functionality.

#### Methods

##### `__init__(window_size=10, embedding_dim=128, reservoir_type='standard', **kwargs)`

Initialize the LSM model with configuration parameters.

**Parameters:**
- `window_size` (int, default=10): Size of the sliding window for sequence processing
- `embedding_dim` (int, default=128): Dimension of the embedding space
- `reservoir_type` (str, default='standard'): Type of reservoir ('standard', 'hierarchical', 'attentive', 'echo_state', 'deep')
- `random_state` (int, optional): Random seed for reproducibility
- `**kwargs`: Additional configuration parameters

##### `save(path)`

Save the trained model to disk.

**Parameters:**
- `path` (str or Path): Directory path to save the model

**Example:**
```python
generator = LSMGenerator()
generator.fit(data)
generator.save("my_model")
```

##### `load(path)` (class method)

Load a previously saved model from disk.

**Parameters:**
- `path` (str or Path): Directory path containing the saved model

**Returns:**
- Loaded model instance

**Example:**
```python
generator = LSMGenerator.load("my_model")
response = generator.generate("Hello!")
```

##### `get_params(deep=True)`

Get parameters for this estimator (sklearn compatibility).

**Parameters:**
- `deep` (bool, default=True): If True, return parameters for sub-estimators

**Returns:**
- `dict`: Parameter names mapped to their values

##### `set_params(**params)`

Set parameters for this estimator (sklearn compatibility).

**Parameters:**
- `**params`: Estimator parameters

**Returns:**
- `self`: Estimator instance

### LSMGenerator

Main class for text generation and conversational AI.

#### Methods

##### `__init__(window_size=10, embedding_dim=128, reservoir_type='hierarchical', system_message_support=True, response_level=True, tokenizer='gpt2', preset=None, **kwargs)`

Initialize the text generator.

**Parameters:**
- `window_size` (int, default=10): Size of the sliding window
- `embedding_dim` (int, default=128): Embedding dimension
- `reservoir_type` (str, default='hierarchical'): Reservoir architecture type
- `system_message_support` (bool, default=True): Enable system message processing
- `response_level` (bool, default=True): Enable response-level generation
- `tokenizer` (str, default='gpt2'): Tokenizer to use ('gpt2', 'bert', etc.)
- `preset` (str, optional): Use preset configuration ('fast', 'balanced', 'quality')
- `**kwargs`: Additional parameters

**Example:**
```python
# Basic usage
generator = LSMGenerator()

# With preset
generator = LSMGenerator(preset='quality')

# Custom configuration
generator = LSMGenerator(
    window_size=20,
    embedding_dim=256,
    reservoir_type='attentive',
    system_message_support=True
)
```

##### `fit(conversations, system_messages=None, validation_split=0.2, epochs=50, batch_size=32, **fit_params)`

Train the generator on conversation data.

**Parameters:**
- `conversations` (list): Training conversations in various formats
- `system_messages` (list, optional): System messages for training
- `validation_split` (float, default=0.2): Fraction of data for validation
- `epochs` (int, default=50): Number of training epochs
- `batch_size` (int, default=32): Training batch size
- `**fit_params`: Additional training parameters

**Supported conversation formats:**
```python
# Simple string list
conversations = ["Hello", "Hi there", "How are you?"]

# Structured format
conversations = [
    {"messages": ["Hello", "Hi"], "system": "Be friendly"},
    {"messages": ["Help me", "Sure"], "system": "Be helpful"}
]

# Raw text with automatic splitting
conversations = ["User: Hello\nAssistant: Hi there\nUser: How are you?"]
```

**Example:**
```python
conversations = [
    "User: Hello\nAssistant: Hi there!",
    "User: How are you?\nAssistant: I'm doing well, thanks!"
]

generator = LSMGenerator()
generator.fit(conversations, epochs=100, batch_size=16)
```

##### `generate(prompt, system_message=None, max_length=50, temperature=1.0, **kwargs)`

Generate a response to a prompt.

**Parameters:**
- `prompt` (str): Input text or conversation history
- `system_message` (str, optional): System context for generation
- `max_length` (int, default=50): Maximum response length
- `temperature` (float, default=1.0): Generation randomness (0.1-2.0)
- `**kwargs`: Additional generation parameters

**Returns:**
- `str`: Generated response

**Example:**
```python
response = generator.generate(
    "Hello, how are you?",
    system_message="You are a helpful assistant",
    max_length=100,
    temperature=0.8
)
```

##### `batch_generate(prompts, **kwargs)`

Generate responses for multiple prompts efficiently.

**Parameters:**
- `prompts` (list): List of input prompts
- `**kwargs`: Generation parameters (same as `generate`)

**Returns:**
- `list`: List of generated responses

**Example:**
```python
prompts = ["Hello!", "How are you?", "What's the weather like?"]
responses = generator.batch_generate(prompts, max_length=50)
```

##### `chat(system_message=None)`

Start an interactive chat session.

**Parameters:**
- `system_message` (str, optional): System context for the chat

**Example:**
```python
generator.chat("You are a helpful coding assistant")
# Starts interactive session in terminal
```

### LSMClassifier

Classifier using LSM reservoir states as features.

#### Methods

##### `__init__(window_size=10, embedding_dim=128, reservoir_type='standard', classifier_type='logistic', n_classes=None, **kwargs)`

Initialize the classifier.

**Parameters:**
- `window_size` (int, default=10): Size of the sliding window
- `embedding_dim` (int, default=128): Embedding dimension
- `reservoir_type` (str, default='standard'): Reservoir architecture type
- `classifier_type` (str, default='logistic'): Downstream classifier ('logistic', 'random_forest')
- `n_classes` (int, optional): Number of classes (auto-detected if not provided)
- `**kwargs`: Additional parameters

##### `fit(X, y, validation_split=0.2, epochs=30, **fit_params)`

Train the classifier on text data.

**Parameters:**
- `X` (list): Text samples or sequences
- `y` (array-like): Class labels
- `validation_split` (float, default=0.2): Fraction for validation
- `epochs` (int, default=30): Training epochs for reservoir
- `**fit_params`: Additional training parameters

**Example:**
```python
texts = ["This is positive", "This is negative", "Neutral text"]
labels = [1, 0, 2]

classifier = LSMClassifier(classifier_type='random_forest')
classifier.fit(texts, labels)
```

##### `predict(X)`

Predict classes for text samples.

**Parameters:**
- `X` (list): Text samples to classify

**Returns:**
- `array`: Predicted class labels

##### `predict_proba(X)`

Predict class probabilities.

**Parameters:**
- `X` (list): Text samples to classify

**Returns:**
- `array`: Class probabilities (n_samples, n_classes)

##### `score(X, y)`

Return the mean accuracy on the given test data and labels.

**Parameters:**
- `X` (list): Test samples
- `y` (array-like): True labels

**Returns:**
- `float`: Mean accuracy

### LSMRegressor

Regressor using LSM temporal dynamics for continuous prediction.

#### Methods

##### `__init__(window_size=10, embedding_dim=128, reservoir_type='echo_state', regressor_type='linear', **kwargs)`

Initialize the regressor.

**Parameters:**
- `window_size` (int, default=10): Size of the sliding window
- `embedding_dim` (int, default=128): Embedding dimension
- `reservoir_type` (str, default='echo_state'): Reservoir type (good for time series)
- `regressor_type` (str, default='linear'): Downstream regressor ('linear', 'ridge', 'random_forest')
- `**kwargs`: Additional parameters

##### `fit(X, y, validation_split=0.2, epochs=30, **fit_params)`

Train the regressor on sequential data.

**Parameters:**
- `X` (array-like): Input sequences or text data
- `y` (array-like): Target values
- `validation_split` (float, default=0.2): Fraction for validation
- `epochs` (int, default=30): Training epochs for reservoir
- `**fit_params`: Additional training parameters

**Example:**
```python
# Time series data
X = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]
y = [4, 5, 6]

regressor = LSMRegressor(regressor_type='ridge')
regressor.fit(X, y)
```

##### `predict(X)`

Predict continuous values.

**Parameters:**
- `X` (array-like): Input sequences

**Returns:**
- `array`: Predicted values

##### `score(X, y)`

Return the coefficient of determination R² of the prediction.

**Parameters:**
- `X` (array-like): Test samples
- `y` (array-like): True values

**Returns:**
- `float`: R² score

## Configuration Management

### ConvenienceConfig

Manages configuration presets and parameter validation.

#### Class Methods

##### `get_preset(name)`

Get a preset configuration.

**Parameters:**
- `name` (str): Preset name ('fast', 'balanced', 'quality')

**Returns:**
- `dict`: Configuration parameters

**Available presets:**
- `'fast'`: Quick training for experimentation
- `'balanced'`: Good balance of speed and quality
- `'quality'`: Maximum quality for production use

**Example:**
```python
from lsm.convenience import ConvenienceConfig

config = ConvenienceConfig.get_preset('quality')
generator = LSMGenerator(**config)
```

##### `list_presets()`

List all available presets with descriptions.

**Returns:**
- `dict`: Preset names mapped to descriptions

##### `validate_params(params)`

Validate parameter combinations.

**Parameters:**
- `params` (dict): Parameters to validate

**Raises:**
- `ConvenienceValidationError`: If parameters are invalid

## Error Handling

### ConvenienceValidationError

Custom exception for parameter validation errors with helpful suggestions.

**Attributes:**
- `message` (str): Error description
- `suggestion` (str): Suggested fix
- `valid_options` (list): Valid parameter options

**Example:**
```python
from lsm.convenience import LSMGenerator, ConvenienceValidationError

try:
    generator = LSMGenerator(reservoir_type='invalid')
except ConvenienceValidationError as e:
    print(f"Error: {e}")
    print(f"Suggestion: {e.suggestion}")
    print(f"Valid options: {e.valid_options}")
```

## Data Format Handling

The convenience API automatically handles various data formats:

### Text Generation Data

```python
# Simple conversations
conversations = [
    "Hello there!",
    "How can I help you?",
    "What's the weather like?"
]

# Structured conversations
conversations = [
    {
        "messages": ["Hello", "Hi there!", "How are you?", "I'm good!"],
        "system": "Be friendly and helpful"
    }
]

# Raw dialogue format
conversations = [
    "User: Hello\nAssistant: Hi there!\nUser: How are you?\nAssistant: I'm good!"
]
```

### Classification Data

```python
# Text classification
texts = [
    "This movie is amazing!",
    "Terrible film, waste of time",
    "It was okay, nothing special"
]
labels = ["positive", "negative", "neutral"]

# Or with numeric labels
labels = [1, 0, 2]
```

### Regression Data

```python
# Time series sequences
X = [
    [1.0, 2.0, 3.0],
    [2.0, 3.0, 4.0],
    [3.0, 4.0, 5.0]
]
y = [4.0, 5.0, 6.0]

# Text with numeric targets
texts = ["Short text", "This is a longer piece of text", "Medium length"]
lengths = [2, 8, 3]  # Word counts as targets
```

## Performance Optimization

### Memory Management

The convenience API includes automatic memory management:

```python
from lsm.convenience import LSMGenerator

# Automatic memory optimization
generator = LSMGenerator(auto_memory_management=True)

# Manual memory settings
generator = LSMGenerator(
    batch_size=16,  # Smaller batches for limited memory
    gradient_accumulation_steps=4  # Accumulate gradients
)
```

### Performance Monitoring

```python
from lsm.convenience import LSMGenerator, monitor_performance

with monitor_performance():
    generator = LSMGenerator()
    generator.fit(data)
    # Performance metrics automatically logged
```

## Integration with Scikit-learn

The convenience API is fully compatible with scikit-learn:

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lsm.convenience import LSMClassifier

# Cross-validation
classifier = LSMClassifier()
scores = cross_val_score(classifier, texts, labels, cv=5)

# Pipeline integration
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lsm', LSMClassifier())
])
```

## Advanced Usage

### Custom Tokenizers

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
generator = LSMGenerator(tokenizer=tokenizer)
```

### Custom Reservoir Configurations

```python
reservoir_config = {
    'n_reservoir': 1000,
    'spectral_radius': 0.95,
    'input_scaling': 0.1,
    'connectivity': 0.1
}

generator = LSMGenerator(
    reservoir_type='echo_state',
    reservoir_config=reservoir_config
)
```

### System Message Processing

```python
generator = LSMGenerator(system_message_support=True)

response = generator.generate(
    "What's the capital of France?",
    system_message="You are a geography expert. Provide accurate, concise answers."
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure `src/` is in your Python path
2. **Memory Issues**: Use smaller batch sizes or enable auto memory management
3. **Training Slow**: Try the 'fast' preset for experimentation
4. **Poor Quality**: Use the 'quality' preset and more training data

### Performance Tips

1. **Use presets**: Start with 'balanced' preset and adjust as needed
2. **Batch processing**: Use `batch_generate()` for multiple predictions
3. **Memory management**: Enable automatic memory optimization
4. **Validation**: Use validation split to monitor training progress

### Getting Help

- Check the examples in `examples/` directory
- Review the troubleshooting guide in `docs/TROUBLESHOOTING_GUIDE.md`
- Enable debug logging for detailed information

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```