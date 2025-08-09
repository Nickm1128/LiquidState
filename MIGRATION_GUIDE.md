# LSM Migration Guide

This comprehensive guide helps you migrate from the legacy root-level imports to the new organized src/ structure and convenience API. Whether you're updating existing code or starting fresh, this guide will help you make the transition smoothly.

## Overview

The LSM project has been completely reorganized to provide:
1. **Clean package structure** - All functionality moved to `src/lsm/`
2. **Convenience API** - Scikit-learn-like interface for easy usage
3. **Advanced API** - Full access to all underlying components
4. **Backward compatibility** - Legacy imports still work with deprecation warnings
5. **Better performance** - Optimized implementations and memory management

## Migration Paths

### Path 1: Simple Migration (Recommended for Most Users)

**Old way:**
```python
from train import LSMTrainer
from data_loader import load_data

trainer = LSMTrainer(window_size=10, embedding_dim=128)
X_train, y_train, X_test, y_test, tokenizer = load_data()
trainer.train(X_train, y_train, X_test, y_test)
```

**New way (Convenience API):**
```python
from lsm import LSMGenerator

generator = LSMGenerator(window_size=10, embedding_dim=128)
generator.fit(conversations)
response = generator.generate("Hello, how are you?")
```

### Path 2: Advanced Migration (For Complex Use Cases)

**Old way:**
```python
from train import LSMTrainer
from data_loader import DialogueTokenizer
from reservoir_manager import ReservoirManager

trainer = LSMTrainer(...)
tokenizer = DialogueTokenizer(...)
reservoir = ReservoirManager(...)
```

**New way (Advanced API):**
```python
from lsm.training.train import LSMTrainer
from lsm.data.tokenization import StandardTokenizerWrapper
from lsm.core.reservoir_manager import ReservoirManager

trainer = LSMTrainer(...)
tokenizer = StandardTokenizerWrapper(...)
reservoir = ReservoirManager(...)
```

### Path 3: Hybrid Approach (Best of Both Worlds)

```python
from lsm import LSMGenerator
from lsm.core.reservoir_manager import ReservoirManager

# Use convenience API with custom components
custom_reservoir = ReservoirManager(reservoir_type='attentive')
generator = LSMGenerator(reservoir_manager=custom_reservoir)
```

### For Advanced Use Cases

**Old way:**
```python
from train import LSMTrainer, run_training
from data_loader import DialogueTokenizer
```

**New way:**
```python
from lsm.training.train import LSMTrainer, run_training
from lsm.data.data_loader import DialogueTokenizer
```

## Complete Import Migration Table

| Legacy Import | New Advanced Import | Convenience Alternative | Notes |
|---------------|---------------------|------------------------|-------|
| `from train import LSMTrainer` | `from lsm.training.train import LSMTrainer` | `from lsm import LSMGenerator` | Use convenience for most cases |
| `from data_loader import load_data` | `from lsm.data.data_loader import load_data` | Built into convenience API | Automatic data handling |
| `from data_loader import DialogueTokenizer` | `from lsm.data.tokenization import StandardTokenizerWrapper` | Built into convenience API | Auto tokenizer selection |
| `from reservoir_manager import ReservoirManager` | `from lsm.core.reservoir_manager import ReservoirManager` | Built into convenience API | Configurable via parameters |
| `from response_generator import ResponseGenerator` | `from lsm.inference.response_generator import ResponseGenerator` | Built into convenience API | Integrated generation |
| `from system_message_processor import SystemMessageProcessor` | `from lsm.core.system_message_processor import SystemMessageProcessor` | Built into convenience API | Enable with `system_message_support=True` |
| `from cnn_3d_processor import CNN3DProcessor` | `from lsm.core.cnn_3d_processor import CNN3DProcessor` | Built into convenience API | Auto-configured |
| `from embedding_modifier_generator import EmbeddingModifierGenerator` | `from lsm.core.embedding_modifier_generator import EmbeddingModifierGenerator` | Built into convenience API | Part of system message processing |

## Command Line Interface

### Old CLI (main.py)

```bash
python main.py train --window-size 10 --epochs 50
python main.py evaluate --model-path ./model
```

### New CLI

```bash
python lsm_cli.py train-generator --data-path conversations.txt --epochs 50
python lsm_cli.py generate --model-path ./model --prompt "Hello!"
```

Or using the convenience module:
```bash
python -c "from lsm.convenience.cli import main; main()" train-generator --help
```

## Examples Migration

### Legacy Examples

The following root-level scripts have been replaced:

- `demonstrate_predictions.py` → `examples/convenience_prediction_demo.py`
- `performance_demo.py` → `examples/convenience_performance_demo.py`
- `show_examples.py` → `examples/convenience_dialogue_examples.py`

### Running New Examples

```bash
# Prediction demonstration
python examples/convenience_prediction_demo.py --preset fast

# Performance benchmarking
python examples/convenience_performance_demo.py

# Dialogue processing examples
python examples/convenience_dialogue_examples.py --window-size 5
```

## Convenience API Benefits

The new convenience API provides:

1. **Sklearn-compatible interface** - Familiar `fit()`, `predict()`, `save()`, `load()` methods
2. **Intelligent defaults** - Preset configurations for different use cases
3. **Simplified data handling** - Automatic preprocessing and format conversion
4. **Better error messages** - Clear guidance when things go wrong
5. **Integrated functionality** - All components work together seamlessly

### Preset Configurations

```python
from lsm import LSMGenerator

# Fast training for experimentation
generator = LSMGenerator(preset='fast')

# Balanced performance and quality
generator = LSMGenerator(preset='balanced')

# Maximum quality for production
generator = LSMGenerator(preset='quality')
```

## Backward Compatibility

### Deprecation Warnings

Legacy imports will show deprecation warnings but continue to work:

```python
# This still works but shows a warning
from train import LSMTrainer  # DeprecationWarning: Use src.lsm.training.train
```

### Compatibility Shims

Backward compatibility shims are provided for:
- `train.py` → `src/lsm/training/train.py`
- `data_loader.py` → `src/lsm/data/data_loader.py`

## Migration Steps

### Step 1: Update Simple Scripts

For simple training/inference scripts, switch to the convenience API:

```python
# Before
from train import LSMTrainer
from data_loader import load_data

# After
from lsm import LSMGenerator
```

### Step 2: Update Advanced Scripts

For scripts using advanced features, update imports:

```python
# Before
from train import LSMTrainer
from data_loader import DialogueTokenizer

# After
from lsm.training.train import LSMTrainer
from lsm.data.data_loader import DialogueTokenizer
```

### Step 3: Update CLI Usage

Replace `main.py` usage with `lsm_cli.py`:

```bash
# Before
python main.py train --window-size 10

# After
python lsm_cli.py train-generator --window-size 10
```

### Step 4: Test and Validate

1. Run your updated scripts
2. Check for deprecation warnings
3. Verify functionality works as expected
4. Update any remaining legacy imports

## Troubleshooting

### Import Errors

If you get import errors:

1. **Check Python path** - Ensure `src/` is in your Python path
2. **Install dependencies** - Run `pip install -r requirements.txt`
3. **Use absolute imports** - Prefer `from lsm.module import Class`

### Missing Functionality

If some functionality seems missing:

1. **Check convenience API** - Many features are built-in now
2. **Use advanced imports** - Import directly from submodules if needed
3. **Check examples** - New examples show recommended patterns

### Performance Issues

If you notice performance changes:

1. **Try different presets** - `fast`, `balanced`, `quality`
2. **Check configuration** - Ensure parameters match your old setup
3. **Use profiling** - Run `examples/convenience_performance_demo.py`

## Getting Help

- **Examples** - Check `examples/` directory for usage patterns
- **Documentation** - See `docs/` for detailed API documentation
- **Issues** - Report problems with specific error messages

## Detailed Migration Examples

### Example 1: Basic Training Script

**Before (Legacy):**
```python
#!/usr/bin/env python3
import sys
from train import LSMTrainer
from data_loader import load_data, DialogueTokenizer

def main():
    # Load data
    X_train, y_train, X_test, y_test, tokenizer = load_data(
        data_path="conversations.txt",
        window_size=10
    )
    
    # Create trainer
    trainer = LSMTrainer(
        window_size=10,
        embedding_dim=128,
        n_reservoir=500,
        spectral_radius=0.9
    )
    
    # Train model
    trainer.train(X_train, y_train, X_test, y_test, epochs=50)
    
    # Save model
    trainer.save_model("my_model")

if __name__ == "__main__":
    main()
```

**After (Convenience API):**
```python
#!/usr/bin/env python3
from lsm import LSMGenerator

def main():
    # Load conversations (automatic format detection)
    with open("conversations.txt", "r") as f:
        conversations = f.read().split("\n\n")  # Split by double newline
    
    # Create and train generator
    generator = LSMGenerator(
        window_size=10,
        embedding_dim=128,
        preset='balanced'  # Intelligent defaults
    )
    
    generator.fit(conversations, epochs=50, validation_split=0.2)
    
    # Save model
    generator.save("my_model")
    
    # Test generation
    response = generator.generate("Hello, how are you?")
    print(f"Generated: {response}")

if __name__ == "__main__":
    main()
```

### Example 2: Classification Task

**Before (Legacy):**
```python
from train import LSMTrainer
from data_loader import load_classification_data
from sklearn.linear_model import LogisticRegression
import numpy as np

# Load and prepare data
texts, labels = load_classification_data("sentiment_data.csv")
trainer = LSMTrainer(window_size=8, embedding_dim=64)

# Extract features using LSM
features = []
for text in texts:
    # Complex feature extraction process
    tokenized = trainer.tokenizer.encode(text)
    reservoir_state = trainer.get_reservoir_state(tokenized)
    features.append(reservoir_state.flatten())

features = np.array(features)

# Train classifier
classifier = LogisticRegression()
classifier.fit(features, labels)

# Make predictions
predictions = classifier.predict(features)
```

**After (Convenience API):**
```python
from lsm import LSMClassifier

# Load data
texts = ["I love this!", "This is terrible", "It's okay"]
labels = ["positive", "negative", "neutral"]

# Create and train classifier
classifier = LSMClassifier(
    window_size=8,
    embedding_dim=64,
    classifier_type='logistic'
)

classifier.fit(texts, labels, epochs=30)

# Make predictions
predictions = classifier.predict(texts)
probabilities = classifier.predict_proba(texts)

print("Predictions:", predictions)
print("Probabilities:", probabilities)
```

### Example 3: Advanced Custom Configuration

**Before (Legacy):**
```python
from train import LSMTrainer
from reservoir_manager import ReservoirManager
from system_message_processor import SystemMessageProcessor
from response_generator import ResponseGenerator

# Complex setup
reservoir = ReservoirManager(
    reservoir_type='hierarchical',
    n_reservoir=1000,
    hierarchy_levels=3
)

system_processor = SystemMessageProcessor(
    embedding_dim=256,
    context_length=50
)

trainer = LSMTrainer(
    reservoir_manager=reservoir,
    system_processor=system_processor,
    window_size=15,
    embedding_dim=256
)

response_gen = ResponseGenerator(
    trainer=trainer,
    max_length=100,
    temperature=0.8
)

# Training and inference...
```

**After (Advanced API):**
```python
from lsm.training.train import LSMTrainer
from lsm.core.reservoir_manager import ReservoirManager
from lsm.core.system_message_processor import SystemMessageProcessor
from lsm.inference.response_generator import ResponseGenerator

# Same advanced configuration with new imports
reservoir = ReservoirManager(
    reservoir_type='hierarchical',
    n_reservoir=1000,
    hierarchy_levels=3
)

system_processor = SystemMessageProcessor(
    embedding_dim=256,
    context_length=50
)

trainer = LSMTrainer(
    reservoir_manager=reservoir,
    system_processor=system_processor,
    window_size=15,
    embedding_dim=256
)

response_gen = ResponseGenerator(
    trainer=trainer,
    max_length=100,
    temperature=0.8
)
```

**Or (Convenience API with Custom Config):**
```python
from lsm import LSMGenerator

# Simplified but still powerful
generator = LSMGenerator(
    window_size=15,
    embedding_dim=256,
    reservoir_type='hierarchical',
    reservoir_config={
        'n_reservoir': 1000,
        'hierarchy_levels': 3
    },
    system_message_support=True,
    system_config={
        'context_length': 50
    }
)

# Same functionality, much simpler interface
generator.fit(conversations)
response = generator.generate(
    "Hello!",
    system_message="You are a helpful assistant",
    max_length=100,
    temperature=0.8
)
```

## Feature Mapping

### Training Features

| Legacy Feature | Convenience API | Advanced API |
|----------------|-----------------|--------------|
| `LSMTrainer.train()` | `LSMGenerator.fit()` | `LSMTrainer.train()` |
| Manual data loading | Automatic format detection | `DataLoader` classes |
| Manual tokenization | Built-in tokenizer selection | `StandardTokenizerWrapper` |
| Manual validation split | `validation_split` parameter | Manual splitting |
| Custom loss functions | Built-in optimization | Custom loss functions |

### Inference Features

| Legacy Feature | Convenience API | Advanced API |
|----------------|-----------------|--------------|
| `ResponseGenerator.generate()` | `LSMGenerator.generate()` | `ResponseGenerator.generate()` |
| Manual system message handling | `system_message` parameter | `SystemMessageProcessor` |
| Manual batch processing | `batch_generate()` method | Manual batching |
| Temperature control | `temperature` parameter | Full generation config |

### Model Management

| Legacy Feature | Convenience API | Advanced API |
|----------------|-----------------|--------------|
| Manual model saving | `save()` method | Component-level saving |
| Manual model loading | `load()` class method | Component-level loading |
| Configuration management | Preset system | Manual configuration |
| Model validation | Automatic validation | Manual validation |

## Data Format Migration

### Conversation Data

**Legacy Format:**
```python
# Required specific preprocessing
X_train = [tokenized_sequences]
y_train = [target_sequences]
```

**Convenience API (Multiple Formats Supported):**
```python
# Simple string list
conversations = ["Hello", "Hi there", "How are you?"]

# Structured format
conversations = [
    {"messages": ["Hello", "Hi"], "system": "Be friendly"},
    {"messages": ["Help me", "Sure"], "system": "Be helpful"}
]

# Raw dialogue format (auto-detected)
conversations = [
    "User: Hello\nAssistant: Hi there!\nUser: How are you?\nAssistant: I'm good!"
]

# File-based (automatic loading)
generator.fit("conversations.txt")  # Auto-detects format
```

### Classification Data

**Legacy Format:**
```python
# Manual preprocessing required
texts = preprocess_texts(raw_texts)
labels = encode_labels(raw_labels)
```

**Convenience API:**
```python
# Direct usage, automatic preprocessing
texts = ["I love this!", "This is terrible", "It's okay"]
labels = ["positive", "negative", "neutral"]  # String labels OK
# or
labels = [1, 0, 2]  # Numeric labels OK
```

## Performance Considerations

### Memory Usage

**Legacy Approach:**
```python
# Manual memory management required
import gc
import tensorflow as tf

# Clear memory manually
gc.collect()
tf.keras.backend.clear_session()

# Adjust batch sizes manually
trainer = LSMTrainer(batch_size=8)  # Trial and error
```

**Convenience API:**
```python
# Automatic memory management
generator = LSMGenerator(auto_memory_management=True)

# Or use presets optimized for your hardware
generator = LSMGenerator(preset='fast')  # Lower memory usage
```

### Training Speed

**Legacy Approach:**
```python
# Manual optimization
trainer = LSMTrainer(
    batch_size=32,  # Manual tuning
    learning_rate=0.001,  # Manual tuning
    gradient_accumulation_steps=4  # Manual setup
)
```

**Convenience API:**
```python
# Optimized presets
generator = LSMGenerator(preset='fast')  # Speed-optimized
generator = LSMGenerator(preset='balanced')  # Balanced
generator = LSMGenerator(preset='quality')  # Quality-optimized
```

## Error Handling Migration

### Legacy Error Handling

**Before:**
```python
try:
    trainer = LSMTrainer(window_size=-1)
except Exception as e:
    print(f"Error: {e}")  # Generic error message
```

**After (Convenience API):**
```python
from lsm.convenience import ConvenienceValidationError

try:
    generator = LSMGenerator(window_size=-1)
except ConvenienceValidationError as e:
    print(f"Error: {e}")
    print(f"Suggestion: {e.suggestion}")
    print(f"Valid options: {e.valid_options}")
```

## Testing Migration

### Legacy Testing

**Before:**
```python
import unittest
from train import LSMTrainer

class TestLSM(unittest.TestCase):
    def test_training(self):
        trainer = LSMTrainer()
        # Complex setup required
        X_train, y_train = prepare_data()
        trainer.train(X_train, y_train)
        self.assertTrue(trainer.is_trained)
```

**After (Convenience API):**
```python
import unittest
from lsm import LSMGenerator

class TestLSM(unittest.TestCase):
    def test_training(self):
        generator = LSMGenerator(preset='fast')
        conversations = ["Hello", "Hi there"]
        generator.fit(conversations, epochs=1)  # Quick test
        
        response = generator.generate("Test")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
```

## Deployment Migration

### Legacy Deployment

**Before:**
```python
# Complex deployment setup
from train import LSMTrainer
from response_generator import ResponseGenerator
import pickle

# Load components separately
with open('trainer.pkl', 'rb') as f:
    trainer = pickle.load(f)
with open('generator.pkl', 'rb') as f:
    generator = pickle.load(f)

# Manual setup
def predict(text):
    # Complex prediction pipeline
    tokenized = trainer.tokenizer.encode(text)
    features = trainer.get_features(tokenized)
    response = generator.generate(features)
    return response
```

**After (Convenience API):**
```python
# Simple deployment
from lsm import LSMGenerator

# Load complete model
generator = LSMGenerator.load("my_model")

# Simple prediction
def predict(text):
    return generator.generate(text)

# Or for web deployment
from flask import Flask, request, jsonify

app = Flask(__name__)
generator = LSMGenerator.load("my_model")

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.json['prompt']
    response = generator.generate(prompt)
    return jsonify({'response': response})
```

## Timeline and Deprecation Schedule

### Current Phase (v2.0+)
- ✅ Convenience API fully available
- ✅ Legacy imports work with deprecation warnings
- ✅ All examples updated to show both approaches
- ✅ Migration guide available

### Next Phase (v2.5+)
- 🔄 Convenience API becomes primary in documentation
- 🔄 Legacy imports show stronger warnings
- 🔄 New features primarily in convenience API

### Future Phase (v3.0+)
- ⏳ Legacy root-level imports may be removed
- ⏳ Full migration to src/ structure required
- ⏳ Convenience API becomes standard interface

### Recommendation Timeline

**Immediate (Now):**
- New projects: Use convenience API
- Existing projects: Start planning migration
- Learning: Focus on convenience API

**Short term (3-6 months):**
- Migrate simple scripts to convenience API
- Update imports to src/ structure
- Test new functionality

**Long term (6-12 months):**
- Complete migration of all projects
- Remove legacy import dependencies
- Adopt new best practices

We recommend migrating to the convenience API for new projects and gradually updating existing code. The convenience API provides the same functionality with much simpler usage patterns.