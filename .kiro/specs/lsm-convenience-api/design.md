# Design Document

## Overview

This design creates a scikit-learn-style convenience API for the LSM (Liquid State Machine) project while cleaning up legacy code. The solution provides a simple, familiar interface that abstracts the complexity of the current multi-component architecture while maintaining access to all advanced features.

The design addresses two main goals:
1. **Legacy Code Cleanup**: Remove redundant root-level scripts and consolidate functionality into the organized src/ structure
2. **Convenience API**: Create a sklearn-like interface that makes LSM models as easy to use as `LinearRegression` or `RandomForestClassifier`

## Architecture

### High-Level Design

The convenience API follows the familiar sklearn pattern with three main classes:

```python
from lsm import LSMClassifier, LSMRegressor, LSMGenerator

# Simple classification/regression
model = LSMClassifier()
model.fit(X, y)
predictions = model.predict(X_test)

# Text generation
generator = LSMGenerator()
generator.fit(conversations)
response = generator.generate("Hello, how are you?")
```

### Component Architecture

```
src/lsm/convenience/
├── __init__.py              # Main API exports
├── base.py                  # Base classes and common functionality
├── classifier.py            # LSMClassifier for classification tasks
├── regressor.py             # LSMRegressor for regression tasks  
├── generator.py             # LSMGenerator for text generation
├── config.py                # Configuration management and defaults
└── utils.py                 # Convenience utilities and helpers
```

## Components and Interfaces

### 1. Base Architecture (base.py)

**LSMBase Class**
- Abstract base class following sklearn's BaseEstimator pattern
- Handles parameter validation, model persistence, and common operations
- Manages the underlying LSMTrainer and inference components

```python
class LSMBase(BaseEstimator):
    def __init__(self, 
                 window_size=10,
                 embedding_dim=128, 
                 reservoir_type='standard',
                 reservoir_config=None,
                 random_state=None,
                 **kwargs):
        # Parameter storage and validation
        
    def fit(self, X, y=None, **fit_params):
        # Abstract method to be implemented by subclasses
        
    def predict(self, X):
        # Abstract method to be implemented by subclasses
        
    def save(self, path):
        # Simple model saving
        
    @classmethod
    def load(cls, path):
        # Simple model loading
        
    def get_params(self, deep=True):
        # sklearn-compatible parameter access
        
    def set_params(self, **params):
        # sklearn-compatible parameter setting
```

### 2. Text Generation Interface (generator.py)

**LSMGenerator Class**
- Primary interface for conversational AI and text generation
- Wraps LSMTrainer, ResponseGenerator, and SystemMessageProcessor
- Provides simple fit/generate interface

```python
class LSMGenerator(LSMBase):
    def __init__(self,
                 window_size=10,
                 embedding_dim=128,
                 reservoir_type='hierarchical',  # Better default for text
                 system_message_support=True,
                 response_level=True,
                 tokenizer='gpt2',
                 **kwargs):
        
    def fit(self, conversations, 
            system_messages=None,
            validation_split=0.2,
            epochs=50,
            batch_size=32,
            **fit_params):
        """
        Train on conversation data.
        
        Args:
            conversations: List of conversation strings or structured data
            system_messages: Optional system messages for training
            validation_split: Fraction for validation
            epochs: Training epochs
            batch_size: Training batch size
        """
        
    def generate(self, 
                 prompt, 
                 system_message=None,
                 max_length=50,
                 temperature=1.0):
        """
        Generate response to a prompt.
        
        Args:
            prompt: Input text or conversation history
            system_message: Optional system context
            max_length: Maximum response length
            temperature: Generation randomness
        """
        
    def chat(self, system_message=None):
        """Start interactive chat session."""
        
    def batch_generate(self, prompts, **kwargs):
        """Generate responses for multiple prompts."""
```

### 3. Classification Interface (classifier.py)

**LSMClassifier Class**
- For classification tasks using LSM features
- Extracts reservoir states as features for downstream classification

```python
class LSMClassifier(LSMBase, ClassifierMixin):
    def __init__(self,
                 window_size=10,
                 embedding_dim=128,
                 reservoir_type='standard',
                 n_classes=None,
                 **kwargs):
        
    def fit(self, X, y, **fit_params):
        """
        Train classifier on text data.
        
        Args:
            X: Text samples or sequences
            y: Class labels
        """
        
    def predict(self, X):
        """Predict classes for text samples."""
        
    def predict_proba(self, X):
        """Predict class probabilities."""
        
    def score(self, X, y):
        """Return accuracy score."""
```

### 4. Regression Interface (regressor.py)

**LSMRegressor Class**
- For regression tasks using LSM temporal dynamics
- Useful for time series prediction and continuous value estimation

```python
class LSMRegressor(LSMBase, RegressorMixin):
    def __init__(self,
                 window_size=10,
                 embedding_dim=128,
                 reservoir_type='echo_state',  # Good for time series
                 **kwargs):
        
    def fit(self, X, y, **fit_params):
        """Train regressor on sequential data."""
        
    def predict(self, X):
        """Predict continuous values."""
        
    def score(self, X, y):
        """Return R² score."""
```

### 5. Configuration Management (config.py)

**ConvenienceConfig Class**
- Manages intelligent defaults and parameter validation
- Provides preset configurations for common use cases

```python
class ConvenienceConfig:
    # Preset configurations
    PRESETS = {
        'fast': {
            'window_size': 5,
            'embedding_dim': 64,
            'reservoir_type': 'standard',
            'epochs': 10
        },
        'balanced': {
            'window_size': 10,
            'embedding_dim': 128,
            'reservoir_type': 'hierarchical',
            'epochs': 50
        },
        'quality': {
            'window_size': 20,
            'embedding_dim': 256,
            'reservoir_type': 'attentive',
            'epochs': 100
        }
    }
    
    @classmethod
    def get_preset(cls, name):
        """Get preset configuration."""
        
    @classmethod
    def validate_params(cls, params):
        """Validate parameter combinations."""
```

## Data Models

### Input Data Formats

The convenience API accepts multiple input formats and automatically handles conversion:

**Text Generation:**
```python
# Simple string list
conversations = ["Hello", "Hi there", "How are you?"]

# Structured conversation format
conversations = [
    {"messages": ["Hello", "Hi"], "system": "Be friendly"},
    {"messages": ["Help me", "Sure"], "system": "Be helpful"}
]

# Raw text with automatic conversation splitting
text_data = "User: Hello\nAssistant: Hi there\nUser: How are you?"
```

**Classification/Regression:**
```python
# Text samples with labels
X = ["This is positive", "This is negative", "Neutral text"]
y = [1, 0, 2]  # Classification labels

# Sequential data for regression
X = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]  # Time series
y = [4, 5, 6]  # Target values
```

### Model Persistence Format

Models are saved in a simplified format while maintaining compatibility with the full system:

```
model_name/
├── convenience_config.json    # Convenience API configuration
├── model/                     # Full LSM model (existing format)
│   ├── reservoir_model/
│   ├── cnn_model/
│   ├── tokenizer/
│   └── config.json
└── metadata.json             # Training metadata and performance
```

## Error Handling

### Validation and Error Messages

The convenience API provides clear, actionable error messages:

```python
class ConvenienceValidationError(ValueError):
    """Clear validation errors with suggestions."""
    
    def __init__(self, message, suggestion=None, valid_options=None):
        self.suggestion = suggestion
        self.valid_options = valid_options
        super().__init__(message)

# Example usage
if reservoir_type not in VALID_RESERVOIR_TYPES:
    raise ConvenienceValidationError(
        f"Invalid reservoir_type: {reservoir_type}",
        suggestion="Try 'hierarchical' for text generation",
        valid_options=VALID_RESERVOIR_TYPES
    )
```

### Automatic Error Recovery

The API includes automatic error recovery for common issues:

1. **Missing Dependencies**: Automatic fallback to simpler configurations
2. **Memory Issues**: Automatic batch size reduction
3. **Invalid Parameters**: Automatic parameter correction with warnings

## Testing Strategy

### Unit Tests

**Test Coverage Areas:**
1. **Parameter Validation**: All parameter combinations and edge cases
2. **Data Format Handling**: Various input formats and automatic conversion
3. **Model Persistence**: Save/load functionality across different configurations
4. **Error Handling**: Validation errors and recovery mechanisms
5. **sklearn Compatibility**: Integration with sklearn pipelines and utilities

**Test Structure:**
```
tests/test_convenience/
├── test_base.py              # Base class functionality
├── test_generator.py         # Text generation interface
├── test_classifier.py        # Classification interface
├── test_regressor.py         # Regression interface
├── test_config.py            # Configuration management
├── test_integration.py       # Integration with existing system
└── test_sklearn_compat.py    # sklearn compatibility
```

### Integration Tests

**End-to-End Workflows:**
1. **Simple Generation**: Basic text generation workflow
2. **Advanced Generation**: System messages and batch processing
3. **Classification Pipeline**: Text classification with preprocessing
4. **sklearn Integration**: Use in sklearn pipelines and cross-validation

### Performance Tests

**Benchmarking:**
1. **API Overhead**: Measure convenience layer performance impact
2. **Memory Usage**: Monitor memory efficiency compared to direct API
3. **Training Speed**: Compare training times across interfaces

## Legacy Code Cleanup Plan

### Files to Remove

**Root-level Python files that duplicate src/ functionality:**
1. `main.py` → Functionality moved to `src/lsm/convenience/cli.py`
2. `demonstrate_predictions.py` → Functionality moved to examples
3. `performance_demo.py` → Functionality moved to examples
4. `show_examples.py` → Functionality moved to examples
5. `test_*.py` files → Moved to appropriate test directories
6. `task_8_3_methods.py` → Remove if obsolete
7. `check_*.py` files → Move to development utilities or remove

**Legacy imports to update:**
- All imports of root-level `train` and `data_loader` modules
- Update to use `src.lsm.training.train` and `src.lsm.data.data_loader`

### Consolidation Strategy

**Phase 1: Identify and Categorize**
1. Scan all root-level Python files
2. Identify functionality that exists in src/ structure
3. Categorize as: duplicate, obsolete, or unique functionality

**Phase 2: Migrate Unique Functionality**
1. Move unique functionality to appropriate src/ modules
2. Update imports and dependencies
3. Create migration guide for users

**Phase 3: Update Examples and Documentation**
1. Update all examples to use convenience API
2. Provide both convenience and advanced API examples
3. Update documentation to feature convenience API prominently

### Backward Compatibility

**Deprecation Strategy:**
1. Keep existing advanced API fully functional
2. Add deprecation warnings to root-level imports
3. Provide clear migration paths in documentation
4. Maintain backward compatibility for at least 2 major versions

## Implementation Phases

### Phase 1: Core Convenience API (Week 1-2)
1. Implement `LSMBase` class with parameter management
2. Create `LSMGenerator` with basic fit/generate functionality
3. Implement model save/load with convenience format
4. Add basic parameter validation and error handling

### Phase 2: Extended Interfaces (Week 2-3)
1. Implement `LSMClassifier` and `LSMRegressor`
2. Add preset configurations and intelligent defaults
3. Implement batch processing and advanced features
4. Add comprehensive input format handling

### Phase 3: Legacy Cleanup (Week 3-4)
1. Identify and categorize all legacy code
2. Migrate unique functionality to src/ structure
3. Remove duplicate root-level files
4. Update all examples and documentation

### Phase 4: Integration and Testing (Week 4-5)
1. Comprehensive test suite for convenience API
2. sklearn compatibility testing
3. Performance benchmarking
4. Documentation and examples update

### Phase 5: Polish and Documentation (Week 5-6)
1. Error message improvement and user experience polish
2. Complete documentation with tutorials
3. Migration guide for existing users
4. Final testing and validation