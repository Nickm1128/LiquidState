# Sparse Sine-Activated Liquid State Machine for Conversational AI

This project implements a novel neural architecture that combines **Liquid State Machines (LSM)** with **Convolutional Neural Networks (CNN)** for advanced conversational AI and response generation. The system uses sparse connectivity patterns and parametric sine activation functions to create complex temporal dynamics, supporting both next-token prediction and complete response generation with system message integration.

## Overview

### What is a Liquid State Machine?
A Liquid State Machine is a type of recurrent neural network inspired by biological neural circuits. Unlike traditional RNNs, LSMs maintain a "reservoir" of randomly connected neurons that create rich temporal dynamics. This project extends the concept with several key innovations:

1. **Sparse Connectivity**: Only a fraction of connections exist between neurons, making the network more efficient and biologically plausible
2. **Parametric Sine Activation**: Learnable sine functions: `A * exp(-α * |x|) * sin(ω * x)` optimized for natural language patterns
3. **Rolling Wave Encoding**: LSM outputs are encoded as 2D/3D spatial-temporal patterns
4. **Multi-Modal CNN Processing**: 2D and 3D CNNs interpret these patterns for both token and response-level prediction
5. **System Message Integration**: Advanced system message processing for context-aware responses
6. **HuggingFace Integration**: Native support for modern datasets and tokenizers

### Architecture Components

#### Core Components
1. **HuggingFace Dataset Integration**: Downloads and processes cosmopedia-v2 and other conversational datasets
2. **Advanced Tokenization System**: StandardTokenizerWrapper with SinusoidalEmbedder for optimized embeddings
3. **Sparse Reservoir Network**: Multiple reservoir types (standard, hierarchical, attentive, echo state, deep)
4. **Multi-Dimensional CNN Processing**: 2D and 3D CNN architectures with attention mechanisms
5. **System Message Processor**: Handles system prompts and context integration
6. **Response Generation System**: Complete response-level inference with ReservoirManager

#### Enhanced Features
7. **Message Annotation System**: Conversation flow markers and metadata handling
8. **Pipeline Orchestration**: Modular architecture for component swapping and experimentation
9. **Colab Compatibility**: Optimized for Google Colab deployment and experimentation
10. **Production Monitoring**: Comprehensive logging, validation, and performance monitoring
11. **Model Management**: Advanced model storage, discovery, and validation utilities

## Installation

### Requirements
- Python 3.11+
- TensorFlow 2.10+
- CUDA support recommended for GPU acceleration
- Optional: psutil for memory monitoring

### Setup
1. Clone or download this project
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Alternative Installation Methods

#### Using uv (Recommended)
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -r requirements.txt
```

#### Development Installation
```bash
# Install in development mode with all optional dependencies
pip install -e .
pip install psutil matplotlib seaborn pytest
```

## Quick Start Guide

### 🚀 Convenience API (Recommended for New Users)

The LSM project now includes a **scikit-learn-compatible convenience API** that makes it easy to get started without dealing with the complexity of the underlying architecture.

#### Simple Text Generation
```python
from lsm import LSMGenerator

# Create and train a generator with intelligent defaults
generator = LSMGenerator(preset='fast')  # or 'balanced', 'quality'
generator.fit(conversations)

# Generate responses
response = generator.generate("Hello, how are you?")
print(response)

# Interactive chat
generator.chat()  # Start interactive session
```

#### Classification Tasks
```python
from lsm import LSMClassifier

# Train a classifier
classifier = LSMClassifier(preset='balanced')
classifier.fit(texts, labels)

# Make predictions
predictions = classifier.predict(new_texts)
probabilities = classifier.predict_proba(new_texts)
```

#### Command Line Interface
```bash
# Train a text generator
python lsm_cli.py train-generator --data-path conversations.txt --preset balanced

# Generate responses
python lsm_cli.py generate --model-path ./model --prompt "Hello there!" --interactive

# Train a classifier
python lsm_cli.py train-classifier --data-path data.csv --preset quality
```

#### Examples and Demos
```bash
# See prediction examples
python examples/convenience_prediction_demo.py --preset fast

# Performance benchmarking
python examples/convenience_performance_demo.py

# Dialogue processing demonstration
python examples/convenience_dialogue_examples.py
```

> **New to LSM?** Start with the [Getting Started Tutorial](docs/GETTING_STARTED_TUTORIAL.md) for a comprehensive introduction.
> 
> **Migration Note**: If you're upgrading from the legacy interface, see [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for help updating your code.

### Advanced Training (Direct API)

#### Basic Training
```bash
# Train with default settings (uses HuggingFace cosmopedia-v2 dataset)
python main.py train

# Train with specific parameters
python main.py train --window-size 8 --batch-size 16 --epochs 10

# Train with advanced reservoir configuration
python main.py train --reservoir-type hierarchical --sparsity 0.15
```

#### Advanced Training Options
```bash
# Train with 3D CNN and system message support
python main.py train --reservoir-type attentive --use-attention --embedding-dim 256

# Train with custom reservoir configuration
python main.py train --reservoir-config '{"layers": [512, 256, 128], "dropout": 0.2}'

# Train with specific dataset configuration
python main.py train --test-size 0.15 --batch-size 64 --epochs 50
```

#### Available Reservoir Types
- `standard`: Basic sparse reservoir with sine activation
- `hierarchical`: Multi-layer hierarchical reservoir
- `attentive`: Attention-enhanced reservoir
- `echo_state`: Echo State Network variant
- `deep`: Deep reservoir with multiple processing layers

### Using Trained Models for Inference

The enhanced inference system provides multiple ways to interact with trained models, supporting both token-level and response-level generation:

#### Interactive Mode
Start an interactive session for continuous dialogue:
```bash
# Enhanced inference with system message support
python -m src.lsm.inference.inference --model-path ./models_20250107_143022 --interactive

# Legacy inference for older models
python inference.py --model-path ./models_20250107_143022 --interactive
```

#### Response-Level Generation
Generate complete responses instead of individual tokens:
```bash
# Generate complete responses with system context
python -m src.lsm.inference.response_generator --model-path ./models_20250107_143022 \
  --system-message "You are a helpful assistant" \
  --conversation "Hello" "How can I help you today?"

# Batch response generation
python -m src.lsm.inference.response_generator --model-path ./models_20250107_143022 \
  --batch-file conversations.json --output-file responses.json
```

#### System Message Integration
Use system messages to guide response generation:
```bash
# With system message context
python -m src.lsm.inference.inference --model-path ./models_20250107_143022 \
  --system-message "You are a technical expert" \
  --input-text "Explain neural networks"

# Multiple system contexts
python -m src.lsm.inference.inference --model-path ./models_20250107_143022 \
  --system-contexts "technical,friendly,concise" \
  --input-text "What is machine learning?"
```

#### Performance Optimized Mode
Use optimized inference for production environments:
```bash
python -m src.lsm.inference.inference --model-path ./models_20250107_143022 \
  --optimized --cache-size 2000 --batch-size 64 --lazy-load
```

### Model Management

#### List Available Models
```bash
# List all models in current directory
python -m src.lsm.management.manage_models list

# List models in specific directory
python -m src.lsm.management.manage_models list --models-dir ./saved_models
```

#### Get Model Information
```bash
# Detailed model information
python -m src.lsm.management.manage_models info --model-path ./models_20250107_143022

# Model summary
python -m src.lsm.management.manage_models summary --model-path ./models_20250107_143022
```

#### Validate Model Integrity
```bash
# Validate single model
python -m src.lsm.management.manage_models validate --model-path ./models_20250107_143022

# Validate all models
python -m src.lsm.management.manage_models validate-all
```

#### Clean Up and Maintenance
```bash
# Clean up incomplete models (dry run)
python -m src.lsm.management.manage_models cleanup --dry-run

# Actually remove incomplete models
python -m src.lsm.management.manage_models cleanup --confirm

# Archive old models
python -m src.lsm.management.manage_models archive --older-than 30
```

## Enhanced Inference Capabilities

### Complete Text Processing Pipeline

The enhanced inference system provides multiple processing modes with advanced features:

#### Core Features
- **StandardTokenizerWrapper Integration**: Modern tokenization with GPT-2/BERT compatibility
- **SinusoidalEmbedder**: Optimized embeddings for sine-activated LSM architecture
- **Response-Level Generation**: Complete response generation instead of token-by-token
- **System Message Processing**: Context-aware responses with system prompt integration
- **3D CNN Support**: Advanced spatial-temporal pattern processing
- **Automatic Model Persistence**: Complete model state saving and loading

#### Advanced Capabilities
- **Multi-Modal Processing**: Support for both 2D and 3D CNN architectures
- **Reservoir Strategy Management**: Intelligent reservoir reuse and coordination
- **Message Annotation**: Conversation flow markers and metadata handling
- **Pipeline Orchestration**: Modular component swapping and experimentation
- **Production Monitoring**: Comprehensive logging and performance tracking
- **Memory Optimization**: Lazy loading, caching, and memory management

### Inference Modes

#### 1. Enhanced Inference (Recommended)
Modern inference with full feature support:
```python
from src.lsm.inference.inference import EnhancedLSMInference

# Initialize enhanced inference
inference = EnhancedLSMInference("./models_20250107_143022")

# Interactive session with system message support
inference.interactive_session()

# Response-level generation
response = inference.generate_response(
    conversation=["Hello", "How are you?"],
    system_message="You are a helpful assistant"
)
print(f"Generated response: {response}")
```

#### 2. Response Generation API
For complete response generation:
```python
from src.lsm.inference.response_generator import ResponseGenerator

# Initialize response generator
generator = ResponseGenerator("./models_20250107_143022")

# Generate complete responses
result = generator.generate_response(
    conversation=["What's the weather like?"],
    system_context="You are a weather expert",
    max_length=50
)

print(f"Response: {result.response}")
print(f"Confidence: {result.confidence:.3f}")
print(f"Generation time: {result.generation_time:.2f}s")
```

#### 3. System Message Processing
Advanced context-aware generation:
```python
from src.lsm.core.system_message_processor import SystemMessageProcessor

# Initialize system message processor
processor = SystemMessageProcessor()

# Process system messages
context = processor.process_system_message(
    "You are a technical expert specializing in AI",
    tokenizer_wrapper=inference.tokenizer
)

# Generate with system context
response = inference.generate_with_system_context(
    conversation=["Explain neural networks"],
    system_context=context
)
```

#### 4. Batch Processing
Efficient processing of multiple conversations:
```python
from src.lsm.inference.inference import EnhancedLSMInference

inference = EnhancedLSMInference("./models_20250107_143022")

# Batch response generation
conversations = [
    {"conversation": ["Hello"], "system": "Be friendly"},
    {"conversation": ["What's AI?"], "system": "Be technical"},
    {"conversation": ["Tell a joke"], "system": "Be humorous"}
]

results = inference.batch_generate_responses(conversations)
for result in results:
    print(f"Response: {result.response}")
```

#### 5. Legacy Compatibility
For backward compatibility with older models:
```python
from src.lsm.inference.inference import LSMInference

# Use legacy interface for older models
inference = LSMInference("./old_model_directory")
inference.interactive_session()
```

### Model Storage Structure

Enhanced models are stored with a comprehensive directory structure:
```
models_YYYYMMDD_HHMMSS/
├── reservoir_model/           # Keras reservoir model files
├── cnn_model/                # Keras CNN model (2D or 3D)
├── response_model/           # Response-level inference model
├── tokenizer/                # StandardTokenizerWrapper components
│   ├── tokenizer.pkl         # Serialized tokenizer
│   ├── embedder.pkl          # SinusoidalEmbedder
│   ├── vocab_mapping.json    # Vocabulary mappings
│   └── config.json          # Tokenizer configuration
├── system_processor/         # System message processing
│   ├── processor.pkl         # SystemMessageProcessor
│   ├── modifier_generator.pkl # EmbeddingModifierGenerator
│   └── system_config.json    # System processing configuration
├── config.json              # Complete model configuration
├── metadata.json            # Training metadata & performance
├── training_history.csv     # Training metrics history
├── validation_results.json  # Model validation results
└── inference_cache/         # Optional: cached embeddings and responses
    ├── embedding_cache.pkl   # Cached embeddings
    └── response_cache.pkl    # Cached responses
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

## Project Structure

### Directory Organization

```
src/lsm/                      # Main package directory
├── core/                     # Core LSM components
│   ├── reservoir.py          # Basic reservoir implementations
│   ├── advanced_reservoir.py # Advanced reservoir architectures
│   ├── cnn_model.py         # 2D CNN processing
│   ├── cnn_3d_processor.py  # 3D CNN with system message support
│   ├── cnn_architecture_factory.py # CNN factory for different architectures
│   ├── rolling_wave.py      # Temporal-spatial encoding
│   ├── system_message_processor.py # System message handling
│   ├── embedding_modifier_generator.py # System context modifiers
│   └── loss_functions.py    # Custom loss functions (cosine similarity)
├── data/                    # Data loading and processing
│   ├── data_loader.py       # Legacy data loading
│   ├── huggingface_loader.py # HuggingFace dataset integration
│   ├── tokenization.py      # StandardTokenizerWrapper & SinusoidalEmbedder
│   └── message_annotator.py # Message annotation system
├── training/                # Training pipeline
│   ├── train.py            # Enhanced LSMTrainer
│   ├── model_config.py     # Configuration management
│   └── train_backup.py     # Backup training utilities
├── inference/              # Inference systems
│   ├── inference.py        # Enhanced inference with system messages
│   ├── response_generator.py # Complete response generation
│   ├── reservoir_manager.py # Reservoir strategy management
│   └── response_inference_model.py # Response-level model
├── management/             # Model management utilities
│   ├── model_manager.py    # Model discovery and validation
│   └── manage_models.py    # CLI model management
├── pipeline/               # Pipeline orchestration
│   ├── pipeline_orchestrator.py # Modular architecture coordinator
│   └── colab_compatibility.py # Google Colab optimizations
└── utils/                  # Utilities and helpers
    ├── lsm_exceptions.py   # Custom exception classes
    ├── lsm_logging.py      # Logging and performance monitoring
    ├── input_validation.py # Input validation utilities
    ├── production_validation.py # Production readiness checks
    └── tensorflow_compat.py # TensorFlow compatibility helpers

examples/                   # Example scripts and demonstrations
├── basic_inference.py      # Basic inference examples
├── enhanced_inference_demo.py # Enhanced inference demonstration
├── response_generator_demo.py # Response generation examples
├── system_message_processor_demo.py # System message examples
├── tokenization_demo.py    # Tokenization system examples
├── huggingface_dataset_demo.py # Dataset integration examples
└── model_management.py     # Model management examples

docs/                      # Comprehensive documentation
├── API_DOCUMENTATION.md   # Complete API reference
├── TROUBLESHOOTING_GUIDE.md # Common issues and solutions
├── DEPLOYMENT_GUIDE.md    # Production deployment guide
├── PERFORMANCE_OPTIMIZATION_SUMMARY.md # Performance tuning
├── HUGGINGFACE_INTEGRATION_SUMMARY.md # Dataset integration guide
└── LSM_TRAINING_TECHNICAL_SPECIFICATION.md # Technical specifications

tests/                     # Test suite
├── test_*.py             # Unit tests for all components
└── integration/          # Integration tests
```

### Key Components Overview

#### Core Architecture
- **ReservoirLayer**: Basic sparse reservoir with sine activation
- **AdvancedReservoir**: Hierarchical, attentive, and deep reservoir variants
- **CNN3DProcessor**: 3D CNN with system message integration
- **SystemMessageProcessor**: Standalone system message handling
- **EmbeddingModifierGenerator**: System context influence on embeddings

#### Data Processing
- **HuggingFaceDatasetLoader**: Modern dataset integration
- **StandardTokenizerWrapper**: GPT-2/BERT compatible tokenization
- **SinusoidalEmbedder**: Optimized embeddings for sine activation
- **MessageAnnotator**: Conversation flow and metadata handling

#### Training & Inference
- **LSMTrainer**: Enhanced training with response-level optimization
- **EnhancedLSMInference**: Modern inference with system message support
- **ResponseGenerator**: Complete response generation orchestrator
- **ReservoirManager**: Intelligent reservoir strategy management

#### Management & Utilities
- **ModelManager**: Model discovery, validation, and maintenance
- **PipelineOrchestrator**: Modular architecture experimentation
- **ColabCompatibilityManager**: Google Colab deployment optimizations

## API Reference

### EnhancedLSMInference Class

The main inference class with full feature support including system messages and response generation.

#### Constructor
```python
EnhancedLSMInference(
    model_path: str,
    lazy_load: bool = True,
    cache_size: int = 1000,
    max_batch_size: int = 32,
    enable_system_messages: bool = True
)
```

#### Core Methods

**generate_response(conversation: List[str], system_message: Optional[str] = None, max_length: int = 50) -> str**
- Generate complete response for a conversation
- Supports system message context
- Returns generated response text

**predict_next_token(dialogue_sequence: List[str]) -> str**
- Predict the next token for a dialogue sequence (legacy compatibility)
- Returns the most likely next token as text

**generate_with_system_context(conversation: List[str], system_context: SystemMessageContext) -> str**
- Generate response with pre-processed system context
- Advanced system message integration
- Returns context-aware response

**batch_generate_responses(conversations: List[Dict[str, Any]], batch_size: Optional[int] = None) -> List[ResponseGenerationResult]**
- Process multiple conversations with system contexts
- Returns list of ResponseGenerationResult objects

**interactive_session()**
- Start interactive dialogue session with system message support
- Includes performance monitoring and help commands

#### System Message Methods

**process_system_message(system_message: str) -> SystemMessageContext**
- Process and validate system message
- Returns SystemMessageContext for reuse

**set_default_system_message(system_message: str)**
- Set default system message for all generations
- Useful for consistent context

**clear_system_context()**
- Clear current system message context
- Reset to default behavior

#### Performance and Monitoring

**get_model_info() -> Dict[str, Any]**
- Get comprehensive model information
- Includes configuration, metadata, and performance stats

**get_cache_stats() -> Dict[str, Any]**
- Get cache performance statistics
- Useful for monitoring and optimization

**get_generation_stats() -> Dict[str, Any]**
- Get response generation statistics
- Includes timing, quality metrics, and system message usage

### ResponseGenerator Class

Advanced response generation orchestrator with system message support.

#### Constructor
```python
ResponseGenerator(
    model_path: str,
    reservoir_strategy: str = "reuse",
    enable_3d_cnn: bool = True,
    cache_responses: bool = True
)
```

#### Methods

**generate_response(conversation: List[str], system_context: Optional[str] = None, max_length: int = 50) -> ResponseGenerationResult**
- Generate complete response with full context
- Returns ResponseGenerationResult with response, confidence, and metadata

**batch_generate(conversations: List[Dict[str, Any]]) -> List[ResponseGenerationResult]**
- Efficient batch processing of multiple conversations
- Supports different system contexts per conversation

**set_reservoir_strategy(strategy: str)**
- Configure reservoir reuse strategy ("reuse", "separate", "adaptive")
- Affects performance and response quality

### ModelManager Class

Enhanced model discovery and management with validation.

#### Constructor
```python
ModelManager(models_root_dir: str = ".")
```

#### Methods

**list_available_models() -> List[Dict[str, Any]]**
- Scan for valid model directories with enhanced metadata
- Returns list of model information dictionaries

**get_model_info(model_path: str) -> Dict[str, Any]**
- Get detailed information about a specific model
- Includes configuration, metadata, system message support, and file information

**validate_model(model_path: str) -> Tuple[bool, List[str]]**
- Check model integrity and completeness including new components
- Returns (is_valid, error_list)

**validate_enhanced_model(model_path: str) -> Tuple[bool, List[str]]**
- Validate enhanced model with system message components
- Checks for StandardTokenizerWrapper, SinusoidalEmbedder, and SystemMessageProcessor

**cleanup_incomplete_models(dry_run: bool = True) -> List[str]**
- Find and optionally remove incomplete models
- Returns list of cleanup candidates

**archive_old_models(older_than_days: int = 30) -> List[str]**
- Archive models older than specified days
- Returns list of archived model paths

**get_model_summary(model_path: str) -> str**
- Get human-readable model summary with enhanced features
- Formatted for display purposes

## Configuration Management

### ModelConfiguration Class

Centralized configuration management for all model parameters including enhanced features.

```python
from src.lsm.training.model_config import ModelConfiguration

# Load configuration from saved model
config = ModelConfiguration.load("./models_20250107_143022/config.json")

# Access configuration parameters
print(f"Window size: {config.window_size}")
print(f"Embedding dimension: {config.embedding_dim}")
print(f"Reservoir type: {config.reservoir_type}")
print(f"System message support: {config.enable_system_messages}")
print(f"CNN architecture: {config.cnn_architecture}")

# Create new enhanced configuration
config = ModelConfiguration(
    # Basic parameters
    window_size=10,
    embedding_dim=128,
    reservoir_type="hierarchical",
    reservoir_config={"layers": [512, 256, 128], "dropout": 0.2},
    reservoir_units=[256, 128, 64],
    sparsity=0.1,
    
    # Enhanced features
    enable_system_messages=True,
    cnn_architecture="3d",
    use_sinusoidal_embedder=True,
    tokenizer_type="standard",
    
    # Training parameters
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    use_cosine_loss=True,
    
    # Dataset configuration
    dataset_name="cosmopedia-v2",
    conversation_aware_split=True
)

# Save configuration
config.save("./new_model/config.json")

# Validate configuration
is_valid, errors = config.validate()
if not is_valid:
    print(f"Configuration errors: {errors}")
```

### Advanced Configuration Options

#### Reservoir Configuration
```python
# Standard reservoir
reservoir_config = {
    "sparsity": 0.1,
    "spectral_radius": 0.9,
    "input_scaling": 1.0
}

# Hierarchical reservoir
reservoir_config = {
    "layers": [512, 256, 128],
    "layer_sparsity": [0.1, 0.15, 0.2],
    "inter_layer_connections": True,
    "dropout": 0.2
}

# Attentive reservoir
reservoir_config = {
    "attention_heads": 8,
    "attention_dim": 64,
    "use_self_attention": True,
    "attention_dropout": 0.1
}
```

#### System Message Configuration
```python
system_config = {
    "enable_system_messages": True,
    "system_embedding_dim": 128,
    "modifier_generator_layers": [256, 128],
    "system_attention_heads": 4,
    "max_system_length": 200
}
```

#### CNN Architecture Configuration
```python
cnn_config = {
    "architecture": "3d",  # "2d" or "3d"
    "filters": [64, 128, 256],
    "kernel_sizes": [(3, 3, 3), (3, 3, 3), (3, 3, 3)],
    "use_attention": True,
    "attention_type": "spatial",
    "dropout": 0.3
}
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

The `examples/` directory contains comprehensive demonstrations of all system capabilities:

### Basic Usage Examples
- `basic_inference.py`: Simple prediction examples with legacy compatibility
- `enhanced_inference_demo.py`: Modern inference with system message support
- `interactive_demo.py`: Interactive session demonstration
- `batch_processing.py`: Efficient batch processing examples

### Advanced Feature Examples
- `response_generator_demo.py`: Complete response generation examples
- `system_message_processor_demo.py`: System message processing and context handling
- `tokenization_demo.py`: StandardTokenizerWrapper and SinusoidalEmbedder usage
- `huggingface_dataset_demo.py`: Dataset integration and conversation-aware splitting

### Architecture and Training Examples
- `cnn_3d_processor_demo.py`: 3D CNN with system message integration
- `cnn_architecture_factory_demo.py`: Different CNN architecture configurations
- `reservoir_manager_demo.py`: Reservoir strategy management
- `pipeline_orchestrator_demo.py`: Modular architecture experimentation

### Management and Deployment Examples
- `model_management.py`: Model discovery, validation, and maintenance
- `performance_optimization.py`: Performance tuning and monitoring
- `colab_compatibility_demo.py`: Google Colab deployment and setup

### Specialized Examples
- `message_annotator_demo.py`: Conversation flow and metadata handling
- `embedding_modifier_generator_demo.py`: System context influence on embeddings
- `cosine_loss_demo.py`: Custom loss function usage
- `response_inference_model_demo.py`: Response-level model training and inference

### Running Examples
```bash
# Basic inference example
python examples/basic_inference.py --model-path ./models_20250107_143022

# Enhanced inference with system messages
python examples/enhanced_inference_demo.py --model-path ./models_20250107_143022

# Response generation demonstration
python examples/response_generator_demo.py --model-path ./models_20250107_143022

# System message processing
python examples/system_message_processor_demo.py

# HuggingFace dataset integration
python examples/huggingface_dataset_demo.py --dataset cosmopedia-v2
```

## Documentation

### Convenience API Documentation (New Users Start Here)

- **[Getting Started Tutorial](docs/GETTING_STARTED_TUTORIAL.md)**: Step-by-step tutorial for new users
- **[Convenience API Documentation](docs/CONVENIENCE_API_DOCUMENTATION.md)**: Complete reference for the simplified API
- **[Advanced Convenience Tutorial](docs/ADVANCED_CONVENIENCE_API_TUTORIAL.md)**: Advanced patterns and production usage
- **[Convenience API Troubleshooting](docs/CONVENIENCE_API_TROUBLESHOOTING.md)**: Solutions for convenience API issues
- **[Migration Guide](MIGRATION_GUIDE.md)**: Migrate from legacy code to convenience API

### Comprehensive Documentation Suite

- **[API Documentation](docs/API_DOCUMENTATION.md)**: Complete API reference for all classes and methods
- **[Troubleshooting Guide](docs/TROUBLESHOOTING_GUIDE.md)**: Solutions for common issues and problems
- **[Deployment Guide](docs/DEPLOYMENT_GUIDE.md)**: Production deployment and scaling guide
- **[Performance Optimization](docs/PERFORMANCE_OPTIMIZATION_SUMMARY.md)**: Performance tuning and monitoring
- **[HuggingFace Integration](docs/HUGGINGFACE_INTEGRATION_SUMMARY.md)**: Dataset integration and usage guide
- **[Technical Specification](docs/LSM_TRAINING_TECHNICAL_SPECIFICATION.md)**: Detailed technical specifications
- **[Enhancement Summary](docs/ENHANCEMENT_SUMMARY.md)**: Overview of recent enhancements and features
- **[Error Handling](docs/ERROR_HANDLING_SUMMARY.md)**: Comprehensive error handling and recovery
- **[Production Monitoring](docs/PRODUCTION_MONITORING_GUIDE.md)**: Monitoring and maintenance in production
- **[Colab Usage Guide](docs/colab_usage_guide.md)**: Google Colab deployment and optimization
- **[Advanced Reservoirs](docs/advanced_reservoirs_summary.md)**: Advanced reservoir architectures and usage

### Quick Reference Guides

#### Training Quick Reference
```bash
# Basic training
python main.py train

# Advanced training with system messages
python main.py train --reservoir-type hierarchical --use-attention --embedding-dim 256

# Custom configuration
python main.py train --reservoir-config '{"layers": [512, 256], "dropout": 0.2}'
```

#### Inference Quick Reference
```bash
# Enhanced inference
python -m src.lsm.inference.inference --model-path ./model --interactive

# Response generation
python -m src.lsm.inference.response_generator --model-path ./model --system-message "Be helpful"

# Batch processing
python -m src.lsm.inference.inference --model-path ./model --batch-file conversations.json
```

#### Model Management Quick Reference
```bash
# List models
python -m src.lsm.management.manage_models list

# Validate model
python -m src.lsm.management.manage_models validate --model-path ./model

# Clean up incomplete models
python -m src.lsm.management.manage_models cleanup --dry-run
```

## Contributing

### Development Guidelines

When contributing to the LSM project:

#### Code Quality Standards
1. **Error Handling**: Add comprehensive error handling with helpful messages using custom exception classes
2. **Input Validation**: Include input validation for all public methods using validation utilities
3. **Logging**: Add performance monitoring and debug logging using the LSM logging system
4. **Testing**: Write comprehensive unit and integration tests for new functionality
5. **Documentation**: Update API documentation and examples for any changes
6. **Backward Compatibility**: Consider impact on existing models and provide migration paths

#### Architecture Principles
1. **Modularity**: Follow the modular architecture pattern for easy component swapping
2. **Configuration**: Use ModelConfiguration for all configurable parameters
3. **System Messages**: Support system message integration in new components
4. **Performance**: Optimize for both memory usage and inference speed
5. **Production Ready**: Include monitoring, validation, and error recovery

#### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd sparse-sine-lsm

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 mypy

# Run tests
python -m pytest tests/

# Run code formatting
black src/ examples/ tests/
flake8 src/ examples/ tests/

# Type checking
mypy src/
```

#### Testing Guidelines
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_core/ -v          # Core component tests
python -m pytest tests/test_inference/ -v    # Inference system tests
python -m pytest tests/test_training/ -v     # Training pipeline tests

# Run integration tests
python -m pytest tests/integration/ -v

# Run performance tests
python -m pytest tests/performance/ -v
```

#### Contribution Process
1. **Fork** the repository and create a feature branch
2. **Implement** your changes following the guidelines above
3. **Test** thoroughly including edge cases and error conditions
4. **Document** your changes in code comments and API documentation
5. **Submit** a pull request with a clear description of changes
6. **Review** process will check code quality, tests, and documentation

## Recent Enhancements

### Version 2.0 - Enhanced Conversational AI (January 2025)

This major release introduces comprehensive enhancements for advanced conversational AI:

#### 🚀 New Features
- **HuggingFace Dataset Integration**: Native support for cosmopedia-v2 and other conversational datasets
- **Advanced Tokenization**: StandardTokenizerWrapper with GPT-2/BERT compatibility and SinusoidalEmbedder
- **System Message Support**: Complete system message processing with context-aware response generation
- **3D CNN Architecture**: Enhanced spatial-temporal processing with system message integration
- **Response-Level Generation**: Complete response generation instead of token-by-token prediction
- **Pipeline Orchestration**: Modular architecture for component experimentation and swapping

#### 🔧 Enhanced Components
- **Multiple Reservoir Types**: Standard, hierarchical, attentive, echo state, and deep reservoirs
- **Advanced CNN Architectures**: 2D and 3D CNNs with attention mechanisms and residual connections
- **Cosine Similarity Loss**: Optimized loss function for response-level training
- **Message Annotation System**: Conversation flow markers and metadata handling
- **Production Monitoring**: Comprehensive logging, validation, and performance tracking

#### 🛠️ Infrastructure Improvements
- **Model Management**: Enhanced model discovery, validation, and maintenance utilities
- **Google Colab Support**: Optimized deployment and experimentation in Colab environments
- **Memory Optimization**: Lazy loading, intelligent caching, and memory management
- **Error Handling**: Comprehensive error handling with helpful error messages and recovery
- **Backward Compatibility**: Support for legacy models with migration utilities

#### 📊 Performance Enhancements
- **Reservoir Strategy Management**: Intelligent reservoir reuse and coordination
- **Batch Processing**: Efficient processing of multiple conversations
- **Caching Systems**: Multi-level caching for embeddings and responses
- **Production Validation**: Comprehensive validation for production deployments

### Migration from Version 1.x

#### Automatic Migration
Most existing models will work with the new system through backward compatibility layers:

```python
# Legacy models automatically use compatibility mode
from src.lsm.inference.inference import LSMInference
inference = LSMInference("./old_model_directory")
```

#### Enhanced Migration
To take advantage of new features, retrain models with the enhanced pipeline:

```bash
# Retrain with enhanced features
python main.py train --reservoir-type hierarchical --use-attention --embedding-dim 256
```

#### Configuration Migration
Update existing configurations to use new features:

```python
# Old configuration
config = ModelConfiguration(window_size=10, embedding_dim=128)

# Enhanced configuration
config = ModelConfiguration(
    window_size=10,
    embedding_dim=128,
    enable_system_messages=True,
    cnn_architecture="3d",
    reservoir_type="hierarchical"
)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **TensorFlow Team**: For the excellent deep learning framework
- **HuggingFace**: For datasets and tokenization libraries
- **Research Community**: For advances in Liquid State Machines and reservoir computing
- **Contributors**: All contributors who have helped improve this project

## Citation

If you use this project in your research, please cite:

```bibtex
@software{sparse_sine_lsm,
  title={Sparse Sine-Activated Liquid State Machine for Conversational AI},
  author={LSM Development Team},
  year={2025},
  url={https://github.com/your-repo/sparse-sine-lsm},
  version={2.0}
}
```