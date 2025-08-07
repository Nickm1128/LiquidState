# Design Document

## Overview

This design document outlines the architecture for enhancing the Sparse Sine-Activated Liquid State Machine (LSM) project with complete inference capabilities. The design focuses on tokenizer persistence, text decoding, model configuration management, and a streamlined inference pipeline that enables seamless text-to-text prediction.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Text Input    │───▶│  Enhanced LSM    │───▶│  Text Output    │
│   "Hello..."    │    │   Inference      │    │ "How are you?"  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Model Storage   │
                    │  ├─ reservoir/   │
                    │  ├─ cnn/         │
                    │  ├─ tokenizer/   │
                    │  ├─ config.json  │
                    │  └─ metadata.json│
                    └──────────────────┘
```

### Enhanced Model Storage Structure

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

## Components and Interfaces

### 1. Enhanced DialogueTokenizer

**Purpose**: Extended tokenizer with persistence and decoding capabilities

**Key Methods**:
```python
class DialogueTokenizer:
    def save(self, save_path: str) -> None
    def load(self, save_path: str) -> None
    def decode_embedding(self, embedding: np.ndarray) -> str
    def decode_embeddings_batch(self, embeddings: np.ndarray) -> List[str]
    def get_closest_texts(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]
```

**Decoding Strategy**:
- Use cosine similarity to find closest vocabulary embeddings
- Implement fallback mechanisms for out-of-vocabulary cases
- Support both single and batch decoding operations

### 2. ModelConfiguration Class

**Purpose**: Centralized configuration management for all model parameters

```python
@dataclass
class ModelConfiguration:
    window_size: int
    embedding_dim: int
    reservoir_type: str
    reservoir_config: Dict
    reservoir_units: List[int]
    sparsity: float
    use_multichannel: bool
    training_params: Dict
    
    def save(self, path: str) -> None
    def load(cls, path: str) -> 'ModelConfiguration'
    def to_dict(self) -> Dict
    def from_dict(cls, data: Dict) -> 'ModelConfiguration'
```

### 3. Enhanced LSMTrainer

**New Methods**:
```python
class LSMTrainer:
    def save_complete_model(self, save_dir: str, tokenizer: DialogueTokenizer, 
                           metadata: Dict = None) -> None
    def load_complete_model(self, save_dir: str) -> Tuple['LSMTrainer', DialogueTokenizer]
    def get_model_info(self) -> Dict
```

**Enhanced Functionality**:
- Automatic tokenizer integration
- Complete model state persistence
- Metadata tracking and retrieval

### 4. LSMInference Class (Redesigned)

**Purpose**: Streamlined inference interface with complete text processing

```python
class LSMInference:
    def __init__(self, model_path: str)
    def predict_next_token(self, dialogue_sequence: List[str]) -> str
    def predict_with_confidence(self, dialogue_sequence: List[str]) -> Tuple[str, float]
    def predict_top_k(self, dialogue_sequence: List[str], k: int = 5) -> List[Tuple[str, float]]
    def interactive_session(self) -> None
    def batch_predict(self, sequences: List[List[str]]) -> List[str]
```

### 5. ModelManager Class

**Purpose**: High-level model management and discovery

```python
class ModelManager:
    def list_available_models(self) -> List[Dict]
    def get_model_info(self, model_path: str) -> Dict
    def validate_model(self, model_path: str) -> bool
    def migrate_old_model(self, old_path: str, new_path: str) -> bool
    def cleanup_incomplete_models(self) -> List[str]
```

## Data Models

### Configuration Schema

```json
{
  "model_version": "1.0",
  "created_at": "2025-01-08T10:30:00Z",
  "model_config": {
    "window_size": 10,
    "embedding_dim": 128,
    "reservoir_type": "standard",
    "reservoir_config": {},
    "reservoir_units": [256, 128, 64],
    "sparsity": 0.1,
    "use_multichannel": true
  },
  "training_config": {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "validation_split": 0.1
  },
  "tokenizer_config": {
    "max_features": 10000,
    "ngram_range": [1, 2],
    "vocabulary_size": 8743
  }
}
```

### Metadata Schema

```json
{
  "training_completed_at": "2025-01-08T12:45:00Z",
  "training_duration_seconds": 3600,
  "dataset_info": {
    "source": "Synthetic-Persona-Chat",
    "num_sequences": 15000,
    "train_samples": 12000,
    "test_samples": 3000
  },
  "performance_metrics": {
    "final_test_mse": 0.0234,
    "final_test_mae": 0.1123,
    "best_val_loss": 0.0198
  },
  "system_info": {
    "python_version": "3.11.9",
    "tensorflow_version": "2.14.0",
    "platform": "Windows"
  }
}
```

## Error Handling

### Error Categories and Responses

1. **Model Loading Errors**
   - Missing files: Detailed list of missing components
   - Corrupted files: Validation errors with recovery suggestions
   - Version mismatches: Migration options or compatibility warnings

2. **Tokenizer Errors**
   - Unfitted tokenizer: Clear instructions for fitting or loading
   - Vocabulary mismatches: Fallback to closest matches with warnings
   - Encoding failures: Graceful degradation with error logging

3. **Inference Errors**
   - Invalid input format: Helpful format examples and correction suggestions
   - Sequence length mismatches: Automatic padding/truncation with warnings
   - Prediction failures: Fallback responses with error context

4. **Configuration Errors**
   - Invalid parameters: Validation with suggested corrections
   - Missing configurations: Default value substitution with warnings
   - Schema violations: Detailed validation error messages

## Testing Strategy

### Unit Tests

1. **DialogueTokenizer Tests**
   - Save/load functionality
   - Encoding/decoding accuracy
   - Vocabulary consistency
   - Error handling for edge cases

2. **ModelConfiguration Tests**
   - Serialization/deserialization
   - Validation logic
   - Default value handling
   - Schema compliance

3. **LSMInference Tests**
   - End-to-end prediction pipeline
   - Batch processing efficiency
   - Error handling robustness
   - Memory usage optimization

### Integration Tests

1. **Complete Workflow Tests**
   - Train → Save → Load → Predict cycle
   - Multiple model format compatibility
   - Cross-platform model portability
   - Performance benchmarking

2. **Backward Compatibility Tests**
   - Old model format loading
   - Migration functionality
   - Graceful degradation scenarios
   - Error message clarity

### Performance Tests

1. **Inference Speed Tests**
   - Single prediction latency
   - Batch processing throughput
   - Memory usage profiling
   - Model loading time

2. **Scalability Tests**
   - Large vocabulary handling
   - High-dimensional embedding processing
   - Concurrent inference requests
   - Resource utilization optimization

## Implementation Phases

### Phase 1: Core Infrastructure
- Enhanced DialogueTokenizer with save/load
- ModelConfiguration class implementation
- Basic error handling framework

### Phase 2: Model Integration
- Enhanced LSMTrainer with complete model persistence
- Updated training pipeline integration
- Backward compatibility layer

### Phase 3: Inference Pipeline
- Redesigned LSMInference class
- Text decoding implementation
- Interactive and batch inference modes

### Phase 4: Management and Utilities
- ModelManager implementation
- CLI enhancements
- Documentation and examples

### Phase 5: Testing and Optimization
- Comprehensive test suite
- Performance optimization
- Production readiness validation