# Advanced LSM Convenience API Tutorial

This tutorial covers advanced usage patterns, customization options, and best practices for the LSM Convenience API. It assumes you're familiar with the basics covered in the Getting Started Tutorial.

## Table of Contents

1. [Advanced Configuration](#advanced-configuration)
2. [Custom Data Formats](#custom-data-formats)
3. [Performance Optimization](#performance-optimization)
4. [Production Deployment](#production-deployment)
5. [Integration Patterns](#integration-patterns)
6. [Monitoring and Debugging](#monitoring-and-debugging)
7. [Best Practices](#best-practices)

## Advanced Configuration

### Custom Reservoir Configurations

While presets work for most cases, you can customize reservoir behavior for specific needs:

```python
from lsm import LSMGenerator

# Custom Echo State Network for time series
generator = LSMGenerator(
    reservoir_type='echo_state',
    reservoir_config={
        'n_reservoir': 1000,
        'spectral_radius': 0.95,  # Close to 1.0 for long memory
        'input_scaling': 0.1,     # Small input scaling
        'connectivity': 0.1,      # Sparse connections
        'leaking_rate': 0.3       # Memory decay rate
    }
)

# Custom Hierarchical Reservoir for complex text
generator = LSMGenerator(
    reservoir_type='hierarchical',
    reservoir_config={
        'hierarchy_levels': 4,
        'level_sizes': [200, 400, 600, 800],
        'inter_level_connectivity': 0.1,
        'intra_level_connectivity': 0.2
    }
)

# Custom Attentive Reservoir for context-aware generation
generator = LSMGenerator(
    reservoir_type='attentive',
    reservoir_config={
        'n_reservoir': 800,
        'attention_heads': 8,
        'attention_dim': 64,
        'context_length': 50
    }
)
```

### Advanced System Message Processing

System messages provide powerful control over generation behavior:

```python
from lsm import LSMGenerator

generator = LSMGenerator(
    system_message_support=True,
    system_config={
        'context_length': 100,      # Longer system context
        'embedding_modifier_strength': 0.8,  # Strong influence
        'context_decay': 0.1        # How quickly context fades
    }
)

# Train with diverse system messages
training_data = [
    {
        "messages": ["Explain quantum physics", "Quantum physics deals with..."],
        "system": "You are a physics professor. Use technical language and provide detailed explanations."
    },
    {
        "messages": ["Explain quantum physics", "Quantum stuff is like..."],
        "system": "You are talking to a 10-year-old. Use simple language and fun analogies."
    }
]

generator.fit(training_data)

# Generate with different personas
technical_response = generator.generate(
    "What is quantum entanglement?",
    system_message="You are a quantum physicist. Be precise and technical."
)

simple_response = generator.generate(
    "What is quantum entanglement?",
    system_message="You are a science teacher for kids. Use simple words and fun examples."
)
```

### Multi-Modal Configuration

Configure different components for different types of processing:

```python
generator = LSMGenerator(
    # Text processing
    tokenizer='gpt2',
    embedding_dim=256,
    
    # Reservoir processing
    reservoir_type='hierarchical',
    reservoir_config={'hierarchy_levels': 3},
    
    # CNN processing for 3D features
    cnn_3d_support=True,
    cnn_config={
        'filters': [32, 64, 128],
        'kernel_sizes': [(3,3,3), (3,3,3), (3,3,3)],
        'activation': 'relu'
    },
    
    # Response generation
    response_level=True,
    response_config={
        'max_length': 100,
        'temperature': 0.8,
        'top_p': 0.9
    }
)
```

## Custom Data Formats

### Advanced Conversation Formats

Handle complex conversation structures:

```python
from lsm import LSMGenerator
from lsm.convenience.data_formats import ConversationFormat

# Multi-turn conversations with metadata
conversations = [
    {
        "conversation_id": "conv_001",
        "participants": ["user", "assistant"],
        "turns": [
            {"speaker": "user", "text": "Hello", "timestamp": "2024-01-01T10:00:00"},
            {"speaker": "assistant", "text": "Hi there!", "timestamp": "2024-01-01T10:00:01"},
            {"speaker": "user", "text": "How are you?", "timestamp": "2024-01-01T10:00:05"},
            {"speaker": "assistant", "text": "I'm doing well!", "timestamp": "2024-01-01T10:00:06"}
        ],
        "system": "You are a helpful assistant",
        "context": {"topic": "greeting", "mood": "friendly"}
    }
]

# Custom format handler
def custom_conversation_parser(data):
    """Parse custom conversation format."""
    parsed_conversations = []
    for conv in data:
        messages = []
        for turn in conv["turns"]:
            messages.append(f"{turn['speaker']}: {turn['text']}")
        
        parsed_conversations.append({
            "messages": messages,
            "system": conv.get("system"),
            "metadata": {
                "conversation_id": conv["conversation_id"],
                "context": conv.get("context", {})
            }
        })
    return parsed_conversations

# Use custom parser
parsed_data = custom_conversation_parser(conversations)
generator = LSMGenerator()
generator.fit(parsed_data)
```

### Structured Data Integration

Integrate with databases and structured data sources:

```python
import pandas as pd
from lsm import LSMClassifier

# Load from database
def load_from_database():
    """Load data from database."""
    # Simulated database query
    data = pd.DataFrame({
        'text': ['Great product!', 'Terrible service', 'Average quality'],
        'sentiment': ['positive', 'negative', 'neutral'],
        'confidence': [0.9, 0.95, 0.6],
        'user_id': [1, 2, 3],
        'product_category': ['electronics', 'service', 'clothing']
    })
    return data

# Custom preprocessing
def preprocess_data(df):
    """Preprocess data with custom logic."""
    # Filter by confidence
    df = df[df['confidence'] > 0.7]
    
    # Add context to text
    df['enhanced_text'] = df.apply(
        lambda row: f"[{row['product_category']}] {row['text']}", 
        axis=1
    )
    
    return df['enhanced_text'].tolist(), df['sentiment'].tolist()

# Load and train
df = load_from_database()
texts, labels = preprocess_data(df)

classifier = LSMClassifier(
    window_size=12,  # Longer window for enhanced text
    classifier_type='random_forest'
)
classifier.fit(texts, labels)
```

### Real-time Data Streaming

Handle streaming data for continuous learning:

```python
from lsm import LSMGenerator
import asyncio
from typing import AsyncGenerator

class StreamingTrainer:
    def __init__(self, generator: LSMGenerator, batch_size: int = 32):
        self.generator = generator
        self.batch_size = batch_size
        self.buffer = []
    
    async def process_stream(self, data_stream: AsyncGenerator):
        """Process streaming data."""
        async for data_point in data_stream:
            self.buffer.append(data_point)
            
            if len(self.buffer) >= self.batch_size:
                # Train on batch
                await self.train_batch()
                self.buffer = []
    
    async def train_batch(self):
        """Train on accumulated batch."""
        if self.buffer:
            # Simulate async training (use thread pool in practice)
            await asyncio.sleep(0.1)  # Simulate training time
            self.generator.partial_fit(self.buffer, epochs=1)
            print(f"Trained on batch of {len(self.buffer)} samples")

# Usage
async def simulate_data_stream():
    """Simulate streaming data."""
    conversations = [
        "User: Hello\nBot: Hi!",
        "User: How are you?\nBot: Good!",
        # ... more data
    ]
    
    for conv in conversations:
        yield conv
        await asyncio.sleep(0.1)  # Simulate real-time arrival

# Run streaming training
generator = LSMGenerator(preset='fast')
trainer = StreamingTrainer(generator)

# In practice, you'd run this continuously
# asyncio.run(trainer.process_stream(simulate_data_stream()))
```

## Performance Optimization

### Memory-Efficient Training

Optimize for large datasets and limited memory:

```python
from lsm import LSMGenerator
from lsm.convenience import AutoMemoryManager, PerformanceProfiler

class MemoryEfficientTrainer:
    def __init__(self, generator: LSMGenerator):
        self.generator = generator
        self.memory_manager = AutoMemoryManager()
        self.profiler = PerformanceProfiler()
    
    def train_large_dataset(self, dataset, chunk_size: int = 1000):
        """Train on large dataset with memory management."""
        self.profiler.start()
        
        with self.memory_manager:
            # Process in chunks
            for i in range(0, len(dataset), chunk_size):
                chunk = dataset[i:i+chunk_size]
                
                # Adjust batch size based on available memory
                available_memory = self.memory_manager.get_available_memory()
                optimal_batch_size = self.memory_manager.calculate_optimal_batch_size(
                    data_size=len(chunk),
                    available_memory=available_memory
                )
                
                print(f"Processing chunk {i//chunk_size + 1}, batch_size={optimal_batch_size}")
                
                if i == 0:
                    # Initial training
                    self.generator.fit(
                        chunk,
                        batch_size=optimal_batch_size,
                        epochs=10
                    )
                else:
                    # Incremental training
                    self.generator.partial_fit(
                        chunk,
                        batch_size=optimal_batch_size,
                        epochs=5
                    )
                
                # Force garbage collection
                self.memory_manager.cleanup()
        
        stats = self.profiler.get_stats()
        print(f"Training completed in {stats['total_time']:.2f}s")
        print(f"Peak memory usage: {stats['peak_memory_mb']:.1f}MB")

# Usage
generator = LSMGenerator(
    preset='balanced',
    auto_memory_management=True
)

trainer = MemoryEfficientTrainer(generator)
# trainer.train_large_dataset(large_dataset)
```

### Parallel Processing

Utilize multiple cores for faster processing:

```python
from lsm import LSMGenerator
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class ParallelGenerator:
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or mp.cpu_count()
        self.generators = []
    
    def train_ensemble(self, datasets):
        """Train multiple generators in parallel."""
        def train_single(data_and_config):
            data, config = data_and_config
            generator = LSMGenerator(**config)
            generator.fit(data)
            return generator
        
        # Different configurations for ensemble
        configs = [
            {'preset': 'fast', 'reservoir_type': 'standard'},
            {'preset': 'balanced', 'reservoir_type': 'hierarchical'},
            {'preset': 'quality', 'reservoir_type': 'attentive'}
        ]
        
        # Prepare data and configs
        tasks = [(datasets[i % len(datasets)], configs[i]) 
                for i in range(len(configs))]
        
        # Train in parallel
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            self.generators = list(executor.map(train_single, tasks))
        
        print(f"Trained {len(self.generators)} generators in parallel")
    
    def generate_ensemble(self, prompt, **kwargs):
        """Generate responses from all generators and combine."""
        def generate_single(generator):
            return generator.generate(prompt, **kwargs)
        
        with ThreadPoolExecutor(max_workers=len(self.generators)) as executor:
            responses = list(executor.map(generate_single, self.generators))
        
        # Simple ensemble: return most common response or best scored
        return self.combine_responses(responses)
    
    def combine_responses(self, responses):
        """Combine multiple responses (simple implementation)."""
        # In practice, you might use more sophisticated combination
        return max(responses, key=len)  # Return longest response

# Usage
parallel_gen = ParallelGenerator(n_workers=4)
# parallel_gen.train_ensemble([dataset1, dataset2, dataset3])
# response = parallel_gen.generate_ensemble("Hello, how are you?")
```

### GPU Optimization

Optimize for GPU usage:

```python
import tensorflow as tf
from lsm import LSMGenerator

class GPUOptimizedGenerator:
    def __init__(self, **kwargs):
        self.setup_gpu()
        self.generator = LSMGenerator(**kwargs)
    
    def setup_gpu(self):
        """Configure GPU for optimal performance."""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                
                # Set mixed precision for faster training
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
                print(f"GPU optimization enabled for {len(gpus)} GPU(s)")
            except RuntimeError as e:
                print(f"GPU setup error: {e}")
        else:
            print("No GPU available, using CPU")
    
    def train_with_gpu_optimization(self, data, **kwargs):
        """Train with GPU-specific optimizations."""
        # Use larger batch sizes on GPU
        if tf.config.list_physical_devices('GPU'):
            kwargs.setdefault('batch_size', 64)
        else:
            kwargs.setdefault('batch_size', 16)
        
        # Enable XLA compilation for faster execution
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            self.generator.fit(data, **kwargs)

# Usage
gpu_generator = GPUOptimizedGenerator(preset='quality')
# gpu_generator.train_with_gpu_optimization(data, epochs=100)
```

## Production Deployment

### Model Serving

Deploy models for production serving:

```python
from lsm import LSMGenerator
from flask import Flask, request, jsonify
import logging
import time
from functools import lru_cache

class ModelServer:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.generator = None
        self.load_model()
        self.setup_logging()
    
    def load_model(self):
        """Load model with error handling."""
        try:
            self.generator = LSMGenerator.load(self.model_path)
            print(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def setup_logging(self):
        """Setup production logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    @lru_cache(maxsize=1000)
    def cached_generate(self, prompt: str, system_message: str = None, **kwargs):
        """Generate with caching for repeated requests."""
        return self.generator.generate(prompt, system_message=system_message, **kwargs)
    
    def generate_with_monitoring(self, prompt: str, **kwargs):
        """Generate with performance monitoring."""
        start_time = time.time()
        
        try:
            response = self.generator.generate(prompt, **kwargs)
            
            # Log metrics
            duration = time.time() - start_time
            self.logger.info(f"Generation completed in {duration:.3f}s")
            
            return {
                'response': response,
                'status': 'success',
                'duration': duration
            }
        
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return {
                'error': str(e),
                'status': 'error',
                'duration': time.time() - start_time
            }

# Flask app
app = Flask(__name__)
model_server = ModelServer("production_model")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    system_message = data.get('system_message')
    max_length = data.get('max_length', 50)
    temperature = data.get('temperature', 0.8)
    
    result = model_server.generate_with_monitoring(
        prompt=prompt,
        system_message=system_message,
        max_length=max_length,
        temperature=temperature
    )
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model_server.generator is not None})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
```

### Batch Processing Service

Handle batch requests efficiently:

```python
from lsm import LSMGenerator
import asyncio
from typing import List, Dict
import uuid
from dataclasses import dataclass
from datetime import datetime

@dataclass
class BatchRequest:
    id: str
    prompts: List[str]
    system_message: str = None
    max_length: int = 50
    temperature: float = 0.8
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class BatchResponse:
    request_id: str
    responses: List[str]
    status: str
    processing_time: float
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class BatchProcessor:
    def __init__(self, model_path: str, max_batch_size: int = 100):
        self.generator = LSMGenerator.load(model_path)
        self.max_batch_size = max_batch_size
        self.pending_requests = {}
        self.completed_requests = {}
    
    async def submit_batch(self, prompts: List[str], **kwargs) -> str:
        """Submit a batch request and return request ID."""
        request_id = str(uuid.uuid4())
        
        # Split large batches
        if len(prompts) > self.max_batch_size:
            # Handle large batches by splitting
            sub_requests = []
            for i in range(0, len(prompts), self.max_batch_size):
                sub_batch = prompts[i:i+self.max_batch_size]
                sub_request = BatchRequest(
                    id=f"{request_id}_{i//self.max_batch_size}",
                    prompts=sub_batch,
                    **kwargs
                )
                sub_requests.append(sub_request)
            
            # Process sub-requests
            tasks = [self.process_batch(req) for req in sub_requests]
            sub_responses = await asyncio.gather(*tasks)
            
            # Combine responses
            all_responses = []
            total_time = 0
            for resp in sub_responses:
                all_responses.extend(resp.responses)
                total_time += resp.processing_time
            
            response = BatchResponse(
                request_id=request_id,
                responses=all_responses,
                status='completed',
                processing_time=total_time
            )
        else:
            request = BatchRequest(id=request_id, prompts=prompts, **kwargs)
            response = await self.process_batch(request)
        
        self.completed_requests[request_id] = response
        return request_id
    
    async def process_batch(self, request: BatchRequest) -> BatchResponse:
        """Process a single batch request."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Use batch_generate for efficiency
            responses = self.generator.batch_generate(
                request.prompts,
                system_message=request.system_message,
                max_length=request.max_length,
                temperature=request.temperature
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return BatchResponse(
                request_id=request.id,
                responses=responses,
                status='completed',
                processing_time=processing_time
            )
        
        except Exception as e:
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return BatchResponse(
                request_id=request.id,
                responses=[f"Error: {str(e)}"] * len(request.prompts),
                status='error',
                processing_time=processing_time
            )
    
    def get_result(self, request_id: str) -> BatchResponse:
        """Get batch processing result."""
        return self.completed_requests.get(request_id)
    
    def get_status(self, request_id: str) -> str:
        """Get batch processing status."""
        if request_id in self.completed_requests:
            return self.completed_requests[request_id].status
        elif request_id in self.pending_requests:
            return 'processing'
        else:
            return 'not_found'

# Usage
# processor = BatchProcessor("production_model")
# request_id = await processor.submit_batch(["Hello", "How are you?", "Goodbye"])
# result = processor.get_result(request_id)
```

## Integration Patterns

### Sklearn Pipeline Integration

Integrate with sklearn pipelines:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from lsm import LSMClassifier
import numpy as np

class HybridTextClassifier:
    """Combine traditional NLP features with LSM features."""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.lsm = LSMClassifier()
        self.final_classifier = None
    
    def fit(self, X, y):
        """Train hybrid classifier."""
        # Extract TF-IDF features
        tfidf_features = self.tfidf.fit_transform(X)
        
        # Extract LSM features
        self.lsm.fit(X, y)
        lsm_features = self.lsm.get_reservoir_features(X)
        
        # Combine features
        combined_features = np.hstack([
            tfidf_features.toarray(),
            lsm_features
        ])
        
        # Train final classifier
        from sklearn.ensemble import RandomForestClassifier
        self.final_classifier = RandomForestClassifier()
        self.final_classifier.fit(combined_features, y)
        
        return self
    
    def predict(self, X):
        """Predict with hybrid features."""
        tfidf_features = self.tfidf.transform(X)
        lsm_features = self.lsm.get_reservoir_features(X)
        
        combined_features = np.hstack([
            tfidf_features.toarray(),
            lsm_features
        ])
        
        return self.final_classifier.predict(combined_features)

# Grid search with LSM
param_grid = {
    'lsm__window_size': [5, 10, 15],
    'lsm__reservoir_type': ['standard', 'hierarchical'],
    'lsm__classifier_type': ['logistic', 'random_forest']
}

pipeline = Pipeline([
    ('lsm', LSMClassifier())
])

grid_search = GridSearchCV(pipeline, param_grid, cv=3)
# grid_search.fit(texts, labels)
```

### Database Integration

Integrate with databases for persistent storage:

```python
import sqlite3
from lsm import LSMGenerator
import json
import pickle
from datetime import datetime

class DatabaseIntegratedGenerator:
    def __init__(self, db_path: str, model_path: str):
        self.db_path = db_path
        self.generator = LSMGenerator.load(model_path)
        self.setup_database()
    
    def setup_database(self):
        """Setup database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                prompt TEXT,
                response TEXT,
                system_message TEXT,
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Training data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_text TEXT,
                system_message TEXT,
                metadata TEXT,
                used_for_training BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def generate_and_store(self, user_id: str, prompt: str, **kwargs):
        """Generate response and store in database."""
        response = self.generator.generate(prompt, **kwargs)
        
        # Store conversation
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations 
            (user_id, prompt, response, system_message, parameters)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            user_id,
            prompt,
            response,
            kwargs.get('system_message'),
            json.dumps(kwargs)
        ))
        
        conn.commit()
        conn.close()
        
        return response
    
    def collect_training_data(self, min_rating: float = 4.0):
        """Collect highly-rated conversations for retraining."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get conversations with high ratings (assuming rating system)
        cursor.execute('''
            SELECT prompt, response, system_message 
            FROM conversations 
            WHERE rating >= ? AND used_for_training = FALSE
        ''', (min_rating,))
        
        conversations = []
        for row in cursor.fetchall():
            prompt, response, system_message = row
            conversation = f"User: {prompt}\nAssistant: {response}"
            conversations.append({
                "messages": [prompt, response],
                "system": system_message
            })
        
        conn.close()
        return conversations
    
    def retrain_from_database(self):
        """Retrain model using database conversations."""
        training_data = self.collect_training_data()
        
        if len(training_data) > 10:  # Minimum data threshold
            print(f"Retraining with {len(training_data)} conversations")
            self.generator.fit(training_data, epochs=20)
            
            # Mark data as used
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE conversations 
                SET used_for_training = TRUE 
                WHERE rating >= 4.0 AND used_for_training = FALSE
            ''')
            conn.commit()
            conn.close()
            
            return True
        else:
            print("Insufficient data for retraining")
            return False

# Usage
# db_generator = DatabaseIntegratedGenerator("conversations.db", "my_model")
# response = db_generator.generate_and_store("user123", "Hello!")
# db_generator.retrain_from_database()
```

## Monitoring and Debugging

### Comprehensive Monitoring

Monitor model performance in production:

```python
from lsm import LSMGenerator
from lsm.convenience import PerformanceProfiler, MemoryMonitor
import logging
import time
from collections import defaultdict, deque
import threading

class ProductionMonitor:
    def __init__(self, generator: LSMGenerator, window_size: int = 1000):
        self.generator = generator
        self.window_size = window_size
        
        # Metrics storage
        self.response_times = deque(maxlen=window_size)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.quality_scores = deque(maxlen=window_size)
        
        # Monitoring components
        self.profiler = PerformanceProfiler()
        self.memory_monitor = MemoryMonitor()
        
        # Setup logging
        self.setup_logging()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitoring_thread.start()
    
    def setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def monitored_generate(self, prompt: str, **kwargs):
        """Generate with comprehensive monitoring."""
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000)}"
        
        try:
            # Start profiling
            self.profiler.start_request(request_id)
            
            # Generate response
            response = self.generator.generate(prompt, **kwargs)
            
            # Calculate metrics
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Quality assessment (simple heuristic)
            quality_score = self.assess_quality(prompt, response)
            self.quality_scores.append(quality_score)
            
            # Log success
            self.logger.info(f"Request {request_id} completed in {response_time:.3f}s")
            self.request_counts['success'] += 1
            
            return {
                'response': response,
                'request_id': request_id,
                'response_time': response_time,
                'quality_score': quality_score,
                'status': 'success'
            }
        
        except Exception as e:
            response_time = time.time() - start_time
            error_type = type(e).__name__
            
            self.error_counts[error_type] += 1
            self.request_counts['error'] += 1
            
            self.logger.error(f"Request {request_id} failed: {e}")
            
            return {
                'error': str(e),
                'error_type': error_type,
                'request_id': request_id,
                'response_time': response_time,
                'status': 'error'
            }
        
        finally:
            self.profiler.end_request(request_id)
    
    def assess_quality(self, prompt: str, response: str) -> float:
        """Simple quality assessment."""
        # Basic heuristics (in practice, use more sophisticated methods)
        score = 1.0
        
        # Penalize very short responses
        if len(response) < 10:
            score -= 0.3
        
        # Penalize repetitive responses
        words = response.split()
        if len(set(words)) < len(words) * 0.5:
            score -= 0.2
        
        # Reward relevant responses (simple keyword matching)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        relevance = len(prompt_words & response_words) / max(len(prompt_words), 1)
        score += relevance * 0.2
        
        return max(0.0, min(1.0, score))
    
    def get_metrics(self) -> dict:
        """Get current performance metrics."""
        if not self.response_times:
            return {'status': 'no_data'}
        
        return {
            'response_time': {
                'mean': sum(self.response_times) / len(self.response_times),
                'p95': sorted(self.response_times)[int(len(self.response_times) * 0.95)],
                'p99': sorted(self.response_times)[int(len(self.response_times) * 0.99)]
            },
            'quality': {
                'mean': sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0,
                'min': min(self.quality_scores) if self.quality_scores else 0,
                'max': max(self.quality_scores) if self.quality_scores else 0
            },
            'requests': dict(self.request_counts),
            'errors': dict(self.error_counts),
            'memory': self.memory_monitor.get_current_usage(),
            'uptime': time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
    
    def monitor_loop(self):
        """Background monitoring loop."""
        self.start_time = time.time()
        
        while True:
            try:
                metrics = self.get_metrics()
                
                # Log metrics periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self.logger.info(f"Performance metrics: {metrics}")
                
                # Check for alerts
                self.check_alerts(metrics)
                
                time.sleep(10)  # Check every 10 seconds
            
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(30)  # Wait longer on error
    
    def check_alerts(self, metrics: dict):
        """Check for performance alerts."""
        if 'response_time' in metrics:
            # Alert on high response times
            if metrics['response_time']['p95'] > 5.0:
                self.logger.warning(f"High response time: {metrics['response_time']['p95']:.2f}s")
            
            # Alert on low quality
            if 'quality' in metrics and metrics['quality']['mean'] < 0.5:
                self.logger.warning(f"Low quality score: {metrics['quality']['mean']:.2f}")
            
            # Alert on high error rate
            total_requests = sum(metrics['requests'].values())
            error_rate = metrics['requests'].get('error', 0) / max(total_requests, 1)
            if error_rate > 0.1:  # 10% error rate
                self.logger.warning(f"High error rate: {error_rate:.2%}")

# Usage
generator = LSMGenerator.load("production_model")
monitor = ProductionMonitor(generator)

# Monitored generation
result = monitor.monitored_generate("Hello, how are you?")
print(f"Response: {result['response']}")
print(f"Quality: {result['quality_score']:.2f}")

# Get metrics
metrics = monitor.get_metrics()
print(f"Current metrics: {metrics}")
```

## Best Practices

### 1. Model Lifecycle Management

```python
from lsm import LSMGenerator
import os
from datetime import datetime
import shutil

class ModelManager:
    def __init__(self, base_path: str = "models"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
    
    def save_versioned_model(self, generator: LSMGenerator, version: str = None):
        """Save model with version control."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = os.path.join(self.base_path, f"model_v{version}")
        generator.save(model_path)
        
        # Create symlink to latest
        latest_path = os.path.join(self.base_path, "latest")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        os.symlink(model_path, latest_path)
        
        return model_path
    
    def load_latest_model(self):
        """Load the latest model version."""
        latest_path = os.path.join(self.base_path, "latest")
        if os.path.exists(latest_path):
            return LSMGenerator.load(latest_path)
        else:
            raise FileNotFoundError("No model versions found")
    
    def cleanup_old_versions(self, keep_versions: int = 5):
        """Keep only the N most recent versions."""
        versions = []
        for item in os.listdir(self.base_path):
            if item.startswith("model_v") and os.path.isdir(os.path.join(self.base_path, item)):
                versions.append(item)
        
        versions.sort(reverse=True)
        
        for old_version in versions[keep_versions:]:
            old_path = os.path.join(self.base_path, old_version)
            shutil.rmtree(old_path)
            print(f"Removed old version: {old_version}")

# Usage
manager = ModelManager()
generator = LSMGenerator()
# generator.fit(data)
# model_path = manager.save_versioned_model(generator)
# latest_generator = manager.load_latest_model()
```

### 2. Configuration Management

```python
import yaml
from lsm import LSMGenerator
from lsm.convenience import ConvenienceConfig

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self.get_default_config()
    
    def get_default_config(self):
        """Get default configuration."""
        return {
            'model': {
                'preset': 'balanced',
                'window_size': 10,
                'embedding_dim': 128,
                'system_message_support': True
            },
            'training': {
                'epochs': 50,
                'batch_size': 32,
                'validation_split': 0.2
            },
            'generation': {
                'max_length': 50,
                'temperature': 0.8
            },
            'performance': {
                'auto_memory_management': True,
                'enable_profiling': False
            }
        }
    
    def save_config(self):
        """Save current configuration."""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def create_generator(self):
        """Create generator from configuration."""
        model_config = self.config['model']
        return LSMGenerator(**model_config)
    
    def get_training_params(self):
        """Get training parameters."""
        return self.config['training']
    
    def get_generation_params(self):
        """Get generation parameters."""
        return self.config['generation']

# Usage
config_manager = ConfigManager()
generator = config_manager.create_generator()
training_params = config_manager.get_training_params()
# generator.fit(data, **training_params)
```

### 3. Testing and Validation

```python
import unittest
from lsm import LSMGenerator, LSMClassifier
import numpy as np

class ConvenienceAPITestSuite(unittest.TestCase):
    def setUp(self):
        """Setup test fixtures."""
        self.test_conversations = [
            "User: Hello\nAssistant: Hi there!",
            "User: How are you?\nAssistant: I'm doing well!",
            "User: Goodbye\nAssistant: See you later!"
        ]
        
        self.test_texts = ["Good product", "Bad service", "Average quality"]
        self.test_labels = ["positive", "negative", "neutral"]
    
    def test_generator_basic_functionality(self):
        """Test basic generator functionality."""
        generator = LSMGenerator(preset='fast')
        
        # Test training
        generator.fit(self.test_conversations, epochs=2)
        
        # Test generation
        response = generator.generate("Hello")
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        
        # Test batch generation
        responses = generator.batch_generate(["Hello", "Hi"])
        self.assertEqual(len(responses), 2)
        self.assertTrue(all(isinstance(r, str) for r in responses))
    
    def test_classifier_functionality(self):
        """Test classifier functionality."""
        classifier = LSMClassifier()
        
        # Test training
        classifier.fit(self.test_texts, self.test_labels, epochs=2)
        
        # Test prediction
        predictions = classifier.predict(self.test_texts)
        self.assertEqual(len(predictions), len(self.test_texts))
        
        # Test probabilities
        probabilities = classifier.predict_proba(self.test_texts)
        self.assertEqual(probabilities.shape, (len(self.test_texts), len(set(self.test_labels))))
    
    def test_model_persistence(self):
        """Test model saving and loading."""
        generator = LSMGenerator(preset='fast')
        generator.fit(self.test_conversations, epochs=1)
        
        # Save model
        model_path = "test_model"
        generator.save(model_path)
        
        # Load model
        loaded_generator = LSMGenerator.load(model_path)
        
        # Test loaded model
        response = loaded_generator.generate("Test")
        self.assertIsInstance(response, str)
        
        # Cleanup
        import shutil
        shutil.rmtree(model_path)
    
    def test_error_handling(self):
        """Test error handling."""
        from lsm.convenience import ConvenienceValidationError
        
        # Test invalid parameters
        with self.assertRaises(ConvenienceValidationError):
            LSMGenerator(window_size=-1)
        
        # Test empty data
        generator = LSMGenerator(preset='fast')
        with self.assertRaises(ConvenienceValidationError):
            generator.fit([])

# Run tests
if __name__ == '__main__':
    unittest.main()
```

This advanced tutorial covers sophisticated usage patterns that go beyond basic functionality. These patterns are designed for production use cases where you need fine-grained control, high performance, and robust error handling.

The key takeaways are:
1. **Start simple** with presets and basic functionality
2. **Customize gradually** as your needs become more specific
3. **Monitor everything** in production environments
4. **Plan for scale** with proper architecture patterns
5. **Test thoroughly** with comprehensive test suites

These patterns will help you build robust, scalable applications using the LSM Convenience API.