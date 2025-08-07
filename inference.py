#!/usr/bin/env python3
"""
Enhanced inference script for the Sparse Sine-Activated LSM.
This script loads a trained model and performs next-token prediction with complete text processing.
Optimized for performance and memory efficiency.
"""

import os
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import argparse
import time
import threading
from functools import lru_cache
import gc

# Optional psutil import for memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from train import LSMTrainer
from data_loader import DialogueTokenizer
from model_config import ModelConfiguration
from lsm_exceptions import (
    ModelLoadError, InferenceError, InvalidInputError, PredictionError,
    TokenizerNotFittedError, create_error_context
)
from lsm_logging import get_logger, log_performance, create_operation_logger
from input_validation import (
    validate_file_path, validate_dialogue_sequence, validate_positive_integer,
    create_helpful_error_message
)

logger = get_logger(__name__)

class OptimizedLSMInference:
    """
    Performance-optimized inference class for trained LSM models.
    Features lazy loading, caching, and memory-efficient batch processing.
    """
    
    def __init__(self, model_path: str, lazy_load: bool = True, cache_size: int = 1000, 
                 max_batch_size: int = 32):
        """
        Initialize inference with performance optimizations.
        
        Args:
            model_path: Path to saved model directory
            lazy_load: If True, load models only when needed
            cache_size: Size of prediction cache
            max_batch_size: Maximum batch size for memory efficiency
        """
        self.model_path = model_path
        self.lazy_load = lazy_load
        self.cache_size = cache_size
        self.max_batch_size = max_batch_size
        
        # Model components (loaded lazily if enabled)
        self.trainer = None
        self.tokenizer = None
        self.config = None
        
        # Performance optimizations
        self._prediction_cache = {}
        self._embedding_cache = {}
        self._model_loaded = False
        self._tokenizer_loaded = False
        self._load_lock = threading.Lock()
        
        # Memory management
        self._memory_threshold_mb = 1024  # 1GB threshold
        self._last_gc_time = time.time()
        self._gc_interval = 30  # Run GC every 30 seconds
        
        # Load immediately if not lazy loading
        if not lazy_load:
            self._load_complete_model()
    
    def _load_complete_model(self):
        """Load the complete trained model including tokenizer and configuration."""
        with self._load_lock:
            if self._model_loaded and self._tokenizer_loaded:
                return
                
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path {self.model_path} does not exist")
            
            logger.info(f"Loading complete model from {self.model_path}")
            start_time = time.time()
            
            try:
                # Load configuration first (lightweight)
                config_path = os.path.join(self.model_path, "config.json")
                if os.path.exists(config_path):
                    self.config = ModelConfiguration.load(config_path)
                    logger.debug("Model configuration loaded")
                
                # Initialize trainer with default values (will be updated from config)
                if self.trainer is None:
                    self.trainer = LSMTrainer()
                
                # Load complete model state
                self.trainer, self.tokenizer = self.trainer.load_complete_model(self.model_path)
                
                self._model_loaded = True
                self._tokenizer_loaded = True
                
                load_time = time.time() - start_time
                logger.info(f"Complete model loaded successfully in {load_time:.2f}s")
                
            except Exception as e:
                logger.exception("Failed to load complete model", model_path=self.model_path)
                raise ModelLoadError(self.model_path, f"Failed to load complete model: {e}")
    
    def _ensure_model_loaded(self):
        """Ensure model is loaded (lazy loading support)."""
        if not self._model_loaded:
            self._load_complete_model()
    
    def _ensure_tokenizer_loaded(self):
        """Ensure tokenizer is loaded (lazy loading support)."""
        if not self._tokenizer_loaded:
            self._load_complete_model()
    
    def _manage_memory(self):
        """Manage memory usage and run garbage collection if needed."""
        current_time = time.time()
        if current_time - self._last_gc_time > self._gc_interval:
            # Check memory usage if psutil is available
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    
                    if memory_mb > self._memory_threshold_mb:
                        logger.debug(f"Memory usage {memory_mb:.1f}MB exceeds threshold, running GC")
                        gc.collect()
                        
                        # Clear caches if memory is still high
                        if process.memory_info().rss / 1024 / 1024 > self._memory_threshold_mb:
                            self._clear_caches()
                except Exception as e:
                    logger.debug(f"Memory monitoring failed: {e}")
            else:
                # Fallback: just run GC periodically
                gc.collect()
            
            self._last_gc_time = current_time
    
    def _clear_caches(self):
        """Clear prediction and embedding caches."""
        cache_size_before = len(self._prediction_cache) + len(self._embedding_cache)
        self._prediction_cache.clear()
        self._embedding_cache.clear()
        logger.debug(f"Cleared {cache_size_before} cached items")
    
    @lru_cache(maxsize=1000)
    def _get_sequence_hash(self, sequence_tuple: tuple) -> str:
        """Get a hash for a sequence for caching purposes."""
        return str(hash(sequence_tuple))
    
    def _cache_prediction(self, sequence_key: str, prediction: str, confidence: float = None):
        """Cache a prediction result."""
        if len(self._prediction_cache) >= self.cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._prediction_cache))
            del self._prediction_cache[oldest_key]
        
        self._prediction_cache[sequence_key] = {
            'prediction': prediction,
            'confidence': confidence,
            'timestamp': time.time()
        }
    
    def _get_cached_prediction(self, sequence_key: str) -> Optional[Dict]:
        """Get cached prediction if available."""
        return self._prediction_cache.get(sequence_key)
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache text embedding."""
        if len(self._embedding_cache) >= self.cache_size:
            # Remove oldest entries
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[text] = embedding.copy()
    
    def _get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        return self._embedding_cache.get(text)
    
    def predict_next_token(self, dialogue_sequence: List[str]) -> str:
        """
        Predict the next token given a dialogue sequence with caching and optimization.
        
        Args:
            dialogue_sequence: List of dialogue turns
            
        Returns:
            Predicted next dialogue turn as text
        """
        self._ensure_model_loaded()
        self._ensure_tokenizer_loaded()
        
        if not self.tokenizer.is_fitted:
            raise TokenizerNotFittedError("predict_next_token")
        
        # Validate input length
        expected_length = self.trainer.window_size
        if len(dialogue_sequence) != expected_length:
            raise InvalidInputError(
                "dialogue_sequence",
                f"list with {expected_length} dialogue turns",
                f"list with {len(dialogue_sequence)} turns"
            )
        
        # Check cache first
        sequence_key = self._get_sequence_hash(tuple(dialogue_sequence))
        cached_result = self._get_cached_prediction(sequence_key)
        if cached_result:
            logger.debug("Returning cached prediction")
            return cached_result['prediction']
        
        try:
            start_time = time.time()
            
            # Convert to embeddings with caching
            sequence_embeddings = self._encode_with_cache(dialogue_sequence)
            sequence_embeddings = np.expand_dims(sequence_embeddings, axis=0)  # Add batch dim
            
            # Make prediction
            prediction_embedding = self.trainer.predict(sequence_embeddings)[0]
            
            # Convert back to text using tokenizer decoding
            predicted_text = self.tokenizer.decode_embedding(prediction_embedding)
            
            # Cache the result
            self._cache_prediction(sequence_key, predicted_text)
            
            prediction_time = time.time() - start_time
            logger.debug("Prediction completed successfully", 
                        input_length=len(dialogue_sequence),
                        predicted_text=predicted_text[:50] + "..." if len(predicted_text) > 50 else predicted_text,
                        prediction_time=f"{prediction_time:.3f}s")
            
            # Memory management
            self._manage_memory()
            
            return predicted_text
            
        except Exception as e:
            logger.exception("Prediction failed", 
                           input_sequence=dialogue_sequence,
                           error_type=type(e).__name__)
            raise PredictionError(f"Failed to predict next token: {e}", 
                                input_shape=sequence_embeddings.shape if 'sequence_embeddings' in locals() else None)
    
    def _encode_with_cache(self, texts: List[str]) -> np.ndarray:
        """Encode texts with embedding caching."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts in batch
        if uncached_texts:
            uncached_embeddings = self.tokenizer.encode(uncached_texts)
            
            # Cache and insert the new embeddings
            for idx, (text, embedding) in enumerate(zip(uncached_texts, uncached_embeddings)):
                self._cache_embedding(text, embedding)
                embeddings[uncached_indices[idx]] = embedding
        
        return np.array(embeddings)
    
    def predict_with_confidence(self, dialogue_sequence: List[str]) -> Tuple[str, float]:
        """
        Predict the next token with confidence score using caching.
        
        Args:
            dialogue_sequence: List of dialogue turns
            
        Returns:
            Tuple of (predicted_text, confidence_score)
        """
        self._ensure_model_loaded()
        self._ensure_tokenizer_loaded()
        
        if not self.tokenizer.is_fitted:
            raise TokenizerNotFittedError("predict_with_confidence")
        
        # Check cache first
        sequence_key = self._get_sequence_hash(tuple(dialogue_sequence))
        cached_result = self._get_cached_prediction(sequence_key)
        if cached_result and cached_result.get('confidence') is not None:
            return cached_result['prediction'], cached_result['confidence']
        
        try:
            # Get prediction embedding with caching
            sequence_embeddings = self._encode_with_cache(dialogue_sequence)
            sequence_embeddings = np.expand_dims(sequence_embeddings, axis=0)
            prediction_embedding = self.trainer.predict(sequence_embeddings)[0]
            
            # Get top candidates with similarities
            top_candidates = self.tokenizer.get_closest_texts(prediction_embedding, top_k=1)
            
            if top_candidates:
                predicted_text, confidence = top_candidates[0]
                # Cache with confidence
                self._cache_prediction(sequence_key, predicted_text, confidence)
                return predicted_text, confidence
            else:
                result = "[UNKNOWN]", 0.0
                self._cache_prediction(sequence_key, result[0], result[1])
                return result
                
        except Exception as e:
            logger.error(f"Prediction with confidence failed: {e}")
            return "[ERROR]", 0.0
    
    def predict_top_k(self, dialogue_sequence: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """
        Predict top-k most likely next tokens with caching.
        
        Args:
            dialogue_sequence: List of dialogue turns
            k: Number of top predictions to return
            
        Returns:
            List of (predicted_text, confidence_score) tuples, sorted by confidence
        """
        self._ensure_model_loaded()
        self._ensure_tokenizer_loaded()
        
        if not self.tokenizer.is_fitted:
            raise TokenizerNotFittedError("predict_top_k")
        
        try:
            # Get prediction embedding with caching
            sequence_embeddings = self._encode_with_cache(dialogue_sequence)
            sequence_embeddings = np.expand_dims(sequence_embeddings, axis=0)
            prediction_embedding = self.trainer.predict(sequence_embeddings)[0]
            
            # Get top-k candidates
            return self.tokenizer.get_closest_texts(prediction_embedding, top_k=k)
            
        except Exception as e:
            logger.error(f"Top-k prediction failed: {e}")
            return [("[ERROR]", 0.0)]
    
    def batch_predict(self, dialogue_sequences: List[List[str]], 
                     batch_size: Optional[int] = None) -> List[str]:
        """
        Predict next tokens for a batch of sequences with memory-efficient processing.
        
        Args:
            dialogue_sequences: List of dialogue sequences
            batch_size: Optional batch size override for memory management
            
        Returns:
            List of predicted next tokens
        """
        if not dialogue_sequences:
            return []
        
        self._ensure_model_loaded()
        self._ensure_tokenizer_loaded()
        
        if not self.tokenizer.is_fitted:
            raise TokenizerNotFittedError("batch_predict")
        
        # Use configured batch size if not specified
        if batch_size is None:
            batch_size = min(self.max_batch_size, len(dialogue_sequences))
        
        predictions = []
        
        # Validate all sequences have correct length
        expected_length = self.trainer.window_size
        for i, seq in enumerate(dialogue_sequences):
            if len(seq) != expected_length:
                raise InvalidInputError(
                    f"dialogue_sequences[{i}]",
                    f"list with {expected_length} dialogue turns",
                    f"list with {len(seq)} turns"
                )
        
        logger.info(f"Processing {len(dialogue_sequences)} sequences in batches of {batch_size}")
        
        # Process in memory-efficient batches
        for batch_start in range(0, len(dialogue_sequences), batch_size):
            batch_end = min(batch_start + batch_size, len(dialogue_sequences))
            batch_sequences = dialogue_sequences[batch_start:batch_end]
            
            try:
                # Check cache for batch items first
                batch_predictions = []
                uncached_sequences = []
                uncached_indices = []
                
                for i, sequence in enumerate(batch_sequences):
                    sequence_key = self._get_sequence_hash(tuple(sequence))
                    cached_result = self._get_cached_prediction(sequence_key)
                    if cached_result:
                        batch_predictions.append(cached_result['prediction'])
                    else:
                        batch_predictions.append(None)  # Placeholder
                        uncached_sequences.append(sequence)
                        uncached_indices.append(i)
                
                # Process uncached sequences
                if uncached_sequences:
                    # Convert sequences to embeddings with caching
                    all_embeddings = []
                    for sequence in uncached_sequences:
                        seq_embeddings = self._encode_with_cache(sequence)
                        all_embeddings.append(seq_embeddings)
                    
                    # Stack into batch
                    batch_embeddings = np.stack(all_embeddings, axis=0)
                    
                    # Make batch prediction
                    prediction_embeddings = self.trainer.predict(batch_embeddings)
                    
                    # Decode predictions
                    uncached_predictions = self.tokenizer.decode_embeddings_batch(prediction_embeddings)
                    
                    # Cache and insert results
                    for idx, (sequence, prediction) in enumerate(zip(uncached_sequences, uncached_predictions)):
                        sequence_key = self._get_sequence_hash(tuple(sequence))
                        self._cache_prediction(sequence_key, prediction)
                        batch_predictions[uncached_indices[idx]] = prediction
                
                predictions.extend(batch_predictions)
                
                # Memory management after each batch
                self._manage_memory()
                
            except Exception as e:
                logger.warning(f"Error in batch prediction for batch {batch_start}-{batch_end}: {e}")
                # Fallback to individual predictions for this batch
                for sequence in batch_sequences:
                    try:
                        pred = self.predict_next_token(sequence)
                        predictions.append(pred)
                    except Exception as seq_error:
                        logger.error(f"Error predicting for sequence: {seq_error}")
                        predictions.append("[ERROR]")
        
        logger.info(f"Batch prediction completed: {len(predictions)} results")
        return predictions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        self._ensure_model_loaded()
        
        if self.trainer is None:
            return {"error": "No model loaded"}
        
        info = self.trainer.get_model_info()
        
        # Add tokenizer info
        if self.tokenizer:
            info['tokenizer'] = self.tokenizer.get_config()
        
        # Add configuration info
        if self.config:
            info['configuration'] = self.config.to_dict()
        
        # Add performance info
        info['performance'] = {
            'cache_size': len(self._prediction_cache),
            'embedding_cache_size': len(self._embedding_cache),
            'lazy_loading_enabled': self.lazy_load,
            'max_batch_size': self.max_batch_size
        }
        
        return info
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for performance monitoring."""
        return {
            'prediction_cache': {
                'size': len(self._prediction_cache),
                'max_size': self.cache_size,
                'hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1)
            },
            'embedding_cache': {
                'size': len(self._embedding_cache),
                'max_size': self.cache_size
            },
            'memory_management': {
                'last_gc_time': self._last_gc_time,
                'gc_interval': self._gc_interval,
                'memory_threshold_mb': self._memory_threshold_mb
            }
        }
    
    def clear_caches(self):
        """Manually clear all caches."""
        self._clear_caches()
        logger.info("All caches cleared manually")
    
    def interactive_session(self):
        """Enhanced interactive session with performance monitoring."""
        self._ensure_model_loaded()
        
        print(f"\n🎯 Interactive Mode (Optimized)")
        print(f"Enter {self.trainer.window_size} dialogue turns separated by spaces")
        print("Commands: 'quit' to exit, 'info' for model info, 'cache' for cache stats, 'clear' to clear cache")
        print("-" * 50)
        
        session_start = time.time()
        prediction_count = 0
        
        while True:
            try:
                user_input = input("\n💬 Enter dialogue sequence: ").strip()
                
                if user_input.lower() == 'quit':
                    session_time = time.time() - session_start
                    print(f"👋 Session completed! {prediction_count} predictions in {session_time:.1f}s")
                    if prediction_count > 0:
                        print(f"Average time per prediction: {session_time/prediction_count:.3f}s")
                    break
                elif user_input.lower() == 'info':
                    info = self.get_model_info()
                    print(f"Model: {info['architecture']['reservoir_type']} reservoir")
                    print(f"Window size: {info['architecture']['window_size']}")
                    print(f"Cache size: {info['performance']['cache_size']}")
                    continue
                elif user_input.lower() == 'cache':
                    stats = self.get_cache_stats()
                    print(f"Prediction cache: {stats['prediction_cache']['size']}/{stats['prediction_cache']['max_size']}")
                    print(f"Embedding cache: {stats['embedding_cache']['size']}/{stats['embedding_cache']['max_size']}")
                    continue
                elif user_input.lower() == 'clear':
                    self.clear_caches()
                    print("✓ Caches cleared")
                    continue
                elif user_input.lower() == 'help':
                    print(f"Enter exactly {self.trainer.window_size} dialogue turns")
                    print("Example: hello how are you doing today")
                    continue
                
                # Parse input
                sequence = user_input.split()
                
                # Validate input
                is_valid, error_msg = self.validate_input(sequence)
                if not is_valid:
                    print(f"❌ {error_msg}")
                    continue
                
                # Make prediction with timing
                start_time = time.time()
                prediction = self.predict_next_token(sequence)
                prediction_time = time.time() - start_time
                prediction_count += 1
                
                print(f"🔮 Predicted next token: '{prediction}' ({prediction_time:.3f}s)")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def validate_input(self, dialogue_sequence: List[str]) -> Tuple[bool, str]:
        """
        Validate input dialogue sequence.
        
        Args:
            dialogue_sequence: List of dialogue turns to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(dialogue_sequence, list):
            return False, "Input must be a list of strings"
        
        if not dialogue_sequence:
            return False, "Input sequence cannot be empty"
        
        expected_length = self.trainer.window_size if self.trainer else 10
        if len(dialogue_sequence) != expected_length:
            return False, f"Sequence length must be {expected_length}, got {len(dialogue_sequence)}"
        
        for i, turn in enumerate(dialogue_sequence):
            if not isinstance(turn, str):
                return False, f"Turn {i} must be a string, got {type(turn)}"
            
            if not turn.strip():
                return False, f"Turn {i} cannot be empty or whitespace only"
        
        return True, ""


# Legacy class for backward compatibility
class LSMInference(OptimizedLSMInference):
    """
    Legacy inference class that maintains backward compatibility.
    Inherits from OptimizedLSMInference but disables optimizations by default.
    """
    
    def __init__(self, model_path: str):
        """Initialize with legacy behavior (no lazy loading, smaller caches)."""
        super().__init__(
            model_path=model_path,
            lazy_load=False,  # Load immediately for backward compatibility
            cache_size=100,   # Smaller cache
            max_batch_size=16  # Smaller batch size
        )
    
    def interactive_session(self):
        """Legacy interactive session without performance monitoring."""
        self._ensure_model_loaded()
        
        print(f"\n🎯 Interactive Mode")
        print(f"Enter {self.trainer.window_size} dialogue turns separated by spaces")
        print("Commands: 'quit' to exit, 'info' for model info, 'help' for help")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Enter dialogue sequence: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'info':
                    info = self.get_model_info()
                    print(f"Model: {info['architecture']['reservoir_type']} reservoir")
                    print(f"Window size: {info['architecture']['window_size']}")
                    continue
                elif user_input.lower() == 'help':
                    print(f"Enter exactly {self.trainer.window_size} dialogue turns")
                    print("Example: hello how are you doing today")
                    continue
                
                # Parse input
                sequence = user_input.split()
                
                # Validate input
                is_valid, error_msg = self.validate_input(sequence)
                if not is_valid:
                    print(f"❌ {error_msg}")
                    continue
                
                # Make prediction
                prediction = self.predict_next_token(sequence)
                print(f"🔮 Predicted next token: '{prediction}'")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Enhanced CLI interface for inference with performance optimizations."""
    parser = argparse.ArgumentParser(
        description="Enhanced LSM Inference with complete text processing and performance optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to saved model directory')
    parser.add_argument('--input-text', type=str, nargs='+',
                       help='Input dialogue sequence (space-separated)')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--batch-file', type=str,
                       help='File containing batch input sequences (one per line)')
    parser.add_argument('--top-k', type=int, default=1,
                       help='Number of top predictions to show')
    parser.add_argument('--show-confidence', action='store_true',
                       help='Show confidence scores with predictions')
    parser.add_argument('--model-info', action='store_true',
                       help='Display model information and exit')
    
    # Performance optimization options
    parser.add_argument('--lazy-load', action='store_true', default=True,
                       help='Enable lazy loading of model components')
    parser.add_argument('--cache-size', type=int, default=1000,
                       help='Size of prediction cache')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Maximum batch size for memory efficiency')
    parser.add_argument('--legacy-mode', action='store_true',
                       help='Use legacy inference mode (no optimizations)')
    
    args = parser.parse_args()
    
    # Initialize inference with optimizations
    try:
        print("Initializing LSM inference...")
        
        if args.legacy_mode:
            print("Using legacy inference mode")
            inference = LSMInference(args.model_path)
        else:
            print("Using optimized inference mode")
            inference = OptimizedLSMInference(
                model_path=args.model_path,
                lazy_load=args.lazy_load,
                cache_size=args.cache_size,
                max_batch_size=args.batch_size
            )
        
        print("✓ Inference system ready!")
    except Exception as e:
        print(f"❌ Failed to initialize inference: {e}")
        return 1
    
    # Show model info if requested
    if args.model_info:
        info = inference.get_model_info()
        print("\n📊 Model Information:")
        print("=" * 50)
        
        if 'architecture' in info:
            arch = info['architecture']
            print(f"Window Size: {arch['window_size']}")
            print(f"Embedding Dim: {arch['embedding_dim']}")
            print(f"Reservoir Type: {arch['reservoir_type']}")
            print(f"Multi-channel: {arch['use_multichannel']}")
        
        if 'tokenizer' in info:
            tok = info['tokenizer']
            print(f"Vocabulary Size: {tok['vocabulary_size']}")
            print(f"Max Features: {tok['max_features']}")
        
        if 'training_summary' in info:
            train = info['training_summary']
            print(f"Epochs Trained: {train['epochs_trained']}")
            print(f"Final Test MSE: {train.get('final_test_mse', 'N/A')}")
        
        return 0
    
    # Interactive mode
    if args.interactive:
        print(f"\n🎯 Interactive Mode")
        print(f"Enter {inference.trainer.window_size} dialogue turns separated by spaces")
        print("Commands: 'quit' to exit, 'info' for model info, 'help' for help")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n💬 Enter dialogue sequence: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'info':
                    info = inference.get_model_info()
                    print(f"Model: {info['architecture']['reservoir_type']} reservoir")
                    print(f"Window size: {info['architecture']['window_size']}")
                    continue
                elif user_input.lower() == 'help':
                    print(f"Enter exactly {inference.trainer.window_size} dialogue turns")
                    print("Example: hello how are you doing today")
                    continue
                
                # Parse input
                sequence = user_input.split()
                
                # Validate input
                is_valid, error_msg = inference.validate_input(sequence)
                if not is_valid:
                    print(f"❌ {error_msg}")
                    continue
                
                # Make prediction
                if args.show_confidence or args.top_k > 1:
                    predictions = inference.predict_top_k(sequence, args.top_k)
                    print(f"\n🔮 Top {len(predictions)} predictions:")
                    for i, (text, confidence) in enumerate(predictions, 1):
                        print(f"  {i}. '{text}' (confidence: {confidence:.3f})")
                else:
                    prediction = inference.predict_next_token(sequence)
                    print(f"🔮 Predicted next token: '{prediction}'")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    # Single prediction mode
    elif args.input_text:
        is_valid, error_msg = inference.validate_input(args.input_text)
        if not is_valid:
            print(f"❌ {error_msg}")
            return 1
        
        try:
            print(f"📝 Input: {' '.join(args.input_text)}")
            
            if args.show_confidence or args.top_k > 1:
                predictions = inference.predict_top_k(args.input_text, args.top_k)
                print(f"\n🔮 Top {len(predictions)} predictions:")
                for i, (text, confidence) in enumerate(predictions, 1):
                    print(f"  {i}. '{text}' (confidence: {confidence:.3f})")
            else:
                prediction = inference.predict_next_token(args.input_text)
                print(f"🔮 Predicted next token: '{prediction}'")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 1
    
    # Batch file mode
    elif args.batch_file:
        if not os.path.exists(args.batch_file):
            print(f"❌ Batch file not found: {args.batch_file}")
            return 1
        
        try:
            with open(args.batch_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            sequences = [line.split() for line in lines]
            
            # Validate all sequences
            for i, seq in enumerate(sequences):
                is_valid, error_msg = inference.validate_input(seq)
                if not is_valid:
                    print(f"❌ Line {i+1}: {error_msg}")
                    return 1
            
            print(f"📁 Processing {len(sequences)} sequences from {args.batch_file}")
            
            # Use custom batch size if specified
            batch_size = args.batch_size if hasattr(args, 'batch_size') else None
            start_time = time.time()
            predictions = inference.batch_predict(sequences, batch_size=batch_size)
            batch_time = time.time() - start_time
            
            print(f"\n🔮 Batch Predictions (completed in {batch_time:.2f}s):")
            for i, (seq, pred) in enumerate(zip(sequences, predictions), 1):
                print(f"  {i}. '{' '.join(seq)}' → '{pred}'")
                
        except Exception as e:
            print(f"❌ Error processing batch file: {e}")
            return 1
    
    else:
        print("❌ Please provide --input-text, --interactive, --batch-file, or --model-info")
        return 1
    
    return 0

if __name__ == "__main__":
    main()