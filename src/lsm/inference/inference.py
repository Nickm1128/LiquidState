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

from ..training.train import LSMTrainer
from ..data.data_loader import DialogueTokenizer
from ..data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from ..inference.response_generator import ResponseGenerator, ResponseGenerationResult
from ..inference.reservoir_manager import ReservoirManager
from ..core.system_message_processor import SystemMessageProcessor, SystemMessageContext
from ..core.cnn_3d_processor import CNN3DProcessor, SystemContext
from ..training.model_config import ModelConfiguration
from ..utils.lsm_exceptions import (
    ModelLoadError, InferenceError, InvalidInputError, PredictionError,
    TokenizerNotFittedError, create_error_context
)
from ..utils.lsm_logging import get_logger, log_performance, create_operation_logger
from ..utils.input_validation import (
    validate_file_path, validate_dialogue_sequence, validate_positive_integer,
    create_helpful_error_message
)

logger = get_logger(__name__)

class EnhancedLSMInference:
    """
    Enhanced inference class supporting new tokenization and response-level generation.
    Features StandardTokenizerWrapper, SinusoidalEmbedder, and complete response generation.
    """
    
    def __init__(self, model_path: str, 
                 use_response_level: bool = True,
                 tokenizer_name: str = 'gpt2',
                 lazy_load: bool = True, 
                 cache_size: int = 1000, 
                 max_batch_size: int = 32):
        """
        Initialize enhanced inference with new tokenization system.
        
        Args:
            model_path: Path to saved model directory
            use_response_level: If True, use response-level inference instead of token-level
            tokenizer_name: Name of standard tokenizer to use
            lazy_load: If True, load models only when needed
            cache_size: Size of prediction cache
            max_batch_size: Maximum batch size for memory efficiency
        """
        self.model_path = model_path
        self.use_response_level = use_response_level
        self.tokenizer_name = tokenizer_name
        self.lazy_load = lazy_load
        self.cache_size = cache_size
        self.max_batch_size = max_batch_size
        
        # Model components (loaded lazily if enabled)
        self.trainer = None
        self.legacy_tokenizer = None  # DialogueTokenizer for backward compatibility
        self.standard_tokenizer = None  # StandardTokenizerWrapper
        self.sinusoidal_embedder = None  # SinusoidalEmbedder
        self.response_generator = None  # ResponseGenerator for response-level inference
        self.reservoir_manager = None  # ReservoirManager
        self.system_message_processor = None  # SystemMessageProcessor for system message handling
        self.cnn_3d_processor = None  # CNN3DProcessor for 3D CNN inference with system context
        self.config = None
        
        # Performance optimizations
        self._prediction_cache = {}
        self._embedding_cache = {}
        self._response_cache = {}
        self._model_loaded = False
        self._tokenizer_loaded = False
        self._enhanced_components_loaded = False
        self._load_lock = threading.Lock()
        
        # Memory management
        self._memory_threshold_mb = 1024  # 1GB threshold
        self._last_gc_time = time.time()
        self._gc_interval = 30  # Run GC every 30 seconds
        
        # Load immediately if not lazy loading
        if not lazy_load:
            self._load_complete_model()
    
    def _load_enhanced_components(self):
        """Load enhanced tokenization and response generation components."""
        with self._load_lock:
            if self._enhanced_components_loaded:
                return
                
            logger.info("Loading enhanced tokenization and response generation components")
            start_time = time.time()
            
            try:
                # Load or create StandardTokenizerWrapper
                tokenizer_path = os.path.join(self.model_path, "standard_tokenizer")
                if os.path.exists(tokenizer_path):
                    self.standard_tokenizer = StandardTokenizerWrapper.load(tokenizer_path)
                    logger.info("Loaded existing StandardTokenizerWrapper")
                else:
                    # Create new tokenizer with specified name
                    self.standard_tokenizer = StandardTokenizerWrapper(
                        tokenizer_name=self.tokenizer_name
                    )
                    logger.info(f"Created new StandardTokenizerWrapper with {self.tokenizer_name}")
                
                # Load or create SinusoidalEmbedder
                embedder_path = os.path.join(self.model_path, "sinusoidal_embedder")
                if os.path.exists(embedder_path):
                    self.sinusoidal_embedder = SinusoidalEmbedder.load(embedder_path)
                    logger.info("Loaded existing SinusoidalEmbedder")
                else:
                    # Create new embedder with tokenizer vocab size
                    vocab_size = self.standard_tokenizer.get_vocab_size()
                    embedding_dim = 128  # Default embedding dimension
                    self.sinusoidal_embedder = SinusoidalEmbedder(
                        vocab_size=vocab_size,
                        embedding_dim=embedding_dim
                    )
                    logger.warning("Created new SinusoidalEmbedder - consider training it on data")
                
                # Initialize ReservoirManager if available
                try:
                    self.reservoir_manager = ReservoirManager(
                        reservoir_model=self.trainer.model if self.trainer else None
                    )
                except Exception as e:
                    logger.warning(f"Could not initialize ReservoirManager: {e}")
                
                # Initialize SystemMessageProcessor
                try:
                    self.system_message_processor = SystemMessageProcessor(
                        tokenizer=self.standard_tokenizer
                    )
                    logger.info("Initialized SystemMessageProcessor")
                except Exception as e:
                    logger.warning(f"Could not initialize SystemMessageProcessor: {e}")
                
                # Initialize CNN3DProcessor for system-aware inference
                try:
                    if self.trainer and self.trainer.model:
                        # Get reservoir output shape from model
                        reservoir_shape = self._get_reservoir_output_shape()
                        self.cnn_3d_processor = CNN3DProcessor(
                            reservoir_shape=reservoir_shape,
                            system_embedding_dim=self.sinusoidal_embedder.embedding_dim if self.sinusoidal_embedder else 128,
                            output_embedding_dim=self.sinusoidal_embedder.embedding_dim if self.sinusoidal_embedder else 128,
                            tokenizer=self.standard_tokenizer,
                            embedder=self.sinusoidal_embedder
                        )
                        logger.info("Initialized CNN3DProcessor for system-aware inference")
                except Exception as e:
                    logger.warning(f"Could not initialize CNN3DProcessor: {e}")

                # Initialize ResponseGenerator for response-level inference
                if self.use_response_level:
                    try:
                        self.response_generator = ResponseGenerator(
                            tokenizer=self.standard_tokenizer,
                            embedder=self.sinusoidal_embedder,
                            reservoir_model=self.trainer.model if self.trainer else None,
                            cnn_3d_processor=self.cnn_3d_processor
                        )
                        logger.info("Initialized ResponseGenerator for response-level inference")
                    except Exception as e:
                        logger.warning(f"Could not initialize ResponseGenerator: {e}")
                        self.use_response_level = False
                
                self._enhanced_components_loaded = True
                load_time = time.time() - start_time
                logger.info(f"Enhanced components loaded successfully in {load_time:.2f}s")
                
            except Exception as e:
                logger.exception("Failed to load enhanced components")
                raise ModelLoadError(self.model_path, f"Failed to load enhanced components: {e}")
    
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
                self.trainer, self.legacy_tokenizer = self.trainer.load_complete_model(self.model_path)
                
                self._model_loaded = True
                self._tokenizer_loaded = True
                
                # Load enhanced components if not lazy loading
                if not self.lazy_load:
                    self._load_enhanced_components()
                
                load_time = time.time() - start_time
                logger.info(f"Complete model loaded successfully in {load_time:.2f}s")
                
            except Exception as e:
                logger.exception("Failed to load complete model", model_path=self.model_path)
                raise ModelLoadError(self.model_path, f"Failed to load complete model: {e}")
    
    def _ensure_enhanced_components_loaded(self):
        """Ensure enhanced components are loaded (lazy loading support)."""
        if not self._enhanced_components_loaded:
            self._load_enhanced_components()
    
    def generate_response(self, input_text: str, system_message: Optional[str] = None) -> str:
        """
        Generate a complete response using the enhanced inference system with system message support.
        
        Args:
            input_text: Input text to generate response for
            system_message: Optional system message to influence generation
            
        Returns:
            Generated response text
        """
        self._ensure_model_loaded()
        self._ensure_enhanced_components_loaded()
        
        if not self.use_response_level or not self.response_generator:
            # Fallback to token-level generation with system message support
            return self._generate_token_level_response(input_text, system_message)
        
        # Check cache first
        cache_key = f"response:{hash(input_text)}:{hash(system_message) if system_message else 'none'}"
        cached_result = self._response_cache.get(cache_key)
        if cached_result:
            logger.debug("Returning cached response")
            return cached_result['response']
        
        try:
            start_time = time.time()
            
            # Tokenize input text
            token_ids = self.standard_tokenizer.encode_single(input_text)
            
            # Convert to embeddings
            embeddings = self.sinusoidal_embedder.embed(token_ids)
            
            # Generate response using ResponseGenerator with system message support
            if system_message and self.system_message_processor:
                # Process system message
                system_context = self._process_system_message(system_message)
                
                # Use 3D CNN inference if available
                if self.cnn_3d_processor:
                    result = self._generate_system_aware_response(embeddings, system_context)
                else:
                    # Fallback to ResponseGenerator with system message
                    result = self.response_generator.generate_with_system_message(
                        embeddings, system_message
                    )
            else:
                result = self.response_generator.generate_complete_response(embeddings)
            
            # Cache the result
            generation_time = time.time() - start_time
            self._response_cache[cache_key] = {
                'response': result.response_text if hasattr(result, 'response_text') else result,
                'confidence': getattr(result, 'confidence_score', 1.0),
                'generation_time': generation_time,
                'timestamp': time.time(),
                'system_influence': getattr(result, 'system_influence', None)
            }
            
            # Manage cache size
            if len(self._response_cache) > self.cache_size:
                oldest_key = min(self._response_cache.keys(), 
                               key=lambda k: self._response_cache[k]['timestamp'])
                del self._response_cache[oldest_key]
            
            response_text = result.response_text if hasattr(result, 'response_text') else result
            logger.debug(f"Response generated successfully in {generation_time:.3f}s")
            return response_text
            
        except Exception as e:
            logger.exception("Response generation failed")
            # Fallback to token-level generation
            return self._generate_token_level_response(input_text, system_message)
    
    def _generate_token_level_response(self, input_text: str, system_message: Optional[str] = None) -> str:
        """Fallback method for token-level response generation with system message support."""
        try:
            # Convert input text to dialogue sequence format expected by legacy system
            # This is a simplified conversion - in practice, you might need more sophisticated parsing
            words = input_text.split()
            
            # If system message is provided, prepend it to the input
            if system_message:
                system_words = system_message.split()
                words = system_words + ["[SEP]"] + words
            
            # Pad or truncate to expected window size
            if self.trainer and hasattr(self.trainer, 'window_size'):
                window_size = self.trainer.window_size
                if len(words) < window_size:
                    words.extend([''] * (window_size - len(words)))
                elif len(words) > window_size:
                    words = words[:window_size]
            
            # Use legacy prediction method
            if self.legacy_tokenizer and self.legacy_tokenizer.is_fitted:
                return self.predict_next_token(words)
            else:
                return "[Unable to generate response - tokenizer not available]"
                
        except Exception as e:
            logger.error(f"Token-level fallback failed: {e}")
            return "[Error generating response]"
    
    def _process_system_message(self, system_message: str) -> SystemMessageContext:
        """Process system message using SystemMessageProcessor."""
        try:
            if not self.system_message_processor:
                raise InferenceError("SystemMessageProcessor not available")
            
            # Process the system message
            context = self.system_message_processor.process_system_message(system_message)
            logger.debug(f"System message processed successfully: {context.parsed_content}")
            return context
            
        except Exception as e:
            logger.error(f"System message processing failed: {e}")
            # Return a basic context as fallback
            token_ids = self.standard_tokenizer.encode_single(system_message)
            embeddings = self.sinusoidal_embedder.embed(token_ids)
            return SystemMessageContext(
                original_message=system_message,
                parsed_content={"type": "unknown", "content": system_message},
                token_ids=token_ids,
                embeddings=embeddings,
                metadata={},
                validation_status=False
            )
    
    def _generate_system_aware_response(self, input_embeddings: np.ndarray, 
                                      system_context: SystemMessageContext) -> ResponseGenerationResult:
        """Generate response using 3D CNN with system context."""
        try:
            start_time = time.time()
            
            # Process input through reservoir first
            if self.trainer and self.trainer.model:
                # Reshape for model input
                embeddings_batch = np.expand_dims(input_embeddings, axis=0)
                reservoir_output = self.trainer.model.predict(embeddings_batch)[0]
            else:
                # Use input embeddings directly if no reservoir model
                reservoir_output = input_embeddings
            
            # Create system context for 3D CNN
            system_ctx = SystemContext(
                message=system_context.original_message,
                embeddings=system_context.embeddings,
                influence_strength=1.0,
                processing_mode="3d_cnn"
            )
            
            # Process through 3D CNN with system context
            processing_result = self.cnn_3d_processor.process_with_system_context(
                reservoir_output, system_ctx
            )
            
            # Convert output embeddings back to text
            response_text = self._decode_embeddings_to_text(processing_result.output_embeddings)
            
            generation_time = time.time() - start_time
            
            return ResponseGenerationResult(
                response_text=response_text,
                confidence_score=0.8,  # Default confidence for 3D CNN processing
                generation_time=generation_time,
                reservoir_strategy_used="3d_cnn_system_aware",
                system_influence=processing_result.system_influence
            )
            
        except Exception as e:
            logger.error(f"System-aware response generation failed: {e}")
            # Fallback to regular response generation
            return ResponseGenerationResult(
                response_text="[System-aware generation failed]",
                confidence_score=0.0,
                generation_time=0.0,
                reservoir_strategy_used="fallback"
            )
    
    def _decode_embeddings_to_text(self, embeddings: np.ndarray) -> str:
        """Decode embeddings back to text using the tokenizer and embedder."""
        try:
            # Get the embedding matrix from the sinusoidal embedder
            if hasattr(self.sinusoidal_embedder, 'get_embedding_matrix'):
                embedding_matrix = self.sinusoidal_embedder.get_embedding_matrix()
            else:
                # Fallback: use the embedder's internal embeddings
                embedding_matrix = self.sinusoidal_embedder.embeddings
            
            # Find closest embeddings for each position
            if len(embeddings.shape) == 1:
                # Single embedding vector
                similarities = np.dot(embedding_matrix, embeddings)
                best_token_id = np.argmax(similarities)
                return self.standard_tokenizer.decode([best_token_id])
            else:
                # Sequence of embeddings
                token_ids = []
                for emb in embeddings:
                    similarities = np.dot(embedding_matrix, emb)
                    best_token_id = np.argmax(similarities)
                    token_ids.append(best_token_id)
                
                return self.standard_tokenizer.decode(token_ids)
                
        except Exception as e:
            logger.error(f"Embedding decoding failed: {e}")
            return "[Decoding error]"
    
    def _get_reservoir_output_shape(self) -> Tuple[int, int, int, int]:
        """Get the output shape of the reservoir model for 3D CNN initialization."""
        try:
            if self.trainer and self.trainer.model:
                # Get a sample input to determine output shape
                sample_input = np.random.randn(1, 128)  # Assuming 128-dim embeddings
                sample_output = self.trainer.model.predict(sample_input)
                
                # Convert to 4D shape for 3D CNN (depth, height, width, channels)
                output_shape = sample_output.shape[1:]
                if len(output_shape) == 1:
                    # 1D output - reshape to 4D
                    dim = output_shape[0]
                    # Create a reasonable 4D shape
                    depth = min(8, dim // 16)
                    height = min(8, dim // (depth * 2))
                    width = min(8, dim // (depth * height))
                    channels = max(1, dim // (depth * height * width))
                    return (depth, height, width, channels)
                elif len(output_shape) == 2:
                    # 2D output - add depth and channels
                    return (1, output_shape[0], output_shape[1], 1)
                elif len(output_shape) == 3:
                    # 3D output - add channels
                    return (*output_shape, 1)
                else:
                    # Already 4D
                    return output_shape
            else:
                # Default shape if no model available
                return (4, 8, 8, 16)
                
        except Exception as e:
            logger.warning(f"Could not determine reservoir output shape: {e}")
            return (4, 8, 8, 16)  # Default fallback shape
    
    def predict_with_enhanced_tokenizer(self, input_text: str) -> Tuple[str, float]:
        """
        Predict using the enhanced tokenization system.
        
        Args:
            input_text: Input text for prediction
            
        Returns:
            Tuple of (predicted_text, confidence_score)
        """
        self._ensure_model_loaded()
        self._ensure_enhanced_components_loaded()
        
        try:
            # Tokenize with standard tokenizer
            token_ids = self.standard_tokenizer.encode_single(input_text)
            
            # Convert to embeddings
            embeddings = self.sinusoidal_embedder.embed(token_ids)
            
            # Process through reservoir if available
            if self.trainer and self.trainer.model:
                # Reshape for model input
                embeddings_batch = np.expand_dims(embeddings, axis=0)
                reservoir_output = self.trainer.model.predict(embeddings_batch)[0]
                
                # Convert reservoir output back to text
                # This is a simplified approach - you might need more sophisticated decoding
                predicted_text = self._decode_reservoir_output(reservoir_output)
                confidence = self._calculate_confidence(reservoir_output)
                
                return predicted_text, confidence
            else:
                return "[No trained model available]", 0.0
                
        except Exception as e:
            logger.error(f"Enhanced prediction failed: {e}")
            return "[Prediction error]", 0.0
    
    def _decode_reservoir_output(self, reservoir_output: np.ndarray) -> str:
        """Decode reservoir output to text using enhanced tokenizer."""
        try:
            # Find closest embedding in the sinusoidal embedder
            embedding_matrix = self.sinusoidal_embedder.get_embedding_matrix()
            
            # Calculate similarities
            similarities = np.dot(embedding_matrix, reservoir_output)
            best_token_id = np.argmax(similarities)
            
            # Decode token ID to text
            predicted_text = self.standard_tokenizer.decode([best_token_id])
            return predicted_text.strip()
            
        except Exception as e:
            logger.error(f"Decoding failed: {e}")
            return "[Decoding error]"
    
    def _calculate_confidence(self, reservoir_output: np.ndarray) -> float:
        """Calculate confidence score for reservoir output."""
        try:
            # Simple confidence based on output magnitude and consistency
            magnitude = np.linalg.norm(reservoir_output)
            consistency = 1.0 - np.std(reservoir_output) / (np.mean(np.abs(reservoir_output)) + 1e-8)
            
            # Normalize to [0, 1] range
            confidence = min(1.0, max(0.0, magnitude * consistency * 0.1))
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def generate_system_aware_response(self, input_text: str, system_message: str) -> Dict[str, Any]:
        """
        Generate a response with explicit system message influence using 3D CNN processing.
        
        Args:
            input_text: Input text to generate response for
            system_message: System message to influence generation
            
        Returns:
            Dictionary containing response and metadata
        """
        self._ensure_model_loaded()
        self._ensure_enhanced_components_loaded()
        
        if not self.system_message_processor or not self.cnn_3d_processor:
            # Fallback to regular response generation
            response = self.generate_response(input_text, system_message)
            return {
                'response': response,
                'method': 'fallback',
                'system_influence': None,
                'confidence': 0.5
            }
        
        try:
            start_time = time.time()
            
            # Process system message
            system_context = self._process_system_message(system_message)
            
            # Tokenize and embed input text
            token_ids = self.standard_tokenizer.encode_single(input_text)
            embeddings = self.sinusoidal_embedder.embed(token_ids)
            
            # Generate system-aware response
            result = self._generate_system_aware_response(embeddings, system_context)
            
            generation_time = time.time() - start_time
            
            return {
                'response': result.response_text,
                'method': '3d_cnn_system_aware',
                'system_influence': result.system_influence,
                'confidence': result.confidence_score,
                'generation_time': generation_time,
                'system_message_parsed': system_context.parsed_content
            }
            
        except Exception as e:
            logger.error(f"System-aware response generation failed: {e}")
            # Fallback to regular response generation
            response = self.generate_response(input_text, system_message)
            return {
                'response': response,
                'method': 'fallback_after_error',
                'system_influence': None,
                'confidence': 0.3,
                'error': str(e)
            }
    
    def get_enhanced_model_info(self) -> Dict[str, Any]:
        """Get information about the enhanced model components."""
        self._ensure_model_loaded()
        
        info = {}
        
        # Base model info
        if self.trainer:
            info.update(self.trainer.get_model_info())
        
        # Enhanced components info
        if self._enhanced_components_loaded:
            info['enhanced_components'] = {
                'standard_tokenizer': {
                    'name': self.standard_tokenizer.tokenizer_name if self.standard_tokenizer else None,
                    'vocab_size': self.standard_tokenizer.get_vocab_size() if self.standard_tokenizer else None
                },
                'sinusoidal_embedder': {
                    'vocab_size': self.sinusoidal_embedder.vocab_size if self.sinusoidal_embedder else None,
                    'embedding_dim': self.sinusoidal_embedder.embedding_dim if self.sinusoidal_embedder else None,
                    'is_fitted': getattr(self.sinusoidal_embedder, '_is_fitted', False) if self.sinusoidal_embedder else False
                },
                'response_generator': self.response_generator is not None,
                'reservoir_manager': self.reservoir_manager is not None,
                'system_message_processor': self.system_message_processor is not None,
                'cnn_3d_processor': self.cnn_3d_processor is not None
            }
        
        # Configuration info
        if self.config:
            info['configuration'] = self.config.to_dict()
        
        # Performance info
        info['performance'] = {
            'use_response_level': self.use_response_level,
            'cache_sizes': {
                'prediction_cache': len(self._prediction_cache),
                'embedding_cache': len(self._embedding_cache),
                'response_cache': len(self._response_cache)
            },
            'lazy_loading_enabled': self.lazy_load,
            'max_batch_size': self.max_batch_size
        }
        
        return info
    
    def save_enhanced_components(self, save_path: Optional[str] = None):
        """
        Save enhanced components to disk.
        
        Args:
            save_path: Optional path to save components. If None, uses model_path.
        """
        if not self._enhanced_components_loaded:
            logger.warning("Enhanced components not loaded - nothing to save")
            return
        
        save_path = save_path or self.model_path
        
        try:
            # Save StandardTokenizerWrapper
            if self.standard_tokenizer:
                tokenizer_path = os.path.join(save_path, "standard_tokenizer")
                self.standard_tokenizer.save(tokenizer_path)
                logger.info(f"StandardTokenizerWrapper saved to {tokenizer_path}")
            
            # Save SinusoidalEmbedder
            if self.sinusoidal_embedder and getattr(self.sinusoidal_embedder, '_is_fitted', False):
                embedder_path = os.path.join(save_path, "sinusoidal_embedder")
                self.sinusoidal_embedder.save(embedder_path)
                logger.info(f"SinusoidalEmbedder saved to {embedder_path}")
            
            logger.info("Enhanced components saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save enhanced components: {e}")
            raise
    
    def interactive_enhanced_session(self):
        """Enhanced interactive session with new tokenization and response generation."""
        self._ensure_model_loaded()
        self._ensure_enhanced_components_loaded()
        
        print(f"\n🚀 Enhanced Interactive Mode")
        print("Features: StandardTokenizer, SinusoidalEmbedder, Response-level generation, System message support")
        print("Commands: 'quit' to exit, 'info' for model info, 'token' for token-level, 'response' for response-level")
        print("         'system <message>' to set system message, 'clear' to clear system message")
        print("-" * 70)
        
        session_start = time.time()
        prediction_count = 0
        current_system_message = None
        
        while True:
            try:
                user_input = input("\n💬 Enter text: ").strip()
                
                if user_input.lower() == 'quit':
                    session_time = time.time() - session_start
                    print(f"👋 Session completed! {prediction_count} predictions in {session_time:.1f}s")
                    if prediction_count > 0:
                        print(f"Average time per prediction: {session_time/prediction_count:.3f}s")
                    break
                elif user_input.lower() == 'info':
                    info = self.get_enhanced_model_info()
                    print(f"Enhanced components loaded: {self._enhanced_components_loaded}")
                    if 'enhanced_components' in info:
                        print(f"Tokenizer: {info['enhanced_components']['standard_tokenizer']['name']}")
                        print(f"Vocab size: {info['enhanced_components']['standard_tokenizer']['vocab_size']}")
                        print(f"Embedder fitted: {info['enhanced_components']['sinusoidal_embedder']['is_fitted']}")
                    continue
                elif user_input.lower() == 'token':
                    print("Switching to token-level prediction mode")
                    self.use_response_level = False
                    continue
                elif user_input.lower() == 'response':
                    print("Switching to response-level generation mode")
                    self.use_response_level = True
                    continue
                elif user_input.lower().startswith('system '):
                    system_msg = user_input[7:].strip()
                    if system_msg:
                        current_system_message = system_msg
                        print(f"✅ System message set: '{current_system_message}'")
                    else:
                        print("❌ Please provide a system message after 'system'")
                    continue
                elif user_input.lower() == 'clear':
                    current_system_message = None
                    print("✅ System message cleared")
                    continue
                elif user_input.lower() == 'help':
                    print("Enter any text to generate a response")
                    print("The system will use enhanced tokenization and embeddings")
                    print("Use 'system <message>' to set a system message that influences responses")
                    continue
                
                if not user_input:
                    print("❌ Please enter some text")
                    continue
                
                # Generate response with timing
                start_time = time.time()
                
                if self.use_response_level:
                    response = self.generate_response(user_input, current_system_message)
                    method = "Response-level"
                    if current_system_message:
                        method += " (with system message)"
                else:
                    response, confidence = self.predict_with_enhanced_tokenizer(user_input)
                    method = f"Token-level (confidence: {confidence:.3f})"
                
                prediction_time = time.time() - start_time
                prediction_count += 1
                
                print(f"🔮 {method}: '{response}' ({prediction_time:.3f}s)")
                if current_system_message:
                    print(f"📋 System: {current_system_message}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
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
        cache_size_before = len(self._prediction_cache) + len(self._embedding_cache) + len(self._response_cache)
        self._prediction_cache.clear()
        self._embedding_cache.clear()
        self._response_cache.clear()
        logger.debug(f"Cleared {cache_size_before} cached items")
    
    # Legacy methods for backward compatibility
    def predict_next_token(self, dialogue_sequence: List[str]) -> str:
        """Legacy method using DialogueTokenizer for backward compatibility."""
        self._ensure_model_loaded()
        self._ensure_tokenizer_loaded()
        
        if not self.legacy_tokenizer or not self.legacy_tokenizer.is_fitted:
            # Try to use enhanced tokenizer as fallback
            if self._enhanced_components_loaded and self.standard_tokenizer:
                input_text = " ".join(dialogue_sequence)
                response, _ = self.predict_with_enhanced_tokenizer(input_text)
                return response
            else:
                raise TokenizerNotFittedError("predict_next_token")
        
        # Use original implementation with legacy tokenizer
        return self._legacy_predict_next_token(dialogue_sequence)
    
    def _legacy_predict_next_token(self, dialogue_sequence: List[str]) -> str:
        """Original predict_next_token implementation."""
        # Validate input length
        expected_length = self.trainer.window_size
        if len(dialogue_sequence) != expected_length:
            raise InvalidInputError(
                "dialogue_sequence",
                f"list with {expected_length} dialogue turns",
                f"list with {len(dialogue_sequence)} turns"
            )
        
        # Check cache first
        sequence_key = str(hash(tuple(dialogue_sequence)))
        cached_result = self._prediction_cache.get(sequence_key)
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
            predicted_text = self.legacy_tokenizer.decode_embedding(prediction_embedding)
            
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
        """Encode texts with embedding caching using legacy tokenizer."""
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self._embedding_cache.get(text)
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Encode uncached texts in batch
        if uncached_texts:
            uncached_embeddings = self.legacy_tokenizer.encode(uncached_texts)
            
            # Cache and insert the new embeddings
            for idx, (text, embedding) in enumerate(zip(uncached_texts, uncached_embeddings)):
                self._cache_embedding(text, embedding)
                embeddings[uncached_indices[idx]] = embedding
        
        return np.array(embeddings)
    
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
    
    def _cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache text embedding."""
        if len(self._embedding_cache) >= self.cache_size:
            # Remove oldest entries
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[text] = embedding.copy()


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
    """Enhanced CLI interface for inference with new tokenization and response generation."""
    parser = argparse.ArgumentParser(
        description="Enhanced LSM Inference with StandardTokenizer, SinusoidalEmbedder, and response-level generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to saved model directory')
    parser.add_argument('--input-text', type=str,
                        help='Input text for response generation')
    parser.add_argument('--system-message', type=str,
                        help='Optional system message to influence generation')
    
    # Mode selection
    parser.add_argument('--interactive', action='store_true',
                        help='Run in enhanced interactive mode')
    parser.add_argument('--legacy-interactive', action='store_true',
                        help='Run in legacy interactive mode')
    parser.add_argument('--model-info', action='store_true',
                        help='Display enhanced model information and exit')
    
    # Enhanced inference options
    parser.add_argument('--use-response-level', action='store_true', default=False,
                        help='Use response-level inference instead of token-level')
    parser.add_argument('--tokenizer-name', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'bert-base-uncased', 'bert-base-cased'],
                        help='Standard tokenizer to use')
    parser.add_argument('--save-components', action='store_true',
                        help='Save enhanced components after loading')
    
    # Performance optimization options
    parser.add_argument('--lazy-load', action='store_true', default=False,
                        help='Enable lazy loading of model components')
    parser.add_argument('--cache-size', type=int, default=1000,
                        help='Size of prediction cache')
    parser.add_argument('--max-batch-size', type=int, default=32,
                        help='Maximum batch size for memory efficiency')
    
    # Legacy options for backward compatibility
    parser.add_argument('--legacy-mode', action='store_true',
                        help='Use legacy OptimizedLSMInference class')
    parser.add_argument('--dialogue-sequence', type=str, nargs='+',
                        help='Legacy: Input dialogue sequence (space-separated)')
    parser.add_argument('--top-k', type=int, default=1,
                        help='Legacy: Number of top predictions to show')
    parser.add_argument('--show-confidence', action='store_true',
                        help='Legacy: Show confidence scores with predictions')
    parser.add_argument('--batch-file', type=str,
                        help='Legacy: File containing batch input sequences (one per line)')
    
    # Added argument from the second code block, but integrated here for clarity
    parser.add_argument('--memory-threshold', type=int, default=512,
                        help='Memory threshold in MB for garbage collection (Legacy mode only)')
    
    args = parser.parse_args()
    
    try:
        # Validate model path
        if not validate_file_path(args.model_path):
            print(f"❌ Error: Model path '{args.model_path}' does not exist")
            return 1
        
        # Choose inference class based on mode
        if args.legacy_mode:
            print("🔄 Using legacy OptimizedLSMInference")
            inference = OptimizedLSMInference(
                model_path=args.model_path,
                lazy_load=args.lazy_load,
                cache_size=args.cache_size,
                max_batch_size=args.max_batch_size
            )
            inference._memory_threshold_mb = args.memory_threshold
        else:
            print("🚀 Using enhanced inference with new tokenization system")
            inference = EnhancedLSMInference(
                model_path=args.model_path,
                use_response_level=args.use_response_level,
                tokenizer_name=args.tokenizer_name,
                lazy_load=args.lazy_load,
                cache_size=args.cache_size,
                max_batch_size=args.max_batch_size
            )
        
        # Handle model info request
        if args.model_info:
            if hasattr(inference, 'get_enhanced_model_info'):
                info = inference.get_enhanced_model_info()
                print("\n📊 Enhanced Model Information:")
                print(f"Model path: {args.model_path}")
                
                if 'enhanced_components' in info:
                    print(f"Standard tokenizer: {info['enhanced_components']['standard_tokenizer']['name']}")
                    print(f"Vocabulary size: {info['enhanced_components']['standard_tokenizer']['vocab_size']}")
                    print(f"Embedder fitted: {'✓' if info['enhanced_components']['sinusoidal_embedder']['is_fitted'] else '✗'}")
                    print(f"Response generator: {'✓' if info['enhanced_components']['response_generator'] else '✗'}")
                
                if 'performance' in info:
                    print(f"Response-level inference: {'✓' if info['performance']['use_response_level'] else '✗'}")
                    print(f"Cache sizes: {info['performance']['cache_sizes']}")
            else:
                info = inference.get_model_info()
                print("\n📊 Model Information:")
                print(f"Architecture: {info.get('architecture', {}).get('reservoir_type', 'Unknown')}")
                print(f"Window size: {info.get('architecture', {}).get('window_size', 'Unknown')}")
                
            return 0
        
        # Save enhanced components if requested
        if args.save_components and hasattr(inference, 'save_enhanced_components'):
            print("💾 Saving enhanced components...")
            inference.save_enhanced_components()
            print("✓ Enhanced components saved")
        
        # Handle interactive modes
        if args.interactive:
            if hasattr(inference, 'interactive_enhanced_session'):
                inference.interactive_enhanced_session()
            else:
                inference.interactive_session()
            return 0
        elif args.legacy_interactive:
            inference.interactive_session()
            return 0
        
        # Handle single text input
        if args.input_text:
            if hasattr(inference, 'generate_response'):
                print(f"📝 Input: {args.input_text}")
                if args.system_message:
                    print(f"🎯 System: {args.system_message}")
                
                start_time = time.time()
                response = inference.generate_response(args.input_text, args.system_message)
                generation_time = time.time() - start_time
                
                print(f"🔮 Response: {response}")
                print(f"⏱️ Generation time: {generation_time:.3f}s")
            else:
                print("❌ Enhanced response generation not available in legacy mode")
                print("Use --dialogue-sequence for legacy token prediction")
            return 0
        
        # Handle legacy dialogue sequence input
        if args.dialogue_sequence:
            if not hasattr(inference, 'predict_next_token'):
                print("❌ Legacy dialogue sequence prediction not available")
                return 1
            
            print(f"📝 Dialogue sequence: {' '.join(args.dialogue_sequence)}")
            
            start_time = time.time()
            if args.show_confidence and hasattr(inference, 'predict_with_confidence'):
                prediction, confidence = inference.predict_with_confidence(args.dialogue_sequence)
                print(f"🔮 Prediction: '{prediction}' (confidence: {confidence:.3f})")
            elif args.top_k > 1 and hasattr(inference, 'predict_top_k'):
                predictions = inference.predict_top_k(args.dialogue_sequence, args.top_k)
                print(f"🔮 Top {args.top_k} predictions:")
                for i, (pred, conf) in enumerate(predictions, 1):
                    print(f"  {i}. '{pred}' (confidence: {conf:.3f})")
            else:
                prediction = inference.predict_next_token(args.dialogue_sequence)
                print(f"🔮 Prediction: '{prediction}'")
            
            prediction_time = time.time() - start_time
            print(f"⏱️ Prediction time: {prediction_time:.3f}s")
            return 0
        
        # Handle batch file processing (legacy)
        if args.batch_file:
            if not hasattr(inference, 'batch_predict'):
                print("❌ Batch prediction not available")
                return 1
            
            try:
                with open(args.batch_file, 'r', encoding='utf-8') as f:
                    sequences = [line.strip().split() for line in f if line.strip()]
                
                print(f"📁 Processing {len(sequences)} sequences from {args.batch_file}")
                
                start_time = time.time()
                predictions = inference.batch_predict(sequences)
                batch_time = time.time() - start_time
                
                print(f"✅ Batch processing completed in {batch_time:.2f}s")
                print(f"Average time per sequence: {batch_time/len(sequences):.3f}s")
                
                # Save results
                output_file = args.batch_file.replace('.txt', '_predictions.txt')
                with open(output_file, 'w', encoding='utf-8') as f:
                    for seq, pred in zip(sequences, predictions):
                        f.write(f"Input: {' '.join(seq)}\n")
                        f.write(f"Prediction: {pred}\n\n")
                
                print(f"💾 Results saved to {output_file}")
                return 0
                
            except FileNotFoundError:
                print(f"❌ Batch file not found: {args.batch_file}")
                return 1
            except Exception as e:
                print(f"❌ Batch processing failed: {e}")
                return 1
        
        # If no specific action requested, show help
        print("ℹ️ No action specified. Use --help for options or --interactive for interactive mode")
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 0
    except Exception as e:
        logger.exception("Main execution failed")
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())