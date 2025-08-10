#!/usr/bin/env python3
"""
Response Generator for Complete Response Generation.

This module provides the main orchestrator for response-level inference,
replacing token-by-token generation with complete response generation.
It integrates with existing CNN architectures and manages reservoir strategies.
"""

import numpy as np
import time
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..core.cnn_architecture_factory import CNNArchitectureFactory
from ..core.cnn_3d_processor import CNN3DProcessor, SystemContext
from ..data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from ..utils.lsm_exceptions import LSMError, InferenceError
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class ReservoirStrategy(Enum):
    """Enumeration of reservoir reuse strategies."""
    REUSE = "reuse"
    SEPARATE = "separate"
    ADAPTIVE = "adaptive"


class ResponseGenerationError(InferenceError):
    """Raised when response generation fails."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {"operation": operation, "reason": reason}
        if details:
            error_details.update(details)
        
        message = f"Response generation failed during {operation}: {reason}"
        super().__init__(message, error_details)
        self.operation = operation


@dataclass
class TokenEmbeddingSequence:
    """Container for token embedding sequences."""
    embeddings: np.ndarray  # Shape: (sequence_length, embedding_dim)
    tokens: List[str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ResponseGenerationResult:
    """Result of response generation."""
    response_text: str
    confidence_score: float
    generation_time: float
    reservoir_strategy_used: str
    intermediate_embeddings: Optional[List[np.ndarray]] = None
    system_influence: Optional[float] = None


class ResponseGenerator:
    """
    Main orchestrator for response-level inference.
    
    This class handles complete response generation by processing token embedding
    sequences through reservoir and CNN architectures, with support for both
    2D and 3D CNN processing and system message integration.
    """
    
    def __init__(self,
                 tokenizer: StandardTokenizerWrapper,
                 embedder: SinusoidalEmbedder,
                 reservoir_model: Any,  # LSM reservoir model
                 cnn_architecture_factory: Optional[CNNArchitectureFactory] = None,
                 cnn_3d_processor: Optional[CNN3DProcessor] = None,
                 default_reservoir_strategy: str = "adaptive",
                 max_response_length: int = 512,
                 confidence_threshold: float = 0.5):
        """
        Initialize ResponseGenerator.
        
        Args:
            tokenizer: Standard tokenizer wrapper for text processing
            embedder: Sinusoidal embedder for token embeddings
            reservoir_model: Trained LSM reservoir model
            cnn_architecture_factory: Factory for creating CNN models
            cnn_3d_processor: Processor for 3D CNN with system messages
            default_reservoir_strategy: Default strategy for reservoir usage
            max_response_length: Maximum length of generated responses
            confidence_threshold: Minimum confidence threshold for responses
        """
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.reservoir_model = reservoir_model
        self.cnn_architecture_factory = cnn_architecture_factory or CNNArchitectureFactory()
        self.cnn_3d_processor = cnn_3d_processor
        
        # Configuration
        self.default_reservoir_strategy = ReservoirStrategy(default_reservoir_strategy)
        self.max_response_length = max_response_length
        self.confidence_threshold = confidence_threshold
        
        # Models (created lazily)
        self._cnn_2d_model = None
        self._response_inference_model = None
        self._reservoir_manager = None
        
        # Performance tracking
        self._generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_generation_time": 0.0,
            "reservoir_strategy_usage": {
                "reuse": 0,
                "separate": 0,
                "adaptive": 0
            }
        }
        
        logger.info(f"ResponseGenerator initialized with strategy: {default_reservoir_strategy}")
    
    def generate_complete_response(self,
                                 input_sequence: Union[List[str], TokenEmbeddingSequence],
                                 system_context: Optional[SystemContext] = None,
                                 reservoir_strategy: Optional[str] = None,
                                 return_intermediate: bool = False) -> ResponseGenerationResult:
        """
        Generate a complete response from input sequence.
        
        Args:
            input_sequence: Input text sequence or token embedding sequence
            system_context: Optional system message context for 3D CNN processing
            reservoir_strategy: Override default reservoir strategy
            return_intermediate: Whether to return intermediate embeddings
            
        Returns:
            ResponseGenerationResult with complete response and metadata
            
        Raises:
            ResponseGenerationError: If response generation fails
        """
        try:
            start_time = time.time()
            self._generation_stats["total_generations"] += 1
            
            # Determine reservoir strategy
            strategy = ReservoirStrategy(reservoir_strategy or self.default_reservoir_strategy.value)
            
            # Process input to token embedding sequence
            if isinstance(input_sequence, list):
                token_embedding_seq = self._process_text_to_embeddings(input_sequence)
            elif isinstance(input_sequence, str):
                # Handle string input by converting to list first
                token_embedding_seq = self._process_text_to_embeddings([input_sequence])
            else:
                # Assume it's already a TokenEmbeddingSequence or compatible object
                token_embedding_seq = input_sequence
            
            # Process through reservoir
            reservoir_output = self._process_through_reservoir(token_embedding_seq, strategy)
            
            # Choose CNN architecture based on system context
            if system_context is not None and self.cnn_3d_processor is not None:
                # Use 3D CNN for system-aware processing
                cnn_output = self._process_with_3d_cnn(reservoir_output, system_context)
                system_influence = cnn_output.system_influence
            else:
                # Use 2D CNN for standard processing
                cnn_output = self._process_with_2d_cnn(reservoir_output)
                system_influence = None
            
            # Generate final response
            response_text, confidence = self._generate_response_from_embeddings(
                cnn_output.output_embeddings if hasattr(cnn_output, 'output_embeddings') else cnn_output
            )
            
            generation_time = time.time() - start_time
            
            # Update statistics
            self._update_generation_stats(strategy, generation_time, confidence >= self.confidence_threshold)
            
            # Prepare intermediate embeddings if requested
            intermediate_embeddings = None
            if return_intermediate:
                intermediate_embeddings = [
                    token_embedding_seq.embeddings,
                    reservoir_output,
                    cnn_output.output_embeddings if hasattr(cnn_output, 'output_embeddings') else cnn_output
                ]
            
            result = ResponseGenerationResult(
                response_text=response_text,
                confidence_score=confidence,
                generation_time=generation_time,
                reservoir_strategy_used=strategy.value,
                intermediate_embeddings=intermediate_embeddings,
                system_influence=system_influence
            )
            
            logger.debug(f"Response generated successfully in {generation_time:.3f}s with confidence {confidence:.3f}")
            return result
            
        except Exception as e:
            logger.exception("Response generation failed")
            raise ResponseGenerationError(
                "complete_response_generation",
                f"Failed to generate complete response: {str(e)}",
                {
                    "input_type": type(input_sequence).__name__,
                    "has_system_context": system_context is not None,
                    "reservoir_strategy": strategy.value if 'strategy' in locals() else None
                }
            )
    
    def process_token_embedding_sequences(self,
                                        embedding_sequences: List[TokenEmbeddingSequence],
                                        batch_size: int = 8) -> List[ResponseGenerationResult]:
        """
        Process multiple token embedding sequences in batches.
        
        Args:
            embedding_sequences: List of token embedding sequences to process
            batch_size: Batch size for processing
            
        Returns:
            List of response generation results
            
        Raises:
            ResponseGenerationError: If batch processing fails
        """
        try:
            results = []
            
            logger.info(f"Processing {len(embedding_sequences)} sequences in batches of {batch_size}")
            
            for i in range(0, len(embedding_sequences), batch_size):
                batch_end = min(i + batch_size, len(embedding_sequences))
                batch_sequences = embedding_sequences[i:batch_end]
                
                batch_results = []
                for j, seq in enumerate(batch_sequences):
                    try:
                        result = self.generate_complete_response(seq)
                        batch_results.append(result)
                    except Exception as e:
                        logger.warning(f"Failed to process sequence {i + j}: {e}")
                        # Create error result
                        error_result = ResponseGenerationResult(
                            response_text="[ERROR]",
                            confidence_score=0.0,
                            generation_time=0.0,
                            reservoir_strategy_used="unknown"
                        )
                        batch_results.append(error_result)
                
                results.extend(batch_results)
                logger.debug(f"Processed batch {i//batch_size + 1}/{(len(embedding_sequences) + batch_size - 1)//batch_size}")
            
            return results
            
        except Exception as e:
            raise ResponseGenerationError(
                "batch_processing",
                f"Failed to process token embedding sequences: {str(e)}",
                {"num_sequences": len(embedding_sequences), "batch_size": batch_size}
            )
    
    def determine_reservoir_strategy(self,
                                   input_sequence: TokenEmbeddingSequence,
                                   system_context: Optional[SystemContext] = None) -> ReservoirStrategy:
        """
        Determine the optimal reservoir strategy for the given input.
        
        Args:
            input_sequence: Input token embedding sequence
            system_context: Optional system context
            
        Returns:
            Recommended reservoir strategy
        """
        try:
            # Simple heuristic-based strategy determination
            # In practice, this could be more sophisticated
            
            sequence_length = len(input_sequence.embeddings)
            embedding_complexity = np.std(input_sequence.embeddings)
            
            # If system context is present, prefer separate reservoir
            if system_context is not None:
                return ReservoirStrategy.SEPARATE
            
            # For short, simple sequences, reuse reservoir
            if sequence_length < 50 and embedding_complexity < 0.5:
                return ReservoirStrategy.REUSE
            
            # For complex sequences, use separate reservoir
            if embedding_complexity > 1.0:
                return ReservoirStrategy.SEPARATE
            
            # Default to adaptive for medium complexity
            return ReservoirStrategy.ADAPTIVE
            
        except Exception as e:
            logger.warning(f"Failed to determine reservoir strategy: {e}")
            return self.default_reservoir_strategy
    
    def get_generation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about response generation performance.
        
        Returns:
            Dictionary with generation statistics
        """
        stats = self._generation_stats.copy()
        
        # Calculate success rate
        if stats["total_generations"] > 0:
            stats["success_rate"] = stats["successful_generations"] / stats["total_generations"]
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset generation statistics."""
        self._generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_generation_time": 0.0,
            "reservoir_strategy_usage": {
                "reuse": 0,
                "separate": 0,
                "adaptive": 0
            }
        }
        logger.info("Generation statistics reset")
    
    # Private helper methods
    
    def _process_text_to_embeddings(self, text_sequence: List[str]) -> TokenEmbeddingSequence:
        """Convert text sequence to token embedding sequence."""
        try:
            # Tokenize text
            token_ids = self.tokenizer.tokenize(text_sequence, padding=False, truncation=True)
            
            # Flatten token IDs if needed
            if isinstance(token_ids[0], list):
                # Multiple sequences - concatenate for now
                flat_token_ids = []
                for seq in token_ids:
                    flat_token_ids.extend(seq)
                token_ids = flat_token_ids
            
            # Convert to embeddings
            embeddings = self.embedder.embed(token_ids)
            
            # Decode tokens for reference
            tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
            
            return TokenEmbeddingSequence(
                embeddings=embeddings,
                tokens=tokens,
                metadata={"original_text": text_sequence}
            )
            
        except Exception as e:
            raise ResponseGenerationError(
                "text_to_embeddings",
                f"Failed to convert text to embeddings: {str(e)}",
                {"text_sequence_length": len(text_sequence)}
            )
    
    def _process_through_reservoir(self,
                                 token_embedding_seq: TokenEmbeddingSequence,
                                 strategy: ReservoirStrategy) -> np.ndarray:
        """Process token embeddings through reservoir."""
        try:
            embeddings = token_embedding_seq.embeddings
            
            # Ensure embeddings have batch dimension
            if len(embeddings.shape) == 2:
                embeddings = np.expand_dims(embeddings, axis=0)
            
            # Reshape embeddings to match reservoir input expectations
            # The reservoir was trained with individual embeddings, not sequences
            if len(embeddings.shape) == 3:  # (batch, seq_len, embedding_dim)
                batch_size, seq_len, embedding_dim = embeddings.shape
                # Reshape to (batch * seq_len, embedding_dim) for reservoir processing
                embeddings_reshaped = embeddings.reshape(-1, embedding_dim)
                
                # Process through reservoir
                if strategy == ReservoirStrategy.REUSE:
                    reservoir_output = self.reservoir_model.predict(embeddings_reshaped, verbose=0)
                elif strategy == ReservoirStrategy.SEPARATE:
                    if hasattr(self.reservoir_model, 'reset_states'):
                        self.reservoir_model.reset_states()
                    reservoir_output = self.reservoir_model.predict(embeddings_reshaped, verbose=0)
                else:  # ADAPTIVE
                    reservoir_output = self.reservoir_model.predict(embeddings_reshaped, verbose=0)
                
                # Reshape back to (batch, seq_len, reservoir_output_dim)
                if len(reservoir_output.shape) == 2:
                    reservoir_output_dim = reservoir_output.shape[1]
                    reservoir_output = reservoir_output.reshape(batch_size, seq_len, reservoir_output_dim)
                
            else:
                # Handle other shapes
                if strategy == ReservoirStrategy.REUSE:
                    reservoir_output = self.reservoir_model.predict(embeddings, verbose=0)
                elif strategy == ReservoirStrategy.SEPARATE:
                    if hasattr(self.reservoir_model, 'reset_states'):
                        self.reservoir_model.reset_states()
                    reservoir_output = self.reservoir_model.predict(embeddings, verbose=0)
                else:  # ADAPTIVE
                    reservoir_output = self.reservoir_model.predict(embeddings, verbose=0)
            
            return reservoir_output
            
        except Exception as e:
            raise ResponseGenerationError(
                "reservoir_processing",
                f"Failed to process through reservoir: {str(e)}",
                {"strategy": strategy.value, "embeddings_shape": embeddings.shape}
            )
    
    def _process_with_2d_cnn(self, reservoir_output: np.ndarray) -> np.ndarray:
        """Process reservoir output through 2D CNN."""
        try:
            # Create 2D CNN model if not exists
            if self._cnn_2d_model is None:
                self._create_2d_cnn_model(reservoir_output.shape)
            
            # Reshape reservoir output to match CNN input shape
            reshaped_input = self._reshape_for_2d_cnn(reservoir_output)
            
            # Process through 2D CNN
            cnn_output = self._cnn_2d_model.predict(reshaped_input, verbose=0)
            
            return cnn_output
            
        except Exception as e:
            raise ResponseGenerationError(
                "2d_cnn_processing",
                f"Failed to process with 2D CNN: {str(e)}",
                {"reservoir_output_shape": reservoir_output.shape}
            )
    
    def _process_with_3d_cnn(self,
                           reservoir_output: np.ndarray,
                           system_context: SystemContext) -> Any:
        """Process reservoir output through 3D CNN with system context."""
        try:
            if self.cnn_3d_processor is None:
                raise ValueError("3D CNN processor not available")
            
            # Process with system context
            result = self.cnn_3d_processor.process_with_system_context(
                reservoir_output, system_context
            )
            
            return result
            
        except Exception as e:
            raise ResponseGenerationError(
                "3d_cnn_processing",
                f"Failed to process with 3D CNN: {str(e)}",
                {"reservoir_output_shape": reservoir_output.shape}
            )
    
    def _generate_response_from_embeddings(self, embeddings: np.ndarray) -> Tuple[str, float]:
        """Generate final response text from embeddings."""
        try:
            # Validate input embeddings
            if embeddings is None:
                raise ValueError("Embeddings cannot be None")
            
            if not isinstance(embeddings, np.ndarray):
                raise ValueError(f"Embeddings must be numpy array, got {type(embeddings)}")
            
            if embeddings.size == 0:
                raise ValueError("Embeddings array is empty")
            
            # Simple implementation - in practice would use more sophisticated decoding
            # For now, find closest tokens in embedding space
            
            # Flatten embeddings if needed
            if len(embeddings.shape) > 2:
                embeddings = embeddings.reshape(embeddings.shape[0], -1)
            
            # Take mean embedding as response representation
            response_embedding = np.mean(embeddings, axis=0)
            
            # Simple confidence calculation based on embedding magnitude
            confidence = min(1.0, np.linalg.norm(response_embedding) / 10.0)
            
            # Generate response text (simplified - would use proper decoding)
            response_text = self._decode_embedding_to_text(response_embedding)
            
            return response_text, confidence
            
        except Exception as e:
            raise ResponseGenerationError(
                "response_generation",
                f"Failed to generate response from embeddings: {str(e)}",
                {"embeddings_shape": embeddings.shape}
            )
    
    def _decode_embedding_to_text(self, embedding: np.ndarray) -> str:
        """Decode embedding to text with improved response generation."""
        try:
            # Improved response generation based on embedding characteristics
            embedding_norm = np.linalg.norm(embedding)
            embedding_mean = np.mean(embedding)
            embedding_std = np.std(embedding)
            
            # Generate more varied responses based on embedding patterns
            if embedding_norm > 8.0:
                if embedding_mean > 0.1:
                    responses = [
                        "I'd be happy to help you with that question.",
                        "That's an interesting topic. Let me share some thoughts.",
                        "I understand what you're asking about.",
                        "Here's what I can tell you about that.",
                        "That's a great question. Let me explain."
                    ]
                else:
                    responses = [
                        "I see what you're getting at.",
                        "That's something worth considering.",
                        "I can provide some insight on that.",
                        "Let me think about that for a moment.",
                        "That's an important point to discuss."
                    ]
            elif embedding_norm > 5.0:
                if embedding_std > 0.5:
                    responses = [
                        "Thank you for your question.",
                        "I appreciate you asking about that.",
                        "That's something I can help with.",
                        "Let me address that for you.",
                        "I'm glad you brought that up."
                    ]
                else:
                    responses = [
                        "I understand your interest in this topic.",
                        "That's a valid concern.",
                        "I can see why you'd ask about that.",
                        "That's worth exploring further.",
                        "I'm here to help with that."
                    ]
            elif embedding_norm > 2.0:
                responses = [
                    "Thank you for your message.",
                    "I hear what you're saying.",
                    "That's noted.",
                    "I acknowledge your input.",
                    "Thanks for sharing that."
                ]
            else:
                responses = [
                    "Hello there!",
                    "Hi, how can I help?",
                    "I'm here to assist.",
                    "What can I do for you?",
                    "How may I help you today?"
                ]
            
            # Use embedding characteristics to select response
            response_index = int(abs(embedding_mean * 1000)) % len(responses)
            return responses[response_index]
            
        except Exception as e:
            logger.warning(f"Failed to decode embedding to text: {e}")
            return "I'm here to help you."
    
    def _create_2d_cnn_model(self, input_shape: Tuple[int, ...]):
        """Create 2D CNN model for standard processing."""
        try:
            # Determine appropriate 2D input shape from reservoir output
            if len(input_shape) == 3:  # (batch, seq_len, features)
                # Convert to 2D CNN input shape - ensure minimum dimensions for pooling
                seq_len = input_shape[1]
                features = input_shape[2]
                
                # Calculate height and width ensuring they're at least 4 for pooling operations
                height = max(4, int(np.sqrt(seq_len)))
                width = max(4, seq_len // height)
                
                # If we still have issues, use a safe default
                if height * width < seq_len:
                    height = max(4, int(np.sqrt(features)))
                    width = max(4, features // height)
                    channels = 1
                else:
                    channels = features
                
                cnn_input_shape = (height, width, channels)
            else:
                # Use default shape
                cnn_input_shape = (8, 8, 1)
            
            self._cnn_2d_model = self.cnn_architecture_factory.create_2d_cnn(
                input_shape=cnn_input_shape,
                output_dim=512,  # Standard embedding dimension
                use_attention=True,
                attention_type="spatial"
            )
            
            # Compile model
            self._cnn_2d_model = self.cnn_architecture_factory.compile_model(
                self._cnn_2d_model,
                loss_type="cosine_similarity",
                learning_rate=0.001
            )
            
            logger.info(f"Created 2D CNN model with input shape: {cnn_input_shape}")
            
        except Exception as e:
            raise ResponseGenerationError(
                "2d_cnn_creation",
                f"Failed to create 2D CNN model: {str(e)}",
                {"input_shape": input_shape}
            )
    
    def _reshape_for_2d_cnn(self, reservoir_output: np.ndarray) -> np.ndarray:
        """Reshape reservoir output to match 2D CNN input requirements."""
        try:
            # Get the expected input shape from the CNN model
            expected_shape = self._cnn_2d_model.input_shape[1:]  # Remove batch dimension
            
            if len(reservoir_output.shape) == 3:  # (batch, seq_len, features)
                batch_size = reservoir_output.shape[0]
                seq_len = reservoir_output.shape[1]
                features = reservoir_output.shape[2]
                
                # Calculate total elements
                total_elements = seq_len * features
                target_elements = expected_shape[0] * expected_shape[1] * expected_shape[2]
                
                if total_elements >= target_elements:
                    # Truncate if we have more elements than needed
                    flattened = reservoir_output.reshape(batch_size, -1)[:, :target_elements]
                else:
                    # Pad if we have fewer elements than needed
                    flattened = reservoir_output.reshape(batch_size, -1)
                    padding_needed = target_elements - total_elements
                    padding = np.zeros((batch_size, padding_needed))
                    flattened = np.concatenate([flattened, padding], axis=1)
                
                # Reshape to expected 2D CNN input shape
                reshaped = flattened.reshape(batch_size, expected_shape[0], expected_shape[1], expected_shape[2])
                
            else:
                # Handle other shapes by flattening and reshaping
                batch_size = reservoir_output.shape[0] if len(reservoir_output.shape) > 1 else 1
                flattened = reservoir_output.reshape(batch_size, -1)
                
                target_elements = expected_shape[0] * expected_shape[1] * expected_shape[2]
                
                if flattened.shape[1] >= target_elements:
                    flattened = flattened[:, :target_elements]
                else:
                    padding_needed = target_elements - flattened.shape[1]
                    padding = np.zeros((batch_size, padding_needed))
                    flattened = np.concatenate([flattened, padding], axis=1)
                
                reshaped = flattened.reshape(batch_size, expected_shape[0], expected_shape[1], expected_shape[2])
            
            return reshaped
            
        except Exception as e:
            raise ResponseGenerationError(
                "2d_cnn_reshape",
                f"Failed to reshape for 2D CNN: {str(e)}",
                {
                    "reservoir_output_shape": reservoir_output.shape,
                    "expected_shape": expected_shape if 'expected_shape' in locals() else None
                }
            )
    
    def _update_generation_stats(self, strategy: ReservoirStrategy, generation_time: float, success: bool):
        """Update generation statistics."""
        # Update strategy usage
        self._generation_stats["reservoir_strategy_usage"][strategy.value] += 1
        
        # Update success count
        if success:
            self._generation_stats["successful_generations"] += 1
        
        # Update average generation time
        current_avg = self._generation_stats["average_generation_time"]
        total_gens = self._generation_stats["total_generations"]
        
        if total_gens > 1:
            self._generation_stats["average_generation_time"] = (
                (current_avg * (total_gens - 1) + generation_time) / total_gens
            )
        else:
            self._generation_stats["average_generation_time"] = generation_time


# Convenience functions for easy usage

def create_response_generator(tokenizer: StandardTokenizerWrapper,
                            embedder: SinusoidalEmbedder,
                            reservoir_model: Any,
                            enable_3d_cnn: bool = False,
                            reservoir_strategy: str = "adaptive") -> ResponseGenerator:
    """
    Create a ResponseGenerator with standard configuration.
    
    Args:
        tokenizer: Standard tokenizer wrapper
        embedder: Sinusoidal embedder
        reservoir_model: Trained LSM reservoir model
        enable_3d_cnn: Whether to enable 3D CNN processing
        reservoir_strategy: Default reservoir strategy
        
    Returns:
        Configured ResponseGenerator instance
    """
    cnn_3d_processor = None
    if enable_3d_cnn:
        # Create 3D CNN processor with default configuration
        reservoir_shape = (64, 64, 64, 1)  # Default shape
        cnn_3d_processor = CNN3DProcessor(
            reservoir_shape=reservoir_shape,
            system_embedding_dim=256,
            output_embedding_dim=512
        )
    
    return ResponseGenerator(
        tokenizer=tokenizer,
        embedder=embedder,
        reservoir_model=reservoir_model,
        cnn_3d_processor=cnn_3d_processor,
        default_reservoir_strategy=reservoir_strategy
    )


def create_system_aware_response_generator(tokenizer: StandardTokenizerWrapper,
                                         embedder: SinusoidalEmbedder,
                                         reservoir_model: Any,
                                         reservoir_shape: Tuple[int, int, int, int] = (64, 64, 64, 1)) -> ResponseGenerator:
    """
    Create a ResponseGenerator with system message support.
    
    Args:
        tokenizer: Standard tokenizer wrapper
        embedder: Sinusoidal embedder
        reservoir_model: Trained LSM reservoir model
        reservoir_shape: Shape of reservoir output for 3D CNN
        
    Returns:
        ResponseGenerator with 3D CNN support for system messages
    """
    cnn_3d_processor = CNN3DProcessor(
        reservoir_shape=reservoir_shape,
        system_embedding_dim=256,
        output_embedding_dim=512
    )
    
    return ResponseGenerator(
        tokenizer=tokenizer,
        embedder=embedder,
        reservoir_model=reservoir_model,
        cnn_3d_processor=cnn_3d_processor,
        default_reservoir_strategy="separate"  # Use separate strategy for system-aware processing
    )