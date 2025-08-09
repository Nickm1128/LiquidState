#!/usr/bin/env python3
"""
Advanced tokenization system for LSM training pipeline enhancement.

This module provides standard tokenizer integration and sinusoidal embedding
optimization for the LSM architecture.
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod

try:
    from transformers import AutoTokenizer, GPT2Tokenizer, BertTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..utils.lsm_exceptions import (
    TokenizerError, TokenizerNotFittedError, 
    TokenizerLoadError, TokenizerSaveError, InvalidInputError
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class StandardTokenizerWrapper:
    """
    Wrapper for standard tokenizers (GPT-2, BERT, etc.) with LSM-specific functionality.
    
    This class integrates popular tokenizers and provides a consistent interface
    for tokenization, decoding, and vocabulary management.
    """
    
    SUPPORTED_TOKENIZERS = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium', 
        'gpt2-large': 'gpt2-large',
        'bert-base-uncased': 'bert-base-uncased',
        'bert-base-cased': 'bert-base-cased',
        'distilbert-base-uncased': 'distilbert-base-uncased'
    }
    
    def __init__(self, tokenizer_name: str = 'gpt2', max_length: int = 512):
        """
        Initialize StandardTokenizerWrapper.
        
        Args:
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
        """
        if not TRANSFORMERS_AVAILABLE:
            raise TokenizerError(
                "transformers library not available. Install with: pip install transformers"
            )
        
        if tokenizer_name not in self.SUPPORTED_TOKENIZERS:
            raise TokenizerError(
                f"Unsupported tokenizer: {tokenizer_name}. "
                f"Supported: {list(self.SUPPORTED_TOKENIZERS.keys())}"
            )
        
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None
        self._vocab_size = None
        self._pad_token_id = None
        self._eos_token_id = None
        self._bos_token_id = None
        
        self._initialize_tokenizer()
        
    def _initialize_tokenizer(self):
        """Initialize the underlying tokenizer."""
        try:
            model_name = self.SUPPORTED_TOKENIZERS[self.tokenizer_name]
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Set special tokens
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            self._vocab_size = len(self._tokenizer)
            self._pad_token_id = self._tokenizer.pad_token_id
            self._eos_token_id = self._tokenizer.eos_token_id
            self._bos_token_id = getattr(self._tokenizer, 'bos_token_id', None)
            
            logger.info(f"Initialized {self.tokenizer_name} tokenizer with vocab size {self._vocab_size}")
            
        except Exception as e:
            raise TokenizerError(f"Failed to initialize tokenizer {self.tokenizer_name}: {str(e)}")
    
    def tokenize(self, texts: Union[str, List[str]], add_special_tokens: bool = True, 
                 padding: bool = True, truncation: bool = True) -> List[List[int]]:
        """
        Tokenize texts to token IDs.
        
        Args:
            texts: Single text or list of texts to tokenize
            add_special_tokens: Whether to add special tokens (BOS, EOS)
            padding: Whether to pad sequences to max_length
            truncation: Whether to truncate sequences to max_length
            
        Returns:
            List of token ID sequences
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            encoded = self._tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=self.max_length,
                return_tensors=None
            )
            
            return encoded['input_ids']
            
        except Exception as e:
            raise TokenizerError(f"Tokenization failed: {str(e)}")
    
    def decode(self, token_ids: Union[List[int], List[List[int]]], 
               skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Single sequence or batch of token ID sequences
            skip_special_tokens: Whether to skip special tokens in decoding
            
        Returns:
            Decoded text(s)
        """
        try:
            # Handle single sequence
            if isinstance(token_ids[0], int):
                return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            
            # Handle batch of sequences
            return [
                self._tokenizer.decode(seq, skip_special_tokens=skip_special_tokens) 
                for seq in token_ids
            ]
            
        except Exception as e:
            raise TokenizerError(f"Decoding failed: {str(e)}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self._tokenizer.get_vocab()
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            'pad_token_id': self._pad_token_id,
            'eos_token_id': self._eos_token_id,
            'bos_token_id': self._bos_token_id,
        }
    
    def encode_single(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode a single text to token IDs.
        
        Args:
            text: Text to encode
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token IDs
        """
        return self.tokenize([text], add_special_tokens=add_special_tokens, 
                           padding=False, truncation=True)[0]
    
    def decode_single(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a single sequence of token IDs.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        return self.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def get_token_embeddings_shape(self, embedding_dim: int) -> Tuple[int, int]:
        """
        Get the shape for token embeddings matrix.
        
        Args:
            embedding_dim: Desired embedding dimension
            
        Returns:
            Tuple of (vocab_size, embedding_dim)
        """
        return (self._vocab_size, embedding_dim)
    
    def save(self, save_path: str):
        """
        Save tokenizer configuration.
        
        Args:
            save_path: Directory path to save tokenizer config
        """
        try:
            # Validate path before creating directories
            if not save_path or save_path.strip() == "":
                raise ValueError("Save path cannot be empty")
            
            # Try to create directory and write file
            os.makedirs(save_path, exist_ok=True)
            
            config = {
                'tokenizer_name': self.tokenizer_name,
                'max_length': self.max_length,
                'vocab_size': self._vocab_size,
                'special_tokens': self.get_special_tokens()
            }
            
            config_path = os.path.join(save_path, 'standard_tokenizer_config.json')
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"StandardTokenizerWrapper config saved to {save_path}")
            
        except Exception as e:
            raise TokenizerSaveError(save_path, str(e))
    
    @classmethod
    def load(cls, load_path: str) -> 'StandardTokenizerWrapper':
        """
        Load tokenizer from saved configuration.
        
        Args:
            load_path: Directory path to load tokenizer config from
            
        Returns:
            Loaded StandardTokenizerWrapper instance
        """
        try:
            # Validate path exists
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Load path does not exist: {load_path}")
            
            config_path = os.path.join(load_path, 'standard_tokenizer_config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            tokenizer = cls(
                tokenizer_name=config['tokenizer_name'],
                max_length=config['max_length']
            )
            
            logger.info(f"StandardTokenizerWrapper loaded from {load_path}")
            return tokenizer
            
        except Exception as e:
            raise TokenizerLoadError(load_path, str(e))
    
    def __repr__(self) -> str:
        return f"StandardTokenizerWrapper(tokenizer={self.tokenizer_name}, vocab_size={self._vocab_size})"


class SinusoidalEmbedder:
    """
    Embedding layer optimized for sinusoidal patterns in natural language.
    
    This class creates embeddings that maximize sinusoidality to enhance
    learnability by sine-activated LSM reservoirs and CNN models.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 max_position: int = 10000, temperature: float = 1.0):
        """
        Initialize SinusoidalEmbedder.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            max_position: Maximum position for positional encoding
            temperature: Temperature parameter for sinusoidal patterns
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_position = max_position
        self.temperature = temperature
        
        # Initialize embedding matrix
        self._embedding_matrix = None
        self._positional_encodings = None
        self._is_fitted = False
        
        # Training parameters
        self._learning_rate = 0.01
        self._sinusoidal_weight = 1.0
        self._diversity_weight = 0.1
        
        logger.info(f"Initialized SinusoidalEmbedder with vocab_size={vocab_size}, "
                   f"embedding_dim={embedding_dim}")
    
    def _create_positional_encodings(self) -> np.ndarray:
        """Create sinusoidal positional encodings."""
        position = np.arange(self.max_position)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * 
                         -(np.log(10000.0) / self.embedding_dim))
        
        pos_encoding = np.zeros((self.max_position, self.embedding_dim))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        return pos_encoding.astype(np.float32)
    
    def _initialize_embeddings(self) -> np.ndarray:
        """Initialize embedding matrix with sinusoidal patterns."""
        # Create base sinusoidal patterns
        embeddings = np.zeros((self.vocab_size, self.embedding_dim), dtype=np.float32)
        
        for i in range(self.vocab_size):
            # Create unique sinusoidal pattern for each token
            phase_shift = (i / self.vocab_size) * 2 * np.pi
            
            for j in range(self.embedding_dim):
                # Alternate between sin and cos with different frequencies
                freq = (j + 1) / self.embedding_dim
                if j % 2 == 0:
                    embeddings[i, j] = np.sin(freq * phase_shift * self.temperature)
                else:
                    embeddings[i, j] = np.cos(freq * phase_shift * self.temperature)
        
        # Add small random noise for diversity
        noise = np.random.normal(0, 0.01, embeddings.shape).astype(np.float32)
        embeddings += noise
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)
        
        return embeddings
    
    def fit(self, training_data: np.ndarray, epochs: int = 100) -> None:
        """
        Fit embeddings to maximize sinusoidality on training data.
        
        Args:
            training_data: Token sequences for training (batch_size, seq_len)
            epochs: Number of training epochs
        """
        logger.info(f"Fitting SinusoidalEmbedder on {len(training_data)} sequences "
                   f"for {epochs} epochs")
        
        # Initialize embeddings and positional encodings
        self._embedding_matrix = self._initialize_embeddings()
        self._positional_encodings = self._create_positional_encodings()
        
        # Convert training data to embeddings
        initial_embeddings = self._embed_sequences(training_data)
        
        # Optimize embeddings for sinusoidality
        for epoch in range(epochs):
            # Calculate sinusoidality score
            sin_score = self._calculate_sinusoidality_score(initial_embeddings)
            
            # Calculate gradients and update embeddings
            gradients = self._calculate_gradients(training_data, initial_embeddings)
            self._embedding_matrix -= self._learning_rate * gradients
            
            # Normalize embeddings
            norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            self._embedding_matrix = self._embedding_matrix / (norms + 1e-8)
            
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Sinusoidality score = {sin_score:.4f}")
        
        self._is_fitted = True
        logger.info("SinusoidalEmbedder fitting completed")
    
    def _embed_sequences(self, token_sequences: np.ndarray) -> np.ndarray:
        """Convert token sequences to embeddings."""
        batch_size, seq_len = token_sequences.shape
        embeddings = np.zeros((batch_size, seq_len, self.embedding_dim), dtype=np.float32)
        
        for i, sequence in enumerate(token_sequences):
            for j, token_id in enumerate(sequence):
                if token_id < self.vocab_size:
                    # Base embedding
                    embedding = self._embedding_matrix[token_id].copy()
                    
                    # Add positional encoding if available
                    if j < self.max_position:
                        embedding += self._positional_encodings[j]
                    
                    embeddings[i, j] = embedding
        
        return embeddings
    
    def _calculate_sinusoidality_score(self, embeddings: np.ndarray) -> float:
        """Calculate how sinusoidal the embeddings are."""
        # Flatten embeddings for analysis
        flat_embeddings = embeddings.reshape(-1, self.embedding_dim)
        
        # Calculate FFT to analyze frequency content
        fft_scores = []
        for i in range(self.embedding_dim):
            signal = flat_embeddings[:, i]
            fft = np.fft.fft(signal)
            # Focus on low-frequency components (more sinusoidal)
            low_freq_power = np.sum(np.abs(fft[:len(fft)//4])**2)
            total_power = np.sum(np.abs(fft)**2)
            fft_scores.append(low_freq_power / (total_power + 1e-8))
        
        return np.mean(fft_scores)
    
    def _calculate_gradients(self, token_sequences: np.ndarray, 
                           current_embeddings: np.ndarray) -> np.ndarray:
        """Calculate gradients for embedding optimization (fast analytical version)."""
        gradients = np.zeros_like(self._embedding_matrix)
        
        # Vectorized gradient calculation for sinusoidality optimization
        for token_id in range(self.vocab_size):
            # Find sequences containing this token
            token_mask = (token_sequences == token_id)
            if not np.any(token_mask):
                continue
            
            current_embedding = self._embedding_matrix[token_id]
            
            # Vectorized sinusoidal target calculation
            freqs = (np.arange(self.embedding_dim) + 1) / self.embedding_dim
            phase = 2 * np.pi * token_id / self.vocab_size
            
            # Create target sinusoidal pattern
            targets = np.zeros(self.embedding_dim)
            targets[::2] = np.sin(freqs[::2] * phase)  # Even indices: sine
            targets[1::2] = np.cos(freqs[1::2] * phase)  # Odd indices: cosine
            
            # Calculate gradients
            sin_grad = self._sinusoidal_weight * (targets - current_embedding)
            diversity_grad = -self._diversity_weight * current_embedding
            
            gradients[token_id] = sin_grad + diversity_grad
        
        return gradients
    
    def embed(self, token_ids: Union[List[int], np.ndarray]) -> np.ndarray:
        """
        Convert token IDs to embeddings.
        
        Args:
            token_ids: Token IDs to embed
            
        Returns:
            Embeddings array
        """
        if not self._is_fitted:
            raise TokenizerNotFittedError("embed")
        
        if isinstance(token_ids, list):
            token_ids = np.array(token_ids)
        
        if token_ids.ndim == 1:
            # Single sequence
            embeddings = np.zeros((len(token_ids), self.embedding_dim), dtype=np.float32)
            for i, token_id in enumerate(token_ids):
                if token_id < self.vocab_size:
                    embedding = self._embedding_matrix[token_id].copy()
                    # Add positional encoding
                    if i < self.max_position:
                        embedding += self._positional_encodings[i]
                    embeddings[i] = embedding
            return embeddings
        else:
            # Batch of sequences
            return self._embed_sequences(token_ids)
    
    def optimize_for_sine_activation(self, reservoir_outputs: np.ndarray) -> None:
        """
        Optimize embeddings specifically for sine-activated LSM architecture.
        
        Args:
            reservoir_outputs: Sample outputs from sine-activated reservoir
        """
        if not self._is_fitted:
            logger.warning("Embedder not fitted. Call fit() first.")
            return
        
        logger.info("Optimizing embeddings for sine-activated LSM architecture")
        
        # Analyze reservoir output patterns
        reservoir_freq_content = self._analyze_frequency_content(reservoir_outputs)
        
        # Adjust embedding frequencies to match reservoir preferences
        for token_id in range(self.vocab_size):
            embedding = self._embedding_matrix[token_id]
            
            # Adjust embedding to better match reservoir frequency preferences
            for j in range(self.embedding_dim):
                if j < len(reservoir_freq_content):
                    # Scale embedding component based on reservoir frequency response
                    embedding[j] *= (1.0 + 0.1 * reservoir_freq_content[j])
            
            # Renormalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                self._embedding_matrix[token_id] = embedding / norm
        
        logger.info("Sine-activation optimization completed")
    
    def _analyze_frequency_content(self, signals: np.ndarray) -> np.ndarray:
        """Analyze frequency content of reservoir outputs."""
        if signals.ndim == 3:
            # Flatten batch and sequence dimensions
            signals = signals.reshape(-1, signals.shape[-1])
        
        freq_content = np.zeros(min(self.embedding_dim, signals.shape[-1]))
        
        for i in range(len(freq_content)):
            if i < signals.shape[-1]:
                signal = signals[:, i]
                fft = np.fft.fft(signal)
                # Focus on dominant frequency components
                freq_content[i] = np.mean(np.abs(fft[:len(fft)//4]))
        
        # Normalize
        freq_content = freq_content / (np.max(freq_content) + 1e-8)
        return freq_content
    
    def get_embedding_matrix(self) -> np.ndarray:
        """Get the embedding matrix."""
        if not self._is_fitted:
            raise TokenizerNotFittedError("get_embedding_matrix")
        return self._embedding_matrix.copy()
    
    def save(self, save_path: str):
        """
        Save embedder to disk.
        
        Args:
            save_path: Directory path to save embedder
        """
        if not self._is_fitted:
            raise TokenizerNotFittedError("save")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save embedding matrix
            np.save(os.path.join(save_path, "embedding_matrix.npy"), 
                   self._embedding_matrix)
            
            # Save positional encodings
            np.save(os.path.join(save_path, "positional_encodings.npy"), 
                   self._positional_encodings)
            
            # Save configuration
            config = {
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'max_position': self.max_position,
                'temperature': self.temperature,
                'is_fitted': self._is_fitted
            }
            
            with open(os.path.join(save_path, "sinusoidal_embedder_config.json"), "w") as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"SinusoidalEmbedder saved to {save_path}")
            
        except Exception as e:
            raise TokenizerSaveError(save_path, str(e))
    
    @classmethod
    def load(cls, load_path: str) -> 'SinusoidalEmbedder':
        """
        Load embedder from disk.
        
        Args:
            load_path: Directory path to load embedder from
            
        Returns:
            Loaded SinusoidalEmbedder instance
        """
        try:
            # Validate path exists
            if not os.path.exists(load_path):
                raise FileNotFoundError(f"Load path does not exist: {load_path}")
            
            config_path = os.path.join(load_path, "sinusoidal_embedder_config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            # Load configuration
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Create embedder instance
            embedder = cls(
                vocab_size=config['vocab_size'],
                embedding_dim=config['embedding_dim'],
                max_position=config['max_position'],
                temperature=config['temperature']
            )
            
            # Load embedding matrix and positional encodings
            embedder._embedding_matrix = np.load(
                os.path.join(load_path, "embedding_matrix.npy")
            )
            embedder._positional_encodings = np.load(
                os.path.join(load_path, "positional_encodings.npy")
            )
            embedder._is_fitted = config['is_fitted']
            
            logger.info(f"SinusoidalEmbedder loaded from {load_path}")
            return embedder
            
        except Exception as e:
            raise TokenizerLoadError(load_path, str(e))
    
    def __repr__(self) -> str:
        return (f"SinusoidalEmbedder(vocab_size={self.vocab_size}, "
                f"embedding_dim={self.embedding_dim}, fitted={self._is_fitted})")


class EmbeddingOptimizer:
    """
    Optimizer for analyzing and optimizing embedding sinusoidality.
    
    This class provides algorithms to evaluate embedding quality for
    reservoir processing and optimize embeddings for sine-activated LSM.
    """
    
    def __init__(self, learning_rate: float = 0.01, max_iterations: int = 1000,
                 convergence_threshold: float = 1e-6):
        """
        Initialize EmbeddingOptimizer.
        
        Args:
            learning_rate: Learning rate for optimization
            max_iterations: Maximum number of optimization iterations
            convergence_threshold: Threshold for convergence detection
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        
        # Optimization history
        self._optimization_history = []
        self._best_score = -np.inf
        self._best_embeddings = None
        
        logger.info(f"Initialized EmbeddingOptimizer with lr={learning_rate}, "
                   f"max_iter={max_iterations}")
    
    def analyze_sinusoidality(self, embeddings: np.ndarray) -> Dict[str, float]:
        """
        Analyze the sinusoidality of embeddings.
        
        Args:
            embeddings: Embedding matrix (vocab_size, embedding_dim)
            
        Returns:
            Dictionary with sinusoidality metrics
        """
        if embeddings.ndim != 2:
            raise InvalidInputError("Embeddings must be 2D array (vocab_size, embedding_dim)")
        
        vocab_size, embedding_dim = embeddings.shape
        
        # Calculate various sinusoidality metrics
        metrics = {}
        
        # 1. Frequency domain analysis
        freq_scores = []
        for i in range(embedding_dim):
            signal = embeddings[:, i]
            fft = np.fft.fft(signal)
            
            # Power spectral density
            psd = np.abs(fft)**2
            
            # Focus on low-frequency components (more sinusoidal)
            low_freq_power = np.sum(psd[:len(psd)//4])
            total_power = np.sum(psd)
            
            freq_score = low_freq_power / (total_power + 1e-8)
            freq_scores.append(freq_score)
        
        metrics['frequency_score'] = np.mean(freq_scores)
        metrics['frequency_std'] = np.std(freq_scores)
        
        # 2. Autocorrelation analysis
        autocorr_scores = []
        for i in range(embedding_dim):
            signal = embeddings[:, i]
            # Normalize signal
            signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
            
            # Calculate autocorrelation
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Look for periodic patterns
            if len(autocorr) > 1:
                # Find peaks in autocorrelation
                peak_score = np.max(autocorr[1:len(autocorr)//4]) if len(autocorr) > 4 else 0
                autocorr_scores.append(peak_score)
        
        metrics['autocorr_score'] = np.mean(autocorr_scores) if autocorr_scores else 0.0
        metrics['autocorr_std'] = np.std(autocorr_scores) if autocorr_scores else 0.0
        
        # 3. Smoothness analysis (sinusoidal signals should be smooth)
        smoothness_scores = []
        for i in range(embedding_dim):
            signal = embeddings[:, i]
            # Calculate second derivative (measure of smoothness)
            if len(signal) > 2:
                second_deriv = np.diff(signal, n=2)
                smoothness = 1.0 / (1.0 + np.mean(np.abs(second_deriv)))
                smoothness_scores.append(smoothness)
        
        metrics['smoothness_score'] = np.mean(smoothness_scores) if smoothness_scores else 0.0
        metrics['smoothness_std'] = np.std(smoothness_scores) if smoothness_scores else 0.0
        
        # 4. Phase coherence analysis
        phase_scores = []
        for i in range(0, embedding_dim-1, 2):
            if i+1 < embedding_dim:
                # Treat pairs as sin/cos components
                sin_component = embeddings[:, i]
                cos_component = embeddings[:, i+1]
                
                # Normalize components
                sin_norm = (sin_component - np.mean(sin_component)) / (np.std(sin_component) + 1e-8)
                cos_norm = (cos_component - np.mean(cos_component)) / (np.std(cos_component) + 1e-8)
                
                # Calculate normalized phase coherence (should be between 0 and 1)
                coherence = np.mean(np.abs(sin_norm * cos_norm))
                phase_scores.append(min(coherence, 1.0))  # Clamp to [0, 1]
        
        metrics['phase_coherence'] = np.mean(phase_scores) if phase_scores else 0.0
        metrics['phase_coherence_std'] = np.std(phase_scores) if phase_scores else 0.0
        
        # 5. Overall sinusoidality score
        overall_score = (
            0.4 * metrics['frequency_score'] +
            0.3 * metrics['autocorr_score'] +
            0.2 * metrics['smoothness_score'] +
            0.1 * metrics['phase_coherence']
        )
        metrics['overall_score'] = np.clip(overall_score, 0.0, 1.0)
        
        return metrics
    
    def evaluate_reservoir_compatibility(self, embeddings: np.ndarray, 
                                       reservoir_outputs: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate how well embeddings work with reservoir processing.
        
        Args:
            embeddings: Embedding matrix
            reservoir_outputs: Optional reservoir outputs for analysis
            
        Returns:
            Dictionary with compatibility metrics
        """
        metrics = {}
        
        # Basic embedding properties
        vocab_size, embedding_dim = embeddings.shape
        
        # 1. Embedding diversity (avoid collapse)
        pairwise_similarities = np.dot(embeddings, embeddings.T)
        np.fill_diagonal(pairwise_similarities, 0)  # Ignore self-similarity
        
        avg_similarity = np.mean(np.abs(pairwise_similarities))
        max_similarity = np.max(np.abs(pairwise_similarities))
        
        metrics['diversity_score'] = 1.0 - avg_similarity  # Higher is better
        metrics['max_similarity'] = max_similarity
        
        # 2. Embedding magnitude consistency
        norms = np.linalg.norm(embeddings, axis=1)
        metrics['norm_mean'] = np.mean(norms)
        metrics['norm_std'] = np.std(norms)
        metrics['norm_consistency'] = 1.0 / (1.0 + metrics['norm_std'])
        
        # 3. Dimension utilization
        dim_variances = np.var(embeddings, axis=0)
        metrics['dim_utilization'] = np.mean(dim_variances > 1e-6)  # Fraction of used dimensions
        metrics['dim_balance'] = 1.0 - np.std(dim_variances) / (np.mean(dim_variances) + 1e-8)
        
        # 4. If reservoir outputs provided, analyze compatibility
        if reservoir_outputs is not None:
            # Analyze frequency matching
            embedding_freqs = self._analyze_frequency_spectrum(embeddings)
            reservoir_freqs = self._analyze_frequency_spectrum(reservoir_outputs.reshape(-1, reservoir_outputs.shape[-1]))
            
            # Calculate frequency alignment
            freq_alignment = self._calculate_frequency_alignment(embedding_freqs, reservoir_freqs)
            metrics['frequency_alignment'] = freq_alignment
        
        # 5. Overall compatibility score
        base_score = (
            0.3 * metrics['diversity_score'] +
            0.2 * metrics['norm_consistency'] +
            0.2 * metrics['dim_utilization'] +
            0.3 * metrics['dim_balance']
        )
        
        if 'frequency_alignment' in metrics:
            metrics['compatibility_score'] = 0.7 * base_score + 0.3 * metrics['frequency_alignment']
        else:
            metrics['compatibility_score'] = base_score
        
        return metrics
    
    def optimize_embeddings(self, embeddings: np.ndarray, 
                          target_metrics: Optional[Dict[str, float]] = None,
                          reservoir_outputs: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Optimize embeddings for better sinusoidality and reservoir compatibility.
        
        Args:
            embeddings: Initial embedding matrix
            target_metrics: Target values for optimization metrics
            reservoir_outputs: Optional reservoir outputs for compatibility optimization
            
        Returns:
            Tuple of (optimized_embeddings, optimization_info)
        """
        logger.info("Starting embedding optimization")
        
        # Initialize optimization
        current_embeddings = embeddings.copy()
        self._optimization_history = []
        self._best_score = -np.inf
        self._best_embeddings = None
        
        # Set default target metrics if not provided
        if target_metrics is None:
            target_metrics = {
                'overall_score': 0.8,
                'compatibility_score': 0.7,
                'diversity_score': 0.6
            }
        
        prev_score = -np.inf
        
        for iteration in range(self.max_iterations):
            # Analyze current embeddings
            sin_metrics = self.analyze_sinusoidality(current_embeddings)
            compat_metrics = self.evaluate_reservoir_compatibility(
                current_embeddings, reservoir_outputs
            )
            
            # Calculate combined objective score
            objective_score = self._calculate_objective_score(
                sin_metrics, compat_metrics, target_metrics
            )
            
            # Track best embeddings
            if objective_score > self._best_score:
                self._best_score = objective_score
                self._best_embeddings = current_embeddings.copy()
            
            # Record history
            self._optimization_history.append({
                'iteration': iteration,
                'objective_score': objective_score,
                'sinusoidality_score': sin_metrics['overall_score'],
                'compatibility_score': compat_metrics['compatibility_score']
            })
            
            # Check convergence
            if abs(objective_score - prev_score) < self.convergence_threshold:
                logger.info(f"Converged at iteration {iteration}")
                break
            
            # Calculate gradients and update embeddings
            gradients = self._calculate_optimization_gradients(
                current_embeddings, sin_metrics, compat_metrics, target_metrics
            )
            
            # Apply gradients
            current_embeddings -= self.learning_rate * gradients
            
            # Normalize embeddings to prevent explosion
            norms = np.linalg.norm(current_embeddings, axis=1, keepdims=True)
            current_embeddings = current_embeddings / (norms + 1e-8)
            
            prev_score = objective_score
            
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: Score = {objective_score:.4f}")
        
        # Return best embeddings found
        final_embeddings = self._best_embeddings if self._best_embeddings is not None else current_embeddings
        
        optimization_info = {
            'final_score': self._best_score,
            'iterations': len(self._optimization_history),
            'converged': abs(objective_score - prev_score) < self.convergence_threshold,
            'history': self._optimization_history
        }
        
        logger.info(f"Optimization completed. Final score: {self._best_score:.4f}")
        
        return final_embeddings, optimization_info
    
    def _analyze_frequency_spectrum(self, signals: np.ndarray) -> np.ndarray:
        """Analyze frequency spectrum of signals."""
        if signals.ndim == 1:
            signals = signals.reshape(1, -1)
        
        freq_spectra = []
        for i in range(signals.shape[1]):
            signal = signals[:, i].flatten()
            fft = np.fft.fft(signal)
            spectrum = np.abs(fft[:len(fft)//2])  # Take positive frequencies
            freq_spectra.append(spectrum)
        
        return np.array(freq_spectra)
    
    def _calculate_frequency_alignment(self, freq1: np.ndarray, freq2: np.ndarray) -> float:
        """Calculate alignment between two frequency spectra."""
        # Normalize spectra
        freq1_norm = freq1 / (np.linalg.norm(freq1, axis=1, keepdims=True) + 1e-8)
        freq2_norm = freq2 / (np.linalg.norm(freq2, axis=1, keepdims=True) + 1e-8)
        
        # Calculate cosine similarity
        min_dims = min(freq1_norm.shape[0], freq2_norm.shape[0])
        min_len = min(freq1_norm.shape[1], freq2_norm.shape[1])
        
        similarities = []
        for i in range(min_dims):
            sim = np.dot(freq1_norm[i, :min_len], freq2_norm[i, :min_len])
            similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_objective_score(self, sin_metrics: Dict[str, float], 
                                 compat_metrics: Dict[str, float],
                                 target_metrics: Dict[str, float]) -> float:
        """Calculate combined objective score for optimization."""
        score = 0.0
        
        # Sinusoidality component
        sin_score = sin_metrics['overall_score']
        sin_target = target_metrics.get('overall_score', 0.8)
        sin_component = 1.0 - abs(sin_score - sin_target)
        
        # Compatibility component
        compat_score = compat_metrics['compatibility_score']
        compat_target = target_metrics.get('compatibility_score', 0.7)
        compat_component = 1.0 - abs(compat_score - compat_target)
        
        # Diversity component
        diversity_score = compat_metrics['diversity_score']
        diversity_target = target_metrics.get('diversity_score', 0.6)
        diversity_component = 1.0 - abs(diversity_score - diversity_target)
        
        # Weighted combination
        score = (
            0.4 * sin_component +
            0.4 * compat_component +
            0.2 * diversity_component
        )
        
        return score
    
    def _calculate_optimization_gradients(self, embeddings: np.ndarray,
                                        sin_metrics: Dict[str, float],
                                        compat_metrics: Dict[str, float],
                                        target_metrics: Dict[str, float]) -> np.ndarray:
        """Calculate gradients for embedding optimization (simplified analytical version)."""
        vocab_size, embedding_dim = embeddings.shape
        gradients = np.zeros_like(embeddings)
        
        # Simplified analytical gradients for key objectives
        
        # 1. Sinusoidality gradient - encourage sinusoidal patterns
        for i in range(vocab_size):
            for j in range(embedding_dim):
                freq = (j + 1) / embedding_dim
                phase = 2 * np.pi * i / vocab_size
                
                if j % 2 == 0:
                    target = np.sin(freq * phase)
                else:
                    target = np.cos(freq * phase)
                
                # Gradient toward sinusoidal target
                gradients[i, j] += 0.5 * (target - embeddings[i, j])
        
        # 2. Diversity gradient - prevent embedding collapse
        mean_embedding = np.mean(embeddings, axis=0)
        for i in range(vocab_size):
            # Push embeddings away from mean
            gradients[i] += 0.3 * (embeddings[i] - mean_embedding)
        
        # 3. Normalization gradient - maintain unit norm
        for i in range(vocab_size):
            norm = np.linalg.norm(embeddings[i])
            if norm > 0:
                # Gradient to maintain unit norm
                gradients[i] += 0.2 * (embeddings[i] / norm - embeddings[i])
        
        return gradients
    
    def create_training_loop(self, embedder: SinusoidalEmbedder, 
                           training_data: np.ndarray,
                           validation_data: Optional[np.ndarray] = None,
                           epochs: int = 50) -> Dict[str, Any]:
        """
        Create a training loop for embedding optimization.
        
        Args:
            embedder: SinusoidalEmbedder to optimize
            training_data: Training token sequences
            validation_data: Optional validation token sequences
            epochs: Number of training epochs
            
        Returns:
            Training history and metrics
        """
        logger.info(f"Starting embedding training loop for {epochs} epochs")
        
        training_history = {
            'train_scores': [],
            'val_scores': [],
            'sinusoidality_scores': [],
            'compatibility_scores': []
        }
        
        for epoch in range(epochs):
            # Get current embeddings
            if not embedder._is_fitted:
                # Initial fit
                embedder.fit(training_data, epochs=5)
            
            current_embeddings = embedder.get_embedding_matrix()
            
            # Analyze current state
            sin_metrics = self.analyze_sinusoidality(current_embeddings)
            compat_metrics = self.evaluate_reservoir_compatibility(current_embeddings)
            
            # Optimize embeddings
            optimized_embeddings, opt_info = self.optimize_embeddings(
                current_embeddings, 
                target_metrics={'overall_score': 0.8, 'compatibility_score': 0.7}
            )
            
            # Update embedder with optimized embeddings
            embedder._embedding_matrix = optimized_embeddings
            
            # Record metrics
            training_history['sinusoidality_scores'].append(sin_metrics['overall_score'])
            training_history['compatibility_scores'].append(compat_metrics['compatibility_score'])
            training_history['train_scores'].append(opt_info['final_score'])
            
            # Validation if provided
            if validation_data is not None:
                val_embeddings = embedder.embed(validation_data)
                val_sin_metrics = self.analyze_sinusoidality(val_embeddings.reshape(-1, val_embeddings.shape[-1]))
                training_history['val_scores'].append(val_sin_metrics['overall_score'])
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Sin={sin_metrics['overall_score']:.4f}, "
                           f"Compat={compat_metrics['compatibility_score']:.4f}")
        
        logger.info("Training loop completed")
        return training_history
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the optimization history."""
        return self._optimization_history.copy()
    
    def __repr__(self) -> str:
        return (f"EmbeddingOptimizer(lr={self.learning_rate}, "
                f"max_iter={self.max_iterations})")