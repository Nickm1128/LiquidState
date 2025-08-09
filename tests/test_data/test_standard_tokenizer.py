#!/usr/bin/env python3
"""
Tests for StandardTokenizerWrapper class.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from src.lsm.data.tokenization import StandardTokenizerWrapper
from src.lsm.utils.lsm_exceptions import TokenizerError, TokenizerSaveError, TokenizerLoadError


class TestStandardTokenizerWrapper:
    """Test cases for StandardTokenizerWrapper."""
    
    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers library for testing."""
        with patch('src.lsm.data.tokenization.TRANSFORMERS_AVAILABLE', True):
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = '<|endoftext|>'
            mock_tokenizer.pad_token_id = 50256
            mock_tokenizer.eos_token_id = 50256
            mock_tokenizer.bos_token_id = None
            mock_tokenizer.__len__ = MagicMock(return_value=50257)
            mock_tokenizer.get_vocab.return_value = {'hello': 15496, 'world': 995}
            
            # Mock tokenization
            mock_tokenizer.return_value = {
                'input_ids': [[15496, 995]]  # hello world
            }
            
            # Mock decoding
            mock_tokenizer.decode.return_value = "hello world"
            
            with patch('src.lsm.data.tokenization.AutoTokenizer') as mock_auto:
                mock_auto.from_pretrained.return_value = mock_tokenizer
                yield mock_tokenizer
    
    def test_init_success(self, mock_transformers):
        """Test successful initialization."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        assert tokenizer.tokenizer_name == 'gpt2'
        assert tokenizer.max_length == 512
        assert tokenizer.get_vocab_size() == 50257
    
    def test_init_unsupported_tokenizer(self):
        """Test initialization with unsupported tokenizer."""
        with patch('src.lsm.data.tokenization.TRANSFORMERS_AVAILABLE', True):
            with pytest.raises(TokenizerError, match="Unsupported tokenizer"):
                StandardTokenizerWrapper('unsupported-tokenizer')
    
    def test_init_no_transformers(self):
        """Test initialization without transformers library."""
        with patch('src.lsm.data.tokenization.TRANSFORMERS_AVAILABLE', False):
            with pytest.raises(TokenizerError, match="transformers library not available"):
                StandardTokenizerWrapper('gpt2')
    
    def test_tokenize_single_text(self, mock_transformers):
        """Test tokenizing a single text."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        # Mock the tokenizer call
        mock_transformers.return_value = {'input_ids': [[15496, 995]]}
        
        result = tokenizer.tokenize("hello world")
        
        assert result == [[15496, 995]]
        mock_transformers.assert_called_once()
    
    def test_tokenize_multiple_texts(self, mock_transformers):
        """Test tokenizing multiple texts."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        # Mock the tokenizer call
        mock_transformers.return_value = {
            'input_ids': [[15496, 995], [31373, 995]]
        }
        
        result = tokenizer.tokenize(["hello world", "goodbye world"])
        
        assert result == [[15496, 995], [31373, 995]]
    
    def test_decode_single_sequence(self, mock_transformers):
        """Test decoding a single sequence."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        result = tokenizer.decode([15496, 995])
        
        assert result == "hello world"
        mock_transformers.decode.assert_called_once_with([15496, 995], skip_special_tokens=True)
    
    def test_decode_batch_sequences(self, mock_transformers):
        """Test decoding batch of sequences."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        # Mock multiple decode calls
        mock_transformers.decode.side_effect = ["hello world", "goodbye world"]
        
        result = tokenizer.decode([[15496, 995], [31373, 995]])
        
        assert result == ["hello world", "goodbye world"]
        assert mock_transformers.decode.call_count == 2
    
    def test_encode_single(self, mock_transformers):
        """Test encoding a single text."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        # Mock the tokenizer call
        mock_transformers.return_value = {'input_ids': [[15496, 995]]}
        
        result = tokenizer.encode_single("hello world")
        
        assert result == [15496, 995]
    
    def test_decode_single(self, mock_transformers):
        """Test decoding a single sequence."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        result = tokenizer.decode_single([15496, 995])
        
        assert result == "hello world"
    
    def test_get_vocab(self, mock_transformers):
        """Test getting vocabulary."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        vocab = tokenizer.get_vocab()
        
        assert vocab == {'hello': 15496, 'world': 995}
    
    def test_get_special_tokens(self, mock_transformers):
        """Test getting special tokens."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        special_tokens = tokenizer.get_special_tokens()
        
        expected = {
            'pad_token_id': 50256,
            'eos_token_id': 50256,
            'bos_token_id': None
        }
        assert special_tokens == expected
    
    def test_get_token_embeddings_shape(self, mock_transformers):
        """Test getting token embeddings shape."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        shape = tokenizer.get_token_embeddings_shape(128)
        
        assert shape == (50257, 128)
    
    def test_save_and_load(self, mock_transformers):
        """Test saving and loading tokenizer configuration."""
        tokenizer = StandardTokenizerWrapper('gpt2', max_length=256)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save tokenizer
            tokenizer.save(temp_dir)
            
            # Check config file exists
            config_path = os.path.join(temp_dir, 'standard_tokenizer_config.json')
            assert os.path.exists(config_path)
            
            # Load tokenizer
            loaded_tokenizer = StandardTokenizerWrapper.load(temp_dir)
            
            assert loaded_tokenizer.tokenizer_name == 'gpt2'
            assert loaded_tokenizer.max_length == 256
            assert loaded_tokenizer.get_vocab_size() == 50257
    
    def test_save_error(self, mock_transformers):
        """Test save error handling."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        # Try to save to invalid path (empty string)
        with pytest.raises(TokenizerSaveError):
            tokenizer.save("")
    
    def test_load_error(self):
        """Test load error handling."""
        with patch('src.lsm.data.tokenization.TRANSFORMERS_AVAILABLE', True):
            with patch('os.path.exists', return_value=False):
                with pytest.raises(TokenizerLoadError):
                    StandardTokenizerWrapper.load("/invalid/path/that/does/not/exist")
    
    def test_repr(self, mock_transformers):
        """Test string representation."""
        tokenizer = StandardTokenizerWrapper('gpt2')
        
        repr_str = repr(tokenizer)
        
        assert "StandardTokenizerWrapper" in repr_str
        assert "gpt2" in repr_str
        assert "50257" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])


class TestSinusoidalEmbedder:
    """Test cases for SinusoidalEmbedder."""
    
    @pytest.fixture
    def embedder(self):
        """Create a SinusoidalEmbedder for testing."""
        from src.lsm.data.tokenization import SinusoidalEmbedder
        return SinusoidalEmbedder(vocab_size=100, embedding_dim=64)
    
    def test_init(self, embedder):
        """Test initialization."""
        assert embedder.vocab_size == 100
        assert embedder.embedding_dim == 64
        assert embedder.max_position == 10000
        assert embedder.temperature == 1.0
        assert not embedder._is_fitted
    
    def test_create_positional_encodings(self, embedder):
        """Test positional encoding creation."""
        pos_encodings = embedder._create_positional_encodings()
        
        assert pos_encodings.shape == (embedder.max_position, embedder.embedding_dim)
        assert pos_encodings.dtype == np.float32
        
        # Check that encodings have sinusoidal patterns
        # First dimension should be sin, second should be cos
        assert not np.allclose(pos_encodings[:, 0], pos_encodings[:, 1])
    
    def test_initialize_embeddings(self, embedder):
        """Test embedding initialization."""
        embeddings = embedder._initialize_embeddings()
        
        assert embeddings.shape == (embedder.vocab_size, embedder.embedding_dim)
        assert embeddings.dtype == np.float32
        
        # Check that embeddings are normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)
        
        # Check that embeddings are different for different tokens
        assert not np.allclose(embeddings[0], embeddings[1])
    
    def test_fit(self, embedder):
        """Test fitting the embedder."""
        # Create sample training data
        training_data = np.random.randint(0, embedder.vocab_size, (10, 20))
        
        embedder.fit(training_data, epochs=5)
        
        assert embedder._is_fitted
        assert embedder._embedding_matrix is not None
        assert embedder._positional_encodings is not None
        assert embedder._embedding_matrix.shape == (embedder.vocab_size, embedder.embedding_dim)
    
    def test_embed_single_sequence(self, embedder):
        """Test embedding a single sequence."""
        # Fit embedder first
        training_data = np.random.randint(0, embedder.vocab_size, (5, 10))
        embedder.fit(training_data, epochs=2)
        
        # Test embedding
        token_ids = [1, 5, 10, 15]
        embeddings = embedder.embed(token_ids)
        
        assert embeddings.shape == (len(token_ids), embedder.embedding_dim)
        assert embeddings.dtype == np.float32
    
    def test_embed_batch_sequences(self, embedder):
        """Test embedding batch of sequences."""
        # Fit embedder first
        training_data = np.random.randint(0, embedder.vocab_size, (5, 10))
        embedder.fit(training_data, epochs=2)
        
        # Test batch embedding
        token_sequences = np.array([[1, 5, 10], [2, 6, 11]])
        embeddings = embedder.embed(token_sequences)
        
        assert embeddings.shape == (2, 3, embedder.embedding_dim)
        assert embeddings.dtype == np.float32
    
    def test_embed_not_fitted_error(self, embedder):
        """Test error when embedding without fitting."""
        with pytest.raises(TokenizerNotFittedError):
            embedder.embed([1, 2, 3])
    
    def test_optimize_for_sine_activation(self, embedder):
        """Test optimization for sine activation."""
        # Fit embedder first
        training_data = np.random.randint(0, embedder.vocab_size, (5, 10))
        embedder.fit(training_data, epochs=2)
        
        # Create sample reservoir outputs
        reservoir_outputs = np.random.randn(10, 20, embedder.embedding_dim)
        
        # Get embeddings before optimization
        original_matrix = embedder.get_embedding_matrix()
        
        # Optimize
        embedder.optimize_for_sine_activation(reservoir_outputs)
        
        # Check that embeddings changed
        optimized_matrix = embedder.get_embedding_matrix()
        assert not np.allclose(original_matrix, optimized_matrix)
    
    def test_calculate_sinusoidality_score(self, embedder):
        """Test sinusoidality score calculation."""
        # Create embeddings with known sinusoidal patterns
        embeddings = np.zeros((10, 5, embedder.embedding_dim))
        for i in range(10):
            for j in range(5):
                for k in range(embedder.embedding_dim):
                    embeddings[i, j, k] = np.sin(2 * np.pi * k / embedder.embedding_dim)
        
        score = embedder._calculate_sinusoidality_score(embeddings)
        
        assert isinstance(score, float)
        assert 0 <= score <= 1
    
    def test_get_embedding_matrix(self, embedder):
        """Test getting embedding matrix."""
        # Test error when not fitted
        with pytest.raises(TokenizerNotFittedError):
            embedder.get_embedding_matrix()
        
        # Fit and test
        training_data = np.random.randint(0, embedder.vocab_size, (5, 10))
        embedder.fit(training_data, epochs=2)
        
        matrix = embedder.get_embedding_matrix()
        assert matrix.shape == (embedder.vocab_size, embedder.embedding_dim)
        
        # Check that it's a copy (modifying shouldn't affect original)
        matrix[0, 0] = 999.0
        original_matrix = embedder.get_embedding_matrix()
        assert original_matrix[0, 0] != 999.0
    
    def test_save_and_load(self, embedder):
        """Test saving and loading embedder."""
        # Fit embedder first
        training_data = np.random.randint(0, embedder.vocab_size, (5, 10))
        embedder.fit(training_data, epochs=2)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save embedder
            embedder.save(temp_dir)
            
            # Check files exist
            assert os.path.exists(os.path.join(temp_dir, "embedding_matrix.npy"))
            assert os.path.exists(os.path.join(temp_dir, "positional_encodings.npy"))
            assert os.path.exists(os.path.join(temp_dir, "sinusoidal_embedder_config.json"))
            
            # Load embedder
            from src.lsm.data.tokenization import SinusoidalEmbedder
            loaded_embedder = SinusoidalEmbedder.load(temp_dir)
            
            assert loaded_embedder.vocab_size == embedder.vocab_size
            assert loaded_embedder.embedding_dim == embedder.embedding_dim
            assert loaded_embedder._is_fitted
            
            # Test that embeddings are the same
            original_matrix = embedder.get_embedding_matrix()
            loaded_matrix = loaded_embedder.get_embedding_matrix()
            assert np.allclose(original_matrix, loaded_matrix)
    
    def test_save_not_fitted_error(self, embedder):
        """Test save error when not fitted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(TokenizerNotFittedError):
                embedder.save(temp_dir)
    
    def test_load_error(self):
        """Test load error handling."""
        from src.lsm.data.tokenization import SinusoidalEmbedder
        
        with pytest.raises(TokenizerLoadError):
            SinusoidalEmbedder.load("/invalid/path/that/does/not/exist")
    
    def test_repr(self, embedder):
        """Test string representation."""
        repr_str = repr(embedder)
        
        assert "SinusoidalEmbedder" in repr_str
        assert "vocab_size=100" in repr_str
        assert "embedding_dim=64" in repr_str
        assert "fitted=False" in repr_str


class TestEmbeddingOptimizer:
    """Test cases for EmbeddingOptimizer."""
    
    @pytest.fixture
    def optimizer(self):
        """Create an EmbeddingOptimizer for testing."""
        from src.lsm.data.tokenization import EmbeddingOptimizer
        return EmbeddingOptimizer(learning_rate=0.1, max_iterations=10)
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(50, 32).astype(np.float32)
    
    def test_init(self, optimizer):
        """Test initialization."""
        assert optimizer.learning_rate == 0.1
        assert optimizer.max_iterations == 10
        assert optimizer.convergence_threshold == 1e-6
        assert optimizer._optimization_history == []
        assert optimizer._best_score == -np.inf
    
    def test_analyze_sinusoidality(self, optimizer, sample_embeddings):
        """Test sinusoidality analysis."""
        metrics = optimizer.analyze_sinusoidality(sample_embeddings)
        
        # Check that all expected metrics are present
        expected_keys = [
            'frequency_score', 'frequency_std',
            'autocorr_score', 'autocorr_std',
            'smoothness_score', 'smoothness_std',
            'phase_coherence', 'phase_coherence_std',
            'overall_score'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (float, np.floating))
            assert not np.isnan(metrics[key])
        
        # Overall score should be between 0 and 1
        assert 0 <= metrics['overall_score'] <= 1
    
    def test_analyze_sinusoidality_invalid_input(self, optimizer):
        """Test sinusoidality analysis with invalid input."""
        # Test with 1D array
        with pytest.raises(InvalidInputError):
            optimizer.analyze_sinusoidality(np.array([1, 2, 3]))
        
        # Test with 3D array
        with pytest.raises(InvalidInputError):
            optimizer.analyze_sinusoidality(np.random.randn(10, 20, 30))
    
    def test_evaluate_reservoir_compatibility(self, optimizer, sample_embeddings):
        """Test reservoir compatibility evaluation."""
        metrics = optimizer.evaluate_reservoir_compatibility(sample_embeddings)
        
        # Check expected metrics
        expected_keys = [
            'diversity_score', 'max_similarity',
            'norm_mean', 'norm_std', 'norm_consistency',
            'dim_utilization', 'dim_balance',
            'compatibility_score'
        ]
        
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], (float, np.floating))
            assert not np.isnan(metrics[key])
        
        # Test with reservoir outputs
        reservoir_outputs = np.random.randn(20, 15, 32)
        metrics_with_reservoir = optimizer.evaluate_reservoir_compatibility(
            sample_embeddings, reservoir_outputs
        )
        
        assert 'frequency_alignment' in metrics_with_reservoir
        assert 'compatibility_score' in metrics_with_reservoir
    
    def test_optimize_embeddings(self, optimizer, sample_embeddings):
        """Test embedding optimization."""
        # Test basic optimization
        optimized_embeddings, opt_info = optimizer.optimize_embeddings(
            sample_embeddings
        )
        
        assert optimized_embeddings.shape == sample_embeddings.shape
        assert optimized_embeddings.dtype == sample_embeddings.dtype
        
        # Check optimization info
        assert 'final_score' in opt_info
        assert 'iterations' in opt_info
        assert 'converged' in opt_info
        assert 'history' in opt_info
        
        assert isinstance(opt_info['final_score'], (float, np.floating))
        assert isinstance(opt_info['iterations'], int)
        assert isinstance(opt_info['converged'], bool)
        assert isinstance(opt_info['history'], list)
        
        # Check that optimization history is recorded
        assert len(opt_info['history']) > 0
        assert len(optimizer.get_optimization_history()) > 0
    
    def test_optimize_embeddings_with_targets(self, optimizer, sample_embeddings):
        """Test optimization with target metrics."""
        target_metrics = {
            'overall_score': 0.9,
            'compatibility_score': 0.8,
            'diversity_score': 0.7
        }
        
        optimized_embeddings, opt_info = optimizer.optimize_embeddings(
            sample_embeddings, target_metrics=target_metrics
        )
        
        assert optimized_embeddings.shape == sample_embeddings.shape
        assert opt_info['final_score'] >= 0  # Should be non-negative
    
    def test_optimize_embeddings_with_reservoir(self, optimizer, sample_embeddings):
        """Test optimization with reservoir outputs."""
        reservoir_outputs = np.random.randn(20, 15, 32)
        
        optimized_embeddings, opt_info = optimizer.optimize_embeddings(
            sample_embeddings, reservoir_outputs=reservoir_outputs
        )
        
        assert optimized_embeddings.shape == sample_embeddings.shape
        assert 'final_score' in opt_info
    
    def test_analyze_frequency_spectrum(self, optimizer):
        """Test frequency spectrum analysis."""
        # Create a simple sinusoidal signal
        t = np.linspace(0, 2*np.pi, 100)
        signal = np.sin(t).reshape(1, -1)
        
        spectrum = optimizer._analyze_frequency_spectrum(signal)
        
        assert spectrum.shape[0] == 1  # One signal
        assert spectrum.shape[1] == 50  # Half of signal length (positive frequencies)
        assert np.all(spectrum >= 0)  # Magnitude spectrum should be non-negative
    
    def test_calculate_frequency_alignment(self, optimizer):
        """Test frequency alignment calculation."""
        # Create two similar frequency spectra
        freq1 = np.random.rand(5, 20)
        freq2 = freq1 + 0.1 * np.random.rand(5, 20)  # Similar but with noise
        
        alignment = optimizer._calculate_frequency_alignment(freq1, freq2)
        
        assert isinstance(alignment, (float, np.floating))
        assert 0 <= alignment <= 1  # Should be normalized similarity
        
        # Test with identical spectra
        alignment_identical = optimizer._calculate_frequency_alignment(freq1, freq1)
        assert alignment_identical > alignment  # Should be higher for identical spectra
    
    def test_create_training_loop(self, optimizer):
        """Test training loop creation."""
        from src.lsm.data.tokenization import SinusoidalEmbedder
        
        # Create embedder and training data
        embedder = SinusoidalEmbedder(vocab_size=50, embedding_dim=32)
        training_data = np.random.randint(0, 50, (10, 15))
        validation_data = np.random.randint(0, 50, (5, 15))
        
        # Run training loop
        history = optimizer.create_training_loop(
            embedder, training_data, validation_data, epochs=3
        )
        
        # Check history structure
        expected_keys = ['train_scores', 'val_scores', 'sinusoidality_scores', 'compatibility_scores']
        for key in expected_keys:
            assert key in history
            assert isinstance(history[key], list)
            assert len(history[key]) == 3  # Should have 3 epochs
        
        # Check that embedder is fitted
        assert embedder._is_fitted
    
    def test_get_optimization_history(self, optimizer, sample_embeddings):
        """Test getting optimization history."""
        # Run optimization to generate history
        optimizer.optimize_embeddings(sample_embeddings)
        
        history = optimizer.get_optimization_history()
        
        assert isinstance(history, list)
        assert len(history) > 0
        
        # Check history entry structure
        entry = history[0]
        assert 'iteration' in entry
        assert 'objective_score' in entry
        assert 'sinusoidality_score' in entry
        assert 'compatibility_score' in entry
    
    def test_repr(self, optimizer):
        """Test string representation."""
        repr_str = repr(optimizer)
        
        assert "EmbeddingOptimizer" in repr_str
        assert "lr=0.1" in repr_str
        assert "max_iter=10" in repr_str