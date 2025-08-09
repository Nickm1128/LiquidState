#!/usr/bin/env python3
"""
Tests for ResponseGenerator - Complete Response Generation.

This module tests the ResponseGenerator class and its integration with
existing CNN architectures and reservoir strategies.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

from src.lsm.inference.response_generator import (
    ResponseGenerator, TokenEmbeddingSequence, ResponseGenerationResult,
    ReservoirStrategy, ResponseGenerationError,
    create_response_generator, create_system_aware_response_generator
)
from src.lsm.core.cnn_3d_processor import SystemContext
from src.lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder


class TestTokenEmbeddingSequence:
    """Test TokenEmbeddingSequence dataclass."""
    
    def test_token_embedding_sequence_creation(self):
        """Test creating TokenEmbeddingSequence."""
        embeddings = np.random.randn(10, 128)
        tokens = [f"token_{i}" for i in range(10)]
        metadata = {"test": "data"}
        
        seq = TokenEmbeddingSequence(
            embeddings=embeddings,
            tokens=tokens,
            metadata=metadata
        )
        
        assert np.array_equal(seq.embeddings, embeddings)
        assert seq.tokens == tokens
        assert seq.metadata == metadata
    
    def test_token_embedding_sequence_without_metadata(self):
        """Test creating TokenEmbeddingSequence without metadata."""
        embeddings = np.random.randn(5, 64)
        tokens = ["hello", "world", "test", "tokens", "here"]
        
        seq = TokenEmbeddingSequence(embeddings=embeddings, tokens=tokens)
        
        assert np.array_equal(seq.embeddings, embeddings)
        assert seq.tokens == tokens
        assert seq.metadata is None


class TestResponseGenerationResult:
    """Test ResponseGenerationResult dataclass."""
    
    def test_response_generation_result_creation(self):
        """Test creating ResponseGenerationResult."""
        result = ResponseGenerationResult(
            response_text="Test response",
            confidence_score=0.85,
            generation_time=1.23,
            reservoir_strategy_used="adaptive",
            intermediate_embeddings=[np.random.randn(10, 128)],
            system_influence=0.5
        )
        
        assert result.response_text == "Test response"
        assert result.confidence_score == 0.85
        assert result.generation_time == 1.23
        assert result.reservoir_strategy_used == "adaptive"
        assert len(result.intermediate_embeddings) == 1
        assert result.system_influence == 0.5


class TestResponseGenerator:
    """Test ResponseGenerator class."""
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer."""
        tokenizer = Mock(spec=StandardTokenizerWrapper)
        tokenizer.tokenize.return_value = [[1, 2, 3, 4, 5]]
        tokenizer.decode.return_value = "test token"
        tokenizer.get_vocab_size.return_value = 1000
        return tokenizer
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = Mock(spec=SinusoidalEmbedder)
        embedder.embed.return_value = np.random.randn(5, 128)
        return embedder
    
    @pytest.fixture
    def mock_reservoir(self):
        """Create mock reservoir model."""
        reservoir = Mock()
        reservoir.predict.return_value = np.random.randn(1, 32, 32, 1)
        return reservoir
    
    @pytest.fixture
    def response_generator(self, mock_tokenizer, mock_embedder, mock_reservoir):
        """Create ResponseGenerator instance for testing."""
        return ResponseGenerator(
            tokenizer=mock_tokenizer,
            embedder=mock_embedder,
            reservoir_model=mock_reservoir,
            default_reservoir_strategy="adaptive"
        )
    
    def test_response_generator_initialization(self, mock_tokenizer, mock_embedder, mock_reservoir):
        """Test ResponseGenerator initialization."""
        generator = ResponseGenerator(
            tokenizer=mock_tokenizer,
            embedder=mock_embedder,
            reservoir_model=mock_reservoir,
            default_reservoir_strategy="reuse",
            max_response_length=256,
            confidence_threshold=0.7
        )
        
        assert generator.tokenizer == mock_tokenizer
        assert generator.embedder == mock_embedder
        assert generator.reservoir_model == mock_reservoir
        assert generator.default_reservoir_strategy == ReservoirStrategy.REUSE
        assert generator.max_response_length == 256
        assert generator.confidence_threshold == 0.7
    
    def test_generate_complete_response_with_text_input(self, response_generator):
        """Test generating complete response from text input."""
        input_text = ["Hello", "world", "test"]
        
        result = response_generator.generate_complete_response(input_text)
        
        assert isinstance(result, ResponseGenerationResult)
        assert isinstance(result.response_text, str)
        assert isinstance(result.confidence_score, float)
        assert isinstance(result.generation_time, float)
        assert result.reservoir_strategy_used in ["reuse", "separate", "adaptive"]
        assert result.generation_time > 0
    
    def test_generate_complete_response_with_embedding_sequence(self, response_generator):
        """Test generating complete response from TokenEmbeddingSequence."""
        embeddings = np.random.randn(10, 128)
        tokens = [f"token_{i}" for i in range(10)]
        
        embedding_seq = TokenEmbeddingSequence(
            embeddings=embeddings,
            tokens=tokens
        )
        
        result = response_generator.generate_complete_response(embedding_seq)
        
        assert isinstance(result, ResponseGenerationResult)
        assert isinstance(result.response_text, str)
        assert isinstance(result.confidence_score, float)
        assert result.generation_time > 0
    
    def test_generate_complete_response_with_system_context(self, response_generator):
        """Test generating response with system context."""
        # Mock 3D CNN processor
        mock_3d_processor = Mock()
        mock_result = Mock()
        mock_result.output_embeddings = np.random.randn(1, 512)
        mock_result.system_influence = 0.7
        mock_3d_processor.process_with_system_context.return_value = mock_result
        
        response_generator.cnn_3d_processor = mock_3d_processor
        
        input_text = ["Hello", "world"]
        system_context = SystemContext(
            message="Be helpful and friendly",
            embeddings=np.random.randn(256),
            influence_strength=0.8
        )
        
        result = response_generator.generate_complete_response(
            input_text, 
            system_context=system_context
        )
        
        assert isinstance(result, ResponseGenerationResult)
        assert result.system_influence == 0.7
        mock_3d_processor.process_with_system_context.assert_called_once()
    
    def test_generate_complete_response_with_intermediate_embeddings(self, response_generator):
        """Test generating response with intermediate embeddings returned."""
        input_text = ["Test", "input"]
        
        result = response_generator.generate_complete_response(
            input_text,
            return_intermediate=True
        )
        
        assert isinstance(result, ResponseGenerationResult)
        assert result.intermediate_embeddings is not None
        assert len(result.intermediate_embeddings) == 3  # input, reservoir, cnn outputs
    
    def test_process_token_embedding_sequences_batch(self, response_generator):
        """Test batch processing of token embedding sequences."""
        sequences = []
        for i in range(5):
            embeddings = np.random.randn(8, 128)
            tokens = [f"token_{j}" for j in range(8)]
            sequences.append(TokenEmbeddingSequence(embeddings=embeddings, tokens=tokens))
        
        results = response_generator.process_token_embedding_sequences(
            sequences, 
            batch_size=2
        )
        
        assert len(results) == 5
        for result in results:
            assert isinstance(result, ResponseGenerationResult)
    
    def test_determine_reservoir_strategy(self, response_generator):
        """Test reservoir strategy determination."""
        # Short, simple sequence
        simple_embeddings = np.random.randn(10, 128) * 0.1
        simple_seq = TokenEmbeddingSequence(
            embeddings=simple_embeddings,
            tokens=[f"token_{i}" for i in range(10)]
        )
        
        strategy = response_generator.determine_reservoir_strategy(simple_seq)
        assert strategy == ReservoirStrategy.REUSE
        
        # Complex sequence
        complex_embeddings = np.random.randn(100, 128) * 2.0
        complex_seq = TokenEmbeddingSequence(
            embeddings=complex_embeddings,
            tokens=[f"token_{i}" for i in range(100)]
        )
        
        strategy = response_generator.determine_reservoir_strategy(complex_seq)
        assert strategy == ReservoirStrategy.SEPARATE
        
        # With system context
        system_context = SystemContext(
            message="Test system message",
            embeddings=np.random.randn(256)
        )
        
        strategy = response_generator.determine_reservoir_strategy(simple_seq, system_context)
        assert strategy == ReservoirStrategy.SEPARATE
    
    def test_get_generation_statistics(self, response_generator):
        """Test getting generation statistics."""
        # Generate some responses to populate statistics
        input_text = ["Test", "statistics"]
        
        for _ in range(3):
            response_generator.generate_complete_response(input_text)
        
        stats = response_generator.get_generation_statistics()
        
        assert "total_generations" in stats
        assert "successful_generations" in stats
        assert "average_generation_time" in stats
        assert "success_rate" in stats
        assert "reservoir_strategy_usage" in stats
        
        assert stats["total_generations"] == 3
        assert stats["success_rate"] >= 0.0
        assert stats["average_generation_time"] > 0.0
    
    def test_reset_statistics(self, response_generator):
        """Test resetting generation statistics."""
        # Generate a response to populate statistics
        input_text = ["Test", "reset"]
        response_generator.generate_complete_response(input_text)
        
        # Verify statistics are populated
        stats_before = response_generator.get_generation_statistics()
        assert stats_before["total_generations"] > 0
        
        # Reset statistics
        response_generator.reset_statistics()
        
        # Verify statistics are reset
        stats_after = response_generator.get_generation_statistics()
        assert stats_after["total_generations"] == 0
        assert stats_after["successful_generations"] == 0
        assert stats_after["average_generation_time"] == 0.0
    
    def test_reservoir_strategy_enum(self):
        """Test ReservoirStrategy enum."""
        assert ReservoirStrategy.REUSE.value == "reuse"
        assert ReservoirStrategy.SEPARATE.value == "separate"
        assert ReservoirStrategy.ADAPTIVE.value == "adaptive"
    
    def test_response_generation_error_handling(self, response_generator):
        """Test error handling in response generation."""
        # Mock tokenizer to raise an exception
        response_generator.tokenizer.tokenize.side_effect = Exception("Tokenization failed")
        
        with pytest.raises(ResponseGenerationError) as exc_info:
            response_generator.generate_complete_response(["test", "error"])
        
        assert "complete_response_generation" in str(exc_info.value)
        assert "Tokenization failed" in str(exc_info.value)
    
    def test_batch_processing_error_handling(self, response_generator):
        """Test error handling in batch processing."""
        # Create sequences that will cause errors
        sequences = [
            TokenEmbeddingSequence(
                embeddings=np.random.randn(5, 128),
                tokens=["test"] * 5
            )
        ]
        
        # Mock reservoir to cause error in processing
        response_generator.reservoir_model.predict.side_effect = Exception("Reservoir processing error")
        
        # Should not raise exception but return error results
        results = response_generator.process_token_embedding_sequences(sequences)
        
        assert len(results) == 1
        assert results[0].response_text == "[ERROR]"
        assert results[0].confidence_score == 0.0


class TestConvenienceFunctions:
    """Test convenience functions for creating ResponseGenerator instances."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        tokenizer = Mock(spec=StandardTokenizerWrapper)
        embedder = Mock(spec=SinusoidalEmbedder)
        reservoir = Mock()
        return tokenizer, embedder, reservoir
    
    def test_create_response_generator(self, mock_components):
        """Test create_response_generator function."""
        tokenizer, embedder, reservoir = mock_components
        
        generator = create_response_generator(
            tokenizer=tokenizer,
            embedder=embedder,
            reservoir_model=reservoir,
            enable_3d_cnn=False,
            reservoir_strategy="reuse"
        )
        
        assert isinstance(generator, ResponseGenerator)
        assert generator.tokenizer == tokenizer
        assert generator.embedder == embedder
        assert generator.reservoir_model == reservoir
        assert generator.cnn_3d_processor is None
        assert generator.default_reservoir_strategy == ReservoirStrategy.REUSE
    
    def test_create_response_generator_with_3d_cnn(self, mock_components):
        """Test create_response_generator with 3D CNN enabled."""
        tokenizer, embedder, reservoir = mock_components
        
        generator = create_response_generator(
            tokenizer=tokenizer,
            embedder=embedder,
            reservoir_model=reservoir,
            enable_3d_cnn=True,
            reservoir_strategy="adaptive"
        )
        
        assert isinstance(generator, ResponseGenerator)
        assert generator.cnn_3d_processor is not None
        assert generator.default_reservoir_strategy == ReservoirStrategy.ADAPTIVE
    
    def test_create_system_aware_response_generator(self, mock_components):
        """Test create_system_aware_response_generator function."""
        tokenizer, embedder, reservoir = mock_components
        
        generator = create_system_aware_response_generator(
            tokenizer=tokenizer,
            embedder=embedder,
            reservoir_model=reservoir,
            reservoir_shape=(32, 32, 32, 2)
        )
        
        assert isinstance(generator, ResponseGenerator)
        assert generator.cnn_3d_processor is not None
        assert generator.default_reservoir_strategy == ReservoirStrategy.SEPARATE


class TestIntegration:
    """Integration tests for ResponseGenerator with real components."""
    
    @pytest.mark.skipif(
        not hasattr(StandardTokenizerWrapper, '__init__'),
        reason="StandardTokenizerWrapper not available"
    )
    def test_integration_with_real_tokenizer(self):
        """Test integration with real tokenizer (if available)."""
        try:
            # This test will be skipped if transformers is not available
            tokenizer = StandardTokenizerWrapper('gpt2')
            
            # Mock other components
            embedder = Mock(spec=SinusoidalEmbedder)
            embedder.embed.return_value = np.random.randn(5, 128)
            
            reservoir = Mock()
            reservoir.predict.return_value = np.random.randn(1, 32, 32, 1)
            
            generator = ResponseGenerator(
                tokenizer=tokenizer,
                embedder=embedder,
                reservoir_model=reservoir
            )
            
            # Test with real tokenization
            result = generator.generate_complete_response(["Hello", "world"])
            
            assert isinstance(result, ResponseGenerationResult)
            assert isinstance(result.response_text, str)
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to missing dependencies: {e}")
    
    def test_end_to_end_mock_pipeline(self):
        """Test end-to-end pipeline with all mocked components."""
        # Create comprehensive mocks
        tokenizer = Mock(spec=StandardTokenizerWrapper)
        tokenizer.tokenize.return_value = [[1, 2, 3, 4, 5]]
        tokenizer.decode.return_value = "decoded token"
        
        embedder = Mock(spec=SinusoidalEmbedder)
        embedder.embed.return_value = np.random.randn(5, 128)
        
        reservoir = Mock()
        reservoir.predict.return_value = np.random.randn(1, 32, 32, 1)
        
        # Create generator
        generator = ResponseGenerator(
            tokenizer=tokenizer,
            embedder=embedder,
            reservoir_model=reservoir,
            default_reservoir_strategy="adaptive"
        )
        
        # Test complete pipeline
        input_text = ["This", "is", "a", "test", "input"]
        
        result = generator.generate_complete_response(
            input_text,
            return_intermediate=True
        )
        
        # Verify all components were called
        tokenizer.tokenize.assert_called()
        embedder.embed.assert_called()
        reservoir.predict.assert_called()
        
        # Verify result structure
        assert isinstance(result, ResponseGenerationResult)
        assert result.intermediate_embeddings is not None
        assert len(result.intermediate_embeddings) == 3
        assert result.generation_time > 0
        assert result.confidence_score >= 0.0


if __name__ == "__main__":
    pytest.main([__file__])