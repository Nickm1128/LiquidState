#!/usr/bin/env python3
"""
Tests for ResponseInferenceModel.

This module tests the secondary model for complete response prediction
from token embedding sequences, including training and evaluation capabilities.
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from src.lsm.inference.response_inference_model import (
    ResponseInferenceModel,
    ResponsePredictionResult,
    TrainingConfig,
    ModelArchitecture,
    ResponseInferenceError,
    create_response_inference_model,
    create_transformer_response_model,
    create_lstm_response_model
)
from src.lsm.data.tokenization import StandardTokenizerWrapper


class TestResponseInferenceModel:
    """Test cases for ResponseInferenceModel."""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "input_embedding_dim": 256,
            "max_sequence_length": 64,
            "vocab_size": 1000,
            "architecture": "transformer"
        }
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer for testing."""
        tokenizer = Mock(spec=StandardTokenizerWrapper)
        tokenizer.tokenize.return_value = [[1, 2, 3, 4, 5]]
        tokenizer.decode.return_value = "This is a test response"
        return tokenizer
    
    @pytest.fixture
    def sample_embeddings(self):
        """Sample embedding sequences for testing."""
        return [
            np.random.randn(32, 256),  # Sequence 1
            np.random.randn(48, 256),  # Sequence 2
            np.random.randn(16, 256),  # Sequence 3
        ]
    
    @pytest.fixture
    def sample_responses(self):
        """Sample response strings for testing."""
        return [
            "This is the first response.",
            "Here is another response.",
            "A third response example."
        ]
    
    def test_initialization(self, sample_config):
        """Test ResponseInferenceModel initialization."""
        model = ResponseInferenceModel(**sample_config)
        
        assert model.input_embedding_dim == 256
        assert model.max_sequence_length == 64
        assert model.vocab_size == 1000
        assert model.architecture == ModelArchitecture.TRANSFORMER
        assert model._model is None
        assert not model._compiled
        assert not model._is_trained
    
    def test_initialization_with_tokenizer(self, sample_config, mock_tokenizer):
        """Test initialization with tokenizer."""
        model = ResponseInferenceModel(tokenizer=mock_tokenizer, **sample_config)
        
        assert model.tokenizer == mock_tokenizer
    
    def test_initialization_invalid_architecture(self, sample_config):
        """Test initialization with invalid architecture."""
        sample_config["architecture"] = "invalid_arch"
        
        with pytest.raises(ResponseInferenceError) as exc_info:
            ResponseInferenceModel(**sample_config)
        
        assert "initialization" in str(exc_info.value)
    
    def test_create_transformer_model(self, sample_config):
        """Test transformer model creation."""
        model = ResponseInferenceModel(**sample_config)
        
        keras_model = model.create_model()
        
        assert keras_model is not None
        assert model._model is not None
        assert model._compiled
        assert keras_model.name == "transformer_response_model"
        
        # Check input/output shapes
        assert keras_model.input_shape == (None, 64, 256)
        assert keras_model.output_shape == (None, 1000)
    
    def test_create_lstm_model(self, sample_config):
        """Test LSTM model creation."""
        sample_config["architecture"] = "lstm"
        model = ResponseInferenceModel(**sample_config)
        
        keras_model = model.create_model()
        
        assert keras_model is not None
        assert keras_model.name == "lstm_response_model"
        assert keras_model.input_shape == (None, 64, 256)
        assert keras_model.output_shape == (None, 1000)
    
    def test_create_gru_model(self, sample_config):
        """Test GRU model creation."""
        sample_config["architecture"] = "gru"
        model = ResponseInferenceModel(**sample_config)
        
        keras_model = model.create_model()
        
        assert keras_model is not None
        assert keras_model.name == "gru_response_model"
    
    def test_create_conv1d_model(self, sample_config):
        """Test 1D CNN model creation."""
        sample_config["architecture"] = "conv1d"
        model = ResponseInferenceModel(**sample_config)
        
        keras_model = model.create_model()
        
        assert keras_model is not None
        assert keras_model.name == "conv1d_response_model"
    
    def test_create_hybrid_model(self, sample_config):
        """Test hybrid model creation."""
        sample_config["architecture"] = "hybrid"
        model = ResponseInferenceModel(**sample_config)
        
        keras_model = model.create_model()
        
        assert keras_model is not None
        assert keras_model.name == "hybrid_response_model"
    
    def test_predict_response_basic(self, sample_config):
        """Test basic response prediction."""
        model = ResponseInferenceModel(**sample_config)
        
        # Create sample input
        input_sequence = np.random.randn(32, 256)
        
        result = model.predict_response(input_sequence)
        
        assert isinstance(result, ResponsePredictionResult)
        assert isinstance(result.predicted_response, str)
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.prediction_time > 0
        assert result.attention_weights is None  # Default
        assert result.intermediate_states is None  # Default
    
    def test_predict_response_with_batch(self, sample_config):
        """Test response prediction with batch input."""
        model = ResponseInferenceModel(**sample_config)
        
        # Create batch input
        input_sequence = np.random.randn(4, 32, 256)
        
        result = model.predict_response(input_sequence)
        
        assert isinstance(result, ResponsePredictionResult)
        assert isinstance(result.predicted_response, str)
    
    def test_predict_response_with_tokenizer(self, sample_config, mock_tokenizer):
        """Test response prediction with tokenizer."""
        model = ResponseInferenceModel(tokenizer=mock_tokenizer, **sample_config)
        
        input_sequence = np.random.randn(32, 256)
        
        result = model.predict_response(input_sequence)
        
        assert result.predicted_response == "This is a test response"
        mock_tokenizer.decode.assert_called_once()
    
    def test_predict_response_with_intermediates(self, sample_config):
        """Test response prediction with intermediate outputs."""
        model = ResponseInferenceModel(**sample_config)
        
        input_sequence = np.random.randn(32, 256)
        
        result = model.predict_response(
            input_sequence,
            return_attention=True,
            return_intermediate=True
        )
        
        assert isinstance(result, ResponsePredictionResult)
        # Note: Current implementation returns None for these, but structure is there
        # In a full implementation, these would contain actual values
    
    def test_prepare_training_data(self, sample_config, sample_embeddings, sample_responses):
        """Test training data preparation."""
        model = ResponseInferenceModel(**sample_config)
        
        X, y = model._prepare_training_data(sample_embeddings, sample_responses)
        
        assert X.shape == (3, 64, 256)  # 3 samples, max_seq_len, embed_dim
        assert y.shape == (3,)  # 3 target tokens
        assert y.dtype in [np.int32, np.int64]
    
    def test_prepare_training_data_with_tokenizer(self, sample_config, sample_embeddings, 
                                                 sample_responses, mock_tokenizer):
        """Test training data preparation with tokenizer."""
        model = ResponseInferenceModel(tokenizer=mock_tokenizer, **sample_config)
        
        X, y = model._prepare_training_data(sample_embeddings, sample_responses)
        
        assert X.shape == (3, 64, 256)
        assert y.shape == (3,)
        
        # Tokenizer should be called for each response
        assert mock_tokenizer.tokenize.call_count == 3
    
    def test_train_on_response_pairs_basic(self, sample_config, sample_embeddings, sample_responses):
        """Test basic training on response pairs."""
        model = ResponseInferenceModel(**sample_config)
        
        # Use minimal training config for speed
        training_config = TrainingConfig(
            batch_size=2,
            epochs=1,
            early_stopping_patience=0
        )
        
        result = model.train_on_response_pairs(
            sample_embeddings,
            sample_responses,
            training_config
        )
        
        assert "history" in result
        assert "metrics" in result
        assert "config" in result
        assert model._is_trained
        assert model._training_history is not None
    
    def test_train_with_validation_data(self, sample_config, sample_embeddings, sample_responses):
        """Test training with validation data."""
        model = ResponseInferenceModel(**sample_config)
        
        # Split data for validation
        train_embeddings = sample_embeddings[:2]
        train_responses = sample_responses[:2]
        val_embeddings = sample_embeddings[2:]
        val_responses = sample_responses[2:]
        
        training_config = TrainingConfig(
            batch_size=1,
            epochs=1,
            early_stopping_patience=0
        )
        
        result = model.train_on_response_pairs(
            train_embeddings,
            train_responses,
            training_config,
            validation_data=(val_embeddings, val_responses)
        )
        
        assert "history" in result
        assert model._is_trained
    
    def test_evaluate_on_test_data(self, sample_config, sample_embeddings, sample_responses):
        """Test model evaluation on test data."""
        model = ResponseInferenceModel(**sample_config)
        
        # Train first
        training_config = TrainingConfig(batch_size=2, epochs=1, early_stopping_patience=0)
        model.train_on_response_pairs(sample_embeddings, sample_responses, training_config)
        
        # Evaluate
        metrics = model.evaluate_on_test_data(sample_embeddings, sample_responses)
        
        assert isinstance(metrics, dict)
        assert "loss" in metrics or len(metrics) > 0
        assert "average_confidence" in metrics
        assert "average_similarity" in metrics
    
    def test_save_and_load_model(self, sample_config):
        """Test model saving and loading."""
        model = ResponseInferenceModel(**sample_config)
        
        # Create and train model
        model.create_model()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.keras")
            
            # Save model
            model.save_model(model_path)
            
            assert os.path.exists(model_path)
            assert os.path.exists(model_path.replace('.keras', '_metadata.json'))
            
            # Create new model and load
            new_model = ResponseInferenceModel(**sample_config)
            new_model.load_model(model_path)
            
            assert new_model._model is not None
            assert new_model._compiled
    
    def test_get_model_summary(self, sample_config):
        """Test model summary generation."""
        model = ResponseInferenceModel(**sample_config)
        
        summary = model.get_model_summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "transformer_response_model" in summary
    
    def test_get_prediction_statistics(self, sample_config):
        """Test prediction statistics."""
        model = ResponseInferenceModel(**sample_config)
        
        # Initially empty stats
        stats = model.get_prediction_statistics()
        assert stats["total_predictions"] == 0
        assert stats["success_rate"] == 0.0
        
        # Make a prediction
        input_sequence = np.random.randn(32, 256)
        model.predict_response(input_sequence)
        
        # Check updated stats
        stats = model.get_prediction_statistics()
        assert stats["total_predictions"] == 1
        assert stats["successful_predictions"] == 1
        assert stats["success_rate"] == 1.0
    
    def test_reset_statistics(self, sample_config):
        """Test statistics reset."""
        model = ResponseInferenceModel(**sample_config)
        
        # Make a prediction to generate stats
        input_sequence = np.random.randn(32, 256)
        model.predict_response(input_sequence)
        
        # Reset stats
        model.reset_statistics()
        
        stats = model.get_prediction_statistics()
        assert stats["total_predictions"] == 0
        assert stats["successful_predictions"] == 0
        assert stats["success_rate"] == 0.0
    
    def test_input_padding_and_truncation(self, sample_config):
        """Test input padding and truncation."""
        model = ResponseInferenceModel(**sample_config)
        
        # Test truncation (longer than max_sequence_length)
        long_input = np.random.randn(100, 256)
        processed = model._prepare_input(long_input)
        assert processed.shape == (1, 64, 256)
        
        # Test padding (shorter than max_sequence_length)
        short_input = np.random.randn(20, 256)
        processed = model._prepare_input(short_input)
        assert processed.shape == (1, 64, 256)
        
        # Test exact length
        exact_input = np.random.randn(64, 256)
        processed = model._prepare_input(exact_input)
        assert processed.shape == (1, 64, 256)
    
    def test_error_handling_invalid_input(self, sample_config):
        """Test error handling with invalid input."""
        model = ResponseInferenceModel(**sample_config)
        
        # Test with None input
        with pytest.raises(ResponseInferenceError):
            model.predict_response(None)
        
        # Test with wrong dimensions
        with pytest.raises(ResponseInferenceError):
            model.predict_response(np.random.randn(256))  # 1D instead of 2D/3D


class TestConvenienceFunctions:
    """Test convenience functions for creating ResponseInferenceModel instances."""
    
    def test_create_response_inference_model(self):
        """Test basic model creation function."""
        model = create_response_inference_model(
            input_embedding_dim=256,
            max_sequence_length=64,
            vocab_size=1000,
            architecture="transformer"
        )
        
        assert isinstance(model, ResponseInferenceModel)
        assert model.input_embedding_dim == 256
        assert model.max_sequence_length == 64
        assert model.vocab_size == 1000
        assert model.architecture == ModelArchitecture.TRANSFORMER
    
    def test_create_transformer_response_model(self):
        """Test transformer model creation function."""
        model = create_transformer_response_model(
            input_embedding_dim=512,
            max_sequence_length=128,
            vocab_size=5000,
            num_heads=12,
            num_layers=8
        )
        
        assert isinstance(model, ResponseInferenceModel)
        assert model.input_embedding_dim == 512
        assert model.architecture == ModelArchitecture.TRANSFORMER
        assert model.model_config["num_heads"] == 12
        assert model.model_config["num_layers"] == 8
    
    def test_create_lstm_response_model(self):
        """Test LSTM model creation function."""
        model = create_lstm_response_model(
            input_embedding_dim=256,
            max_sequence_length=64,
            vocab_size=1000,
            lstm_units=512,
            num_layers=3
        )
        
        assert isinstance(model, ResponseInferenceModel)
        assert model.architecture == ModelArchitecture.LSTM
        assert model.model_config["lstm_units"] == 512
        assert model.model_config["num_layers"] == 3


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""
    
    def test_default_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        
        assert config.batch_size == 32
        assert config.epochs == 100
        assert config.learning_rate == 0.001
        assert config.validation_split == 0.2
        assert config.early_stopping_patience == 10
        assert config.loss_type == "response_level_cosine"
        assert config.optimizer == "adam"
    
    def test_custom_config(self):
        """Test custom training configuration."""
        config = TrainingConfig(
            batch_size=16,
            epochs=50,
            learning_rate=0.0001,
            loss_type="cosine_similarity"
        )
        
        assert config.batch_size == 16
        assert config.epochs == 50
        assert config.learning_rate == 0.0001
        assert config.loss_type == "cosine_similarity"


class TestModelArchitecture:
    """Test ModelArchitecture enum."""
    
    def test_architecture_values(self):
        """Test architecture enumeration values."""
        assert ModelArchitecture.TRANSFORMER.value == "transformer"
        assert ModelArchitecture.LSTM.value == "lstm"
        assert ModelArchitecture.GRU.value == "gru"
        assert ModelArchitecture.CONV1D.value == "conv1d"
        assert ModelArchitecture.HYBRID.value == "hybrid"
    
    def test_architecture_from_string(self):
        """Test creating architecture from string."""
        arch = ModelArchitecture("transformer")
        assert arch == ModelArchitecture.TRANSFORMER
        
        arch = ModelArchitecture("lstm")
        assert arch == ModelArchitecture.LSTM


if __name__ == "__main__":
    pytest.main([__file__])