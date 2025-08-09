#!/usr/bin/env python3
"""
Tests for Enhanced LSM Inference with new tokenization and embeddings.
"""

import os
import tempfile
import shutil
import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.lsm.inference.inference import EnhancedLSMInference
from src.lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from src.lsm.utils.lsm_exceptions import ModelLoadError, TokenizerError


class TestEnhancedLSMInference:
    """Test cases for EnhancedLSMInference class."""
    
    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary model directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_trainer(self):
        """Create mock trainer."""
        trainer = Mock()
        trainer.window_size = 5
        trainer.model = Mock()
        trainer.model.predict = Mock(return_value=np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))
        trainer.get_model_info = Mock(return_value={
            'architecture': {'reservoir_type': 'sparse', 'window_size': 5}
        })
        return trainer
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock legacy tokenizer."""
        tokenizer = Mock()
        tokenizer.is_fitted = True
        tokenizer.encode = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        tokenizer.decode_embedding = Mock(return_value="test response")
        return tokenizer
    
    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        config = Mock()
        config.to_dict = Mock(return_value={'test': 'config'})
        return config
    
    def test_initialization(self, temp_model_dir):
        """Test EnhancedLSMInference initialization."""
        inference = EnhancedLSMInference(
            model_path=temp_model_dir,
            use_response_level=True,
            tokenizer_name='gpt2',
            lazy_load=True
        )
        
        assert inference.model_path == temp_model_dir
        assert inference.use_response_level is True
        assert inference.tokenizer_name == 'gpt2'
        assert inference.lazy_load is True
        assert not inference._model_loaded
        assert not inference._enhanced_components_loaded
    
    @patch('src.lsm.inference.inference.StandardTokenizerWrapper')
    @patch('src.lsm.inference.inference.SinusoidalEmbedder')
    def test_load_enhanced_components_new(self, mock_embedder_class, mock_tokenizer_class, 
                                         temp_model_dir, mock_trainer):
        """Test loading enhanced components when they don't exist."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.tokenizer_name = 'gpt2'
        mock_tokenizer.get_vocab_size = Mock(return_value=50257)
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_embedder = Mock()
        mock_embedder.vocab_size = 50257
        mock_embedder.embedding_dim = 128
        mock_embedder._is_fitted = False
        mock_embedder_class.return_value = mock_embedder
        
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference._model_loaded = True
        
        # Load enhanced components
        inference._load_enhanced_components()
        
        assert inference._enhanced_components_loaded
        assert inference.standard_tokenizer is not None
        assert inference.sinusoidal_embedder is not None
        mock_tokenizer_class.assert_called_once_with(tokenizer_name='gpt2')
        mock_embedder_class.assert_called_once_with(vocab_size=50257, embedding_dim=128)
    
    @patch('src.lsm.inference.inference.StandardTokenizerWrapper')
    @patch('src.lsm.inference.inference.SinusoidalEmbedder')
    def test_load_enhanced_components_existing(self, mock_embedder_class, mock_tokenizer_class, 
                                              temp_model_dir):
        """Test loading existing enhanced components."""
        # Create directories to simulate existing components
        os.makedirs(os.path.join(temp_model_dir, "standard_tokenizer"))
        os.makedirs(os.path.join(temp_model_dir, "sinusoidal_embedder"))
        
        # Setup mocks for loading
        mock_tokenizer = Mock()
        mock_tokenizer_class.load = Mock(return_value=mock_tokenizer)
        
        mock_embedder = Mock()
        mock_embedder_class.load = Mock(return_value=mock_embedder)
        
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference._model_loaded = True
        
        # Load enhanced components
        inference._load_enhanced_components()
        
        assert inference._enhanced_components_loaded
        mock_tokenizer_class.load.assert_called_once()
        mock_embedder_class.load.assert_called_once()
    
    def test_generate_response_fallback(self, temp_model_dir, mock_trainer, mock_tokenizer):
        """Test response generation fallback to token-level."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference.legacy_tokenizer = mock_tokenizer
        inference._model_loaded = True
        inference._tokenizer_loaded = True
        inference.use_response_level = False  # Force fallback
        
        # Mock the legacy prediction method
        inference._legacy_predict_next_token = Mock(return_value="fallback response")
        
        response = inference.generate_response("test input")
        
        assert "fallback" in response.lower()
    
    @patch('src.lsm.inference.inference.ResponseGenerator')
    def test_generate_response_with_response_generator(self, mock_response_gen_class, 
                                                      temp_model_dir, mock_trainer):
        """Test response generation with ResponseGenerator."""
        # Setup mock response generator
        mock_result = Mock()
        mock_result.response_text = "generated response"
        mock_result.confidence_score = 0.8
        
        mock_response_gen = Mock()
        mock_response_gen.generate_complete_response = Mock(return_value=mock_result)
        mock_response_gen_class.return_value = mock_response_gen
        
        # Setup inference
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference._model_loaded = True
        
        # Mock enhanced components
        mock_tokenizer = Mock()
        mock_tokenizer.encode_single = Mock(return_value=[1, 2, 3])
        inference.standard_tokenizer = mock_tokenizer
        
        mock_embedder = Mock()
        mock_embedder.embed = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        inference.sinusoidal_embedder = mock_embedder
        
        inference.response_generator = mock_response_gen
        inference._enhanced_components_loaded = True
        inference.use_response_level = True
        
        response = inference.generate_response("test input")
        
        assert response == "generated response"
        mock_tokenizer.encode_single.assert_called_once_with("test input")
        mock_embedder.embed.assert_called_once_with([1, 2, 3])
        mock_response_gen.generate_complete_response.assert_called_once()
    
    def test_predict_with_enhanced_tokenizer(self, temp_model_dir, mock_trainer):
        """Test prediction with enhanced tokenizer."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference._model_loaded = True
        
        # Mock enhanced components
        mock_tokenizer = Mock()
        mock_tokenizer.encode_single = Mock(return_value=[1, 2, 3])
        mock_tokenizer.decode = Mock(return_value="decoded response")
        inference.standard_tokenizer = mock_tokenizer
        
        mock_embedder = Mock()
        mock_embedder.embed = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]))
        mock_embedder.get_embedding_matrix = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))
        inference.sinusoidal_embedder = mock_embedder
        
        inference._enhanced_components_loaded = True
        
        # Mock decode method
        inference._decode_reservoir_output = Mock(return_value="predicted text")
        inference._calculate_confidence = Mock(return_value=0.75)
        
        prediction, confidence = inference.predict_with_enhanced_tokenizer("test input")
        
        assert prediction == "predicted text"
        assert confidence == 0.75
        mock_tokenizer.encode_single.assert_called_once_with("test input")
        mock_embedder.embed.assert_called_once_with([1, 2, 3])
    
    def test_decode_reservoir_output(self, temp_model_dir):
        """Test decoding reservoir output to text."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        
        # Mock enhanced components
        mock_tokenizer = Mock()
        mock_tokenizer.decode = Mock(return_value="  decoded text  ")
        inference.standard_tokenizer = mock_tokenizer
        
        mock_embedder = Mock()
        embedding_matrix = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mock_embedder.get_embedding_matrix = Mock(return_value=embedding_matrix)
        inference.sinusoidal_embedder = mock_embedder
        
        reservoir_output = np.array([0.4, 0.5])  # Should match closest to [0.5, 0.6] (index 2)
        
        result = inference._decode_reservoir_output(reservoir_output)
        
        assert result == "decoded text"
        mock_tokenizer.decode.assert_called_once_with([2])
    
    def test_calculate_confidence(self, temp_model_dir):
        """Test confidence calculation."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        
        # Test with various reservoir outputs
        high_confidence_output = np.array([1.0, 1.0, 1.0])
        low_confidence_output = np.array([0.1, 0.9, 0.2])
        
        high_conf = inference._calculate_confidence(high_confidence_output)
        low_conf = inference._calculate_confidence(low_confidence_output)
        
        assert 0.0 <= high_conf <= 1.0
        assert 0.0 <= low_conf <= 1.0
        assert high_conf >= low_conf  # High consistency should give higher confidence
    
    def test_get_enhanced_model_info(self, temp_model_dir, mock_trainer, mock_config):
        """Test getting enhanced model information."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference.config = mock_config
        inference._model_loaded = True
        
        # Mock enhanced components
        mock_tokenizer = Mock()
        mock_tokenizer.tokenizer_name = 'gpt2'
        mock_tokenizer.get_vocab_size = Mock(return_value=50257)
        inference.standard_tokenizer = mock_tokenizer
        
        mock_embedder = Mock()
        mock_embedder.vocab_size = 50257
        mock_embedder.embedding_dim = 128
        mock_embedder._is_fitted = True
        inference.sinusoidal_embedder = mock_embedder
        
        inference.response_generator = Mock()
        inference.reservoir_manager = Mock()
        inference._enhanced_components_loaded = True
        
        info = inference.get_enhanced_model_info()
        
        assert 'enhanced_components' in info
        assert info['enhanced_components']['standard_tokenizer']['name'] == 'gpt2'
        assert info['enhanced_components']['standard_tokenizer']['vocab_size'] == 50257
        assert info['enhanced_components']['sinusoidal_embedder']['vocab_size'] == 50257
        assert info['enhanced_components']['sinusoidal_embedder']['is_fitted'] is True
        assert info['enhanced_components']['response_generator'] is True
        assert info['enhanced_components']['reservoir_manager'] is True
        assert 'performance' in info
        assert info['performance']['use_response_level'] is True
    
    def test_save_enhanced_components(self, temp_model_dir):
        """Test saving enhanced components."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        
        # Mock enhanced components
        mock_tokenizer = Mock()
        mock_tokenizer.save = Mock()
        inference.standard_tokenizer = mock_tokenizer
        
        mock_embedder = Mock()
        mock_embedder._is_fitted = True
        mock_embedder.save = Mock()
        inference.sinusoidal_embedder = mock_embedder
        
        inference._enhanced_components_loaded = True
        
        inference.save_enhanced_components()
        
        mock_tokenizer.save.assert_called_once()
        mock_embedder.save.assert_called_once()
    
    def test_legacy_predict_next_token(self, temp_model_dir, mock_trainer, mock_tokenizer):
        """Test legacy predict_next_token method."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference.legacy_tokenizer = mock_tokenizer
        inference._model_loaded = True
        inference._tokenizer_loaded = True
        
        # Mock encoding method
        inference._encode_with_cache = Mock(return_value=np.array([[0.1, 0.2], [0.3, 0.4]]))
        
        dialogue_sequence = ["hello", "how", "are", "you", "today"]
        
        result = inference.predict_next_token(dialogue_sequence)
        
        assert result == "test response"
        mock_trainer.predict.assert_called_once()
        mock_tokenizer.decode_embedding.assert_called_once()
    
    def test_legacy_predict_next_token_wrong_length(self, temp_model_dir, mock_trainer, mock_tokenizer):
        """Test legacy predict_next_token with wrong sequence length."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference.legacy_tokenizer = mock_tokenizer
        inference._model_loaded = True
        inference._tokenizer_loaded = True
        
        dialogue_sequence = ["hello", "world"]  # Wrong length (should be 5)
        
        with pytest.raises(Exception):  # Should raise InvalidInputError
            inference.predict_next_token(dialogue_sequence)
    
    def test_enhanced_fallback_when_no_legacy_tokenizer(self, temp_model_dir, mock_trainer):
        """Test fallback to enhanced tokenizer when legacy tokenizer not available."""
        inference = EnhancedLSMInference(temp_model_dir, lazy_load=True)
        inference.trainer = mock_trainer
        inference.legacy_tokenizer = None  # No legacy tokenizer
        inference._model_loaded = True
        inference._tokenizer_loaded = True
        inference._enhanced_components_loaded = True
        
        # Mock enhanced prediction
        inference.predict_with_enhanced_tokenizer = Mock(return_value=("enhanced response", 0.8))
        
        dialogue_sequence = ["hello", "how", "are", "you", "today"]
        
        result = inference.predict_next_token(dialogue_sequence)
        
        assert result == "enhanced response"
        inference.predict_with_enhanced_tokenizer.assert_called_once_with("hello how are you today")


if __name__ == "__main__":
    pytest.main([__file__])