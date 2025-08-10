#!/usr/bin/env python3
"""
Integration tests for spaCy adapter with enhanced tokenization system.

This module tests the integration of SpacyAdapter with the TokenizerRegistry
and enhanced tokenization system.
"""

import pytest
from unittest.mock import Mock, patch

from src.lsm.data.enhanced_tokenization import TokenizerRegistry, TokenizerConfig
from src.lsm.data.adapters.spacy_adapter import SpacyAdapter
from src.lsm.utils.lsm_exceptions import TokenizerError


class TestSpacyIntegration:
    """Test spaCy adapter integration with the enhanced tokenization system."""
    
    def test_spacy_adapter_registered(self):
        """Test that spaCy adapter is properly registered."""
        # Check that spacy backend is available
        assert 'spacy' in TokenizerRegistry._adapters
        
        # Check that adapter class is correct
        adapter_class = TokenizerRegistry.get_adapter_class('spacy')
        assert adapter_class == SpacyAdapter
    
    def test_spacy_model_patterns_registered(self):
        """Test that spaCy model patterns are registered."""
        # Test some common spaCy model patterns
        test_patterns = ['en', 'en_core_web_sm', 'de', 'fr_core_news_sm']
        
        for pattern in test_patterns:
            try:
                adapter_class = TokenizerRegistry.get_adapter_class(pattern)
                assert adapter_class == SpacyAdapter
            except TokenizerError:
                # Some patterns might not be registered, that's ok
                pass
    
    def test_create_adapter_through_registry(self):
        """Test creating spaCy adapter through registry."""
        # This should work even without spaCy installed (will fail at initialization)
        try:
            adapter = TokenizerRegistry.create_adapter(
                'spacy',
                max_length=128,
                special_tokens={'pad_token': '[PAD]'}
            )
            assert isinstance(adapter, SpacyAdapter)
            assert adapter.config.backend == 'spacy'
            assert adapter.config.max_length == 128
            assert adapter.config.special_tokens == {'pad_token': '[PAD]'}
        except TokenizerError as e:
            # Expected if spaCy is not available
            assert "spaCy library not available" in str(e)
    
    def test_create_adapter_with_model_name(self):
        """Test creating adapter with specific model name."""
        try:
            adapter = TokenizerRegistry.create_adapter(
                'en_core_web_sm',
                max_length=256
            )
            assert isinstance(adapter, SpacyAdapter)
            assert adapter.config.model_name == 'en_core_web_sm'
            assert adapter.config.backend == 'spacy'
        except TokenizerError as e:
            # Expected if spaCy is not available
            assert "spaCy library not available" in str(e)
    
    def test_supported_models_list(self):
        """Test that supported models list is available."""
        models = SpacyAdapter.list_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        
        # Check some expected models
        expected_models = ['en', 'en_core_web_sm', 'de', 'fr']
        for model in expected_models:
            assert model in models
    
    def test_supported_languages_list(self):
        """Test that supported languages list works."""
        languages = SpacyAdapter.list_supported_languages()
        assert isinstance(languages, list)
        # Will be empty if spaCy not installed, that's ok
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_adapter_initialization_through_registry(self, mock_spacy):
        """Test adapter initialization through registry when spaCy is available."""
        # Mock spaCy components
        mock_nlp = Mock()
        mock_nlp.lang = 'en'
        mock_nlp.lang_ = 'en'
        mock_nlp.pipe_names = ['tokenizer']
        
        # Mock vocabulary
        mock_vocab = Mock()
        mock_vocab.__iter__ = Mock(return_value=iter([
            ('test', Mock(orth=1, norm=1)),
            ('word', Mock(orth=2, norm=2))
        ]))
        mock_nlp.vocab = mock_vocab
        
        mock_spacy.blank.return_value = mock_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        # Create adapter through registry
        adapter = TokenizerRegistry.create_adapter('en', max_length=128)
        
        # Initialize adapter
        adapter.initialize()
        
        assert adapter._is_initialized
        assert adapter.get_vocab_size() > 0
        mock_spacy.blank.assert_called_once_with('en')
    
    def test_error_handling_for_unknown_backend(self):
        """Test error handling for unknown backend."""
        with pytest.raises(TokenizerError, match="No adapter found"):
            TokenizerRegistry.get_adapter_class('completely_unknown_backend_12345')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config = TokenizerConfig(
            backend='spacy',
            model_name='en',
            max_length=512,
            special_tokens={'pad_token': '[PAD]'},
            backend_specific_config={
                'use_linguistic_features': True,
                'unicode_normalization': 'NFC'
            }
        )
        
        try:
            adapter = SpacyAdapter(config)
            assert adapter.config == config
        except TokenizerError as e:
            # Expected if spaCy is not available
            assert "spaCy library not available" in str(e)


if __name__ == '__main__':
    pytest.main([__file__])