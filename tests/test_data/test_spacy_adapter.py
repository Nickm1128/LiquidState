#!/usr/bin/env python3
"""
Tests for spaCy tokenizer adapter.

This module contains comprehensive tests for the SpacyAdapter class,
including tokenization, decoding, linguistic features, and error handling.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Import the adapter and related classes
from src.lsm.data.adapters.spacy_adapter import SpacyAdapter, register_spacy_adapter
from src.lsm.data.enhanced_tokenization import TokenizerConfig
from src.lsm.utils.lsm_exceptions import TokenizerError, TokenizerLoadError


class TestSpacyAdapter:
    """Test cases for SpacyAdapter."""
    
    @pytest.fixture
    def basic_config(self):
        """Basic tokenizer configuration for testing."""
        return TokenizerConfig(
            backend='spacy',
            model_name='en',
            max_length=128
        )
    
    @pytest.fixture
    def advanced_config(self):
        """Advanced tokenizer configuration with backend-specific options."""
        return TokenizerConfig(
            backend='spacy',
            model_name='en_core_web_sm',
            max_length=256,
            special_tokens={
                'pad_token': '[PAD]',
                'unk_token': '[UNK]',
                'bos_token': '[BOS]',
                'eos_token': '[EOS]'
            },
            backend_specific_config={
                'use_linguistic_features': True,
                'unicode_normalization': 'NFC',
                'custom_components': []
            }
        )
    
    @pytest.fixture
    def mock_spacy_nlp(self):
        """Mock spaCy nlp object for testing."""
        mock_nlp = Mock()
        mock_nlp.lang = 'en'
        mock_nlp.lang_ = 'en'
        mock_nlp.pipe_names = ['tokenizer', 'tagger', 'parser']
        
        # Mock vocabulary
        mock_vocab = Mock()
        mock_vocab.__iter__ = Mock(return_value=iter([
            ('hello', Mock(orth=1, norm=1)),
            ('world', Mock(orth=2, norm=2)),
            ('test', Mock(orth=3, norm=3)),
            ('spacy', Mock(orth=4, norm=4))
        ]))
        mock_nlp.vocab = mock_vocab
        
        # Mock tokenization
        def mock_call(text):
            mock_doc = Mock()
            mock_tokens = []
            
            # Simple tokenization for testing
            words = text.lower().split()
            for i, word in enumerate(words):
                mock_token = Mock()
                mock_token.text = word
                mock_token.lemma_ = word
                mock_token.pos_ = 'NOUN'
                mock_token.dep_ = 'ROOT' if i == 0 else 'dep'
                mock_token.head = mock_tokens[0] if mock_tokens else mock_token
                mock_tokens.append(mock_token)
            
            mock_doc.__iter__ = Mock(return_value=iter(mock_tokens))
            mock_doc.ents = []
            mock_doc.sents = [Mock(text=text)]
            
            return mock_doc
        
        mock_nlp.__call__ = mock_call
        return mock_nlp
    
    def test_spacy_not_available(self):
        """Test error when spaCy is not available."""
        with patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', False):
            config = TokenizerConfig(backend='spacy', model_name='en')
            
            with pytest.raises(TokenizerError, match="spaCy library not available"):
                SpacyAdapter(config)
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_initialization_blank_model(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test initialization with blank spaCy model."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        assert adapter._is_initialized
        assert adapter._language_code == 'en'
        assert adapter.get_vocab_size() > 0
        mock_spacy.blank.assert_called_once_with('en')
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_initialization_trained_model(self, mock_spacy, advanced_config, mock_spacy_nlp):
        """Test initialization with trained spaCy model."""
        mock_spacy.load.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(advanced_config)
        adapter.initialize()
        
        assert adapter._is_initialized
        assert adapter._language_code == 'en'
        mock_spacy.load.assert_called_once_with('en_core_web_sm')
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_initialization_model_not_found(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test fallback when model is not found."""
        mock_spacy.load.side_effect = OSError("Can't find model 'en_core_web_sm'")
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        config = TokenizerConfig(backend='spacy', model_name='en_core_web_sm')
        adapter = SpacyAdapter(config)
        adapter.initialize()
        
        assert adapter._is_initialized
        assert adapter._language_code == 'en'
        mock_spacy.blank.assert_called_once_with('en')
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_tokenization(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test text tokenization."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        # Test single text
        result = adapter.tokenize("hello world")
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert all(isinstance(token_id, int) for token_id in result[0])
        
        # Test multiple texts
        result = adapter.tokenize(["hello world", "test spacy"])
        assert len(result) == 2
        assert all(isinstance(seq, list) for seq in result)
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_tokenization_with_special_tokens(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test tokenization with special tokens."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        # Test with special tokens
        result = adapter.tokenize("hello world", add_special_tokens=True)
        assert len(result[0]) > 2  # Should include BOS/EOS tokens
        
        # Test without special tokens
        result_no_special = adapter.tokenize("hello world", add_special_tokens=False)
        assert len(result_no_special[0]) < len(result[0])
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_tokenization_with_padding(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test tokenization with padding."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        # Test with padding
        result = adapter.tokenize(["hello", "hello world test"], padding=True)
        assert len(result[0]) == len(result[1])  # Should be same length due to padding
        
        # Test without padding
        result_no_pad = adapter.tokenize(["hello", "hello world test"], padding=False)
        assert len(result_no_pad[0]) != len(result_no_pad[1])  # Different lengths
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_tokenization_with_truncation(self, mock_spacy, mock_spacy_nlp):
        """Test tokenization with truncation."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        config = TokenizerConfig(backend='spacy', model_name='en', max_length=5)
        adapter = SpacyAdapter(config)
        adapter.initialize()
        
        # Test with truncation
        long_text = "this is a very long text that should be truncated"
        result = adapter.tokenize(long_text, truncation=True, add_special_tokens=True)
        assert len(result[0]) <= 5  # Should be truncated to max_length
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_decoding(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test token ID decoding."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        # Test single sequence decoding
        token_ids = [1, 2, 3]
        result = adapter.decode(token_ids)
        assert isinstance(result, str)
        
        # Test batch decoding
        batch_ids = [[1, 2], [3, 4]]
        result = adapter.decode(batch_ids)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(text, str) for text in result)
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_vocabulary_methods(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test vocabulary-related methods."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        # Test vocab size
        vocab_size = adapter.get_vocab_size()
        assert isinstance(vocab_size, int)
        assert vocab_size > 0
        
        # Test vocab mapping
        vocab = adapter.get_vocab()
        assert isinstance(vocab, dict)
        assert len(vocab) == vocab_size
        
        # Test special tokens
        special_tokens = adapter.get_special_tokens()
        assert isinstance(special_tokens, dict)
        assert 'pad_token_id' in special_tokens
        assert 'unk_token_id' in special_tokens
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_linguistic_features(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test linguistic feature extraction."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        features = adapter.get_linguistic_features("hello world")
        assert isinstance(features, dict)
        assert 'tokens' in features
        assert 'pos_tags' in features
        assert 'lemmas' in features
        assert 'entities' in features
        assert 'sentences' in features
        assert 'dependencies' in features
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_unicode_normalization(self, mock_spacy, mock_spacy_nlp):
        """Test Unicode normalization."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        config = TokenizerConfig(
            backend='spacy',
            model_name='en',
            backend_specific_config={'unicode_normalization': 'NFC'}
        )
        adapter = SpacyAdapter(config)
        adapter.initialize()
        
        # Test with Unicode text
        unicode_text = "café naïve résumé"
        result = adapter.tokenize(unicode_text)
        assert isinstance(result, list)
        assert len(result) == 1
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_tokenizer_info(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test tokenizer information retrieval."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        info = adapter.get_tokenizer_info()
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'language_code' in info
        assert 'vocab_size' in info
        assert 'backend' in info
        assert info['backend'] == 'spacy'
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_save_and_load_config(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test saving and loading adapter configuration."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test saving
            adapter.save_adapter_config(temp_dir)
            
            # Check files were created
            config_path = os.path.join(temp_dir, 'spacy_adapter_config.json')
            vocab_path = os.path.join(temp_dir, 'spacy_vocab.json')
            assert os.path.exists(config_path)
            assert os.path.exists(vocab_path)
            
            # Test loading
            loaded_adapter = SpacyAdapter.load_adapter_config(temp_dir)
            assert loaded_adapter._is_initialized
            assert loaded_adapter.config.model_name == basic_config.model_name
    
    def test_not_initialized_errors(self, basic_config):
        """Test errors when adapter is not initialized."""
        adapter = SpacyAdapter(basic_config)
        
        with pytest.raises(TokenizerError, match="not initialized"):
            adapter.tokenize("test")
        
        with pytest.raises(TokenizerError, match="not initialized"):
            adapter.decode([1, 2, 3])
        
        with pytest.raises(TokenizerError, match="not initialized"):
            adapter.get_vocab_size()
        
        with pytest.raises(TokenizerError, match="not initialized"):
            adapter.get_vocab()
        
        with pytest.raises(TokenizerError, match="not initialized"):
            adapter.get_special_tokens()
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    def test_list_supported_models(self):
        """Test listing supported models."""
        models = SpacyAdapter.list_supported_models()
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'en_core_web_sm' in models
        assert 'en' in models
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.LANGUAGES', {'en': 'English', 'de': 'German'})
    def test_list_supported_languages(self):
        """Test listing supported languages."""
        languages = SpacyAdapter.list_supported_languages()
        assert isinstance(languages, list)
        assert 'en' in languages
        assert 'de' in languages
    
    def test_load_config_file_not_found(self):
        """Test loading config when file doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(TokenizerLoadError):
                SpacyAdapter.load_adapter_config(temp_dir)
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    def test_register_adapter(self):
        """Test adapter registration."""
        # This should not raise an error
        register_spacy_adapter()
    
    @patch('src.lsm.data.adapters.spacy_adapter.SPACY_AVAILABLE', True)
    @patch('src.lsm.data.adapters.spacy_adapter.spacy')
    def test_repr(self, mock_spacy, basic_config, mock_spacy_nlp):
        """Test string representation."""
        mock_spacy.blank.return_value = mock_spacy_nlp
        mock_spacy.LANGUAGES = {'en': 'English'}
        
        adapter = SpacyAdapter(basic_config)
        adapter.initialize()
        
        repr_str = repr(adapter)
        assert 'SpacyAdapter' in repr_str
        assert 'en' in repr_str
        assert 'initialized=True' in repr_str


if __name__ == '__main__':
    pytest.main([__file__])