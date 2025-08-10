#!/usr/bin/env python3
"""
Tests for streaming tokenizer fitting functionality.

This module tests the streaming tokenizer fitting implementation including
incremental vocabulary building, statistics collection, progress tracking,
and memory usage monitoring.
"""

import pytest
import tempfile
import json
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper, TokenizerConfig, TokenizerAdapter
from src.lsm.data.streaming_data_iterator import StreamingDataIterator
from src.lsm.utils.lsm_exceptions import TokenizerError, InvalidInputError


class MockTokenizerAdapter(TokenizerAdapter):
    """Mock tokenizer adapter for testing."""
    
    def __init__(self, config):
        self.config = config
        self._vocab = {'hello': 1, 'world': 2, 'test': 3, 'streaming': 4, '[PAD]': 0, '[EOS]': 5}
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}
        self._vocab_size = len(self._vocab)
        self._is_initialized = True
    
    def initialize(self):
        pass
    
    def tokenize(self, texts, add_special_tokens=True, padding=True, truncation=True):
        if isinstance(texts, str):
            texts = [texts]
        
        result = []
        for text in texts:
            tokens = []
            words = text.lower().split()
            for word in words:
                tokens.append(self._vocab.get(word, 3))  # 3 is 'test' as UNK
            
            if add_special_tokens:
                tokens.append(5)  # EOS token
            
            if padding and len(tokens) < 8:
                tokens.extend([0] * (8 - len(tokens)))  # PAD to length 8
            
            result.append(tokens)
        
        return result
    
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids[0], int):
            words = []
            for token_id in token_ids:
                if skip_special_tokens and token_id in [0, 5]:
                    continue
                words.append(self._reverse_vocab.get(token_id, '[UNK]'))
            return ' '.join(words)
        else:
            return [self.decode(seq, skip_special_tokens) for seq in token_ids]
    
    def get_vocab_size(self):
        return self._vocab_size
    
    def get_vocab(self):
        return self._vocab.copy()
    
    def get_special_tokens(self):
        return {'pad_token_id': 0, 'eos_token_id': 5}
    
    @classmethod
    def load_adapter_config(cls, load_path):
        config = TokenizerConfig(backend='mock', model_name='mock-model')
        adapter = cls(config)
        adapter.initialize()
        return adapter


class TestStreamingTokenizerFitting:
    """Test cases for streaming tokenizer fitting."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock adapter
        config = TokenizerConfig(backend='mock', model_name='mock-model')
        self.mock_adapter = MockTokenizerAdapter(config)
        
        # Create enhanced tokenizer wrapper
        self.tokenizer = EnhancedTokenizerWrapper(
            tokenizer=self.mock_adapter,
            embedding_dim=64,
            max_length=8
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data_file(self, filename: str, texts: list) -> Path:
        """Create a test data file with text content."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(f"{text}\n")
        return file_path
    
    def create_test_json_file(self, filename: str, data: list) -> Path:
        """Create a test JSON file."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return file_path
    
    def test_fit_streaming_basic(self):
        """Test basic streaming tokenizer fitting."""
        # Create test data
        test_texts = [
            "hello world test",
            "streaming test data",
            "hello streaming world",
            "test hello world streaming"
        ]
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Track progress calls
        progress_calls = []
        def progress_callback(progress_info):
            progress_calls.append(progress_info)
        
        # Fit streaming tokenizer
        embedder = self.tokenizer.fit_streaming(
            data_source=str(test_file),
            batch_size=2,
            epochs=2,
            progress_callback=progress_callback
        )
        
        # Verify embedder was created and fitted
        assert embedder is not None
        assert self.tokenizer._is_fitted
        assert self.tokenizer._sinusoidal_embedder is not None
        
        # Verify progress tracking
        assert len(progress_calls) > 0
        
        # Verify training statistics
        stats = self.tokenizer.get_training_stats()
        assert stats is not None
        # Total sequences = sequences per epoch * epochs (each epoch processes all data)
        assert stats['total_sequences'] == len(test_texts) * 2  # 2 epochs
        assert stats['epochs'] == 2
        assert 'total_tokens' in stats
        assert 'avg_sequence_length' in stats
        assert 'vocab_coverage' in stats
    
    def test_fit_streaming_with_streaming_iterator(self):
        """Test streaming fitting with pre-created StreamingDataIterator."""
        # Create test data
        test_texts = [
            "hello world",
            "test streaming",
            "world hello test"
        ]
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Create streaming iterator
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        )
        
        # Fit with iterator
        embedder = self.tokenizer.fit_streaming(
            data_source=iterator,
            epochs=1
        )
        
        assert embedder is not None
        assert self.tokenizer._is_fitted
    
    def test_fit_streaming_multiple_files(self):
        """Test streaming fitting with multiple files."""
        # Create multiple test files
        file1_texts = ["hello world", "test data"]
        file2_texts = ["streaming test", "world hello"]
        
        file1 = self.create_test_data_file("file1.txt", file1_texts)
        file2 = self.create_test_data_file("file2.txt", file2_texts)
        
        # Fit with multiple files
        embedder = self.tokenizer.fit_streaming(
            data_source=[str(file1), str(file2)],
            batch_size=2,
            epochs=1
        )
        
        assert embedder is not None
        
        # Verify all texts were processed (1 epoch)
        stats = self.tokenizer.get_training_stats()
        assert stats['total_sequences'] == len(file1_texts) + len(file2_texts)
    
    def test_fit_streaming_json_data(self):
        """Test streaming fitting with JSON data."""
        # Create JSON test data
        json_data = [
            {"text": "hello world test"},
            {"text": "streaming data processing"},
            {"text": "test hello streaming"}
        ]
        json_file = self.create_test_json_file("test.json", json_data)
        
        # Fit with JSON data
        embedder = self.tokenizer.fit_streaming(
            data_source=str(json_file),
            batch_size=2,
            epochs=1
        )
        
        assert embedder is not None
        
        stats = self.tokenizer.get_training_stats()
        assert stats['total_sequences'] == len(json_data)
    
    @patch('psutil.Process')
    def test_fit_streaming_memory_monitoring(self, mock_process):
        """Test memory monitoring during streaming fitting."""
        # Mock memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 500 * 1024 * 1024  # 500 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Create test data
        test_texts = ["hello world test"] * 10
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Fit with memory monitoring
        embedder = self.tokenizer.fit_streaming(
            data_source=str(test_file),
            batch_size=5,
            epochs=1,
            memory_threshold_mb=400.0,  # Lower than mocked usage
            auto_adjust_batch_size=True
        )
        
        assert embedder is not None
        
        # Verify memory monitoring was called
        mock_process.assert_called()
    
    def test_fit_streaming_parameter_validation(self):
        """Test parameter validation for streaming fitting."""
        test_file = self.create_test_data_file("test.txt", ["hello world"])
        
        # Test invalid batch_size
        with pytest.raises(InvalidInputError):
            self.tokenizer.fit_streaming(
                data_source=str(test_file),
                batch_size=0
            )
        
        # Test invalid epochs
        with pytest.raises(InvalidInputError):
            self.tokenizer.fit_streaming(
                data_source=str(test_file),
                epochs=0
            )
        
        # Test invalid memory_threshold_mb
        with pytest.raises(InvalidInputError):
            self.tokenizer.fit_streaming(
                data_source=str(test_file),
                memory_threshold_mb=-1.0
            )
        
        # Test invalid min_batch_size
        with pytest.raises(InvalidInputError):
            self.tokenizer.fit_streaming(
                data_source=str(test_file),
                min_batch_size=0
            )
    
    def test_fit_streaming_batch_size_adjustment(self):
        """Test automatic batch size adjustment."""
        # Create test data
        test_texts = ["hello world test"] * 20
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Fit with auto batch size adjustment
        embedder = self.tokenizer.fit_streaming(
            data_source=str(test_file),
            batch_size=10,
            epochs=1,
            auto_adjust_batch_size=True,
            min_batch_size=5,
            max_batch_size=15
        )
        
        assert embedder is not None
        
        stats = self.tokenizer.get_training_stats()
        assert 'final_batch_size' in stats
    
    def test_fit_streaming_statistics_collection(self):
        """Test comprehensive statistics collection."""
        # Create test data with varied content
        test_texts = [
            "hello world",
            "test streaming data processing",
            "hello test",
            "world streaming hello test data"
        ]
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Fit streaming tokenizer
        embedder = self.tokenizer.fit_streaming(
            data_source=str(test_file),
            batch_size=2,
            epochs=1
        )
        
        # Verify comprehensive statistics
        stats = self.tokenizer.get_training_stats()
        
        required_stats = [
            'total_sequences', 'total_tokens', 'avg_sequence_length',
            'vocab_stats', 'vocab_coverage', 'training_time',
            'batches_processed', 'final_batch_size', 'epochs',
            'memory_threshold_mb'
        ]
        
        for stat in required_stats:
            assert stat in stats, f"Missing statistic: {stat}"
        
        # Verify statistics make sense
        assert stats['total_sequences'] == len(test_texts)
        assert stats['total_tokens'] > 0
        assert stats['avg_sequence_length'] > 0
        assert stats['vocab_coverage'] > 0
        assert stats['training_time'] > 0
        assert stats['batches_processed'] > 0
    
    def test_fit_streaming_progress_callback(self):
        """Test detailed progress callback functionality."""
        # Create test data
        test_texts = ["hello world test"] * 8
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Track detailed progress
        progress_history = []
        def detailed_progress_callback(progress_info):
            progress_history.append(progress_info.copy())
        
        # Fit with detailed progress tracking
        embedder = self.tokenizer.fit_streaming(
            data_source=str(test_file),
            batch_size=3,
            epochs=2,
            progress_callback=detailed_progress_callback
        )
        
        assert embedder is not None
        assert len(progress_history) > 0
        
        # Verify progress info structure
        for progress in progress_history:
            required_fields = [
                'epoch', 'total_epochs', 'batch', 'sequences_processed',
                'tokens_processed', 'current_batch_size', 'memory_usage_mb',
                'batch_time_seconds', 'avg_sequence_length'
            ]
            
            for field in required_fields:
                assert field in progress, f"Missing progress field: {field}"
    
    def test_fit_streaming_error_handling(self):
        """Test error handling during streaming fitting."""
        # Test with non-existent file
        with pytest.raises(Exception):  # Should raise DataLoadError or similar
            self.tokenizer.fit_streaming(
                data_source="/nonexistent/file.txt",
                batch_size=2,
                epochs=1
            )
    
    def test_get_training_stats_before_fitting(self):
        """Test getting training stats before fitting."""
        stats = self.tokenizer.get_training_stats()
        assert stats is None
    
    def test_fit_streaming_empty_data(self):
        """Test streaming fitting with empty data."""
        # Create empty file
        empty_file = self.create_test_data_file("empty.txt", [])
        
        # Should handle empty data gracefully
        embedder = self.tokenizer.fit_streaming(
            data_source=str(empty_file),
            batch_size=2,
            epochs=1
        )
        
        # Should still create embedder but with zero sequences processed
        assert embedder is not None
        stats = self.tokenizer.get_training_stats()
        assert stats['total_sequences'] == 0


if __name__ == "__main__":
    pytest.main([__file__])