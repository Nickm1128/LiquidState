#!/usr/bin/env python3
"""
Tests for streaming consistency functionality.

This module tests the deterministic processing, checkpointing, and validation
features for streaming tokenizer fitting to ensure consistent results across
streaming batches.
"""

import pytest
import tempfile
import json
import os
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.lsm.data.enhanced_tokenization import EnhancedTokenizerWrapper, TokenizerConfig, TokenizerAdapter
from src.lsm.data.streaming_data_iterator import StreamingDataIterator
from src.lsm.utils.lsm_exceptions import TokenizerError, InvalidInputError


class MockTokenizerAdapter(TokenizerAdapter):
    """Mock tokenizer adapter for testing."""
    
    def __init__(self, config):
        super().__init__(config)
        self._vocab = {'hello': 1, 'world': 2, 'test': 3, 'streaming': 4, 'consistency': 5, '[PAD]': 0, '[EOS]': 6}
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
                tokens.append(6)  # EOS token
            
            if padding and len(tokens) < 8:
                tokens.extend([0] * (8 - len(tokens)))  # PAD to length 8
            
            result.append(tokens)
        
        return result
    
    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids[0], int):
            words = []
            for token_id in token_ids:
                if skip_special_tokens and token_id in [0, 6]:
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
        return {'pad_token_id': 0, 'eos_token_id': 6}
    
    @classmethod
    def load_adapter_config(cls, load_path):
        config = TokenizerConfig(backend='mock', model_name='mock-model')
        adapter = cls(config)
        adapter.initialize()
        return adapter


class TestStreamingConsistency:
    """Test cases for streaming consistency functionality."""
    
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
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_data_file(self, filename: str, texts: list) -> Path:
        """Create a test data file with text content."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(f"{text}\n")
        return file_path
    
    def test_deterministic_seed_creation(self):
        """Test deterministic seed creation for consistent processing."""
        # Test with same parameters
        seed1 = self.tokenizer._create_deterministic_seed("test.txt", 100, 10)
        seed2 = self.tokenizer._create_deterministic_seed("test.txt", 100, 10)
        assert seed1 == seed2, "Same parameters should produce same seed"
        
        # Test with different parameters
        seed3 = self.tokenizer._create_deterministic_seed("test.txt", 200, 10)
        assert seed1 != seed3, "Different parameters should produce different seeds"
        
        # Test with list data source
        seed4 = self.tokenizer._create_deterministic_seed(["file1.txt", "file2.txt"], 100, 10)
        seed5 = self.tokenizer._create_deterministic_seed(["file2.txt", "file1.txt"], 100, 10)
        assert seed4 == seed5, "Order of files in list should not affect seed"
    
    def test_checkpoint_save_and_load(self):
        """Test checkpointing functionality for resumable streaming."""
        checkpoint_path = self.temp_path / "test_checkpoint.pkl"
        
        # Test data for checkpoint
        test_data = {
            'epoch': 2,
            'batch_idx': 5,
            'total_sequences': 100,
            'total_tokens': 500,
            'vocab_stats': {1: 50, 2: 30, 3: 20},
            'sequence_lengths': [10, 12, 8, 15]
        }
        
        # Save checkpoint
        self.tokenizer._save_checkpoint(
            str(checkpoint_path),
            test_data['epoch'],
            test_data['batch_idx'],
            test_data['total_sequences'],
            test_data['total_tokens'],
            test_data['vocab_stats'],
            test_data['sequence_lengths']
        )
        
        # Verify checkpoint file exists
        assert checkpoint_path.exists(), "Checkpoint file should be created"
        
        # Load checkpoint
        loaded_data = self.tokenizer._load_checkpoint(str(checkpoint_path))
        
        # Verify loaded data
        assert loaded_data is not None, "Checkpoint should load successfully"
        assert loaded_data['epoch'] == test_data['epoch']
        assert loaded_data['batch_idx'] == test_data['batch_idx']
        assert loaded_data['total_sequences'] == test_data['total_sequences']
        assert loaded_data['total_tokens'] == test_data['total_tokens']
        assert loaded_data['vocab_stats'] == test_data['vocab_stats']
        assert loaded_data['sequence_lengths'] == test_data['sequence_lengths']
    
    def test_checkpoint_compatibility_validation(self):
        """Test checkpoint compatibility validation."""
        checkpoint_path = self.temp_path / "incompatible_checkpoint.pkl"
        
        # Create checkpoint with incompatible configuration
        self.tokenizer._save_checkpoint(
            str(checkpoint_path), 1, 0, 10, 50, {}, []
        )
        
        # Modify the checkpoint to have incompatible config
        import pickle
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        # Change embedding dimension to make it incompatible
        checkpoint_data['tokenizer_config']['embedding_dim'] = 128  # Different from current 64
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Try to load incompatible checkpoint
        loaded_data = self.tokenizer._load_checkpoint(str(checkpoint_path))
        assert loaded_data is None, "Incompatible checkpoint should not load"
    
    def test_streaming_consistency_validation(self):
        """Test validation that streaming results match batch processing."""
        # Create test validation data
        validation_data = [
            "hello world test",
            "streaming consistency validation",
            "test hello streaming world"
        ]
        
        # Create a simple embedder for testing
        embedder = self.tokenizer.create_sinusoidal_embedder()
        
        # Tokenize validation data for training
        validation_tokens = self.tokenizer.tokenize(validation_data, padding=True, truncation=True)
        validation_tokens = np.array(validation_tokens)
        
        # Train the embedder
        embedder.fit(validation_tokens, epochs=5)
        
        # Run validation
        validation_metrics = self.tokenizer._validate_streaming_consistency(
            embedder, validation_data
        )
        
        # Verify validation metrics structure
        required_metrics = [
            'embedding_mse', 'embedding_mae', 'avg_cosine_similarity',
            'matrix_mse', 'matrix_correlation', 'validation_samples'
        ]
        
        for metric in required_metrics:
            assert metric in validation_metrics, f"Missing validation metric: {metric}"
        
        # Verify metric values are reasonable
        assert validation_metrics['validation_samples'] == len(validation_data)
        assert 0 <= validation_metrics['avg_cosine_similarity'] <= 1
        assert -1 <= validation_metrics['matrix_correlation'] <= 1
    
    def test_fit_streaming_with_consistency_basic(self):
        """Test basic streaming fitting with consistency features."""
        # Create test data
        test_texts = [
            "hello world test",
            "streaming consistency test",
            "hello streaming world",
            "test consistency validation"
        ]
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Track progress calls
        progress_calls = []
        def progress_callback(progress_info):
            progress_calls.append(progress_info)
        
        # Fit with consistency features
        embedder = self.tokenizer.fit_streaming_with_consistency(
            data_source=str(test_file),
            batch_size=2,
            epochs=2,
            progress_callback=progress_callback,
            enable_checkpointing=True,
            validate_consistency=True,
            validation_sample_size=4,
            deterministic_seed=42
        )
        
        # Verify embedder was created and fitted
        assert embedder is not None
        assert self.tokenizer._is_fitted
        assert self.tokenizer._sinusoidal_embedder is not None
        
        # Verify progress tracking includes consistency info
        assert len(progress_calls) > 0
        for progress in progress_calls:
            assert 'deterministic_seed' in progress
            assert 'checkpoint_enabled' in progress
            assert progress['deterministic_seed'] == 42
            assert progress['checkpoint_enabled'] is True
        
        # Verify training statistics include consistency metrics
        stats = self.tokenizer.get_training_stats()
        assert stats is not None
        assert 'deterministic_seed' in stats
        assert 'checkpointing_enabled' in stats
        assert 'validation_metrics' in stats
        assert stats['deterministic_seed'] == 42
        assert stats['checkpointing_enabled'] is True
    
    def test_fit_streaming_with_checkpointing(self):
        """Test streaming fitting with checkpointing enabled."""
        # Create test data
        test_texts = ["hello world test"] * 10
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Create checkpoint directory
        checkpoint_dir = self.temp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        # Fit with checkpointing
        embedder = self.tokenizer.fit_streaming_with_consistency(
            data_source=str(test_file),
            batch_size=3,
            epochs=1,
            enable_checkpointing=True,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_frequency=2  # Save every 2 batches
        )
        
        assert embedder is not None
        
        # Verify training completed successfully
        stats = self.tokenizer.get_training_stats()
        assert stats['total_sequences'] == len(test_texts)
    
    def test_deterministic_processing_consistency(self):
        """Test that deterministic processing produces consistent results."""
        # Create test data
        test_texts = [
            "hello world",
            "test streaming",
            "consistency validation"
        ]
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # First run with fixed seed
        embedder1 = self.tokenizer.fit_streaming_with_consistency(
            data_source=str(test_file),
            batch_size=2,
            epochs=1,
            deterministic_seed=123,
            validate_consistency=False  # Skip validation for speed
        )
        
        # Get embedding matrix from first run
        matrix1 = embedder1.get_embedding_matrix()
        
        # Reset tokenizer for second run
        self.tokenizer._sinusoidal_embedder = None
        self.tokenizer._is_fitted = False
        
        # Second run with same seed
        embedder2 = self.tokenizer.fit_streaming_with_consistency(
            data_source=str(test_file),
            batch_size=2,
            epochs=1,
            deterministic_seed=123,
            validate_consistency=False  # Skip validation for speed
        )
        
        # Get embedding matrix from second run
        matrix2 = embedder2.get_embedding_matrix()
        
        # Verify matrices are identical (or very close due to floating point)
        mse = np.mean((matrix1 - matrix2) ** 2)
        assert mse < 1e-10, f"Deterministic runs should produce identical results, MSE: {mse}"
    
    def test_parameter_validation_consistency_features(self):
        """Test parameter validation for consistency features."""
        test_file = self.create_test_data_file("test.txt", ["hello world"])
        
        # Test invalid checkpoint_frequency
        with pytest.raises(InvalidInputError):
            self.tokenizer.fit_streaming_with_consistency(
                data_source=str(test_file),
                checkpoint_frequency=0
            )
        
        # Test invalid validation_sample_size
        with pytest.raises(InvalidInputError):
            self.tokenizer.fit_streaming_with_consistency(
                data_source=str(test_file),
                validation_sample_size=-1
            )
    
    def test_checkpoint_resume_functionality(self):
        """Test resuming training from checkpoint."""
        # Create test data
        test_texts = ["hello world test"] * 8
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Create checkpoint directory
        checkpoint_dir = self.temp_path / "resume_checkpoints"
        checkpoint_dir.mkdir()
        
        # Create a mock checkpoint file
        checkpoint_path = checkpoint_dir / "streaming_checkpoint.pkl"
        
        # Create initial checkpoint data
        import pickle
        checkpoint_data = {
            'epoch': 0,
            'batch_idx': 2,  # Resume from batch 2
            'total_sequences': 4,  # Already processed 4 sequences
            'total_tokens': 20,
            'vocab_stats': {1: 2, 2: 2},
            'sequence_lengths': [5, 5, 5, 5],
            'embedder_state': None,
            'tokenizer_config': {
                'embedding_dim': 64,
                'max_length': 8,
                'vocab_size': 7
            }
        }
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        # Resume training from checkpoint
        embedder = self.tokenizer.fit_streaming_with_consistency(
            data_source=str(test_file),
            batch_size=2,
            epochs=1,
            enable_checkpointing=True,
            checkpoint_dir=str(checkpoint_dir),
            validate_consistency=False  # Skip validation for speed
        )
        
        assert embedder is not None
        
        # Verify training completed
        stats = self.tokenizer.get_training_stats()
        # Should process all sequences (checkpoint sequences + remaining)
        assert stats['total_sequences'] >= len(test_texts)
    
    def test_emergency_checkpoint_on_error(self):
        """Test that emergency checkpoint is saved on training error."""
        # Create test data
        test_texts = ["hello world test"] * 5
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Create checkpoint directory
        checkpoint_dir = self.temp_path / "emergency_checkpoints"
        checkpoint_dir.mkdir()
        
        # Mock an error during training by patching the embedder
        with patch.object(self.tokenizer, 'create_sinusoidal_embedder') as mock_create:
            mock_embedder = Mock()
            mock_embedder.fit_batch.side_effect = Exception("Simulated training error")
            mock_create.return_value = mock_embedder
            
            # Attempt training (should fail and save emergency checkpoint)
            with pytest.raises(TokenizerError):
                self.tokenizer.fit_streaming_with_consistency(
                    data_source=str(test_file),
                    batch_size=2,
                    epochs=1,
                    enable_checkpointing=True,
                    checkpoint_dir=str(checkpoint_dir)
                )
            
            # Verify emergency checkpoint was created
            emergency_checkpoint = checkpoint_dir / "streaming_checkpoint.pkl.emergency"
            assert emergency_checkpoint.exists(), "Emergency checkpoint should be created on error"
    
    def test_validation_with_insufficient_data(self):
        """Test validation behavior with insufficient validation data."""
        # Create minimal test data
        test_texts = ["hello"]
        test_file = self.create_test_data_file("test.txt", test_texts)
        
        # Fit with validation but insufficient data
        embedder = self.tokenizer.fit_streaming_with_consistency(
            data_source=str(test_file),
            batch_size=1,
            epochs=1,
            validate_consistency=True,
            validation_sample_size=10  # More than available data
        )
        
        assert embedder is not None
        
        # Verify training completed despite insufficient validation data
        stats = self.tokenizer.get_training_stats()
        assert stats is not None
        # Validation should still run with available data
        assert 'validation_metrics' in stats


if __name__ == "__main__":
    pytest.main([__file__])