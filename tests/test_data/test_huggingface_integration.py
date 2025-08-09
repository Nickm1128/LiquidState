#!/usr/bin/env python3
"""
Unit tests for HuggingFace dataset integration components.

Tests the HuggingFaceDatasetLoader, ConversationSplitter, and DatasetProcessor
classes for proper functionality and error handling.
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
import json
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from lsm.data.huggingface_loader import (
    HuggingFaceDatasetLoader, 
    ConversationSplitter, 
    DatasetProcessor
)
from lsm.utils.lsm_exceptions import (
    HuggingFaceDatasetError, 
    ConversationSplitError, 
    DatasetValidationError
)


class TestHuggingFaceDatasetLoader(unittest.TestCase):
    """Test cases for HuggingFaceDatasetLoader."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.loader = HuggingFaceDatasetLoader(cache_dir=self.temp_dir)
        
        # Create mock dataset
        self.mock_data = {
            'text': [
                "Hello, how are you today?",
                "I'm doing well, thank you!",
                "What's your favorite hobby?",
                "I enjoy reading books."
            ],
            'token_count': [6, 5, 5, 4],
            'url': ['https://example.com'] * 4
        }
        self.mock_df = pd.DataFrame(self.mock_data)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test loader initialization."""
        self.assertEqual(self.loader.dataset_name, "HuggingFaceTB/smollm-corpus")
        self.assertEqual(self.loader.dataset_config, "cosmopedia-v2")
        self.assertTrue(Path(self.temp_dir).exists())
    
    def test_load_cached_datasets_no_cache(self):
        """Test loading when no cache exists."""
        with self.assertRaises(HuggingFaceDatasetError):
            self.loader.load_cached_datasets()
    
    def test_load_cached_datasets_with_cache(self):
        """Test loading with cached data."""
        # Create mock cache
        cache_dir = Path(self.temp_dir) / "cosmopedia-v2"
        cache_dir.mkdir(parents=True)
        
        csv_path = cache_dir / "train.csv"
        self.mock_df.to_csv(csv_path, index=False)
        
        # Load cached data
        loaded_df = self.loader.load_cached_datasets()
        
        self.assertEqual(len(loaded_df), 4)
        self.assertIn('source_file', loaded_df.columns)
        self.assertEqual(loaded_df['source_file'].iloc[0], 'train.csv')
    
    def test_validate_dataset_integrity_valid(self):
        """Test dataset validation with valid data."""
        # Create larger mock dataset to pass validation
        large_mock_data = {
            'text': [f"Sample text number {i} for testing purposes." for i in range(15)],
            'token_count': [8] * 15,
            'url': ['https://example.com'] * 15
        }
        large_mock_df = pd.DataFrame(large_mock_data)
        
        result = self.loader.validate_dataset_integrity(large_mock_df)
        self.assertTrue(result)
    
    def test_validate_dataset_integrity_empty(self):
        """Test dataset validation with empty data."""
        empty_df = pd.DataFrame()
        with self.assertRaises(DatasetValidationError):
            self.loader.validate_dataset_integrity(empty_df)
    
    def test_validate_dataset_integrity_too_small(self):
        """Test dataset validation with insufficient data."""
        small_df = pd.DataFrame({'text': ['hello']})
        with self.assertRaises(DatasetValidationError):
            self.loader.validate_dataset_integrity(small_df)
    
    def test_get_dataset_info_no_metadata(self):
        """Test getting dataset info when no metadata exists."""
        info = self.loader.get_dataset_info()
        self.assertEqual(info['status'], 'No cached dataset found')
    
    def test_get_dataset_info_with_metadata(self):
        """Test getting dataset info with metadata."""
        # Create mock metadata
        cache_dir = Path(self.temp_dir) / "cosmopedia-v2"
        cache_dir.mkdir(parents=True)
        
        metadata = {
            "dataset_name": "test_dataset",
            "total_files": 1
        }
        
        with open(cache_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        info = self.loader.get_dataset_info()
        self.assertEqual(info['dataset_name'], 'test_dataset')
        self.assertEqual(info['total_files'], 1)


class TestConversationSplitter(unittest.TestCase):
    """Test cases for ConversationSplitter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.splitter = ConversationSplitter(conversation_id_column='conversation_id')
        
        # Create mock conversation data
        self.mock_data = {
            'text': [
                "Hello", "Hi there", "How are you?", "I'm good",
                "Good morning", "Morning!", "Nice weather", "Yes indeed",
                "Goodbye", "See you later"
            ],
            'conversation_id': [
                'conv_1', 'conv_1', 'conv_1', 'conv_1',
                'conv_2', 'conv_2', 'conv_2', 'conv_2',
                'conv_3', 'conv_3'
            ]
        }
        self.mock_df = pd.DataFrame(self.mock_data)
    
    def test_initialization(self):
        """Test splitter initialization."""
        self.assertEqual(self.splitter.conversation_id_column, 'conversation_id')
    
    def test_identify_conversation_boundaries_with_ids(self):
        """Test conversation boundary identification with explicit IDs."""
        boundaries = self.splitter.identify_conversation_boundaries(self.mock_df)
        expected_boundaries = [0, 4, 8]  # Start of each conversation
        self.assertEqual(boundaries, expected_boundaries)
    
    def test_identify_conversation_boundaries_without_ids(self):
        """Test conversation boundary identification without explicit IDs."""
        # Remove conversation_id column
        df_no_ids = self.mock_df.drop('conversation_id', axis=1)
        splitter_no_ids = ConversationSplitter(conversation_id_column='nonexistent')
        
        boundaries = splitter_no_ids.identify_conversation_boundaries(df_no_ids)
        self.assertGreaterEqual(len(boundaries), 1)  # At least one boundary (start)
        self.assertEqual(boundaries[0], 0)  # First boundary should be at start
    
    def test_split_by_conversation_valid(self):
        """Test conversation splitting with valid data."""
        # Use 0.5 ratio to ensure we get at least 1 conversation in test set
        train_df, test_df = self.splitter.split_by_conversation(self.mock_df, test_ratio=0.5)
        
        # Should have data in both sets
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        
        # Total rows should be preserved
        self.assertEqual(len(train_df) + len(test_df), len(self.mock_df))
    
    def test_split_by_conversation_empty_dataframe(self):
        """Test conversation splitting with empty DataFrame."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ConversationSplitError):
            self.splitter.split_by_conversation(empty_df)
    
    def test_split_by_conversation_single_conversation(self):
        """Test conversation splitting with only one conversation."""
        single_conv_data = {
            'text': ['Hello', 'Hi', 'Bye'],
            'conversation_id': ['conv_1', 'conv_1', 'conv_1']
        }
        single_conv_df = pd.DataFrame(single_conv_data)
        
        with self.assertRaises(ConversationSplitError):
            self.splitter.split_by_conversation(single_conv_df)
    
    def test_ensure_conversation_integrity_valid(self):
        """Test conversation integrity check with valid split."""
        train_df, test_df = self.splitter.split_by_conversation(self.mock_df, test_ratio=0.33)
        result = self.splitter.ensure_conversation_integrity(train_df, test_df)
        self.assertTrue(result)
    
    def test_ensure_conversation_integrity_empty_test(self):
        """Test conversation integrity check with empty test set."""
        train_df = self.mock_df.copy()
        test_df = pd.DataFrame()
        result = self.splitter.ensure_conversation_integrity(train_df, test_df)
        self.assertTrue(result)
    
    def test_ensure_conversation_integrity_no_id_column(self):
        """Test conversation integrity check without ID column."""
        df_no_ids = self.mock_df.drop('conversation_id', axis=1)
        train_df = df_no_ids.iloc[:5].copy()
        test_df = df_no_ids.iloc[5:].copy()
        
        result = self.splitter.ensure_conversation_integrity(train_df, test_df)
        self.assertTrue(result)  # Should pass with warning


class TestDatasetProcessor(unittest.TestCase):
    """Test cases for DatasetProcessor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.processor = DatasetProcessor()
        
        # Create mock dataset
        self.mock_data = {
            'text': [
                "This is a sample text for testing purposes.",
                "Another text with different content and length.",
                "Short text.",
                "A much longer text that contains more words and should demonstrate the text analysis capabilities of the processor."
            ],
            'token_count': [8, 7, 2, 19],
            'url': ['https://example.com'] * 4
        }
        self.mock_df = pd.DataFrame(self.mock_data)
    
    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsInstance(self.processor, DatasetProcessor)
    
    def test_validate_dataset_structure_valid(self):
        """Test dataset structure validation with valid data."""
        results = self.processor.validate_dataset_structure(self.mock_df)
        
        self.assertTrue(results['is_valid'])
        self.assertEqual(len(results['errors']), 0)
        self.assertEqual(results['statistics']['total_rows'], 4)
        self.assertEqual(results['statistics']['total_columns'], 3)
        self.assertIn('text_analysis', results['statistics'])
    
    def test_validate_dataset_structure_empty(self):
        """Test dataset structure validation with empty data."""
        empty_df = pd.DataFrame()
        results = self.processor.validate_dataset_structure(empty_df)
        
        self.assertFalse(results['is_valid'])
        self.assertIn('Dataset is empty', results['errors'])
    
    def test_validate_dataset_structure_no_text(self):
        """Test dataset structure validation without text column."""
        no_text_df = pd.DataFrame({'numbers': [1, 2, 3]})
        results = self.processor.validate_dataset_structure(no_text_df)
        
        self.assertTrue(results['is_valid'])  # Should still be valid
        self.assertNotIn('text_analysis', results['statistics'])
    
    def test_extract_conversation_metadata(self):
        """Test conversation metadata extraction."""
        # Add conversation IDs to mock data
        self.mock_df['conversation_id'] = ['conv_1', 'conv_1', 'conv_2', 'conv_2']
        
        metadata = self.processor.extract_conversation_metadata(self.mock_df)
        
        self.assertEqual(metadata['total_rows'], 4)
        self.assertGreater(metadata['estimated_conversations'], 0)
        self.assertIn('avg_conversation_length', metadata)
    
    def test_prepare_for_training_valid(self):
        """Test dataset preparation with valid data."""
        prepared_df = self.processor.prepare_for_training(self.mock_df)
        
        self.assertEqual(len(prepared_df), 4)  # All rows should be preserved
        self.assertIn('processed_timestamp', prepared_df.columns)
        self.assertIn('row_id', prepared_df.columns)
    
    def test_prepare_for_training_with_nulls(self):
        """Test dataset preparation with null values."""
        # Add null values
        df_with_nulls = self.mock_df.copy()
        df_with_nulls.loc[1, 'text'] = None
        df_with_nulls.loc[2, 'text'] = ''
        
        prepared_df = self.processor.prepare_for_training(df_with_nulls)
        
        # Should remove null and empty texts
        self.assertLess(len(prepared_df), len(df_with_nulls))
        self.assertTrue(all(prepared_df['text'].str.len() >= 5))
    
    def test_prepare_for_training_short_texts(self):
        """Test dataset preparation with very short texts."""
        short_text_df = pd.DataFrame({
            'text': ['Hi', 'Hello there', 'A', 'This is long enough'],
            'other_col': [1, 2, 3, 4]
        })
        
        prepared_df = self.processor.prepare_for_training(short_text_df)
        
        # Should remove texts shorter than 5 characters
        self.assertEqual(len(prepared_df), 2)  # Only 'Hello there' and 'This is long enough'


class TestIntegration(unittest.TestCase):
    """Integration tests for all components working together."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create comprehensive mock dataset
        self.mock_data = {
            'text': [
                "Hello, how are you today?", "I'm doing well, thank you!",
                "What's your favorite hobby?", "I enjoy reading books.",
                "That sounds interesting!", "Yes, I love science fiction.",
                "Good morning! How can I help?", "I need some advice.",
                "I'd be happy to help.", "What do you need advice about?",
                "How do I learn programming?", "Start with Python basics."
            ],
            'conversation_id': [
                'conv_1', 'conv_1', 'conv_1', 'conv_1', 'conv_1', 'conv_1',
                'conv_2', 'conv_2', 'conv_2', 'conv_2',
                'conv_3', 'conv_3'
            ],
            'token_count': [6, 5, 5, 4, 4, 5, 6, 4, 5, 7, 5, 4],
            'url': ['https://example.com'] * 12
        }
        self.mock_df = pd.DataFrame(self.mock_data)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_pipeline(self):
        """Test the complete pipeline from loading to splitting."""
        # 1. Create cached dataset
        loader = HuggingFaceDatasetLoader(cache_dir=self.temp_dir)
        cache_dir = Path(self.temp_dir) / "cosmopedia-v2"
        cache_dir.mkdir(parents=True)
        
        csv_path = cache_dir / "train.csv"
        self.mock_df.to_csv(csv_path, index=False)
        
        # 2. Load and validate
        loaded_df = loader.load_cached_datasets()
        is_valid = loader.validate_dataset_integrity(loaded_df)
        self.assertTrue(is_valid)
        
        # 3. Process dataset
        processor = DatasetProcessor()
        prepared_df = processor.prepare_for_training(loaded_df)
        
        # 4. Split by conversations (use 0.5 to ensure test set gets data)
        splitter = ConversationSplitter(conversation_id_column='conversation_id')
        train_df, test_df = splitter.split_by_conversation(prepared_df, test_ratio=0.5)
        
        # 5. Verify integrity
        integrity_ok = splitter.ensure_conversation_integrity(train_df, test_df)
        
        # Assertions
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)
        self.assertTrue(integrity_ok)
        self.assertEqual(len(train_df) + len(test_df), len(prepared_df))


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)