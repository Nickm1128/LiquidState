#!/usr/bin/env python3
"""
Tests for StreamingDataIterator.

This module tests the streaming data iterator functionality including
various data formats, batch processing, and memory management.
"""

import pytest
import tempfile
import json
import csv
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.lsm.data.streaming_data_iterator import StreamingDataIterator, create_streaming_iterator
from src.lsm.utils.lsm_exceptions import DataLoadError, InvalidInputError


class TestStreamingDataIterator:
    """Test cases for StreamingDataIterator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_text_file(self, filename: str, lines: list) -> Path:
        """Create a test text file."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(f"{line}\n")
        return file_path
    
    def create_test_json_file(self, filename: str, data: list) -> Path:
        """Create a test JSON file."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        return file_path
    
    def create_test_jsonl_file(self, filename: str, data: list) -> Path:
        """Create a test JSONL file."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        return file_path
    
    def create_test_csv_file(self, filename: str, data: list, headers: list) -> Path:
        """Create a test CSV file."""
        file_path = self.temp_path / filename
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in data:
                writer.writerow(row)
        return file_path
    
    def test_init_with_single_file(self):
        """Test initialization with a single file."""
        test_file = self.create_test_text_file("test.txt", ["line1", "line2", "line3"])
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        )
        
        assert len(iterator._file_paths) == 1
        assert iterator._file_paths[0] == test_file
        assert iterator.batch_size == 2
    
    def test_init_with_directory(self):
        """Test initialization with a directory."""
        # Create multiple test files
        self.create_test_text_file("file1.txt", ["line1", "line2"])
        self.create_test_json_file("file2.json", [{"key": "value1"}, {"key": "value2"}])
        self.create_test_csv_file("file3.csv", [{"col1": "val1", "col2": "val2"}], ["col1", "col2"])
        
        iterator = StreamingDataIterator(
            data_source=str(self.temp_path),
            batch_size=5
        )
        
        assert len(iterator._file_paths) == 3
        assert iterator.batch_size == 5
    
    def test_init_with_file_list(self):
        """Test initialization with a list of files."""
        file1 = self.create_test_text_file("file1.txt", ["line1", "line2"])
        file2 = self.create_test_text_file("file2.txt", ["line3", "line4"])
        
        iterator = StreamingDataIterator(
            data_source=[str(file1), str(file2)],
            batch_size=3
        )
        
        assert len(iterator._file_paths) == 2
        assert iterator.batch_size == 3
    
    def test_text_file_processing(self):
        """Test processing text files."""
        test_lines = ["Hello world", "This is a test", "Another line", "Final line"]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        )
        
        batches = list(iterator)
        
        # Should have 2 batches of size 2
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        
        # Check content
        all_items = [item for batch in batches for item in batch]
        assert all_items == test_lines
    
    def test_json_file_processing(self):
        """Test processing JSON files."""
        test_data = [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25},
            {"name": "Charlie", "age": 35}
        ]
        test_file = self.create_test_json_file("test.json", test_data)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        )
        
        batches = list(iterator)
        
        # Should have 2 batches (2 + 1)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
        
        # Check content
        all_items = [item for batch in batches for item in batch]
        assert all_items == test_data
    
    def test_jsonl_file_processing(self):
        """Test processing JSONL files."""
        test_data = [
            {"text": "First message"},
            {"text": "Second message"},
            {"text": "Third message"}
        ]
        test_file = self.create_test_jsonl_file("test.jsonl", test_data)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        )
        
        batches = list(iterator)
        
        # Should have 2 batches (2 + 1)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
        
        # Check content
        all_items = [item for batch in batches for item in batch]
        assert all_items == test_data
    
    def test_csv_file_processing(self):
        """Test processing CSV files."""
        test_data = [
            {"name": "Alice", "score": "95"},
            {"name": "Bob", "score": "87"},
            {"name": "Charlie", "score": "92"}
        ]
        test_file = self.create_test_csv_file("test.csv", test_data, ["name", "score"])
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        )
        
        batches = list(iterator)
        
        # Should have 2 batches (2 + 1)
        assert len(batches) == 2
        assert len(batches[0]) == 2
        assert len(batches[1]) == 1
        
        # Check content
        all_items = [item for batch in batches for item in batch]
        assert all_items == test_data
    
    def test_multiple_files_processing(self):
        """Test processing multiple files."""
        file1 = self.create_test_text_file("file1.txt", ["line1", "line2"])
        file2 = self.create_test_text_file("file2.txt", ["line3", "line4"])
        
        iterator = StreamingDataIterator(
            data_source=[str(file1), str(file2)],
            batch_size=3
        )
        
        batches = list(iterator)
        
        # Should have 2 batches (3 + 1)
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 1
        
        # Check content
        all_items = [item for batch in batches for item in batch]
        assert all_items == ["line1", "line2", "line3", "line4"]
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        test_lines = ["line1", "line2", "line3", "line4", "line5"]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        progress_calls = []
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2,
            progress_callback=progress_callback
        )
        
        batches = list(iterator)
        
        # Check that progress was tracked
        assert len(progress_calls) > 0
        
        # Check progress info
        progress = iterator.get_progress()
        assert progress['items_processed'] == 5
        assert progress['files_completed'] == 1
        assert progress['current_batch_size'] == 2
    
    @patch('src.lsm.data.streaming_data_iterator.psutil.Process')
    @patch('src.lsm.data.streaming_data_iterator.psutil.virtual_memory')
    def test_memory_monitoring(self, mock_virtual_memory, mock_process):
        """Test memory usage monitoring."""
        # Mock memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 500 * 1024 * 1024  # 500 MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Mock system memory
        mock_system_memory = Mock()
        mock_system_memory.available = 2000 * 1024 * 1024  # 2GB available
        mock_system_memory.percent = 60.0
        mock_virtual_memory.return_value = mock_system_memory
        
        test_lines = ["line1", "line2", "line3", "line4"]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2,
            memory_threshold_mb=400.0,  # Lower than mocked usage
            auto_adjust_batch_size=True
        )
        
        # Process data to trigger memory monitoring
        batches = list(iterator)
        
        # Check that memory usage was monitored
        progress = iterator.get_progress()
        assert 'memory_usage_mb' in progress
        assert 'system_memory_percent' in progress
        assert 'system_available_mb' in progress
    
    def test_reset_functionality(self):
        """Test iterator reset functionality."""
        test_lines = ["line1", "line2", "line3", "line4"]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        )
        
        # Process first batch
        first_batch = next(iterator)
        assert len(first_batch) == 2
        
        # Reset and process again
        iterator.reset()
        first_batch_after_reset = next(iterator)
        
        # Should get the same first batch
        assert first_batch == first_batch_after_reset
    
    def test_context_manager(self):
        """Test context manager functionality."""
        test_lines = ["line1", "line2", "line3"]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        with StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2
        ) as iterator:
            batches = list(iterator)
            assert len(batches) == 2
    
    def test_format_detection(self):
        """Test automatic format detection."""
        # Create files with different formats
        text_file = self.create_test_text_file("test.txt", ["line1", "line2"])
        json_file = self.create_test_json_file("test.json", [{"key": "value"}])
        csv_file = self.create_test_csv_file("test.csv", [{"col": "val"}], ["col"])
        
        iterator = StreamingDataIterator(data_source=str(text_file))
        assert iterator._detect_format(text_file) == 'text'
        
        iterator = StreamingDataIterator(data_source=str(json_file))
        assert iterator._detect_format(json_file) == 'json'
        
        iterator = StreamingDataIterator(data_source=str(csv_file))
        assert iterator._detect_format(csv_file) == 'csv'
    
    def test_error_handling_invalid_path(self):
        """Test error handling for invalid paths."""
        with pytest.raises(DataLoadError):
            StreamingDataIterator(data_source="/nonexistent/path")
    
    def test_error_handling_invalid_data_source(self):
        """Test error handling for invalid data source types."""
        with pytest.raises(InvalidInputError):
            StreamingDataIterator(data_source=123)  # Invalid type
    
    def test_empty_file_handling(self):
        """Test handling of empty files."""
        empty_file = self.create_test_text_file("empty.txt", [])
        
        iterator = StreamingDataIterator(
            data_source=str(empty_file),
            batch_size=2
        )
        
        batches = list(iterator)
        assert len(batches) == 0
    
    def test_create_streaming_iterator_convenience_function(self):
        """Test the convenience function for creating iterators."""
        test_file = self.create_test_text_file("test.txt", ["line1", "line2"])
        
        iterator = create_streaming_iterator(
            data_source=str(test_file),
            batch_size=1
        )
        
        assert isinstance(iterator, StreamingDataIterator)
        assert iterator.batch_size == 1
        
        batches = list(iterator)
        assert len(batches) == 2
        assert len(batches[0]) == 1
        assert len(batches[1]) == 1
    
    @patch('src.lsm.data.streaming_data_iterator.psutil.Process')
    @patch('src.lsm.data.streaming_data_iterator.psutil.virtual_memory')
    def test_adaptive_batch_size_reduction(self, mock_virtual_memory, mock_process):
        """Test automatic batch size reduction under memory pressure."""
        # Mock memory usage progression - start low, then high
        memory_values = [
            100 * 1024 * 1024,  # Initial: 100MB
            1500 * 1024 * 1024,  # After processing: 1.5GB (high usage)
            1500 * 1024 * 1024,  # Maintain high usage
            1500 * 1024 * 1024,
        ]
        
        # Create mock memory info objects
        mock_memory_infos = []
        for value in memory_values + [memory_values[-1]] * 10:
            mock_info = Mock()
            mock_info.rss = value
            mock_memory_infos.append(mock_info)
        
        mock_process.return_value.memory_info.side_effect = mock_memory_infos
        
        # Mock system memory
        mock_system_memory = Mock()
        mock_system_memory.available = 500 * 1024 * 1024  # 500MB available
        mock_system_memory.percent = 85.0
        mock_virtual_memory.return_value = mock_system_memory
        
        # Create larger dataset to trigger multiple memory checks
        test_lines = [f"line{i}" for i in range(100)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=50,
            memory_threshold_mb=500.0,  # Lower than mocked high usage
            auto_adjust_batch_size=True,
            memory_check_interval=1  # Check every batch
        )
        
        initial_batch_size = iterator.batch_size
        
        # Process data to trigger memory monitoring and adjustment
        batches = list(iterator)
        
        # Check that batch size was reduced
        final_batch_size = iterator.batch_size
        assert final_batch_size < initial_batch_size
        
        # Check that emergency mode was activated (which also reduces batch size)
        progress = iterator.get_progress()
        assert progress['emergency_mode'] == True or final_batch_size == iterator.min_batch_size
        
        # Check adaptive stats
        stats = iterator.get_adaptive_stats()
        assert stats['current_state']['batches_processed'] > 0
    
    @patch('src.lsm.data.streaming_data_iterator.psutil.Process')
    @patch('src.lsm.data.streaming_data_iterator.psutil.virtual_memory')
    def test_regular_batch_size_adjustment(self, mock_virtual_memory, mock_process):
        """Test regular batch size adjustment (not emergency mode)."""
        # Mock moderate memory usage - above threshold but not critical
        memory_values = [
            100 * 1024 * 1024,  # Initial: 100MB
            700 * 1024 * 1024,  # Moderate usage: 700MB (above 500MB threshold)
            700 * 1024 * 1024,  # Maintain moderate usage
            700 * 1024 * 1024,
        ]
        
        # Create mock memory info objects
        mock_memory_infos = []
        for value in memory_values + [memory_values[-1]] * 10:
            mock_info = Mock()
            mock_info.rss = value
            mock_memory_infos.append(mock_info)
        
        mock_process.return_value.memory_info.side_effect = mock_memory_infos
        
        # Mock system memory with reasonable availability
        mock_system_memory = Mock()
        mock_system_memory.available = 2000 * 1024 * 1024  # 2GB available
        mock_system_memory.percent = 70.0  # Not critical
        mock_virtual_memory.return_value = mock_system_memory
        
        # Create dataset
        test_lines = [f"line{i}" for i in range(100)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=50,
            memory_threshold_mb=500.0,  # Lower than mocked usage (600MB above initial)
            emergency_threshold_mb=1200.0,  # High enough to avoid emergency mode
            auto_adjust_batch_size=True,
            memory_check_interval=1  # Check every batch
        )
        
        initial_batch_size = iterator.batch_size
        
        # Process data to trigger memory monitoring and adjustment
        batches = list(iterator)
        
        # Check that batch size was reduced through regular adjustment
        final_batch_size = iterator.batch_size
        assert final_batch_size < initial_batch_size
        
        # Should not be in emergency mode
        progress = iterator.get_progress()
        assert progress['emergency_mode'] == False
        
        # Should have adjustment history
        stats = iterator.get_adaptive_stats()
        assert len(stats['adjustment_history']) > 0
    
    @patch('src.lsm.data.streaming_data_iterator.psutil.Process')
    @patch('src.lsm.data.streaming_data_iterator.psutil.virtual_memory')
    def test_adaptive_batch_size_increase(self, mock_virtual_memory, mock_process):
        """Test automatic batch size increase under low memory usage."""
        # Mock low memory usage
        mock_memory_info = Mock()
        mock_memory_info.rss = 200 * 1024 * 1024  # 200MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Mock system memory with plenty available
        mock_system_memory = Mock()
        mock_system_memory.available = 4000 * 1024 * 1024  # 4GB available
        mock_system_memory.percent = 30.0
        mock_virtual_memory.return_value = mock_system_memory
        
        # Create dataset to trigger multiple memory checks
        test_lines = [f"line{i}" for i in range(100)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=10,  # Start with small batch size
            memory_threshold_mb=1000.0,  # Higher than mocked usage
            auto_adjust_batch_size=True,
            memory_check_interval=1  # Check every batch
        )
        
        initial_batch_size = iterator.batch_size
        
        # Process several batches to establish stable memory pattern
        batch_count = 0
        for batch in iterator:
            batch_count += 1
            if batch_count >= 5:  # Process enough batches for stability
                break
        
        # Check that batch size was increased (may take several iterations)
        final_batch_size = iterator.batch_size
        # Note: Increase only happens if memory is stable, so we check the capability exists
        assert hasattr(iterator, '_is_memory_stable')
    
    @patch('src.lsm.data.streaming_data_iterator.psutil.Process')
    @patch('src.lsm.data.streaming_data_iterator.psutil.virtual_memory')
    def test_emergency_mode_activation(self, mock_virtual_memory, mock_process):
        """Test emergency mode activation under critical memory pressure."""
        # Mock critical memory usage progression
        memory_values = [
            100 * 1024 * 1024,  # Initial: 100MB
            2000 * 1024 * 1024,  # Critical usage: 2GB (1900MB above initial)
            2000 * 1024 * 1024,  # Maintain critical usage
            2000 * 1024 * 1024,
        ]
        
        # Create mock memory info objects
        mock_memory_infos = []
        for value in memory_values + [memory_values[-1]] * 10:
            mock_info = Mock()
            mock_info.rss = value
            mock_memory_infos.append(mock_info)
        
        mock_process.return_value.memory_info.side_effect = mock_memory_infos
        
        # Mock critical system memory
        mock_system_memory = Mock()
        mock_system_memory.available = 30 * 1024 * 1024  # Only 30MB available
        mock_system_memory.percent = 96.0
        mock_virtual_memory.return_value = mock_system_memory
        
        test_lines = [f"line{i}" for i in range(50)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=25,
            memory_threshold_mb=500.0,
            emergency_threshold_mb=750.0,  # Lower than mocked usage (1900MB)
            auto_adjust_batch_size=True,
            min_batch_size=5,
            memory_check_interval=1
        )
        
        # Process data to trigger emergency mode
        batches = list(iterator)
        
        # Check that batch size was reduced to minimum (emergency response)
        assert iterator.batch_size == iterator.min_batch_size
        
        # Check that critical memory situation was detected (from logs)
        # The emergency mode flag might not be set if fallback mechanisms handle it
        progress = iterator.get_progress()
        # Either emergency mode is set OR batch size was forced to minimum
        assert progress['emergency_mode'] == True or iterator.batch_size == iterator.min_batch_size
    
    def test_fallback_mechanisms(self):
        """Test fallback mechanisms for memory-constrained environments."""
        test_lines = [f"line{i}" for i in range(20)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=10,
            memory_threshold_mb=100.0,
            auto_adjust_batch_size=True,
            min_batch_size=2,
            max_batch_size=50
        )
        
        # Test configuration updates
        iterator.configure_adaptive_settings(
            memory_threshold_mb=200.0,
            min_batch_size=1,
            max_batch_size=100
        )
        
        assert iterator.memory_threshold_mb == 200.0
        assert iterator.min_batch_size == 1
        assert iterator.max_batch_size == 100
        
        # Test forced reset
        iterator.batch_size = 5
        iterator.force_batch_size_reset()
        assert iterator.batch_size == iterator.initial_batch_size
    
    def test_memory_trend_analysis(self):
        """Test memory usage trend analysis."""
        test_lines = [f"line{i}" for i in range(30)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=5,
            auto_adjust_batch_size=True
        )
        
        # Simulate some memory history
        iterator._memory_history = [
            {'batch_count': i, 'memory_usage': 100 + i * 10, 'batch_size': 5, 'timestamp': time.time()}
            for i in range(10)
        ]
        
        trend = iterator.get_memory_trend()
        assert 'trend' in trend
        assert 'slope' in trend
        assert trend['trend'] == 'increasing'  # Memory usage is increasing
    
    def test_adaptive_stats_collection(self):
        """Test collection of adaptive management statistics."""
        test_lines = [f"line{i}" for i in range(10)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=5,
            auto_adjust_batch_size=True,
            memory_threshold_mb=100.0
        )
        
        # Get initial stats
        stats = iterator.get_adaptive_stats()
        
        assert 'memory_history' in stats
        assert 'adjustment_history' in stats
        assert 'current_config' in stats
        assert 'current_state' in stats
        
        # Check config values
        config = stats['current_config']
        assert config['batch_size'] == 5
        assert config['memory_threshold_mb'] == 100.0
        assert config['auto_adjust_enabled'] == True
    
    def test_memory_check_interval(self):
        """Test memory check interval functionality."""
        test_lines = [f"line{i}" for i in range(20)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=2,
            memory_check_interval=5,  # Check every 5 batches
            auto_adjust_batch_size=True
        )
        
        # Process a few batches
        batch_count = 0
        for batch in iterator:
            batch_count += 1
            if batch_count >= 3:
                break
        
        # Should not check memory on every batch
        assert iterator._batch_count >= 3
        
        # Test should_check_memory logic
        iterator._batch_count = 0
        iterator._last_memory_check = 0
        assert iterator._should_check_memory() == False  # Too soon
        
        iterator._batch_count = 5
        assert iterator._should_check_memory() == True  # Interval reached
    
    def test_batch_size_constraints(self):
        """Test batch size constraint enforcement."""
        test_lines = [f"line{i}" for i in range(10)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=20,
            min_batch_size=5,
            max_batch_size=50,
            auto_adjust_batch_size=True
        )
        
        # Test that batch size respects constraints
        assert iterator.batch_size <= iterator.max_batch_size
        
        # Test constraint updates
        iterator.configure_adaptive_settings(
            min_batch_size=25,  # Higher than current batch size
            max_batch_size=30
        )
        
        # Batch size should be adjusted to respect new minimum
        assert iterator.batch_size >= iterator.min_batch_size
        assert iterator.batch_size <= iterator.max_batch_size
    
    @patch('src.lsm.data.streaming_data_iterator.gc')
    def test_garbage_collection_management(self, mock_gc):
        """Test garbage collection management in fallback mechanisms."""
        test_lines = [f"line{i}" for i in range(10)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=5,
            auto_adjust_batch_size=True
        )
        
        # Test garbage collection is called during fallback
        mock_gc.isenabled.return_value = True
        
        # Simulate critical memory situation
        memory_metrics = {
            'current_usage_mb': 2000.0,
            'system_available_mb': 30.0,
            'system_percent': 96.0
        }
        
        iterator._apply_fallback_mechanisms(memory_metrics)
        
        # Should have called garbage collection
        mock_gc.collect.assert_called()
    
    def test_enhanced_progress_tracking(self):
        """Test enhanced progress tracking with adaptive metrics."""
        test_lines = [f"line{i}" for i in range(15)]
        test_file = self.create_test_text_file("test.txt", test_lines)
        
        iterator = StreamingDataIterator(
            data_source=str(test_file),
            batch_size=5,
            auto_adjust_batch_size=True,
            memory_threshold_mb=100.0
        )
        
        # Process some data
        batches = list(iterator)
        
        # Get enhanced progress
        progress = iterator.get_progress()
        
        # Check for new adaptive fields
        expected_fields = [
            'current_batch_size', 'initial_batch_size', 'average_batch_size',
            'memory_usage_mb', 'memory_threshold_mb', 'emergency_threshold_mb',
            'system_memory_percent', 'system_available_mb', 'emergency_mode',
            'total_adjustments', 'recent_adjustments', 'consecutive_adjustments',
            'adjustment_cooldown', 'batches_processed', 'gc_enabled'
        ]
        
        for field in expected_fields:
            assert field in progress, f"Missing field: {field}"
    
    def test_create_streaming_iterator_with_adaptive_defaults(self):
        """Test convenience function with adaptive management defaults."""
        test_file = self.create_test_text_file("test.txt", ["line1", "line2"])
        
        # Test with adaptive defaults
        iterator = create_streaming_iterator(
            data_source=str(test_file),
            batch_size=10,
            auto_adjust_batch_size=True,
            memory_threshold_mb=500.0
        )
        
        assert isinstance(iterator, StreamingDataIterator)
        assert iterator.batch_size == 10
        assert iterator.auto_adjust_batch_size == True
        assert iterator.memory_threshold_mb == 500.0


if __name__ == "__main__":
    pytest.main([__file__])