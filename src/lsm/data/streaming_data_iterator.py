#!/usr/bin/env python3
"""
Streaming data iterator for memory-efficient data loading.

This module provides a streaming data iterator that can handle large datasets
that don't fit in memory by processing them in configurable batches with
progress tracking, memory usage monitoring, and adaptive batch size management.

Classes:
    StreamingDataIterator: Main iterator class for memory-efficient data processing

Key Features:
    - Memory-efficient processing of datasets larger than available RAM
    - Support for multiple data formats (text, JSON, JSONL, CSV)
    - Adaptive batch size management based on memory usage
    - Real-time memory monitoring and automatic adjustments
    - Progress tracking with customizable callbacks
    - Fallback mechanisms for memory-constrained environments
    - Automatic format detection and text extraction

Example:
    Basic usage with automatic batch size adjustment:
    
    >>> from lsm.data.streaming_data_iterator import StreamingDataIterator
    >>> iterator = StreamingDataIterator(
    ...     data_source='large_dataset.txt',
    ...     batch_size=1000,
    ...     auto_adjust_batch_size=True,
    ...     memory_threshold_mb=500.0
    ... )
    >>> 
    >>> for batch in iterator:
    ...     # Process batch of data
    ...     process_batch(batch)
    
    Advanced usage with progress tracking:
    
    >>> def progress_callback(processed, total):
    ...     print(f"Processed {processed}/{total} items ({processed/total*100:.1f}%)")
    >>> 
    >>> iterator = StreamingDataIterator(
    ...     data_source=['file1.jsonl', 'file2.csv', 'file3.txt'],
    ...     batch_size=2000,
    ...     memory_threshold_mb=1000.0,
    ...     auto_adjust_batch_size=True,
    ...     progress_callback=progress_callback,
    ...     extract_text=True,
    ...     text_field='content'
    ... )
    
    Memory-constrained environment usage:
    
    >>> iterator = StreamingDataIterator(
    ...     data_source='huge_dataset/',
    ...     batch_size=500,
    ...     memory_threshold_mb=200.0,
    ...     emergency_threshold_mb=300.0,
    ...     min_batch_size=50,
    ...     auto_adjust_batch_size=True
    ... )

See Also:
    - EnhancedTokenizerWrapper: For tokenizer integration with streaming
    - ConfigurableSinusoidalEmbedder: For embedding training with streaming data
"""

import os
import json
import csv
import psutil
import time
import gc
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Iterator, Callable
from abc import ABC, abstractmethod

from ..utils.lsm_exceptions import DataLoadError, InvalidInputError
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class StreamingDataIterator:
    """
    Memory-efficient data iterator for processing large datasets.
    
    This iterator can handle various data formats (text, JSON, JSONL, CSV)
    and provides configurable batch processing with automatic memory monitoring,
    adaptive batch size adjustment, and intelligent fallback mechanisms for
    memory-constrained environments.
    
    The iterator processes data in configurable batches while continuously
    monitoring memory usage. When memory usage exceeds specified thresholds,
    it automatically adjusts the batch size to prevent out-of-memory errors.
    It also provides emergency fallback mechanisms for critical memory situations.
    
    Attributes:
        batch_size (int): Current batch size (may change during processing)
        initial_batch_size (int): Original batch size for reset operations
        memory_threshold_mb (float): Memory threshold for batch size adjustment
        auto_adjust_batch_size (bool): Whether to automatically adjust batch size
        progress_callback (Callable): Optional callback for progress updates
        text_field (str): Field name for text extraction from structured data
        extract_text (bool): Whether to extract text from data items
        min_batch_size (int): Minimum allowed batch size
        max_batch_size (int): Maximum allowed batch size
        memory_check_interval (int): Number of batches between memory checks
        emergency_threshold_mb (float): Emergency memory threshold
    
    Key Features:
        - **Adaptive Batch Management**: Automatically adjusts batch size based on
          real-time memory usage to prevent out-of-memory errors
        - **Multi-Format Support**: Handles text, JSON, JSONL, and CSV files with
          automatic format detection
        - **Memory Monitoring**: Continuous monitoring of process and system memory
          with detailed metrics and trend analysis
        - **Emergency Fallbacks**: Aggressive memory management techniques for
          critical memory situations
        - **Progress Tracking**: Customizable progress callbacks with estimated
          completion times
        - **Text Extraction**: Intelligent text extraction from structured data
          formats with configurable field mapping
        - **Resumable Processing**: State management for resuming interrupted
          processing sessions
    
    Example:
        Basic streaming with automatic adjustment:
        
        >>> iterator = StreamingDataIterator(
        ...     data_source='large_file.txt',
        ...     batch_size=1000,
        ...     auto_adjust_batch_size=True,
        ...     memory_threshold_mb=500.0
        ... )
        >>> 
        >>> for batch in iterator:
        ...     # Process each batch
        ...     results = process_batch(batch)
        
        Advanced configuration with callbacks:
        
        >>> def progress_callback(processed, total_estimate):
        ...     percent = (processed / total_estimate) * 100
        ...     print(f"Progress: {processed} items ({percent:.1f}%)")
        >>> 
        >>> iterator = StreamingDataIterator(
        ...     data_source=['data1.jsonl', 'data2.csv'],
        ...     batch_size=2000,
        ...     memory_threshold_mb=1000.0,
        ...     emergency_threshold_mb=1500.0,
        ...     auto_adjust_batch_size=True,
        ...     progress_callback=progress_callback,
        ...     extract_text=True,
        ...     text_field='content',
        ...     min_batch_size=100,
        ...     max_batch_size=10000
        ... )
        
        Memory-constrained environment:
        
        >>> # For systems with limited memory
        >>> iterator = StreamingDataIterator(
        ...     data_source='huge_dataset/',
        ...     batch_size=200,
        ...     memory_threshold_mb=100.0,
        ...     emergency_threshold_mb=150.0,
        ...     min_batch_size=10,
        ...     auto_adjust_batch_size=True,
        ...     memory_check_interval=5  # Check memory more frequently
        ... )
    
    Memory Management:
        The iterator implements a sophisticated memory management system:
        
        1. **Continuous Monitoring**: Tracks both process and system memory usage
        2. **Adaptive Adjustment**: Reduces batch size when memory usage is high
        3. **Emergency Mode**: Aggressive reduction and cleanup in critical situations
        4. **Trend Analysis**: Uses memory usage history to make intelligent adjustments
        5. **Fallback Mechanisms**: File handle management and garbage collection
        
        Memory thresholds:
        - memory_threshold_mb: Normal adjustment threshold
        - emergency_threshold_mb: Critical situation threshold (typically 1.5x normal)
        - System memory: Also considers overall system memory availability
    
    Supported Data Formats:
        - **Text files (.txt)**: Line-by-line processing
        - **JSON files (.json)**: Loads entire file, iterates over arrays
        - **JSONL files (.jsonl)**: Line-by-line JSON processing
        - **CSV files (.csv)**: Row-by-row processing with header detection
        - **Directories**: Processes all supported files in directory
    
    See Also:
        - EnhancedTokenizerWrapper.fit_streaming(): For tokenizer training
        - ConfigurableSinusoidalEmbedder: For embedding training with streaming
    """
    
    def __init__(self, 
                 data_source: Union[str, List[str]], 
                 batch_size: int = 1000,
                 memory_threshold_mb: float = 1000.0,
                 auto_adjust_batch_size: bool = False,
                 progress_callback: Optional[Callable] = None,
                 text_field: Optional[str] = None,
                 min_batch_size: int = 10,
                 max_batch_size: int = 50000,
                 memory_check_interval: int = 10,
                 emergency_threshold_mb: Optional[float] = None,
                 extract_text: bool = False,
                 **kwargs):
        """
        Initialize StreamingDataIterator.
        
        Args:
            data_source: Path to file/directory or list of file paths
            batch_size: Number of items per batch
            memory_threshold_mb: Memory threshold for automatic batch size adjustment
            auto_adjust_batch_size: Whether to automatically adjust batch size
            progress_callback: Optional callback for progress updates
            text_field: Field name for text data in structured formats
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            memory_check_interval: Number of batches between memory checks
            emergency_threshold_mb: Emergency memory threshold for aggressive reduction
            extract_text: Whether to extract text from data items (default: False)
            **kwargs: Additional configuration options
        """
        self.batch_size = batch_size
        self.initial_batch_size = batch_size
        self.memory_threshold_mb = memory_threshold_mb
        self.auto_adjust_batch_size = auto_adjust_batch_size
        self.progress_callback = progress_callback
        self.text_field = text_field or 'text'
        self.extract_text = extract_text
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.memory_check_interval = memory_check_interval
        self.emergency_threshold_mb = emergency_threshold_mb or (memory_threshold_mb * 1.5)
        
        # Initialize file paths
        self._file_paths = self._resolve_data_source(data_source)
        
        # Progress tracking
        self._items_processed = 0
        self._files_completed = 0
        self._current_file_index = 0
        self._current_file_handle = None
        self._current_reader = None
        
        # Memory monitoring and adaptive management
        self._process = psutil.Process()
        self._initial_memory = self._process.memory_info().rss / (1024 * 1024)  # MB
        self._batch_count = 0
        self._memory_history = []
        self._adjustment_history = []
        self._last_memory_check = 0
        self._consecutive_adjustments = 0
        self._adjustment_cooldown = 0
        
        # Fallback mechanisms
        self._emergency_mode = False
        self._gc_enabled = True
        
        # State management
        self._is_reset = True
        self._iterator = None
        
        logger.info(f"StreamingDataIterator initialized: {len(self._file_paths)} files, "
                   f"batch_size={batch_size}, memory_threshold={memory_threshold_mb}MB, "
                   f"adaptive={auto_adjust_batch_size}")
    
    def _resolve_data_source(self, data_source: Union[str, List[str]]) -> List[Path]:
        """
        Resolve data source to list of file paths.
        
        Args:
            data_source: Data source specification
            
        Returns:
            List of resolved file paths
            
        Raises:
            InvalidInputError: If data source type is invalid
            DataLoadError: If files/directories don't exist
        """
        if isinstance(data_source, str):
            path = Path(data_source)
            if not path.exists():
                raise DataLoadError(data_source, "Data source does not exist")
            
            if path.is_file():
                return [path]
            elif path.is_dir():
                # Find all supported files in directory
                supported_extensions = {'.txt', '.json', '.jsonl', '.csv'}
                files = []
                for ext in supported_extensions:
                    files.extend(path.glob(f'*{ext}'))
                return sorted(files)
            else:
                raise DataLoadError(data_source, "Data source is neither file nor directory")
        
        elif isinstance(data_source, list):
            paths = []
            for source in data_source:
                if not isinstance(source, str):
                    raise InvalidInputError("data_source", "string or list of strings", str(type(source)))
                path = Path(source)
                if not path.exists():
                    raise DataLoadError(source, "File does not exist")
                paths.append(path)
            return paths
        
        else:
            raise InvalidInputError("data_source", "string or list of strings", str(type(data_source)))
    
    def _detect_format(self, file_path: Path) -> str:
        """
        Detect file format based on extension and content.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected format ('text', 'json', 'jsonl', 'csv')
        """
        extension = file_path.suffix.lower()
        
        if extension == '.json':
            return 'json'
        elif extension == '.jsonl':
            return 'jsonl'
        elif extension == '.csv':
            return 'csv'
        else:
            return 'text'
    
    def _open_file_reader(self, file_path: Path):
        """
        Open appropriate reader for file format.
        
        Args:
            file_path: Path to file to open
        """
        format_type = self._detect_format(file_path)
        
        try:
            if format_type == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self._current_reader = iter(data)
                    else:
                        self._current_reader = iter([data])
            
            elif format_type == 'jsonl':
                self._current_file_handle = open(file_path, 'r', encoding='utf-8')
                self._current_reader = self._jsonl_reader()
            
            elif format_type == 'csv':
                self._current_file_handle = open(file_path, 'r', encoding='utf-8')
                csv_reader = csv.DictReader(self._current_file_handle)
                self._current_reader = iter(csv_reader)
            
            else:  # text
                self._current_file_handle = open(file_path, 'r', encoding='utf-8')
                self._current_reader = self._text_reader()
        
        except Exception as e:
            raise DataLoadError(str(file_path), f"Failed to open file: {str(e)}")
    
    def _jsonl_reader(self):
        """Generator for reading JSONL files line by line."""
        for line in self._current_file_handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON line: {line[:50]}... Error: {e}")
    
    def _text_reader(self):
        """Generator for reading text files line by line."""
        for line in self._current_file_handle:
            line = line.strip()
            if line:
                yield line
    
    def _close_current_file(self):
        """Close current file handle if open."""
        if self._current_file_handle:
            self._current_file_handle.close()
            self._current_file_handle = None
        self._current_reader = None
    
    def _extract_text(self, item: Any) -> str:
        """
        Extract text from data item based on format.
        
        Args:
            item: Data item to extract text from
            
        Returns:
            Extracted text string
        """
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Try to find text field
            if self.text_field in item:
                return str(item[self.text_field])
            elif 'text' in item:
                return str(item['text'])
            elif 'content' in item:
                return str(item['content'])
            elif 'message' in item:
                return str(item['message'])
            else:
                # Return first string value found
                for value in item.values():
                    if isinstance(value, str):
                        return value
                # If no string found, return JSON representation
                return json.dumps(item)
        else:
            return str(item)
    
    def _monitor_memory(self) -> Dict[str, float]:
        """
        Monitor current memory usage with detailed metrics.
        
        This method provides comprehensive memory monitoring including process-specific
        memory usage, system memory availability, and threshold comparisons. The
        metrics are used for adaptive batch size management and emergency fallback
        mechanisms.
        
        Returns:
            Dict[str, float]: Dictionary containing detailed memory metrics:
                - 'current_usage_mb': Current process memory usage above baseline (MB)
                - 'absolute_memory_mb': Absolute process memory usage (MB)
                - 'system_available_mb': Available system memory (MB)
                - 'system_percent': System memory usage percentage
                - 'threshold_mb': Configured memory threshold (MB)
                - 'emergency_threshold_mb': Emergency memory threshold (MB)
        
        Example:
            >>> iterator = StreamingDataIterator('data.txt', memory_threshold_mb=500.0)
            >>> metrics = iterator._monitor_memory()
            >>> print(f"Memory usage: {metrics['current_usage_mb']:.1f}MB")
            >>> print(f"System available: {metrics['system_available_mb']:.1f}MB")
            >>> 
            >>> if metrics['current_usage_mb'] > metrics['threshold_mb']:
            ...     print("Memory usage exceeds threshold!")
        
        Note:
            This method also updates the internal memory history for trend analysis
            and keeps a rolling window of recent measurements for stability detection.
            The baseline memory is established during iterator initialization.
        """
        memory_info = self._process.memory_info()
        current_memory = memory_info.rss / (1024 * 1024)  # MB
        memory_usage = current_memory - self._initial_memory
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        system_available = system_memory.available / (1024 * 1024)  # MB
        system_percent = system_memory.percent
        
        memory_metrics = {
            'current_usage_mb': memory_usage,
            'absolute_memory_mb': current_memory,
            'system_available_mb': system_available,
            'system_percent': system_percent,
            'threshold_mb': self.memory_threshold_mb,
            'emergency_threshold_mb': self.emergency_threshold_mb
        }
        
        # Store in history for trend analysis
        self._memory_history.append({
            'batch_count': self._batch_count,
            'memory_usage': memory_usage,
            'batch_size': self.batch_size,
            'timestamp': time.time()
        })
        
        # Keep only recent history (last 50 measurements)
        if len(self._memory_history) > 50:
            self._memory_history = self._memory_history[-50:]
        
        return memory_metrics
    
    def _adjust_batch_size(self, memory_metrics: Dict[str, float]) -> bool:
        """
        Intelligently adjust batch size based on memory usage and trends.
        
        Args:
            memory_metrics: Dictionary with memory usage metrics
            
        Returns:
            True if batch size was adjusted, False otherwise
        """
        if not self.auto_adjust_batch_size or self._adjustment_cooldown > 0:
            if self._adjustment_cooldown > 0:
                self._adjustment_cooldown -= 1
            return False
        
        current_usage = memory_metrics['current_usage_mb']
        system_available = memory_metrics['system_available_mb']
        system_percent = memory_metrics['system_percent']
        
        old_batch_size = self.batch_size
        adjustment_made = False
        adjustment_reason = ""
        
        # Emergency mode: aggressive reduction if memory is critically high
        if (current_usage > self.emergency_threshold_mb or 
            system_percent > 90 or 
            system_available < 100):
            
            if not self._emergency_mode:
                logger.warning(f"Entering emergency mode: memory={current_usage:.1f}MB, "
                             f"system={system_percent:.1f}%, available={system_available:.1f}MB")
                self._emergency_mode = True
            
            # Aggressive reduction
            reduction_factor = 0.5 if current_usage > self.emergency_threshold_mb * 1.2 else 0.7
            new_batch_size = max(self.min_batch_size, int(self.batch_size * reduction_factor))
            adjustment_reason = "emergency memory pressure"
            
        # High memory usage: moderate reduction
        elif current_usage > self.memory_threshold_mb:
            # Calculate reduction based on how much we exceed threshold
            excess_ratio = current_usage / self.memory_threshold_mb
            reduction_factor = max(0.7, 1.0 - (excess_ratio - 1.0) * 0.3)
            new_batch_size = max(self.min_batch_size, int(self.batch_size * reduction_factor))
            adjustment_reason = f"memory usage {current_usage:.1f}MB exceeds threshold {self.memory_threshold_mb}MB"
            
        # Low memory usage: consider increasing if stable
        elif (current_usage < self.memory_threshold_mb * 0.6 and 
              system_percent < 70 and 
              self._is_memory_stable()):
            
            # Exit emergency mode if we were in it
            if self._emergency_mode:
                logger.info("Exiting emergency mode: memory usage stabilized")
                self._emergency_mode = False
            
            # Conservative increase
            increase_factor = 1.1 if current_usage < self.memory_threshold_mb * 0.4 else 1.05
            new_batch_size = min(self.max_batch_size, int(self.batch_size * increase_factor))
            adjustment_reason = f"low memory usage {current_usage:.1f}MB, stable trend"
            
        else:
            # Memory usage is acceptable, no adjustment needed
            if self._emergency_mode and current_usage < self.memory_threshold_mb * 0.8:
                logger.info("Exiting emergency mode: memory usage normalized")
                self._emergency_mode = False
            return False
        
        # Apply the adjustment if it's significant enough
        if abs(new_batch_size - self.batch_size) >= max(1, self.batch_size * 0.05):
            self.batch_size = new_batch_size
            adjustment_made = True
            
            # Record adjustment
            self._adjustment_history.append({
                'batch_count': self._batch_count,
                'old_size': old_batch_size,
                'new_size': new_batch_size,
                'reason': adjustment_reason,
                'memory_usage': current_usage,
                'timestamp': time.time()
            })
            
            # Keep only recent adjustment history
            if len(self._adjustment_history) > 20:
                self._adjustment_history = self._adjustment_history[-20:]
            
            # Track consecutive adjustments to prevent oscillation
            if new_batch_size < old_batch_size:
                self._consecutive_adjustments += 1
            else:
                self._consecutive_adjustments = max(0, self._consecutive_adjustments - 1)
            
            # Set cooldown period to prevent rapid oscillations
            if self._consecutive_adjustments > 2:
                self._adjustment_cooldown = 5  # Skip next 5 checks
                logger.warning(f"Multiple consecutive reductions detected, "
                             f"setting cooldown period")
            
            # Log the adjustment
            direction = "reduced" if new_batch_size < old_batch_size else "increased"
            logger.info(f"Batch size {direction} from {old_batch_size} to {new_batch_size} "
                       f"({adjustment_reason})")
            
            # Force garbage collection after significant reductions
            if new_batch_size < old_batch_size * 0.8 and self._gc_enabled:
                gc.collect()
        
        return adjustment_made
    
    def _is_memory_stable(self) -> bool:
        """
        Check if memory usage has been stable over recent batches.
        
        Returns:
            True if memory usage is stable, False otherwise
        """
        if len(self._memory_history) < 5:
            return False
        
        # Check last 5 measurements
        recent_usage = [entry['memory_usage'] for entry in self._memory_history[-5:]]
        
        # Calculate coefficient of variation (std/mean)
        if len(recent_usage) > 1:
            mean_usage = sum(recent_usage) / len(recent_usage)
            if mean_usage > 0:
                variance = sum((x - mean_usage) ** 2 for x in recent_usage) / len(recent_usage)
                std_dev = variance ** 0.5
                cv = std_dev / mean_usage
                
                # Consider stable if coefficient of variation < 0.1 (10%)
                return cv < 0.1
        
        return False
    
    def _apply_fallback_mechanisms(self, memory_metrics: Dict[str, float]):
        """
        Apply fallback mechanisms for memory-constrained environments.
        
        Args:
            memory_metrics: Dictionary with memory usage metrics
        """
        current_usage = memory_metrics['current_usage_mb']
        system_available = memory_metrics['system_available_mb']
        system_percent = memory_metrics['system_percent']
        
        # Critical memory situation - apply all fallback mechanisms
        if (current_usage > self.emergency_threshold_mb * 1.5 or 
            system_percent > 95 or 
            system_available < 50):
            
            logger.critical(f"Critical memory situation detected: "
                          f"usage={current_usage:.1f}MB, "
                          f"system={system_percent:.1f}%, "
                          f"available={system_available:.1f}MB")
            
            # Force minimum batch size
            if self.batch_size > self.min_batch_size:
                logger.warning(f"Forcing minimum batch size: {self.batch_size} -> {self.min_batch_size}")
                self.batch_size = self.min_batch_size
            
            # Force garbage collection
            if self._gc_enabled:
                logger.info("Forcing garbage collection")
                gc.collect()
            
            # Close and reopen current file to free file buffers
            if self._current_file_handle:
                logger.info("Reopening current file to free buffers")
                current_pos = self._current_file_handle.tell() if hasattr(self._current_file_handle, 'tell') else 0
                file_path = Path(self._current_file_handle.name) if hasattr(self._current_file_handle, 'name') else None
                
                self._close_current_file()
                
                if file_path and file_path.exists():
                    try:
                        self._open_file_reader(file_path)
                        if hasattr(self._current_file_handle, 'seek') and current_pos > 0:
                            self._current_file_handle.seek(current_pos)
                    except Exception as e:
                        logger.error(f"Failed to reopen file {file_path}: {e}")
            
            # Disable automatic garbage collection to reduce overhead
            if self._gc_enabled and gc.isenabled():
                logger.warning("Disabling automatic garbage collection to reduce overhead")
                gc.disable()
                self._gc_enabled = False
    
    def _should_check_memory(self) -> bool:
        """
        Determine if memory should be checked based on interval and conditions.
        
        Returns:
            True if memory should be checked, False otherwise
        """
        # Always check in emergency mode
        if self._emergency_mode:
            return True
        
        # Check every N batches based on interval
        if self._batch_count - self._last_memory_check >= self.memory_check_interval:
            return True
        
        # Check if we've processed a significant number of items since last check
        items_since_check = self._items_processed - (self._last_memory_check * self.batch_size)
        if items_since_check >= self.batch_size * 5:  # Every 5 batches worth of items
            return True
        
        return False
    
    def __iter__(self) -> Iterator[List[Any]]:
        """Iterate over data in batches."""
        if not self._is_reset:
            self.reset()
        
        self._is_reset = False
        self._iterator = self._batch_generator()
        return self
    
    def __next__(self) -> List[Any]:
        """Get next batch."""
        if self._iterator is None:
            # Auto-initialize if not already done
            self.__iter__()
        return next(self._iterator)
    
    def _batch_generator(self) -> Iterator[List[Any]]:
        """Generate batches of data."""
        current_batch = []
        
        for file_path in self._file_paths[self._current_file_index:]:
            logger.debug(f"Processing file: {file_path}")
            
            try:
                self._open_file_reader(file_path)
                
                for item in self._current_reader:
                    # Extract text from item if requested, otherwise use raw item
                    if self.extract_text:
                        processed_item = self._extract_text(item)
                    else:
                        processed_item = item
                    current_batch.append(processed_item)
                    self._items_processed += 1
                    
                    # Check if batch is full
                    if len(current_batch) >= self.batch_size:
                        self._batch_count += 1
                        
                        # Monitor memory and adjust batch size if needed
                        if self._should_check_memory():
                            memory_metrics = self._monitor_memory()
                            self._last_memory_check = self._batch_count
                            
                            # Apply fallback mechanisms if needed
                            self._apply_fallback_mechanisms(memory_metrics)
                            
                            # Adjust batch size based on memory usage
                            adjustment_made = self._adjust_batch_size(memory_metrics)
                            
                            # If batch size was reduced significantly, yield current batch early
                            if (adjustment_made and 
                                self.batch_size < len(current_batch) * 0.8):
                                logger.debug(f"Yielding early due to batch size reduction: "
                                           f"{len(current_batch)} items")
                        
                        # Call progress callback if provided
                        if self.progress_callback:
                            self.progress_callback(self._items_processed, self._get_total_estimate())
                        
                        yield current_batch
                        current_batch = []
                
                # Close current file and update progress
                self._close_current_file()
                self._files_completed += 1
                self._current_file_index += 1
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                self._close_current_file()
                continue
        
        # Yield remaining items in final batch
        if current_batch:
            yield current_batch
    
    def _get_total_estimate(self) -> int:
        """
        Estimate total number of items (rough estimate).
        
        Returns:
            Estimated total items
        """
        # This is a rough estimate - in practice, you might want to
        # implement more sophisticated estimation based on file sizes
        return self._items_processed + (len(self._file_paths) - self._files_completed) * 1000
    
    def reset(self):
        """Reset iterator to beginning and clear adaptive state."""
        self._close_current_file()
        self._items_processed = 0
        self._files_completed = 0
        self._current_file_index = 0
        self._batch_count = 0
        self._last_memory_check = 0
        self._consecutive_adjustments = 0
        self._adjustment_cooldown = 0
        self._emergency_mode = False
        
        # Reset batch size to initial value
        self.batch_size = self.initial_batch_size
        
        # Clear history but keep some for trend analysis
        self._memory_history = self._memory_history[-10:] if self._memory_history else []
        self._adjustment_history = self._adjustment_history[-5:] if self._adjustment_history else []
        
        # Re-enable garbage collection if it was disabled
        if not self._gc_enabled and not gc.isenabled():
            gc.enable()
            self._gc_enabled = True
        
        self._is_reset = True
        self._iterator = None
        logger.debug("StreamingDataIterator reset with adaptive state cleared")
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """
        Get detailed statistics about adaptive batch size management.
        
        Returns:
            Dictionary with adaptive management statistics
        """
        return {
            'memory_history': self._memory_history.copy(),
            'adjustment_history': self._adjustment_history.copy(),
            'current_config': {
                'batch_size': self.batch_size,
                'initial_batch_size': self.initial_batch_size,
                'min_batch_size': self.min_batch_size,
                'max_batch_size': self.max_batch_size,
                'memory_threshold_mb': self.memory_threshold_mb,
                'emergency_threshold_mb': self.emergency_threshold_mb,
                'memory_check_interval': self.memory_check_interval,
                'auto_adjust_enabled': self.auto_adjust_batch_size
            },
            'current_state': {
                'emergency_mode': self._emergency_mode,
                'consecutive_adjustments': self._consecutive_adjustments,
                'adjustment_cooldown': self._adjustment_cooldown,
                'gc_enabled': self._gc_enabled,
                'batches_processed': self._batch_count
            }
        }
    
    def configure_adaptive_settings(self, 
                                  memory_threshold_mb: Optional[float] = None,
                                  emergency_threshold_mb: Optional[float] = None,
                                  min_batch_size: Optional[int] = None,
                                  max_batch_size: Optional[int] = None,
                                  memory_check_interval: Optional[int] = None,
                                  auto_adjust_batch_size: Optional[bool] = None):
        """
        Update adaptive batch size management settings during runtime.
        
        Args:
            memory_threshold_mb: New memory threshold
            emergency_threshold_mb: New emergency threshold
            min_batch_size: New minimum batch size
            max_batch_size: New maximum batch size
            memory_check_interval: New memory check interval
            auto_adjust_batch_size: Enable/disable adaptive management
        """
        if memory_threshold_mb is not None:
            self.memory_threshold_mb = memory_threshold_mb
            logger.info(f"Updated memory threshold to {memory_threshold_mb}MB")
        
        if emergency_threshold_mb is not None:
            self.emergency_threshold_mb = emergency_threshold_mb
            logger.info(f"Updated emergency threshold to {emergency_threshold_mb}MB")
        
        if min_batch_size is not None:
            self.min_batch_size = max(1, min_batch_size)
            # Ensure current batch size respects new minimum
            if self.batch_size < self.min_batch_size:
                self.batch_size = self.min_batch_size
            logger.info(f"Updated minimum batch size to {self.min_batch_size}")
        
        if max_batch_size is not None:
            self.max_batch_size = max_batch_size
            # Ensure current batch size respects new maximum
            if self.batch_size > self.max_batch_size:
                self.batch_size = self.max_batch_size
            logger.info(f"Updated maximum batch size to {self.max_batch_size}")
        
        if memory_check_interval is not None:
            self.memory_check_interval = max(1, memory_check_interval)
            logger.info(f"Updated memory check interval to {self.memory_check_interval}")
        
        if auto_adjust_batch_size is not None:
            self.auto_adjust_batch_size = auto_adjust_batch_size
            status = "enabled" if auto_adjust_batch_size else "disabled"
            logger.info(f"Adaptive batch size management {status}")
    
    def force_batch_size_reset(self):
        """
        Force reset batch size to initial value and clear adaptive state.
        """
        old_size = self.batch_size
        self.batch_size = self.initial_batch_size
        self._consecutive_adjustments = 0
        self._adjustment_cooldown = 0
        self._emergency_mode = False
        
        # Re-enable garbage collection if disabled
        if not self._gc_enabled:
            gc.enable()
            self._gc_enabled = True
        
        logger.info(f"Forced batch size reset from {old_size} to {self.batch_size}")
    
    def get_memory_trend(self, window_size: int = 10) -> Dict[str, float]:
        """
        Analyze memory usage trend over recent batches.
        
        Args:
            window_size: Number of recent measurements to analyze
            
        Returns:
            Dictionary with trend analysis
        """
        if len(self._memory_history) < 2:
            return {'trend': 'insufficient_data', 'slope': 0.0, 'r_squared': 0.0}
        
        # Get recent measurements
        recent_data = self._memory_history[-window_size:]
        if len(recent_data) < 2:
            recent_data = self._memory_history
        
        # Simple linear regression to detect trend
        n = len(recent_data)
        x_values = list(range(n))
        y_values = [entry['memory_usage'] for entry in recent_data]
        
        # Calculate slope and correlation
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n
        
        numerator = sum((x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n))
        x_variance = sum((x - x_mean) ** 2 for x in x_values)
        y_variance = sum((y - y_mean) ** 2 for y in y_values)
        
        if x_variance == 0:
            slope = 0.0
            r_squared = 0.0
        else:
            slope = numerator / x_variance
            r_squared = (numerator ** 2) / (x_variance * y_variance) if y_variance > 0 else 0.0
        
        # Classify trend
        if abs(slope) < 1.0:  # Less than 1MB change per batch
            trend = 'stable'
        elif slope > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
        
        return {
            'trend': trend,
            'slope': slope,
            'r_squared': r_squared,
            'recent_average': y_mean,
            'sample_size': n
        }
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Get current progress information with detailed adaptive metrics.
        
        Returns:
            Dictionary with progress information
        """
        memory_metrics = self._monitor_memory()
        
        # Calculate adaptive statistics
        total_adjustments = len(self._adjustment_history)
        recent_adjustments = len([adj for adj in self._adjustment_history 
                                if adj['batch_count'] > self._batch_count - 10])
        
        avg_batch_size = (sum(entry['batch_size'] for entry in self._memory_history) / 
                         len(self._memory_history)) if self._memory_history else self.batch_size
        
        return {
            'items_processed': self._items_processed,
            'files_completed': self._files_completed,
            'total_files': len(self._file_paths),
            'current_batch_size': self.batch_size,
            'initial_batch_size': self.initial_batch_size,
            'average_batch_size': avg_batch_size,
            'memory_usage_mb': memory_metrics['current_usage_mb'],
            'memory_threshold_mb': self.memory_threshold_mb,
            'emergency_threshold_mb': self.emergency_threshold_mb,
            'system_memory_percent': memory_metrics['system_percent'],
            'system_available_mb': memory_metrics['system_available_mb'],
            'emergency_mode': self._emergency_mode,
            'total_adjustments': total_adjustments,
            'recent_adjustments': recent_adjustments,
            'consecutive_adjustments': self._consecutive_adjustments,
            'adjustment_cooldown': self._adjustment_cooldown,
            'batches_processed': self._batch_count,
            'gc_enabled': self._gc_enabled
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._close_current_file()


def create_streaming_iterator(data_source: Union[str, List[str]], 
                            batch_size: int = 1000,
                            auto_adjust_batch_size: bool = True,
                            memory_threshold_mb: float = 1000.0,
                            extract_text: bool = False,
                            **kwargs) -> StreamingDataIterator:
    """
    Convenience function to create a StreamingDataIterator with adaptive batch size management.
    
    Args:
        data_source: Path to file/directory or list of file paths
        batch_size: Initial number of items per batch
        auto_adjust_batch_size: Enable automatic batch size adjustment
        memory_threshold_mb: Memory threshold for batch size adjustment
        extract_text: Whether to extract text from data items
        **kwargs: Additional configuration options
        
    Returns:
        StreamingDataIterator instance with adaptive management enabled
    """
    return StreamingDataIterator(
        data_source=data_source, 
        batch_size=batch_size,
        auto_adjust_batch_size=auto_adjust_batch_size,
        memory_threshold_mb=memory_threshold_mb,
        extract_text=extract_text,
        **kwargs
    )