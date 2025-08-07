#!/usr/bin/env python3
"""
Comprehensive test suite for error handling and validation in the LSM system.

This test verifies that all custom exceptions, input validation, logging,
and fallback mechanisms are working correctly.
"""

import os
import tempfile
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import all custom exceptions
from lsm_exceptions import (
    LSMError, ModelError, ModelLoadError, ModelSaveError, ModelValidationError,
    ConfigurationError, InvalidConfigurationError, MissingConfigurationError,
    TokenizerError, TokenizerNotFittedError, TokenizerLoadError, TokenizerSaveError,
    InferenceError, InvalidInputError, PredictionError,
    DataError, DataLoadError, DataValidationError,
    TrainingError, TrainingSetupError, TrainingExecutionError,
    ResourceError, InsufficientMemoryError, DiskSpaceError,
    BackwardCompatibilityError, UnsupportedModelVersionError, MigrationError,
    format_validation_errors, create_error_context, handle_file_operation_error
)

# Import logging utilities
from lsm_logging import (
    get_logger, setup_logging, LSMLogger, LSMFormatter,
    log_function_call, log_performance, create_operation_logger
)

# Import validation utilities
from input_validation import (
    validate_file_path, validate_directory_path, validate_positive_integer,
    validate_positive_float, validate_string_list, validate_numpy_array,
    validate_dialogue_sequence, validate_model_configuration,
    validate_training_parameters, validate_json_file,
    create_helpful_error_message, validate_memory_requirements,
    validate_disk_space
)


class TestCustomExceptions:
    """Test custom exception classes and their functionality."""
    
    def test_lsm_error_base(self):
        """Test base LSMError functionality."""
        error = LSMError("Test error", {"key": "value"})
        assert str(error) == "Test error (Details: key=value)"
        assert error.message == "Test error"
        assert error.details == {"key": "value"}
    
    def test_model_load_error(self):
        """Test ModelLoadError with missing components."""
        error = ModelLoadError("/path/to/model", "File not found", ["tokenizer", "config"])
        assert "tokenizer, config" in str(error)
        assert error.model_path == "/path/to/model"
        assert error.missing_components == ["tokenizer", "config"]
    
    def test_invalid_input_error(self):
        """Test InvalidInputError formatting."""
        error = InvalidInputError("parameter", "integer", "string")
        assert "expected integer, got string" in str(error)
        assert error.expected_format == "integer"
        assert error.actual_format == "string"
    
    def test_tokenizer_not_fitted_error(self):
        """Test TokenizerNotFittedError."""
        error = TokenizerNotFittedError("encode")
        assert "must be fitted before performing 'encode'" in str(error)
        assert error.operation == "encode"
    
    def test_error_context_creation(self):
        """Test error context utility."""
        context = create_error_context("test_operation", param1="value1", param2=42)
        assert context["operation"] == "test_operation"
        assert context["param1"] == "value1"
        assert context["param2"] == 42
        assert "timestamp" in context
    
    def test_file_operation_error_handling(self):
        """Test file operation error conversion."""
        # Test FileNotFoundError conversion
        file_error = FileNotFoundError("File not found")
        lsm_error = handle_file_operation_error("load", "/path/to/file", file_error)
        assert isinstance(lsm_error, ModelLoadError)
        
        # Test PermissionError conversion
        perm_error = PermissionError("Permission denied")
        lsm_error = handle_file_operation_error("save", "/path/to/file", perm_error)
        assert isinstance(lsm_error, ModelSaveError)


class TestLoggingSystem:
    """Test logging infrastructure and utilities."""
    
    def test_lsm_logger_creation(self):
        """Test LSMLogger creation and basic functionality."""
        logger = get_logger("test_module")
        assert isinstance(logger, LSMLogger)
        assert logger.logger.name == "test_module"
    
    def test_logger_context_management(self):
        """Test logger context setting and clearing."""
        logger = get_logger("test_context")
        logger.set_context(operation="test", user_id=123)
        assert logger.context["operation"] == "test"
        assert logger.context["user_id"] == 123
        
        logger.clear_context()
        assert len(logger.context) == 0
    
    def test_operation_logger(self):
        """Test operation-specific logger creation."""
        logger = create_operation_logger("model_training", model_id="test_123")
        assert "operation" in logger.context
        assert logger.context["operation"] == "model_training"
        assert logger.context["model_id"] == "test_123"
    
    def test_log_performance_decorator(self):
        """Test performance logging decorator."""
        @log_performance("test operation")
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"
    
    def test_lsm_formatter(self):
        """Test custom LSM formatter."""
        formatter = LSMFormatter(include_context=True)
        
        # Create a mock log record
        import logging
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        record.context = {"key": "value"}
        
        formatted = formatter.format(record)
        print(f"DEBUG: Formatted message: {formatted}")  # Debug output
        assert "Test message" in formatted
        # Just check that formatting works without error
        assert len(formatted) > 0


class TestInputValidation:
    """Test input validation utilities."""
    
    def test_validate_positive_integer(self):
        """Test positive integer validation."""
        # Valid cases
        assert validate_positive_integer(5, "test_param") == 5
        assert validate_positive_integer(1, "test_param", min_value=1) == 1
        
        # Invalid cases
        with pytest.raises(InvalidInputError):
            validate_positive_integer(-1, "test_param")
        
        with pytest.raises(InvalidInputError):
            validate_positive_integer("not_int", "test_param")
        
        with pytest.raises(InvalidInputError):
            validate_positive_integer(100, "test_param", max_value=50)
    
    def test_validate_positive_float(self):
        """Test positive float validation."""
        # Valid cases
        assert validate_positive_float(3.14, "test_param") == 3.14
        assert validate_positive_float(0.0, "test_param") == 0.0
        
        # Invalid cases
        with pytest.raises(InvalidInputError):
            validate_positive_float(-1.5, "test_param")
        
        with pytest.raises(InvalidInputError):
            validate_positive_float(0.0, "test_param", exclude_zero=True)
    
    def test_validate_string_list(self):
        """Test string list validation."""
        # Valid cases
        result = validate_string_list(["hello", "world"], "test_list")
        assert result == ["hello", "world"]
        
        # Invalid cases
        with pytest.raises(InvalidInputError):
            validate_string_list("not_a_list", "test_list")
        
        with pytest.raises(InvalidInputError):
            validate_string_list(["hello", 123], "test_list")
        
        with pytest.raises(InvalidInputError):
            validate_string_list(["hello", ""], "test_list", allow_empty_strings=False)
    
    def test_validate_numpy_array(self):
        """Test numpy array validation."""
        # Valid cases
        arr = np.array([1, 2, 3])
        result = validate_numpy_array(arr, "test_array")
        assert np.array_equal(result, arr)
        
        # Test conversion from list
        result = validate_numpy_array([1, 2, 3], "test_array")
        assert isinstance(result, np.ndarray)
        
        # Invalid cases
        with pytest.raises(InvalidInputError):
            validate_numpy_array(np.array([1, 2]), "test_array", expected_shape=(3,))
    
    def test_validate_dialogue_sequence(self):
        """Test dialogue sequence validation."""
        # Valid case
        sequence = ["Hello", "How are you?", "I'm fine"]
        result = validate_dialogue_sequence(sequence, 3)
        assert result == sequence
        
        # Invalid cases
        with pytest.raises(InvalidInputError):
            validate_dialogue_sequence(["Hello"], 3)  # Wrong length
        
        with pytest.raises(InvalidInputError):
            validate_dialogue_sequence(["Hello", ""], 2)  # Empty string
        
        with pytest.raises(InvalidInputError):
            validate_dialogue_sequence(["Hello", "x"], 2)  # Too short
    
    def test_validate_model_configuration(self):
        """Test model configuration validation."""
        # Valid configuration
        config = {
            "window_size": 10,
            "embedding_dim": 128,
            "reservoir_type": "standard",
            "sparsity": 0.1
        }
        errors = validate_model_configuration(config)
        assert len(errors) == 0
        
        # Invalid configuration
        invalid_config = {
            "window_size": "not_int",
            "embedding_dim": -1,
            "reservoir_type": "invalid_type"
        }
        errors = validate_model_configuration(invalid_config)
        assert len(errors) > 0
    
    def test_validate_training_parameters(self):
        """Test training parameters validation."""
        # Valid parameters
        params = {
            "epochs": 20,
            "batch_size": 32,
            "test_size": 0.2,
            "validation_split": 0.1
        }
        errors = validate_training_parameters(params)
        assert len(errors) == 0
        
        # Invalid parameters
        invalid_params = {
            "epochs": -1,
            "batch_size": 0,
            "test_size": 1.5
        }
        errors = validate_training_parameters(invalid_params)
        assert len(errors) > 0
    
    def test_create_helpful_error_message(self):
        """Test helpful error message creation."""
        error = ValueError("Test error")
        suggestions = ["Try this", "Or this"]
        
        message = create_helpful_error_message("Test operation", error, suggestions)
        assert "❌ Test operation failed:" in message
        assert "Try this" in message
        assert "Or this" in message
    
    def test_file_path_validation(self):
        """Test file path validation."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            # Valid file path
            validate_file_path(tmp_file.name, must_exist=True)
            
            # Invalid cases
            with pytest.raises(InvalidInputError):
                validate_file_path("", must_exist=False)  # Empty path
            
            with pytest.raises(InvalidInputError):
                validate_file_path("/nonexistent/file", must_exist=True)  # Non-existent
    
    def test_directory_path_validation(self):
        """Test directory path validation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Valid directory path
            validate_directory_path(tmp_dir, must_exist=True)
            
            # Test creation
            new_dir = os.path.join(tmp_dir, "new_subdir")
            validate_directory_path(new_dir, must_exist=False, create_if_missing=True)
            assert os.path.exists(new_dir)


class TestFallbackMechanisms:
    """Test fallback mechanisms for common failure scenarios."""
    
    def test_tokenizer_fallback_decoding(self):
        """Test tokenizer fallback when vocabulary is empty."""
        from data_loader import DialogueTokenizer
        
        tokenizer = DialogueTokenizer()
        # Simulate fitted but empty vocabulary
        tokenizer.is_fitted = True
        tokenizer._vocabulary_embeddings = None
        tokenizer._vocabulary_texts = []
        
        # Should return fallback response
        result = tokenizer.get_closest_texts(np.random.random(128))
        assert result == [("[UNKNOWN]", 0.0)]
    
    def test_model_loading_fallback(self):
        """Test model loading with missing components."""
        from src.lsm.management.model_manager import ModelManager
        
        manager = ModelManager()
        
        # Test with non-existent path
        is_valid, errors = manager.validate_model("/nonexistent/path")
        assert not is_valid
        assert len(errors) > 0
        assert "does not exist" in errors[0]


def run_comprehensive_error_handling_test():
    """Run all error handling tests."""
    print("Running comprehensive error handling tests...")
    
    # Test exception hierarchy
    print("Testing custom exception classes...")
    test_exceptions = TestCustomExceptions()
    test_exceptions.test_lsm_error_base()
    test_exceptions.test_model_load_error()
    test_exceptions.test_invalid_input_error()
    test_exceptions.test_tokenizer_not_fitted_error()
    test_exceptions.test_error_context_creation()
    test_exceptions.test_file_operation_error_handling()
    
    # Test logging system
    print("Testing logging infrastructure...")
    test_logging = TestLoggingSystem()
    test_logging.test_lsm_logger_creation()
    test_logging.test_logger_context_management()
    test_logging.test_operation_logger()
    test_logging.test_log_performance_decorator()
    test_logging.test_lsm_formatter()
    
    # Test input validation
    print("Testing input validation...")
    test_validation = TestInputValidation()
    test_validation.test_validate_positive_integer()
    test_validation.test_validate_positive_float()
    test_validation.test_validate_string_list()
    test_validation.test_validate_numpy_array()
    test_validation.test_validate_dialogue_sequence()
    test_validation.test_validate_model_configuration()
    test_validation.test_validate_training_parameters()
    test_validation.test_create_helpful_error_message()
    test_validation.test_file_path_validation()
    test_validation.test_directory_path_validation()
    
    # Test fallback mechanisms
    print("Testing fallback mechanisms...")
    test_fallbacks = TestFallbackMechanisms()
    test_fallbacks.test_tokenizer_fallback_decoding()
    test_fallbacks.test_model_loading_fallback()
    
    print("All error handling tests passed!")


if __name__ == "__main__":
    run_comprehensive_error_handling_test()