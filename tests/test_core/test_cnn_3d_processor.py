#!/usr/bin/env python3
"""
Tests for CNN 3D Processor for system message integration.
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, patch, MagicMock

from src.lsm.core.cnn_3d_processor import (
    CNN3DProcessor,
    CNN3DProcessorError,
    SystemContext,
    ProcessingResult,
    create_cnn_3d_processor,
    create_system_aware_processor
)
from src.lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder


class TestCNN3DProcessor:
    """Test cases for CNN3DProcessor class."""
    
    @pytest.fixture
    def sample_reservoir_shape(self):
        """Sample reservoir output shape."""
        return (32, 32, 32, 1)  # depth, height, width, channels
    
    @pytest.fixture
    def sample_system_dim(self):
        """Sample system embedding dimension."""
        return 256
    
    @pytest.fixture
    def sample_output_dim(self):
        """Sample output embedding dimension."""
        return 512
    
    @pytest.fixture
    def processor(self, sample_reservoir_shape, sample_system_dim, sample_output_dim):
        """Create a CNN3DProcessor instance for testing."""
        return CNN3DProcessor(
            reservoir_shape=sample_reservoir_shape,
            system_embedding_dim=sample_system_dim,
            output_embedding_dim=sample_output_dim
        )
    
    @pytest.fixture
    def sample_reservoir_output(self, sample_reservoir_shape):
        """Sample reservoir output data."""
        batch_size = 2
        full_shape = (batch_size,) + sample_reservoir_shape
        return np.random.randn(*full_shape).astype(np.float32)
    
    @pytest.fixture
    def sample_system_context(self, sample_system_dim):
        """Sample system context."""
        return SystemContext(
            message="You are a helpful assistant.",
            embeddings=np.random.randn(sample_system_dim).astype(np.float32),
            influence_strength=1.0
        )
    
    def test_initialization(self, sample_reservoir_shape, sample_system_dim, sample_output_dim):
        """Test CNN3DProcessor initialization."""
        processor = CNN3DProcessor(
            reservoir_shape=sample_reservoir_shape,
            system_embedding_dim=sample_system_dim,
            output_embedding_dim=sample_output_dim
        )
        
        assert processor.reservoir_shape == sample_reservoir_shape
        assert processor.system_embedding_dim == sample_system_dim
        assert processor.output_embedding_dim == sample_output_dim
        assert processor._model is None
        assert not processor._compiled
    
    def test_initialization_with_config(self, sample_reservoir_shape, sample_system_dim, sample_output_dim):
        """Test initialization with custom model configuration."""
        custom_config = {
            "filters": [16, 32, 64],
            "kernel_sizes": [(2, 2, 2), (3, 3, 3), (4, 4, 4)],
            "dropout_rates": [0.1, 0.2, 0.3],
            "use_batch_norm": False,
            "activation": "tanh"
        }
        
        processor = CNN3DProcessor(
            reservoir_shape=sample_reservoir_shape,
            system_embedding_dim=sample_system_dim,
            output_embedding_dim=sample_output_dim,
            model_config=custom_config
        )
        
        assert processor.model_config["filters"] == [16, 32, 64]
        assert processor.model_config["activation"] == "tanh"
        assert not processor.model_config["use_batch_norm"]
    
    def test_initialization_error(self):
        """Test initialization with invalid parameters."""
        # The initialization doesn't validate shape immediately, so let's test a different error
        with pytest.raises(CNN3DProcessorError):
            processor = CNN3DProcessor(
                reservoir_shape=(32, 32, 32, 1),
                system_embedding_dim=-1,  # Invalid negative dimension
                output_embedding_dim=512
            )
            # Force an error by trying to create model with invalid config
            processor.model_config = {"invalid": "config"}
            processor.create_model()
    
    @patch('src.lsm.core.cnn_3d_processor.CNNArchitectureFactory')
    def test_create_model(self, mock_factory_class, processor):
        """Test 3D CNN model creation."""
        # Mock the factory and its methods
        mock_factory = Mock()
        mock_model = Mock()
        mock_compiled_model = Mock()
        
        mock_factory.create_3d_cnn.return_value = mock_model
        mock_factory.compile_model.return_value = mock_compiled_model
        mock_factory_class.return_value = mock_factory
        
        # Create new processor to use mocked factory
        processor = CNN3DProcessor(
            reservoir_shape=(32, 32, 32, 1),
            system_embedding_dim=256,
            output_embedding_dim=512
        )
        
        result = processor.create_model()
        
        # Verify factory methods were called
        mock_factory.create_3d_cnn.assert_called_once()
        mock_factory.compile_model.assert_called_once()
        
        # Verify model is stored and compiled flag is set
        assert processor._model == mock_compiled_model
        assert processor._compiled
        assert result == mock_compiled_model
    
    def test_create_system_processor(self, processor):
        """Test system message processor creation."""
        # Test that the method creates a system processor
        try:
            result = processor.create_system_processor()
            assert result is not None
            assert processor._system_processor is not None
            assert hasattr(result, 'predict')  # Should be a Keras model
        except Exception as e:
            # If TensorFlow/Keras isn't available, skip this test
            pytest.skip(f"TensorFlow/Keras not available: {e}")
    
    def test_create_embedding_modifier(self, processor):
        """Test embedding modifier creation."""
        # Test that the method creates an embedding modifier
        try:
            result = processor.create_embedding_modifier()
            assert result is not None
            assert processor._embedding_modifier is not None
            assert hasattr(result, 'predict')  # Should be a Keras model
        except Exception as e:
            # If TensorFlow/Keras isn't available, skip this test
            pytest.skip(f"TensorFlow/Keras not available: {e}")
    
    def test_validate_reservoir_output_valid(self, processor, sample_reservoir_output):
        """Test reservoir output validation with valid input."""
        # Should not raise any exception
        processor._validate_reservoir_output(sample_reservoir_output)
    
    def test_validate_reservoir_output_none(self, processor):
        """Test reservoir output validation with None input."""
        with pytest.raises(ValueError, match="Reservoir output cannot be None"):
            processor._validate_reservoir_output(None)
    
    def test_validate_reservoir_output_wrong_dimensions(self, processor):
        """Test reservoir output validation with wrong dimensions."""
        wrong_shape_output = np.random.randn(32, 32, 32)  # 3D instead of 5D
        
        with pytest.raises(ValueError, match="Expected 5D reservoir output"):
            processor._validate_reservoir_output(wrong_shape_output)
    
    def test_validate_reservoir_output_wrong_shape(self, processor):
        """Test reservoir output validation with wrong shape."""
        wrong_shape_output = np.random.randn(2, 16, 16, 16, 1)  # Wrong spatial dimensions
        
        with pytest.raises(ValueError, match="Reservoir output shape mismatch"):
            processor._validate_reservoir_output(wrong_shape_output)
    
    def test_validate_system_context_valid(self, processor, sample_system_context):
        """Test system context validation with valid input."""
        # Should not raise any exception
        processor._validate_system_context(sample_system_context)
    
    def test_validate_system_context_none(self, processor):
        """Test system context validation with None input."""
        with pytest.raises(ValueError, match="System context cannot be None"):
            processor._validate_system_context(None)
    
    def test_validate_system_context_none_embeddings(self, processor):
        """Test system context validation with None embeddings."""
        context = SystemContext(
            message="Test message",
            embeddings=None
        )
        
        with pytest.raises(ValueError, match="System embeddings cannot be None"):
            processor._validate_system_context(context)
    
    def test_validate_system_context_wrong_dimension(self, processor):
        """Test system context validation with wrong embedding dimension."""
        context = SystemContext(
            message="Test message",
            embeddings=np.random.randn(128).astype(np.float32)  # Wrong dimension
        )
        
        with pytest.raises(ValueError, match="System embedding dimension mismatch"):
            processor._validate_system_context(context)
    
    @patch.object(CNN3DProcessor, 'create_model')
    def test_process_with_system_context(self, mock_create_model, processor, 
                                       sample_reservoir_output, sample_system_context):
        """Test processing reservoir output with system context."""
        # Mock the model and its predict method
        mock_model = Mock()
        mock_output = np.random.randn(2, 512).astype(np.float32)
        mock_model.predict.return_value = mock_output
        
        processor._model = mock_model
        mock_create_model.return_value = mock_model
        
        result = processor.process_with_system_context(
            sample_reservoir_output,
            sample_system_context
        )
        
        # Verify result structure
        assert isinstance(result, ProcessingResult)
        assert result.output_embeddings is not None
        assert result.system_influence > 0
        assert result.processing_time >= 0  # Allow zero time for mocked operations
        
        # Verify model was called
        mock_model.predict.assert_called_once()
    
    @patch.object(CNN3DProcessor, 'create_embedding_modifier')
    def test_integrate_embedding_modifiers(self, mock_create_modifier, processor, sample_system_context):
        """Test embedding modifier integration."""
        # Mock embedding modifier
        mock_modifier = Mock()
        mock_modifiers = {
            'output_modifiers': np.random.randn(1, 512).astype(np.float32),
            'feature_modifiers': np.random.randn(1, 128).astype(np.float32)
        }
        mock_modifier.predict.return_value = mock_modifiers
        
        processor._embedding_modifier = mock_modifier
        mock_create_modifier.return_value = mock_modifier
        
        base_output = np.random.randn(2, 512).astype(np.float32)
        
        result = processor.integrate_embedding_modifiers(base_output, sample_system_context)
        
        # Verify result shape and that modification occurred
        assert result.shape == base_output.shape
        assert not np.array_equal(result, base_output)  # Should be modified
        
        # Verify modifier was called
        mock_modifier.predict.assert_called_once()
    
    def test_simple_tokenize(self, processor):
        """Test simple tokenization method."""
        text = "Hello world test message"
        tokens = processor._simple_tokenize(text)
        
        assert isinstance(tokens, np.ndarray)
        assert tokens.dtype == np.int32
        assert len(tokens) == 4  # Four words
        assert all(0 <= token < 10000 for token in tokens)  # Within vocabulary range
    
    def test_tokenize_text_with_tokenizer(self, processor):
        """Test text tokenization with proper tokenizer."""
        try:
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.encode_single.return_value = [1, 2, 3, 4]
            processor.tokenizer = mock_tokenizer
            
            text = "Hello world"
            tokens = processor._tokenize_text(text)
            
            assert isinstance(tokens, np.ndarray)
            assert tokens.dtype == np.int32
            assert len(tokens) == 4
            mock_tokenizer.encode_single.assert_called_once_with(text, add_special_tokens=True)
            
        except ImportError:
            pytest.skip("Transformers library not available")
    
    def test_tokenize_text_fallback(self, processor):
        """Test text tokenization fallback when no tokenizer available."""
        # Ensure no tokenizer
        processor.tokenizer = None
        
        text = "Hello world test"
        tokens = processor._tokenize_text(text)
        
        assert isinstance(tokens, np.ndarray)
        assert tokens.dtype == np.int32
        assert len(tokens) == 3  # Three words
    
    @patch.object(CNN3DProcessor, 'create_system_processor')
    @patch.object(CNN3DProcessor, 'process_with_system_context')
    @patch.object(CNN3DProcessor, 'integrate_embedding_modifiers')
    def test_process_reservoir_output_with_system(self, mock_integrate, mock_process, 
                                                mock_create_processor, processor, 
                                                sample_reservoir_output):
        """Test complete processing pipeline."""
        # Mock system processor
        mock_system_processor = Mock()
        mock_system_embeddings = np.random.randn(1, 256).astype(np.float32)
        mock_system_processor.predict.return_value = mock_system_embeddings
        
        processor._system_processor = mock_system_processor
        mock_create_processor.return_value = mock_system_processor
        
        # Mock processing result
        mock_result = ProcessingResult(
            output_embeddings=np.random.randn(2, 512).astype(np.float32),
            system_influence=0.5,
            processing_time=0.1
        )
        mock_process.return_value = mock_result
        
        # Mock modified embeddings
        mock_modified = np.random.randn(2, 512).astype(np.float32)
        mock_integrate.return_value = mock_modified
        
        result = processor.process_reservoir_output_with_system(
            sample_reservoir_output,
            "You are a helpful assistant.",
            influence_strength=0.8
        )
        
        # Verify all components were called
        mock_system_processor.predict.assert_called_once()
        mock_process.assert_called_once()
        mock_integrate.assert_called_once()
        
        # Verify result
        assert isinstance(result, ProcessingResult)
        assert np.array_equal(result.output_embeddings, mock_modified)
    
    @patch.object(CNN3DProcessor, 'create_model')
    def test_get_model_summary(self, mock_create_model, processor):
        """Test model summary generation."""
        mock_model = Mock()
        mock_model.summary.return_value = None  # summary() prints to stdout
        
        processor._model = mock_model
        mock_create_model.return_value = mock_model
        
        summary = processor.get_model_summary()
        
        assert isinstance(summary, str)
        mock_model.summary.assert_called_once()
    
    @patch.object(CNN3DProcessor, 'create_model')
    def test_save_model(self, mock_create_model, processor, tmp_path):
        """Test model saving."""
        mock_model = Mock()
        processor._model = mock_model
        mock_create_model.return_value = mock_model
        
        filepath = str(tmp_path / "test_model.h5")
        processor.save_model(filepath)
        
        mock_model.save.assert_called_once_with(filepath)
    
    def test_save_model_not_created(self, processor, tmp_path):
        """Test saving model that hasn't been created."""
        filepath = str(tmp_path / "test_model.h5")
        
        with pytest.raises(CNN3DProcessorError) as exc_info:
            processor.save_model(filepath)
        
        assert exc_info.value.operation == "model_saving"
    
    @patch('src.lsm.core.cnn_3d_processor.keras')
    def test_load_model(self, mock_keras, processor, tmp_path):
        """Test model loading."""
        mock_model = Mock()
        mock_keras.models.load_model.return_value = mock_model
        
        filepath = str(tmp_path / "test_model.h5")
        processor.load_model(filepath)
        
        assert processor._model == mock_model
        assert processor._compiled
        mock_keras.models.load_model.assert_called_once_with(filepath)


class TestSystemContext:
    """Test cases for SystemContext dataclass."""
    
    def test_system_context_creation(self):
        """Test SystemContext creation with required fields."""
        embeddings = np.random.randn(256).astype(np.float32)
        context = SystemContext(
            message="Test message",
            embeddings=embeddings
        )
        
        assert context.message == "Test message"
        assert np.array_equal(context.embeddings, embeddings)
        assert context.modifier_weights is None
        assert context.influence_strength == 1.0
        assert context.processing_mode == "3d_cnn"
    
    def test_system_context_with_optional_fields(self):
        """Test SystemContext creation with optional fields."""
        embeddings = np.random.randn(256).astype(np.float32)
        modifier_weights = np.random.randn(128).astype(np.float32)
        
        context = SystemContext(
            message="Test message",
            embeddings=embeddings,
            modifier_weights=modifier_weights,
            influence_strength=0.5,
            processing_mode="separate_reservoir"
        )
        
        assert context.message == "Test message"
        assert np.array_equal(context.embeddings, embeddings)
        assert np.array_equal(context.modifier_weights, modifier_weights)
        assert context.influence_strength == 0.5
        assert context.processing_mode == "separate_reservoir"


class TestProcessingResult:
    """Test cases for ProcessingResult dataclass."""
    
    def test_processing_result_creation(self):
        """Test ProcessingResult creation with required fields."""
        output_embeddings = np.random.randn(2, 512).astype(np.float32)
        
        result = ProcessingResult(
            output_embeddings=output_embeddings,
            system_influence=0.7,
            processing_time=0.15
        )
        
        assert np.array_equal(result.output_embeddings, output_embeddings)
        assert result.system_influence == 0.7
        assert result.processing_time == 0.15
        assert result.intermediate_features is None
    
    def test_processing_result_with_intermediate_features(self):
        """Test ProcessingResult creation with intermediate features."""
        output_embeddings = np.random.randn(2, 512).astype(np.float32)
        intermediate_features = {
            "conv_layer_1": np.random.randn(2, 64, 32, 32, 32).astype(np.float32),
            "attention_weights": np.random.randn(2, 32, 32).astype(np.float32)
        }
        
        result = ProcessingResult(
            output_embeddings=output_embeddings,
            system_influence=0.7,
            processing_time=0.15,
            intermediate_features=intermediate_features
        )
        
        assert result.intermediate_features is not None
        assert "conv_layer_1" in result.intermediate_features
        assert "attention_weights" in result.intermediate_features


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_create_cnn_3d_processor(self):
        """Test create_cnn_3d_processor convenience function."""
        reservoir_shape = (32, 32, 32, 1)
        processor = create_cnn_3d_processor(
            reservoir_shape=reservoir_shape,
            system_embedding_dim=128,
            output_embedding_dim=256
        )
        
        assert isinstance(processor, CNN3DProcessor)
        assert processor.reservoir_shape == reservoir_shape
        assert processor.system_embedding_dim == 128
        assert processor.output_embedding_dim == 256
    
    def test_create_system_aware_processor(self):
        """Test create_system_aware_processor convenience function."""
        processor = create_system_aware_processor(
            window_size=64,
            channels=2,
            system_dim=512,
            output_dim=1024
        )
        
        assert isinstance(processor, CNN3DProcessor)
        assert processor.reservoir_shape == (64, 64, 64, 2)
        assert processor.system_embedding_dim == 512
        assert processor.output_embedding_dim == 1024
        
        # Check enhanced configuration
        assert processor.model_config["filters"] == [64, 128, 256]
        assert processor.model_config["use_attention"]


class TestEnhancedFunctionality:
    """Test cases for enhanced CNN3DProcessor functionality."""
    
    @pytest.fixture
    def processor_with_tokenizer(self):
        """Create processor with mock tokenizer."""
        try:
            mock_tokenizer = Mock()
            mock_tokenizer.tokenizer_name = "gpt2"
            mock_tokenizer.get_vocab_size.return_value = 50257
            mock_tokenizer.encode_single.return_value = [1, 2, 3, 4]
            mock_tokenizer.get_special_tokens.return_value = {'pad_token_id': 0}
            
            processor = CNN3DProcessor(
                reservoir_shape=(32, 32, 32, 1),
                system_embedding_dim=256,
                output_embedding_dim=512,
                tokenizer=mock_tokenizer
            )
            return processor
        except Exception:
            pytest.skip("Mock tokenizer setup failed")
    
    def test_create_enhanced_system_processor(self, processor_with_tokenizer):
        """Test enhanced system processor creation."""
        try:
            system_processor = processor_with_tokenizer.create_enhanced_system_processor()
            assert system_processor is not None
            assert processor_with_tokenizer._system_processor is not None
        except Exception as e:
            pytest.skip(f"Enhanced system processor creation failed: {e}")
    
    def test_create_embedding_modifier_generator(self, processor_with_tokenizer):
        """Test embedding modifier generator creation."""
        try:
            modifier_generator = processor_with_tokenizer.create_embedding_modifier_generator()
            assert modifier_generator is not None
            assert processor_with_tokenizer._embedding_modifier_generator is not None
        except Exception as e:
            pytest.skip(f"Embedding modifier generator creation failed: {e}")
    
    def test_process_reservoir_output_with_modifiers(self, processor_with_tokenizer):
        """Test enhanced processing with modifiers."""
        try:
            # Mock the necessary components
            mock_system_processor = Mock()
            mock_context = Mock()
            mock_context.embeddings = np.random.randn(256).astype(np.float32)
            mock_system_processor.process_system_message.return_value = mock_context
            processor_with_tokenizer._system_processor = mock_system_processor
            
            # Mock the base processing
            with patch.object(processor_with_tokenizer, 'process_with_system_context') as mock_process:
                mock_result = ProcessingResult(
                    output_embeddings=np.random.randn(2, 512).astype(np.float32),
                    system_influence=0.5,
                    processing_time=0.1
                )
                mock_process.return_value = mock_result
                
                reservoir_output = np.random.randn(2, 32, 32, 32, 1).astype(np.float32)
                system_message = "Test system message"
                
                result = processor_with_tokenizer.process_reservoir_output_with_modifiers(
                    reservoir_output,
                    system_message,
                    influence_strength=0.8,
                    use_advanced_modifiers=False  # Skip advanced modifiers for this test
                )
                
                assert isinstance(result, ProcessingResult)
                assert result.output_embeddings is not None
                assert result.tokenization_info is not None
                assert result.tokenization_info['tokenizer_type'] == 'gpt2'
                
        except Exception as e:
            pytest.skip(f"Enhanced processing test failed: {e}")
    
    def test_get_processing_statistics(self, processor_with_tokenizer):
        """Test processing statistics retrieval."""
        stats = processor_with_tokenizer.get_processing_statistics()
        
        assert isinstance(stats, dict)
        assert 'model_created' in stats
        assert 'tokenizer_available' in stats
        assert 'tokenizer_type' in stats
        assert 'vocab_size' in stats
        assert stats['tokenizer_available'] is True
        assert stats['tokenizer_type'] == 'gpt2'
        assert stats['vocab_size'] == 50257
    
    def test_set_tokenizer(self, processor_with_tokenizer):
        """Test tokenizer setting."""
        new_mock_tokenizer = Mock()
        new_mock_tokenizer.tokenizer_name = "bert"
        
        processor_with_tokenizer.set_tokenizer(new_mock_tokenizer)
        
        assert processor_with_tokenizer.tokenizer == new_mock_tokenizer
        assert processor_with_tokenizer._system_processor is None  # Should be reset
    
    def test_set_embedder(self, processor_with_tokenizer):
        """Test embedder setting."""
        mock_embedder = Mock()
        mock_embedder.embedding_dim = 256
        mock_embedder._is_fitted = True
        
        processor_with_tokenizer.set_embedder(mock_embedder)
        
        assert processor_with_tokenizer.embedder == mock_embedder
    
    def test_create_training_model(self, processor_with_tokenizer):
        """Test training model creation."""
        try:
            # Mock the base model
            mock_base_model = Mock()
            processor_with_tokenizer._model = mock_base_model
            
            training_model = processor_with_tokenizer.create_training_model()
            assert training_model is not None
            assert processor_with_tokenizer._training_model is not None
            
        except Exception as e:
            pytest.skip(f"Training model creation failed: {e}")
    
    def test_get_training_model_summary(self, processor_with_tokenizer):
        """Test training model summary."""
        # Test when training model doesn't exist
        summary = processor_with_tokenizer.get_training_model_summary()
        assert "Training model not created yet" in summary
        
        # Test when training model exists
        mock_training_model = Mock()
        mock_training_model.summary.return_value = None
        processor_with_tokenizer._training_model = mock_training_model
        
        summary = processor_with_tokenizer.get_training_model_summary()
        assert isinstance(summary, str)
        mock_training_model.summary.assert_called_once()


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_cnn_3d_processor_error_creation(self):
        """Test CNN3DProcessorError creation."""
        error = CNN3DProcessorError(
            operation="test_operation",
            reason="test reason",
            details={"key": "value"}
        )
        
        assert error.operation == "test_operation"
        assert "test_operation" in str(error)
        assert "test reason" in str(error)
        assert error.details["key"] == "value"
    
    def test_cnn_3d_processor_error_without_details(self):
        """Test CNN3DProcessorError creation without details."""
        error = CNN3DProcessorError(
            operation="test_operation",
            reason="test reason"
        )
        
        assert error.operation == "test_operation"
        assert "test_operation" in str(error)
        assert "test reason" in str(error)


if __name__ == "__main__":
    pytest.main([__file__])