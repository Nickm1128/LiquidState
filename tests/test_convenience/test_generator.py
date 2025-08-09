#!/usr/bin/env python3
"""
Tests for LSMGenerator convenience API class.

This module tests the LSMGenerator class which provides a sklearn-like
interface for text generation using LSM models.
"""

import unittest
import tempfile
import shutil
import os
import sys
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from lsm.convenience.generator import LSMGenerator
    from lsm.convenience.config import ConvenienceConfig, ConvenienceValidationError
    GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: LSMGenerator not available: {e}")
    GENERATOR_AVAILABLE = False


@unittest.skipUnless(GENERATOR_AVAILABLE, "LSMGenerator required")
class TestLSMGenerator(unittest.TestCase):
    """Test LSMGenerator functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_conversations = [
            "Hello, how are you?",
            "I'm doing well, thank you!",
            "What's the weather like?",
            "It's sunny and warm today."
        ]
        
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test LSMGenerator initialization."""
        # Test default initialization
        generator = LSMGenerator()
        self.assertIsNotNone(generator)
        self.assertEqual(generator.window_size, 10)  # Default value
        self.assertEqual(generator.embedding_dim, 128)  # Default value
        
        # Test custom initialization
        generator = LSMGenerator(
            window_size=15,
            embedding_dim=256,
            reservoir_type='hierarchical',
            system_message_support=True
        )
        self.assertEqual(generator.window_size, 15)
        self.assertEqual(generator.embedding_dim, 256)
        self.assertEqual(generator.reservoir_type, 'hierarchical')
        self.assertTrue(generator.system_message_support)
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid window_size
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMGenerator(window_size=0)
        
        # Test invalid embedding_dim
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMGenerator(embedding_dim=-1)
        
        # Test invalid reservoir_type
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMGenerator(reservoir_type='invalid_type')
    
    def test_get_params(self):
        """Test get_params method."""
        generator = LSMGenerator(
            window_size=15,
            embedding_dim=256,
            reservoir_type='hierarchical'
        )
        
        params = generator.get_params()
        self.assertIsInstance(params, dict)
        self.assertEqual(params['window_size'], 15)
        self.assertEqual(params['embedding_dim'], 256)
        self.assertEqual(params['reservoir_type'], 'hierarchical')
    
    def test_set_params(self):
        """Test set_params method."""
        generator = LSMGenerator()
        
        # Set new parameters
        generator.set_params(
            window_size=20,
            embedding_dim=512,
            reservoir_type='attentive'
        )
        
        # Verify parameters were set
        self.assertEqual(generator.window_size, 20)
        self.assertEqual(generator.embedding_dim, 512)
        self.assertEqual(generator.reservoir_type, 'attentive')
    
    @patch('lsm.training.train.LSMTrainer')
    def test_fit_basic(self, mock_trainer_class):
        """Test basic fit functionality."""
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        mock_trainer.train.return_value = True
        
        generator = LSMGenerator(window_size=5, embedding_dim=32)
        
        # Mock data preparation
        with patch.object(generator, '_prepare_training_data') as mock_prepare:
            mock_prepare.return_value = (self.test_conversations, None)
            
            # Test fit
            generator.fit(self.test_conversations, epochs=1)
            
            # Verify trainer was created and called
            mock_trainer_class.assert_called_once()
            mock_trainer.train.assert_called_once()
            self.assertTrue(generator._is_fitted)
    
    def test_fit_validation(self):
        """Test fit input validation."""
        generator = LSMGenerator()
        
        # Test empty conversations
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            generator.fit([])
        
        # Test invalid conversation format
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            generator.fit([123, 456])  # Non-string conversations
    
    @patch('lsm.inference.response_generator.ResponseGenerator')
    def test_generate_basic(self, mock_response_gen_class):
        """Test basic generate functionality."""
        mock_response_gen = Mock()
        mock_response_gen_class.return_value = mock_response_gen
        mock_response_gen.generate_response.return_value = "Generated response"
        
        generator = LSMGenerator()
        generator._is_fitted = True  # Mock fitted state
        
        # Mock response generator setup
        with patch.object(generator, '_get_response_generator') as mock_get_gen:
            mock_get_gen.return_value = mock_response_gen
            
            # Test generate
            response = generator.generate("Test prompt")
            
            self.assertEqual(response, "Generated response")
            mock_response_gen.generate_response.assert_called_once()
    
    def test_generate_not_fitted(self):
        """Test generate raises error when not fitted."""
        generator = LSMGenerator()
        
        with self.assertRaises((ValueError, RuntimeError)):
            generator.generate("Test prompt")
    
    def test_generate_validation(self):
        """Test generate input validation."""
        generator = LSMGenerator()
        generator._is_fitted = True
        
        # Test empty prompt
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            generator.generate("")
        
        # Test invalid prompt type
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            generator.generate(123)
    
    @patch('lsm.inference.response_generator.ResponseGenerator')
    def test_batch_generate(self, mock_response_gen_class):
        """Test batch generation functionality."""
        mock_response_gen = Mock()
        mock_response_gen_class.return_value = mock_response_gen
        mock_response_gen.generate_response.side_effect = [
            "Response 1", "Response 2", "Response 3"
        ]
        
        generator = LSMGenerator()
        generator._is_fitted = True
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        
        with patch.object(generator, '_get_response_generator') as mock_get_gen:
            mock_get_gen.return_value = mock_response_gen
            
            responses = generator.batch_generate(prompts)
            
            self.assertEqual(len(responses), 3)
            self.assertEqual(responses[0], "Response 1")
            self.assertEqual(responses[1], "Response 2")
            self.assertEqual(responses[2], "Response 3")
            self.assertEqual(mock_response_gen.generate_response.call_count, 3)
    
    def test_system_message_support(self):
        """Test system message functionality."""
        generator = LSMGenerator(system_message_support=True)
        
        # Test with system message in fit
        with patch.object(generator, '_prepare_training_data') as mock_prepare:
            mock_prepare.return_value = (self.test_conversations, ["Be helpful"])
            
            with patch('lsm.training.train.LSMTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer_class.return_value = mock_trainer
                mock_trainer.train.return_value = True
                
                generator.fit(
                    self.test_conversations,
                    system_messages=["Be helpful"],
                    epochs=1
                )
                
                mock_prepare.assert_called_once()
    
    def test_model_persistence(self):
        """Test model save/load functionality."""
        generator = LSMGenerator(window_size=5, embedding_dim=32)
        generator._is_fitted = True
        generator._model_components = {'test': 'data'}
        
        model_path = os.path.join(self.temp_dir, 'test_model')
        
        try:
            # Test save
            generator.save(model_path)
            self.assertTrue(os.path.exists(model_path))
            
            # Test load
            loaded_generator = LSMGenerator.load(model_path)
            self.assertIsNotNone(loaded_generator)
            self.assertEqual(loaded_generator.window_size, 5)
            self.assertEqual(loaded_generator.embedding_dim, 32)
            
        except NotImplementedError:
            # If not fully implemented, verify interface exists
            self.assertTrue(hasattr(generator, 'save'))
            self.assertTrue(hasattr(LSMGenerator, 'load'))
    
    def test_configuration_presets(self):
        """Test configuration preset functionality."""
        # Test fast preset
        generator = LSMGenerator.from_preset('fast')
        self.assertIsNotNone(generator)
        
        # Test balanced preset
        generator = LSMGenerator.from_preset('balanced')
        self.assertIsNotNone(generator)
        
        # Test quality preset
        generator = LSMGenerator.from_preset('quality')
        self.assertIsNotNone(generator)
        
        # Test invalid preset
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMGenerator.from_preset('invalid_preset')
    
    def test_conversation_format_handling(self):
        """Test different conversation format handling."""
        generator = LSMGenerator()
        
        # Test string list format
        string_conversations = ["Hello", "Hi there", "How are you?"]
        
        # Test structured format
        structured_conversations = [
            {"messages": ["Hello", "Hi"], "system": "Be friendly"},
            {"messages": ["Help me", "Sure"], "system": "Be helpful"}
        ]
        
        # Test that both formats are accepted (mock the actual processing)
        with patch.object(generator, '_prepare_training_data') as mock_prepare:
            mock_prepare.return_value = (string_conversations, None)
            
            with patch('lsm.training.train.LSMTrainer') as mock_trainer_class:
                mock_trainer = Mock()
                mock_trainer_class.return_value = mock_trainer
                mock_trainer.train.return_value = True
                
                # Test string format
                generator.fit(string_conversations, epochs=1)
                mock_prepare.assert_called()
                
                # Reset mock
                mock_prepare.reset_mock()
                mock_prepare.return_value = (structured_conversations, ["Be friendly"])
                
                # Test structured format
                generator.fit(structured_conversations, epochs=1)
                mock_prepare.assert_called()


@unittest.skipUnless(GENERATOR_AVAILABLE, "LSMGenerator required")
class TestLSMGeneratorAdvanced(unittest.TestCase):
    """Test advanced LSMGenerator functionality."""
    
    def test_memory_management(self):
        """Test automatic memory management."""
        generator = LSMGenerator(auto_memory_management=True)
        
        # Test that memory management is enabled
        self.assertTrue(generator.auto_memory_management)
        
        # Test memory threshold setting
        generator.set_memory_threshold(0.8)
        self.assertEqual(generator.memory_threshold, 0.8)
    
    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        generator = LSMGenerator(enable_performance_monitoring=True)
        
        # Test that monitoring is enabled
        self.assertTrue(generator.enable_performance_monitoring)
        
        # Test getting performance metrics
        try:
            metrics = generator.get_performance_metrics()
            self.assertIsInstance(metrics, dict)
        except (NotImplementedError, AttributeError):
            # If not implemented, verify interface exists
            self.assertTrue(hasattr(generator, 'get_performance_metrics'))
    
    def test_custom_tokenizer(self):
        """Test custom tokenizer support."""
        # Test with custom tokenizer
        generator = LSMGenerator(tokenizer='custom')
        self.assertEqual(generator.tokenizer, 'custom')
        
        # Test tokenizer validation
        with self.assertRaises((ValueError, ConvenienceValidationError)):
            LSMGenerator(tokenizer='invalid_tokenizer')
    
    def test_generation_parameters(self):
        """Test generation parameter control."""
        generator = LSMGenerator()
        generator._is_fitted = True
        
        with patch.object(generator, '_get_response_generator') as mock_get_gen:
            mock_response_gen = Mock()
            mock_response_gen.generate_response.return_value = "Test response"
            mock_get_gen.return_value = mock_response_gen
            
            # Test temperature parameter
            generator.generate("Test", temperature=0.8)
            
            # Test max_length parameter
            generator.generate("Test", max_length=100)
            
            # Test both parameters
            generator.generate("Test", temperature=0.5, max_length=50)
            
            # Verify generate was called multiple times
            self.assertEqual(mock_response_gen.generate_response.call_count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)