"""
Tests for the ColabCompatibilityManager.

This module tests Google Colab-specific optimizations and setup,
methods for easy cloning and environment setup, and simplified interfaces
for experimentation.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import os
import json
import sys
import subprocess
from pathlib import Path

from src.lsm.pipeline.colab_compatibility import (
    ColabCompatibilityManager,
    ColabEnvironmentInfo,
    ColabSetupConfig,
    ColabCompatibilityError,
    EnvironmentSetupError,
    DependencyInstallError,
    setup_colab_environment,
    quick_start_colab
)
from src.lsm.pipeline.pipeline_orchestrator import ArchitectureType


class TestColabEnvironmentInfo(unittest.TestCase):
    """Test ColabEnvironmentInfo dataclass."""
    
    def test_environment_info_creation(self):
        """Test creating environment info."""
        env_info = ColabEnvironmentInfo(
            is_colab=True,
            gpu_available=True,
            tpu_available=False,
            python_version="3.8.10",
            available_memory_gb=12.0,
            disk_space_gb=100.0,
            runtime_type="gpu"
        )
        
        self.assertTrue(env_info.is_colab)
        self.assertTrue(env_info.gpu_available)
        self.assertFalse(env_info.tpu_available)
        self.assertEqual(env_info.python_version, "3.8.10")
        self.assertEqual(env_info.available_memory_gb, 12.0)
        self.assertEqual(env_info.disk_space_gb, 100.0)
        self.assertEqual(env_info.runtime_type, "gpu")


class TestColabSetupConfig(unittest.TestCase):
    """Test ColabSetupConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ColabSetupConfig()
        
        self.assertTrue(config.install_dependencies)
        self.assertTrue(config.setup_gpu)
        self.assertTrue(config.download_sample_data)
        self.assertTrue(config.create_examples)
        self.assertTrue(config.enable_logging)
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.workspace_dir, "/content/lsm_workspace")
        self.assertEqual(config.data_dir, "/content/lsm_data")
        self.assertEqual(config.models_dir, "/content/lsm_models")
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ColabSetupConfig(
            install_dependencies=False,
            setup_gpu=False,
            log_level="DEBUG",
            workspace_dir="/custom/workspace"
        )
        
        self.assertFalse(config.install_dependencies)
        self.assertFalse(config.setup_gpu)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.workspace_dir, "/custom/workspace")


class TestColabCompatibilityManager(unittest.TestCase):
    """Test ColabCompatibilityManager functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ColabSetupConfig(
            workspace_dir=f"{self.temp_dir}/workspace",
            data_dir=f"{self.temp_dir}/data",
            models_dir=f"{self.temp_dir}/models"
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.lsm.pipeline.colab_compatibility.sys.modules')
    def test_detect_environment_colab(self, mock_modules):
        """Test environment detection in Colab."""
        mock_modules.__contains__ = lambda self, x: x == 'google.colab'
        
        # Mock TensorFlow import and GPU detection
        with patch('builtins.__import__') as mock_import:
            def side_effect(name, *args, **kwargs):
                if name == 'tensorflow':
                    mock_tf = Mock()
                    mock_tf.config.list_physical_devices.return_value = ['GPU:0']
                    return mock_tf
                return __import__(name, *args, **kwargs)
            
            mock_import.side_effect = side_effect
            
            manager = ColabCompatibilityManager(self.config)
            
            self.assertTrue(manager.env_info.is_colab)
            self.assertTrue(manager.env_info.gpu_available)
            self.assertEqual(manager.env_info.runtime_type, "gpu")
    
    @patch('src.lsm.pipeline.colab_compatibility.sys.modules')
    def test_detect_environment_local(self, mock_modules):
        """Test environment detection in local environment."""
        mock_modules.__contains__ = lambda self, x: False
        
        manager = ColabCompatibilityManager(self.config)
        
        self.assertFalse(manager.env_info.is_colab)
        self.assertEqual(manager.env_info.runtime_type, "standard")
    
    def test_create_workspace_directories(self):
        """Test workspace directory creation."""
        manager = ColabCompatibilityManager(self.config)
        manager._create_workspace_directories()
        
        # Check that directories were created
        self.assertTrue(os.path.exists(self.config.workspace_dir))
        self.assertTrue(os.path.exists(self.config.data_dir))
        self.assertTrue(os.path.exists(self.config.models_dir))
        self.assertTrue(os.path.exists(f"{self.config.workspace_dir}/examples"))
        self.assertTrue(os.path.exists(f"{self.config.workspace_dir}/experiments"))
        self.assertTrue(os.path.exists(f"{self.config.workspace_dir}/logs"))
    
    @patch('subprocess.run')
    def test_install_dependencies_success(self, mock_run):
        """Test successful dependency installation."""
        mock_run.return_value = Mock(returncode=0, stderr="")
        
        manager = ColabCompatibilityManager(self.config)
        
        # Should not raise exception
        manager._install_dependencies()
        
        # Check that subprocess.run was called
        self.assertTrue(mock_run.called)
    
    @patch('subprocess.run')
    def test_install_dependencies_failure(self, mock_run):
        """Test dependency installation with some failures."""
        mock_run.return_value = Mock(returncode=1, stderr="Installation failed")
        
        manager = ColabCompatibilityManager(self.config)
        
        # Should not raise exception (warnings only)
        manager._install_dependencies()
        
        self.assertTrue(mock_run.called)
    
    def test_download_sample_data(self):
        """Test sample data download."""
        manager = ColabCompatibilityManager(self.config)
        manager._create_workspace_directories()
        manager._download_sample_data()
        
        # Check that sample data file was created
        sample_file = f"{self.config.data_dir}/sample_conversations.json"
        self.assertTrue(os.path.exists(sample_file))
        
        # Check content
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn("conversations", data)
        self.assertIsInstance(data["conversations"], list)
        self.assertGreater(len(data["conversations"]), 0)
    
    def test_create_colab_examples(self):
        """Test Colab example creation."""
        manager = ColabCompatibilityManager(self.config)
        manager._create_workspace_directories()
        manager._create_colab_examples()
        
        # Check that example files were created
        examples_dir = f"{self.config.workspace_dir}/examples"
        self.assertTrue(os.path.exists(f"{examples_dir}/getting_started.py"))
        self.assertTrue(os.path.exists(f"{examples_dir}/experimentation.py"))
        
        # Check content of getting started script
        with open(f"{examples_dir}/getting_started.py", 'r') as f:
            content = f.read()
        
        self.assertIn("ColabCompatibilityManager", content)
        self.assertIn("create_simple_pipeline", content)
    
    @patch('src.lsm.pipeline.colab_compatibility.create_pipeline_orchestrator')
    def test_create_simple_pipeline(self, mock_create):
        """Test simple pipeline creation."""
        mock_orchestrator = Mock()
        mock_create.return_value = mock_orchestrator
        
        manager = ColabCompatibilityManager(self.config)
        result = manager.create_simple_pipeline()
        
        self.assertEqual(result, mock_orchestrator)
        mock_create.assert_called_once()
        mock_orchestrator.setup_pipeline.assert_called_once()
    
    @patch('src.lsm.pipeline.colab_compatibility.create_pipeline_orchestrator')
    def test_create_pipeline_with_architecture(self, mock_create):
        """Test pipeline creation with specific architecture."""
        mock_orchestrator = Mock()
        mock_create.return_value = mock_orchestrator
        
        manager = ColabCompatibilityManager(self.config)
        result = manager.create_pipeline(ArchitectureType.SYSTEM_AWARE_3D)
        
        self.assertEqual(result, mock_orchestrator)
        mock_create.assert_called_once_with(
            architecture_type=ArchitectureType.SYSTEM_AWARE_3D,
            experiment_name="colab_system_aware_3d",
            description="Colab pipeline with system_aware_3d architecture"
        )
    
    @patch('src.lsm.pipeline.colab_compatibility.create_pipeline_orchestrator')
    def test_create_pipeline_with_custom_config(self, mock_create):
        """Test pipeline creation with custom configuration."""
        mock_orchestrator = Mock()
        mock_create.return_value = mock_orchestrator
        
        manager = ColabCompatibilityManager(self.config)
        custom_config = {"reservoir": {"units": 200}}
        
        result = manager.create_pipeline(ArchitectureType.STANDARD_2D, custom_config)
        
        self.assertEqual(result, mock_orchestrator)
        # Check that setup_pipeline was called with merged config
        mock_orchestrator.setup_pipeline.assert_called_once()
        call_args = mock_orchestrator.setup_pipeline.call_args[0][0]
        self.assertEqual(call_args["reservoir"]["units"], 200)
    
    @patch('src.lsm.pipeline.colab_compatibility.create_pipeline_orchestrator')
    def test_quick_experiment(self, mock_create):
        """Test quick experiment functionality."""
        mock_orchestrator = Mock()
        mock_orchestrator.process_input.return_value = "Test response"
        mock_create.return_value = mock_orchestrator
        
        manager = ColabCompatibilityManager(self.config)
        
        results = manager.quick_experiment(
            "Test input",
            [ArchitectureType.STANDARD_2D]
        )
        
        self.assertIn("standard_2d", results)
        self.assertTrue(results["standard_2d"]["success"])
        self.assertEqual(results["standard_2d"]["response"], "Test response")
        self.assertIsNotNone(results["standard_2d"]["processing_time"])
    
    @patch('src.lsm.pipeline.colab_compatibility.create_pipeline_orchestrator')
    def test_quick_experiment_with_error(self, mock_create):
        """Test quick experiment with error handling."""
        mock_create.side_effect = Exception("Pipeline creation failed")
        
        manager = ColabCompatibilityManager(self.config)
        
        results = manager.quick_experiment(
            "Test input",
            [ArchitectureType.STANDARD_2D]
        )
        
        self.assertIn("standard_2d", results)
        self.assertFalse(results["standard_2d"]["success"])
        self.assertIsNotNone(results["standard_2d"]["error"])
    
    def test_get_recommended_config_high_memory(self):
        """Test recommended config for high memory environment."""
        # Mock high memory environment
        manager = ColabCompatibilityManager(self.config)
        manager.env_info.available_memory_gb = 15.0
        manager.env_info.runtime_type = "gpu"
        
        config = manager.get_recommended_config()
        
        self.assertEqual(config["reservoir"]["units"], 200)
        self.assertEqual(config["embedder"]["embedding_dim"], 256)
        self.assertTrue(config["use_mixed_precision"])
        self.assertEqual(config["batch_size"], 32)
    
    def test_get_recommended_config_low_memory(self):
        """Test recommended config for low memory environment."""
        # Mock low memory environment
        manager = ColabCompatibilityManager(self.config)
        manager.env_info.available_memory_gb = 6.0
        manager.env_info.runtime_type = "standard"
        
        config = manager.get_recommended_config()
        
        self.assertEqual(config["reservoir"]["units"], 50)
        self.assertEqual(config["embedder"]["embedding_dim"], 64)
        self.assertEqual(config["batch_size"], 16)
    
    def test_get_recommended_config_tpu(self):
        """Test recommended config for TPU environment."""
        # Mock TPU environment
        manager = ColabCompatibilityManager(self.config)
        manager.env_info.runtime_type = "tpu"
        
        config = manager.get_recommended_config()
        
        self.assertTrue(config["use_mixed_precision"])
        self.assertEqual(config["batch_size"], 64)
    
    def test_is_setup_complete(self):
        """Test setup completion status."""
        manager = ColabCompatibilityManager(self.config)
        
        # Initially not complete
        self.assertFalse(manager.is_setup_complete())
        
        # Mark as complete
        manager._setup_complete = True
        self.assertTrue(manager.is_setup_complete())
    
    @patch('subprocess.run')
    def test_clone_and_setup_success(self, mock_run):
        """Test successful repository cloning and setup."""
        mock_run.return_value = Mock(returncode=0)
        
        manager = ColabCompatibilityManager(self.config)
        
        with patch.object(manager, 'setup_colab_environment') as mock_setup:
            with patch('os.chdir'):
                with patch('os.path.exists', return_value=False):
                    manager.clone_and_setup("https://github.com/test/repo.git")
                    
                    mock_run.assert_called_once()
                    mock_setup.assert_called_once()
    
    @patch('subprocess.run')
    def test_clone_and_setup_failure(self, mock_run):
        """Test repository cloning failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git clone")
        
        manager = ColabCompatibilityManager(self.config)
        
        with self.assertRaises(ColabCompatibilityError):
            manager.clone_and_setup("https://github.com/test/repo.git")
    
    @patch('builtins.print')
    def test_show_environment_info(self, mock_print):
        """Test environment info display."""
        manager = ColabCompatibilityManager(self.config)
        manager.show_environment_info()
        
        # Check that print was called multiple times
        self.assertTrue(mock_print.called)
        
        # Check that environment info was printed
        call_args = [call[0][0] for call in mock_print.call_args_list]
        info_text = " ".join(call_args)
        
        self.assertIn("LSM Colab Environment Information", info_text)
        self.assertIn("Running in Colab", info_text)
        self.assertIn("GPU Available", info_text)
    
    @patch('builtins.print')
    @patch('src.lsm.pipeline.colab_compatibility.create_pipeline_orchestrator')
    def test_compare_architectures(self, mock_create, mock_print):
        """Test architecture comparison."""
        mock_orchestrator = Mock()
        mock_orchestrator.process_input.return_value = "Test response"
        mock_create.return_value = mock_orchestrator
        
        manager = ColabCompatibilityManager(self.config)
        manager.compare_architectures("Test input")
        
        # Check that print was called
        self.assertTrue(mock_print.called)
        
        # Check that comparison info was printed
        call_args = [call[0][0] for call in mock_print.call_args_list]
        comparison_text = " ".join(call_args)
        
        self.assertIn("Architecture Comparison", comparison_text)


class TestColabCompatibilityExceptions(unittest.TestCase):
    """Test exception classes."""
    
    def test_colab_compatibility_error(self):
        """Test ColabCompatibilityError."""
        error = ColabCompatibilityError("Test error")
        self.assertEqual(str(error), "Test error")
        self.assertIsInstance(error, Exception)
    
    def test_environment_setup_error(self):
        """Test EnvironmentSetupError."""
        error = EnvironmentSetupError("Setup failed")
        self.assertEqual(str(error), "Setup failed")
        self.assertIsInstance(error, ColabCompatibilityError)
    
    def test_dependency_install_error(self):
        """Test DependencyInstallError."""
        error = DependencyInstallError("Install failed")
        self.assertEqual(str(error), "Install failed")
        self.assertIsInstance(error, ColabCompatibilityError)


class TestColabCompatibilityConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    @patch('src.lsm.pipeline.colab_compatibility.ColabCompatibilityManager')
    def test_setup_colab_environment(self, mock_manager_class):
        """Test setup_colab_environment convenience function."""
        mock_manager = Mock()
        mock_manager_class.return_value = mock_manager
        
        result = setup_colab_environment(force_reinstall=True)
        
        self.assertEqual(result, mock_manager)
        mock_manager.setup_colab_environment.assert_called_once_with(True)
    
    @patch('src.lsm.pipeline.colab_compatibility.setup_colab_environment')
    @patch('builtins.print')
    def test_quick_start_colab(self, mock_print, mock_setup):
        """Test quick_start_colab convenience function."""
        mock_manager = Mock()
        mock_pipeline = Mock()
        mock_manager.create_simple_pipeline.return_value = mock_pipeline
        mock_setup.return_value = mock_manager
        
        result = quick_start_colab()
        
        self.assertEqual(result, mock_pipeline)
        mock_setup.assert_called_once()
        mock_manager.create_simple_pipeline.assert_called_once()
        self.assertTrue(mock_print.called)


class TestColabCompatibilityIntegration(unittest.TestCase):
    """Integration tests for ColabCompatibilityManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = ColabSetupConfig(
            workspace_dir=f"{self.temp_dir}/workspace",
            data_dir=f"{self.temp_dir}/data",
            models_dir=f"{self.temp_dir}/models",
            install_dependencies=False,  # Skip dependency installation in tests
            setup_gpu=False  # Skip GPU setup in tests
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_setup_workflow(self):
        """Test the complete setup workflow."""
        manager = ColabCompatibilityManager(self.config)
        
        # Setup environment (without dependencies and GPU)
        manager.setup_colab_environment()
        
        # Check that setup completed
        self.assertTrue(manager.is_setup_complete())
        
        # Check that directories were created
        self.assertTrue(os.path.exists(self.config.workspace_dir))
        self.assertTrue(os.path.exists(self.config.data_dir))
        self.assertTrue(os.path.exists(self.config.models_dir))
        
        # Check that sample data was created
        sample_file = f"{self.config.data_dir}/sample_conversations.json"
        self.assertTrue(os.path.exists(sample_file))
        
        # Check that examples were created
        examples_dir = f"{self.config.workspace_dir}/examples"
        self.assertTrue(os.path.exists(f"{examples_dir}/getting_started.py"))
        self.assertTrue(os.path.exists(f"{examples_dir}/experimentation.py"))


if __name__ == '__main__':
    unittest.main()