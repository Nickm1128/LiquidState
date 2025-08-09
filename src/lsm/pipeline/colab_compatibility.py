"""
Google Colab compatibility manager for the LSM project.

This module provides Google Colab-specific optimizations and setup,
methods for easy cloning and environment setup, and simplified interfaces
for experimentation in Colab environments.
"""

import os
import sys
import subprocess
import logging
import json
import shutil
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import importlib.util

from ..utils.lsm_exceptions import LSMError
from ..utils.lsm_logging import get_logger
from .pipeline_orchestrator import (
    PipelineOrchestrator, ArchitectureType, ComponentType,
    create_pipeline_orchestrator, PipelineConfiguration
)


class ColabCompatibilityError(LSMError):
    """Base exception for Colab compatibility errors."""
    pass


class EnvironmentSetupError(ColabCompatibilityError):
    """Exception raised when environment setup fails."""
    pass


class DependencyInstallError(ColabCompatibilityError):
    """Exception raised when dependency installation fails."""
    pass


@dataclass
class ColabEnvironmentInfo:
    """Information about the Colab environment."""
    is_colab: bool
    gpu_available: bool
    tpu_available: bool
    python_version: str
    available_memory_gb: float
    disk_space_gb: float
    runtime_type: str  # 'standard', 'gpu', 'tpu'


@dataclass
class ColabSetupConfig:
    """Configuration for Colab setup."""
    install_dependencies: bool = True
    setup_gpu: bool = True
    download_sample_data: bool = True
    create_examples: bool = True
    enable_logging: bool = True
    log_level: str = "INFO"
    workspace_dir: str = "/content/lsm_workspace"
    data_dir: str = "/content/lsm_data"
    models_dir: str = "/content/lsm_models"


class ColabCompatibilityManager:
    """
    Manager for Google Colab-specific optimizations and setup.
    
    Provides methods for easy cloning and environment setup,
    and creates simplified interfaces for experimentation.
    """
    
    def __init__(self, setup_config: Optional[ColabSetupConfig] = None):
        """
        Initialize the Colab compatibility manager.
        
        Args:
            setup_config: Configuration for Colab setup. If None, uses defaults.
        """
        self.logger = get_logger(__name__)
        self.setup_config = setup_config or ColabSetupConfig()
        self.env_info = self._detect_environment()
        self._orchestrator: Optional[PipelineOrchestrator] = None
        self._setup_complete = False
        
        self.logger.info(f"Colab compatibility manager initialized. Colab detected: {self.env_info.is_colab}")
    
    def _detect_environment(self) -> ColabEnvironmentInfo:
        """Detect the current environment and gather system information."""
        try:
            # Check if running in Colab
            is_colab = 'google.colab' in sys.modules
            
            # Check GPU availability
            gpu_available = False
            try:
                import tensorflow as tf
                gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                try:
                    import torch
                    gpu_available = torch.cuda.is_available()
                except ImportError:
                    pass
            
            # Check TPU availability (Colab specific)
            tpu_available = False
            if is_colab:
                try:
                    import tensorflow as tf
                    tpu_available = tf.config.list_logical_devices('TPU') != []
                except:
                    pass
            
            # Get system info
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Estimate available memory (simplified)
            available_memory_gb = 12.0 if is_colab else 8.0  # Colab typically has ~12GB RAM
            
            # Estimate disk space
            if is_colab:
                disk_space_gb = 100.0  # Colab typically provides ~100GB
            else:
                try:
                    statvfs = os.statvfs('/')
                    disk_space_gb = (statvfs.f_frsize * statvfs.f_bavail) / (1024**3)
                except:
                    disk_space_gb = 50.0  # Default estimate
            
            # Determine runtime type
            runtime_type = "standard"
            if tpu_available:
                runtime_type = "tpu"
            elif gpu_available:
                runtime_type = "gpu"
            
            return ColabEnvironmentInfo(
                is_colab=is_colab,
                gpu_available=gpu_available,
                tpu_available=tpu_available,
                python_version=python_version,
                available_memory_gb=available_memory_gb,
                disk_space_gb=disk_space_gb,
                runtime_type=runtime_type
            )
            
        except Exception as e:
            self.logger.warning(f"Error detecting environment: {e}")
            # Get python version safely
            try:
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            except:
                python_version = "unknown"
            
            return ColabEnvironmentInfo(
                is_colab=False,
                gpu_available=False,
                tpu_available=False,
                python_version=python_version,
                available_memory_gb=8.0,
                disk_space_gb=50.0,
                runtime_type="standard"
            )
    
    def setup_colab_environment(self, force_reinstall: bool = False) -> None:
        """
        Set up the complete Colab environment for LSM experimentation.
        
        Args:
            force_reinstall: Whether to force reinstallation of dependencies
        """
        try:
            self.logger.info("Setting up Colab environment...")
            
            # Create workspace directories
            self._create_workspace_directories()
            
            # Install dependencies
            if self.setup_config.install_dependencies:
                self._install_dependencies(force_reinstall)
            
            # Setup GPU/TPU if available
            if self.setup_config.setup_gpu and (self.env_info.gpu_available or self.env_info.tpu_available):
                self._setup_accelerators()
            
            # Download sample data
            if self.setup_config.download_sample_data:
                self._download_sample_data()
            
            # Create example notebooks/scripts
            if self.setup_config.create_examples:
                self._create_colab_examples()
            
            # Setup logging
            if self.setup_config.enable_logging:
                self._setup_colab_logging()
            
            self._setup_complete = True
            self.logger.info("Colab environment setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Colab environment setup failed: {e}")
            raise EnvironmentSetupError(f"Failed to setup Colab environment: {e}") from e
    
    def _create_workspace_directories(self) -> None:
        """Create necessary workspace directories."""
        directories = [
            self.setup_config.workspace_dir,
            self.setup_config.data_dir,
            self.setup_config.models_dir,
            f"{self.setup_config.workspace_dir}/examples",
            f"{self.setup_config.workspace_dir}/experiments",
            f"{self.setup_config.workspace_dir}/logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def _install_dependencies(self, force_reinstall: bool = False) -> None:
        """Install required dependencies for LSM in Colab."""
        try:
            # Core dependencies for LSM
            dependencies = [
                "tensorflow>=2.12.0",
                "numpy>=1.21.0",
                "pandas>=1.3.0",
                "scikit-learn>=1.0.0",
                "matplotlib>=3.5.0",
                "seaborn>=0.11.0",
                "tqdm>=4.62.0",
                "datasets>=2.0.0",  # HuggingFace datasets
                "transformers>=4.20.0",  # HuggingFace transformers
                "tokenizers>=0.13.0",  # HuggingFace tokenizers
                "ipywidgets>=7.6.0",  # For interactive widgets
                "plotly>=5.0.0"  # For interactive plots
            ]
            
            # Install each dependency
            for dep in dependencies:
                try:
                    if force_reinstall:
                        cmd = f"pip install --upgrade --force-reinstall {dep}"
                    else:
                        cmd = f"pip install --upgrade {dep}"
                    
                    self.logger.debug(f"Installing: {dep}")
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        self.logger.warning(f"Failed to install {dep}: {result.stderr}")
                    else:
                        self.logger.debug(f"Successfully installed: {dep}")
                        
                except Exception as e:
                    self.logger.warning(f"Error installing {dep}: {e}")
            
            # Install LSM package in development mode if source is available
            if os.path.exists("/content/lsm") or os.path.exists("./src/lsm"):
                try:
                    subprocess.run("pip install -e .", shell=True, check=True)
                    self.logger.info("Installed LSM package in development mode")
                except subprocess.CalledProcessError:
                    self.logger.warning("Could not install LSM package in development mode")
            
        except Exception as e:
            raise DependencyInstallError(f"Failed to install dependencies: {e}") from e
    
    def _setup_accelerators(self) -> None:
        """Setup GPU/TPU acceleration."""
        try:
            if self.env_info.tpu_available:
                self._setup_tpu()
            elif self.env_info.gpu_available:
                self._setup_gpu()
                
        except Exception as e:
            self.logger.warning(f"Failed to setup accelerators: {e}")
    
    def _setup_gpu(self) -> None:
        """Setup GPU acceleration."""
        try:
            import tensorflow as tf
            
            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.info(f"Configured {len(gpus)} GPU(s) with memory growth")
            
        except Exception as e:
            self.logger.warning(f"GPU setup failed: {e}")
    
    def _setup_tpu(self) -> None:
        """Setup TPU acceleration."""
        try:
            import tensorflow as tf
            
            # Initialize TPU
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            
            self.logger.info("TPU initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"TPU setup failed: {e}")
    
    def _download_sample_data(self) -> None:
        """Download sample data for experimentation."""
        try:
            # Create a simple sample dataset for quick testing
            sample_data = {
                "conversations": [
                    {"input": "Hello, how are you?", "output": "I'm doing well, thank you for asking!"},
                    {"input": "What's the weather like?", "output": "I don't have access to current weather data."},
                    {"input": "Can you help me with coding?", "output": "Of course! I'd be happy to help with coding questions."},
                    {"input": "Tell me a joke", "output": "Why don't scientists trust atoms? Because they make up everything!"},
                    {"input": "What is machine learning?", "output": "Machine learning is a subset of AI that enables computers to learn from data."}
                ]
            }
            
            # Save sample data
            sample_file = f"{self.setup_config.data_dir}/sample_conversations.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            self.logger.info(f"Sample data saved to {sample_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to download sample data: {e}")
    
    def _create_colab_examples(self) -> None:
        """Create example notebooks and scripts for Colab."""
        try:
            examples_dir = f"{self.setup_config.workspace_dir}/examples"
            
            # Create a simple getting started script
            getting_started = '''"""
Getting Started with LSM in Google Colab

This script demonstrates basic usage of the LSM pipeline in Colab.
"""

from lsm.pipeline import ColabCompatibilityManager, create_pipeline_orchestrator, ArchitectureType

# Initialize Colab compatibility manager
colab_manager = ColabCompatibilityManager()

# Setup the environment (if not already done)
if not colab_manager.is_setup_complete():
    colab_manager.setup_colab_environment()

# Create a simple pipeline
pipeline = colab_manager.create_simple_pipeline()

# Test the pipeline
test_input = "Hello, how can I help you today?"
response = pipeline.process_input(test_input)
print(f"Input: {test_input}")
print(f"Response: {response}")

# Show environment info
colab_manager.show_environment_info()
'''
            
            with open(f"{examples_dir}/getting_started.py", 'w') as f:
                f.write(getting_started)
            
            # Create an experimentation script
            experimentation_script = '''"""
LSM Experimentation Script for Colab

This script shows how to experiment with different LSM architectures.
"""

from lsm.pipeline import ColabCompatibilityManager, ArchitectureType

# Initialize manager
colab_manager = ColabCompatibilityManager()

# Try different architectures
architectures = [
    ArchitectureType.STANDARD_2D,
    ArchitectureType.SYSTEM_AWARE_3D,
    ArchitectureType.HYBRID
]

for arch in architectures:
    print(f"\\nTesting {arch.value} architecture:")
    try:
        pipeline = colab_manager.create_pipeline(arch)
        result = pipeline.process_input("Test message")
        print(f"Success: {result[:50]}...")
    except Exception as e:
        print(f"Error: {e}")

# Show performance comparison
colab_manager.compare_architectures()
'''
            
            with open(f"{examples_dir}/experimentation.py", 'w') as f:
                f.write(experimentation_script)
            
            self.logger.info(f"Example scripts created in {examples_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create examples: {e}")
    
    def _setup_colab_logging(self) -> None:
        """Setup logging configuration optimized for Colab."""
        try:
            # Configure logging for Colab display
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            logging.basicConfig(
                level=getattr(logging, self.setup_config.log_level),
                format=log_format,
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(f"{self.setup_config.workspace_dir}/logs/lsm_colab.log")
                ]
            )
            
            self.logger.info("Colab logging configured")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup logging: {e}")
    
    def create_simple_pipeline(self, architecture: ArchitectureType = ArchitectureType.STANDARD_2D) -> PipelineOrchestrator:
        """
        Create a simple pipeline optimized for Colab experimentation.
        
        Args:
            architecture: Architecture type to use
            
        Returns:
            Configured PipelineOrchestrator
        """
        try:
            # Create pipeline with Colab-optimized settings
            config = {
                "reservoir": {
                    "units": 50,  # Smaller for Colab memory constraints
                    "sparsity": 0.1,
                    "frequency": 1.0
                },
                "embedder": {
                    "embedding_dim": 64  # Smaller embedding dimension
                }
            }
            
            orchestrator = create_pipeline_orchestrator(
                architecture_type=architecture,
                experiment_name="colab_experiment",
                description="Simple pipeline for Colab experimentation"
            )
            
            orchestrator.setup_pipeline(config)
            self._orchestrator = orchestrator
            
            self.logger.info(f"Simple pipeline created with {architecture.value} architecture")
            return orchestrator
            
        except Exception as e:
            self.logger.error(f"Failed to create simple pipeline: {e}")
            raise ColabCompatibilityError(f"Pipeline creation failed: {e}") from e
    
    def create_pipeline(self, architecture: ArchitectureType, 
                       custom_config: Optional[Dict[str, Any]] = None) -> PipelineOrchestrator:
        """
        Create a pipeline with the specified architecture and configuration.
        
        Args:
            architecture: Architecture type to use
            custom_config: Custom configuration parameters
            
        Returns:
            Configured PipelineOrchestrator
        """
        try:
            # Merge custom config with Colab defaults
            default_config = {
                "reservoir": {
                    "units": 100 if self.env_info.available_memory_gb > 10 else 50,
                    "sparsity": 0.1
                },
                "embedder": {
                    "embedding_dim": 128 if self.env_info.available_memory_gb > 10 else 64
                }
            }
            
            if custom_config:
                # Deep merge configurations
                for key, value in custom_config.items():
                    if key in default_config and isinstance(value, dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
            
            orchestrator = create_pipeline_orchestrator(
                architecture_type=architecture,
                experiment_name=f"colab_{architecture.value}",
                description=f"Colab pipeline with {architecture.value} architecture"
            )
            
            orchestrator.setup_pipeline(default_config)
            
            self.logger.info(f"Pipeline created with {architecture.value} architecture")
            return orchestrator
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            raise ColabCompatibilityError(f"Pipeline creation failed: {e}") from e
    
    def quick_experiment(self, input_text: str, 
                        architectures: Optional[List[ArchitectureType]] = None) -> Dict[str, Any]:
        """
        Run a quick experiment comparing different architectures.
        
        Args:
            input_text: Text to process
            architectures: List of architectures to test. If None, tests all available.
            
        Returns:
            Dictionary with results from each architecture
        """
        if architectures is None:
            architectures = [
                ArchitectureType.STANDARD_2D,
                ArchitectureType.SYSTEM_AWARE_3D
            ]
        
        results = {}
        
        for arch in architectures:
            try:
                self.logger.info(f"Testing {arch.value} architecture...")
                
                # Create pipeline
                pipeline = self.create_pipeline(arch)
                
                # Process input
                import time
                start_time = time.time()
                response = pipeline.process_input(input_text)
                processing_time = time.time() - start_time
                
                results[arch.value] = {
                    "response": response,
                    "processing_time": processing_time,
                    "success": True,
                    "error": None
                }
                
                self.logger.info(f"{arch.value}: Success in {processing_time:.2f}s")
                
            except Exception as e:
                results[arch.value] = {
                    "response": None,
                    "processing_time": None,
                    "success": False,
                    "error": str(e)
                }
                
                self.logger.warning(f"{arch.value}: Failed - {e}")
        
        return results
    
    def show_environment_info(self) -> None:
        """Display information about the current environment."""
        print("=== LSM Colab Environment Information ===")
        print(f"Running in Colab: {self.env_info.is_colab}")
        print(f"GPU Available: {self.env_info.gpu_available}")
        print(f"TPU Available: {self.env_info.tpu_available}")
        print(f"Runtime Type: {self.env_info.runtime_type}")
        print(f"Python Version: {self.env_info.python_version}")
        print(f"Available Memory: {self.env_info.available_memory_gb:.1f} GB")
        print(f"Disk Space: {self.env_info.disk_space_gb:.1f} GB")
        print(f"Setup Complete: {self._setup_complete}")
        print("=" * 45)
    
    def compare_architectures(self, test_input: str = "Hello, how are you?") -> None:
        """
        Compare different architectures and display results.
        
        Args:
            test_input: Input text to test with
        """
        print("=== Architecture Comparison ===")
        
        results = self.quick_experiment(test_input)
        
        for arch_name, result in results.items():
            print(f"\n{arch_name.upper()}:")
            if result["success"]:
                print(f"  Response: {result['response'][:100]}...")
                print(f"  Time: {result['processing_time']:.3f}s")
            else:
                print(f"  Error: {result['error']}")
        
        print("\n" + "=" * 35)
    
    def clone_and_setup(self, repo_url: str = "https://github.com/your-org/lsm-project.git") -> None:
        """
        Clone the LSM repository and set up the environment.
        
        Args:
            repo_url: URL of the LSM repository to clone
        """
        try:
            self.logger.info(f"Cloning repository from {repo_url}")
            
            # Clone repository
            clone_dir = "/content/lsm-project"
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir)
            
            subprocess.run(f"git clone {repo_url} {clone_dir}", shell=True, check=True)
            
            # Change to project directory
            os.chdir(clone_dir)
            
            # Setup environment
            self.setup_colab_environment()
            
            self.logger.info("Repository cloned and environment setup completed")
            
        except subprocess.CalledProcessError as e:
            raise ColabCompatibilityError(f"Failed to clone repository: {e}") from e
        except Exception as e:
            raise ColabCompatibilityError(f"Setup failed after cloning: {e}") from e
    
    def is_setup_complete(self) -> bool:
        """Check if the Colab environment setup is complete."""
        return self._setup_complete
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """
        Get recommended configuration based on the current environment.
        
        Returns:
            Dictionary with recommended configuration parameters
        """
        config = {}
        
        # Adjust based on available memory
        if self.env_info.available_memory_gb > 12:
            config["reservoir"] = {"units": 200, "sparsity": 0.1}
            config["embedder"] = {"embedding_dim": 256}
        elif self.env_info.available_memory_gb > 8:
            config["reservoir"] = {"units": 100, "sparsity": 0.1}
            config["embedder"] = {"embedding_dim": 128}
        else:
            config["reservoir"] = {"units": 50, "sparsity": 0.15}
            config["embedder"] = {"embedding_dim": 64}
        
        # Adjust based on runtime type
        if self.env_info.runtime_type == "tpu":
            config["use_mixed_precision"] = True
            config["batch_size"] = 64
        elif self.env_info.runtime_type == "gpu":
            config["use_mixed_precision"] = True
            config["batch_size"] = 32
        else:
            config["batch_size"] = 16
        
        return config
    
    def create_interactive_demo(self) -> None:
        """Create an interactive demo for Colab users."""
        try:
            if not self.env_info.is_colab:
                self.logger.warning("Interactive demo is optimized for Colab environment")
            
            # Create interactive widgets if available
            try:
                import ipywidgets as widgets
                from IPython.display import display
                
                # Architecture selection
                arch_dropdown = widgets.Dropdown(
                    options=[arch.value for arch in ArchitectureType],
                    value=ArchitectureType.STANDARD_2D.value,
                    description='Architecture:'
                )
                
                # Input text
                input_text = widgets.Textarea(
                    value='Hello, how are you?',
                    placeholder='Enter your message here...',
                    description='Input:',
                    layout=widgets.Layout(width='100%', height='80px')
                )
                
                # Output area
                output = widgets.Output()
                
                # Process button
                def on_process_click(b):
                    with output:
                        output.clear_output()
                        try:
                            arch = ArchitectureType(arch_dropdown.value)
                            pipeline = self.create_pipeline(arch)
                            response = pipeline.process_input(input_text.value)
                            print(f"Response: {response}")
                        except Exception as e:
                            print(f"Error: {e}")
                
                process_button = widgets.Button(description="Process")
                process_button.on_click(on_process_click)
                
                # Display widgets
                display(widgets.VBox([
                    widgets.HTML("<h3>LSM Interactive Demo</h3>"),
                    arch_dropdown,
                    input_text,
                    process_button,
                    output
                ]))
                
                self.logger.info("Interactive demo created")
                
            except ImportError:
                self.logger.warning("ipywidgets not available, creating simple demo")
                self._create_simple_demo()
                
        except Exception as e:
            self.logger.error(f"Failed to create interactive demo: {e}")
    
    def _create_simple_demo(self) -> None:
        """Create a simple text-based demo."""
        print("=== LSM Simple Demo ===")
        print("Available architectures:")
        for i, arch in enumerate(ArchitectureType, 1):
            print(f"{i}. {arch.value}")
        
        try:
            choice = input("Select architecture (1-4): ")
            arch_index = int(choice) - 1
            arch = list(ArchitectureType)[arch_index]
            
            user_input = input("Enter your message: ")
            
            pipeline = self.create_pipeline(arch)
            response = pipeline.process_input(user_input)
            
            print(f"\nResponse: {response}")
            
        except (ValueError, IndexError):
            print("Invalid selection")
        except Exception as e:
            print(f"Error: {e}")


def setup_colab_environment(force_reinstall: bool = False) -> ColabCompatibilityManager:
    """
    Convenience function to quickly setup Colab environment.
    
    Args:
        force_reinstall: Whether to force reinstallation of dependencies
        
    Returns:
        Configured ColabCompatibilityManager
    """
    manager = ColabCompatibilityManager()
    manager.setup_colab_environment(force_reinstall)
    return manager


def quick_start_colab() -> PipelineOrchestrator:
    """
    Quick start function for Colab users.
    
    Returns:
        Ready-to-use PipelineOrchestrator
    """
    manager = setup_colab_environment()
    pipeline = manager.create_simple_pipeline()
    
    print("LSM pipeline ready! Try:")
    print("response = pipeline.process_input('Hello, how are you?')")
    
    return pipeline