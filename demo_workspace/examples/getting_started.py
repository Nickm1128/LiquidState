"""
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
