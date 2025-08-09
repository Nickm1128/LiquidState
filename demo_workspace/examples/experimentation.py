"""
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
    print(f"\nTesting {arch.value} architecture:")
    try:
        pipeline = colab_manager.create_pipeline(arch)
        result = pipeline.process_input("Test message")
        print(f"Success: {result[:50]}...")
    except Exception as e:
        print(f"Error: {e}")

# Show performance comparison
colab_manager.compare_architectures()
