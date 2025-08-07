"""
Core LSM components including reservoir layers, advanced architectures, and CNN models.
"""

# Basic reservoir components
from .reservoir import (
    SparseDense,
    ParametricSineActivation,
    generate_sparse_mask,
    build_reservoir,
    ReservoirLayer
)

# Advanced reservoir architectures
from .advanced_reservoir import (
    HierarchicalReservoir,
    AttentiveReservoir,
    EchoStateReservoir,
    DeepReservoir,
    create_advanced_reservoir
)

# Rolling wave buffer for temporal pattern storage
from .rolling_wave import (
    RollingWaveBuffer,
    MultiChannelRollingWaveBuffer
)

# CNN models for processing LSM waveforms
from .cnn_model import (
    SpatialAttentionBlock,
    create_cnn_model,
    create_residual_cnn_model,
    compile_cnn_model,
    create_multi_scale_cnn
)

__all__ = [
    # Basic reservoir components
    'SparseDense',
    'ParametricSineActivation',
    'generate_sparse_mask',
    'build_reservoir',
    'ReservoirLayer',
    
    # Advanced reservoir architectures
    'HierarchicalReservoir',
    'AttentiveReservoir',
    'EchoStateReservoir',
    'DeepReservoir',
    'create_advanced_reservoir',
    
    # Rolling wave buffer
    'RollingWaveBuffer',
    'MultiChannelRollingWaveBuffer',
    
    # CNN models
    'SpatialAttentionBlock',
    'create_cnn_model',
    'create_residual_cnn_model',
    'compile_cnn_model',
    'create_multi_scale_cnn'
]