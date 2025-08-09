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

# CNN Architecture Factory for enhanced model creation
from .cnn_architecture_factory import (
    CNNArchitectureFactory,
    CNNArchitectureError,
    CNNType,
    AttentionType,
    LossType,
    create_standard_2d_cnn,
    create_system_aware_3d_cnn,
    create_residual_cnn_model as create_factory_residual_cnn
)

# Loss functions for enhanced training
from .loss_functions import (
    CosineSimilarityLoss,
    CNNLossCalculator,
    LossFunctionError,
    LossType,
    create_response_level_loss,
    create_cosine_similarity_loss
)

# CNN 3D Processor for system message integration
from .cnn_3d_processor import (
    CNN3DProcessor,
    CNN3DProcessorError,
    SystemContext,
    ProcessingResult,
    create_cnn_3d_processor,
    create_system_aware_processor
)

# System Message Processor for standalone system message handling
from .system_message_processor import (
    SystemMessageProcessor,
    SystemMessageError,
    SystemMessageContext,
    SystemMessageConfig,
    create_system_message_processor,
    process_system_message_simple
)

# Embedding Modifier Generator for system influence
from .embedding_modifier_generator import (
    EmbeddingModifierGenerator,
    EmbeddingModifierError,
    ModifierConfig,
    ModifierOutput,
    TrainingBatch,
    create_embedding_modifier_generator,
    create_training_batch_from_prompts
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
    'create_multi_scale_cnn',
    
    # CNN Architecture Factory
    'CNNArchitectureFactory',
    'CNNArchitectureError',
    'CNNType',
    'AttentionType',
    'LossType',
    'create_standard_2d_cnn',
    'create_system_aware_3d_cnn',
    'create_factory_residual_cnn',
    
    # Loss functions
    'CosineSimilarityLoss',
    'CNNLossCalculator',
    'LossFunctionError',
    'LossType',
    'create_response_level_loss',
    'create_cosine_similarity_loss',
    
    # CNN 3D Processor
    'CNN3DProcessor',
    'CNN3DProcessorError',
    'SystemContext',
    'ProcessingResult',
    'create_cnn_3d_processor',
    'create_system_aware_processor',
    
    # System Message Processor
    'SystemMessageProcessor',
    'SystemMessageError',
    'SystemMessageContext',
    'SystemMessageConfig',
    'create_system_message_processor',
    'process_system_message_simple',
    
    # Embedding Modifier Generator
    'EmbeddingModifierGenerator',
    'EmbeddingModifierError',
    'ModifierConfig',
    'ModifierOutput',
    'TrainingBatch',
    'create_embedding_modifier_generator',
    'create_training_batch_from_prompts'
]