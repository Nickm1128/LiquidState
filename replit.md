# Overview

This project implements a novel neural architecture that combines Liquid State Machines (LSM) with Convolutional Neural Networks (CNN) for next-token prediction on dialogue data. The system uses sparse connectivity patterns, parametric sine activation functions, and rolling wave encoding to create spatial-temporal patterns that are processed by a CNN. The architecture now supports **multiple advanced reservoir types** including hierarchical, attentive, echo state, and deep reservoir architectures, each designed to leverage different aspects of biological plausibility while providing specialized learning capabilities for natural language processing tasks.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Core Architecture Components

### Liquid State Machine (LSM) Design
- **Sparse Connectivity**: Uses custom `SparseDense` Keras layers with fixed binary masks to maintain sparse connections between neurons, improving efficiency and biological plausibility
- **Parametric Sine Activation**: Custom activation function `A * exp(-α * |x|) * sin(ω * x)` with learnable parameters (frequency ω, amplitude A, decay α) for rich temporal dynamics
- **Multi-layer Reservoir**: Configurable stack of sparse layers with sine activations to create complex temporal patterns

### Advanced Reservoir Architectures (NEW)
- **Hierarchical Reservoir**: Multi-scale temporal processing with different frequency ranges and inter-scale connections
- **Attentive Reservoir**: Self-attention mechanism applied to reservoir units for dynamic weighting of neural activity
- **Echo State Reservoir**: Classical echo state network with fixed random weights and controlled spectral radius for stability
- **Deep Reservoir**: Multi-layer reservoir with skip connections and layer-wise processing for enhanced representational capacity

### Spatial-Temporal Encoding
- **Rolling Wave Buffer**: Converts temporal LSM outputs into 2D spatial patterns by time-shifting waves and storing them in a window_size × window_size matrix
- **Multi-channel Support**: Optional multi-channel buffer for handling multiple reservoir output streams
- **Pattern Formation**: Each timestep creates a row in the 2D buffer, forming interpretable waveforms over time

### CNN Processing Pipeline
- **2D Pattern Recognition**: CNN processes the spatial-temporal patterns from the rolling wave buffer
- **Spatial Attention**: Optional attention mechanism to focus on important regions of the waveform
- **Embedding Prediction**: Final dense layers output next-token embeddings matching the input embedding dimension

### Data Processing Architecture
- **Custom Tokenizer**: Uses TF-IDF vectorization with n-grams for dialogue text conversion to embeddings
- **Sequence Generation**: Creates input-target pairs from dialogue turns for next-token prediction training
- **Fixed Embedding Size**: Configurable embedding dimension with padding/truncation for consistent input shapes

### Training Pipeline
- **End-to-End Learning**: Integrates LSM reservoir, rolling wave encoding, and CNN in a unified training loop
- **GPU Support**: Configurable GPU memory growth and multi-GPU support
- **Reproducibility**: Comprehensive random seed setting across NumPy, TensorFlow, and Python

## Design Patterns

### Modular Architecture
- Clear separation of concerns with dedicated modules for data loading, reservoir computing, wave encoding, and CNN processing
- Keras Layer abstraction for custom components (SparseDense, ParametricSineActivation, SpatialAttentionBlock)
- Factory functions for model creation with configurable parameters

### State Management
- Rolling buffer maintains temporal state across sequences
- Reservoir state is reset between sequences for clean training
- Training history tracking with comprehensive metrics storage

### Extensibility
- Configurable reservoir architectures with variable layer sizes and sparsity levels
- Optional attention mechanisms and multi-channel processing
- Pluggable activation functions and connectivity patterns

# Latest Updates (August 2025)

## Advanced LSM Reservoir Architectures - COMPLETED ✅
- **Hierarchical Reservoir**: Multi-scale temporal processing with configurable frequency ranges per scale
- **Attentive Reservoir**: Self-attention mechanism for dynamic weighting of reservoir units 
- **Echo State Reservoir**: Classical ESN with fixed weights and controlled spectral radius
- **Deep Reservoir**: Multi-layer architecture with skip connections and layer-wise processing
- **CLI Integration**: Full command-line support with JSON configuration for all reservoir types
- **100% Test Success**: All 4 advanced architectures validated and working correctly
- **Seamless Integration**: Advanced reservoirs work with existing LSM training pipeline

# External Dependencies

## Core Machine Learning Stack
- **TensorFlow 2.10+**: Primary deep learning framework for model implementation and training
- **NumPy**: Numerical computing for array operations and sparse mask generation
- **Pandas**: Data manipulation and CSV processing for dialogue dataset handling

## Data Processing
- **scikit-learn**: TF-IDF vectorization, train-test splitting, and preprocessing utilities
- **requests**: HTTP client for downloading HuggingFace dataset files
- **TensorFlow Text** (implied): Text processing and tokenization utilities

## Dataset Integration
- **HuggingFace Synthetic-Persona-Chat**: Training dataset accessed via direct CSV download from HuggingFace repository
- **Google Synthetic-Persona-Chat**: Specific dataset variant for dialogue turn sequences

## Development Dependencies
- **Python 3.9+**: Minimum Python version requirement
- **CUDA Support**: Optional GPU acceleration for TensorFlow operations

## File I/O and Caching
- **pickle**: Model and tokenizer serialization for persistence
- **json**: Configuration and results storage
- **os/sys**: File system operations and environment configuration