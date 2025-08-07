import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional

class SpatialAttentionBlock(layers.Layer):
    """
    Spatial attention mechanism for focusing on important regions of the LSM waveform.
    """
    
    def __init__(self, **kwargs):
        super(SpatialAttentionBlock, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.conv_attention = layers.Conv2D(
            filters=1,
            kernel_size=7,
            padding='same',
            activation='sigmoid',
            name='spatial_attention_conv'
        )
        super(SpatialAttentionBlock, self).build(input_shape)
        
    def call(self, inputs):
        # Compute spatial attention weights
        attention_weights = self.conv_attention(inputs)
        
        # Apply attention
        attended_features = inputs * attention_weights
        
        return attended_features, attention_weights

def create_cnn_model(window_size: int, embedding_dim: int, 
                    num_channels: int = 1, use_attention: bool = True) -> keras.Model:
    """
    Create a CNN model for processing LSM waveforms and predicting next-token embeddings.
    
    Args:
        window_size: Size of the input waveform (window_size x window_size)
        embedding_dim: Dimension of the output token embedding
        num_channels: Number of input channels
        use_attention: Whether to use spatial attention
        
    Returns:
        Compiled Keras CNN model
    """
    
    # Input layer
    inputs = keras.Input(
        shape=(window_size, window_size, num_channels),
        name='waveform_input'
    )
    
    x = inputs
    
    # Optional spatial attention
    if use_attention:
        attention_block = SpatialAttentionBlock(name='spatial_attention')
        x, attention_weights = attention_block(x)
    
    # First convolutional block
    x = layers.Conv2D(
        filters=32,
        kernel_size=5,
        padding='same',
        activation='relu',
        name='conv1'
    )(x)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.MaxPool2D(pool_size=2, name='pool1')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)
    
    # Second convolutional block
    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='conv2'
    )(x)
    x = layers.BatchNormalization(name='bn2')(x)
    x = layers.MaxPool2D(pool_size=2, name='pool2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)
    
    # Third convolutional block
    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        padding='same',
        activation='relu',
        name='conv3'
    )(x)
    x = layers.BatchNormalization(name='bn3')(x)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.5, name='dropout3')(x)
    
    # Dense layers for embedding prediction
    x = layers.Dense(
        units=256,
        activation='relu',
        name='dense1'
    )(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Dropout(0.5, name='dropout_dense1')(x)
    
    x = layers.Dense(
        units=128,
        activation='relu',
        name='dense2'
    )(x)
    x = layers.BatchNormalization(name='bn_dense2')(x)
    x = layers.Dropout(0.3, name='dropout_dense2')(x)
    
    # Output layer - predict next token embedding
    outputs = layers.Dense(
        units=embedding_dim,
        activation='linear',  # Linear activation for regression
        name='embedding_output'
    )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='lsm_waveform_cnn')
    
    return model

def create_residual_cnn_model(window_size: int, embedding_dim: int,
                             num_channels: int = 1) -> keras.Model:
    """
    Create a CNN model with residual connections for better gradient flow.
    """
    
    inputs = keras.Input(
        shape=(window_size, window_size, num_channels),
        name='waveform_input'
    )
    
    # Initial convolution
    x = layers.Conv2D(64, 7, padding='same', name='initial_conv')(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.Activation('relu', name='initial_activation')(x)
    x = layers.MaxPool2D(3, strides=2, padding='same', name='initial_pool')(x)
    
    # Residual block 1
    shortcut = x
    x = layers.Conv2D(64, 3, padding='same', name='res1_conv1')(x)
    x = layers.BatchNormalization(name='res1_bn1')(x)
    x = layers.Activation('relu', name='res1_activation1')(x)
    x = layers.Conv2D(64, 3, padding='same', name='res1_conv2')(x)
    x = layers.BatchNormalization(name='res1_bn2')(x)
    x = layers.Add(name='res1_add')([x, shortcut])
    x = layers.Activation('relu', name='res1_activation2')(x)
    
    # Residual block 2
    shortcut = layers.Conv2D(128, 1, strides=2, name='res2_shortcut_conv')(x)
    shortcut = layers.BatchNormalization(name='res2_shortcut_bn')(shortcut)
    
    x = layers.Conv2D(128, 3, strides=2, padding='same', name='res2_conv1')(x)
    x = layers.BatchNormalization(name='res2_bn1')(x)
    x = layers.Activation('relu', name='res2_activation1')(x)
    x = layers.Conv2D(128, 3, padding='same', name='res2_conv2')(x)
    x = layers.BatchNormalization(name='res2_bn2')(x)
    x = layers.Add(name='res2_add')([x, shortcut])
    x = layers.Activation('relu', name='res2_activation2')(x)
    
    # Global pooling and final layers
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(embedding_dim, activation='linear', name='embedding_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name='residual_lsm_cnn')
    
    return model

def compile_cnn_model(model: keras.Model, learning_rate: float = 0.001) -> keras.Model:
    """
    Compile the CNN model with appropriate optimizer and loss function.
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for optimizer
        
    Returns:
        Compiled model
    """
    
    # Use Adam optimizer with custom learning rate
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Compile with MSE loss for embedding prediction
    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae', 'mse']
    )
    
    return model

def create_multi_scale_cnn(window_size: int, embedding_dim: int,
                          num_channels: int = 1) -> keras.Model:
    """
    Create a multi-scale CNN that processes the waveform at different scales.
    """
    
    inputs = keras.Input(
        shape=(window_size, window_size, num_channels),
        name='waveform_input'
    )
    
    # Different scale branches
    branches = []
    
    # Fine scale (small kernels)
    fine_branch = layers.Conv2D(32, 3, padding='same', activation='relu', name='fine_conv1')(inputs)
    fine_branch = layers.Conv2D(32, 3, padding='same', activation='relu', name='fine_conv2')(fine_branch)
    fine_branch = layers.MaxPool2D(2, name='fine_pool')(fine_branch)
    branches.append(fine_branch)
    
    # Medium scale
    medium_branch = layers.Conv2D(32, 5, padding='same', activation='relu', name='medium_conv1')(inputs)
    medium_branch = layers.Conv2D(32, 5, padding='same', activation='relu', name='medium_conv2')(medium_branch)
    medium_branch = layers.MaxPool2D(2, name='medium_pool')(medium_branch)
    branches.append(medium_branch)
    
    # Coarse scale (large kernels)
    coarse_branch = layers.Conv2D(32, 7, padding='same', activation='relu', name='coarse_conv1')(inputs)
    coarse_branch = layers.Conv2D(32, 7, padding='same', activation='relu', name='coarse_conv2')(coarse_branch)
    coarse_branch = layers.MaxPool2D(2, name='coarse_pool')(coarse_branch)
    branches.append(coarse_branch)
    
    # Concatenate multi-scale features
    x = layers.Concatenate(axis=-1, name='multi_scale_concat')(branches)
    
    # Process combined features
    x = layers.Conv2D(128, 3, padding='same', activation='relu', name='combined_conv1')(x)
    x = layers.BatchNormalization(name='combined_bn1')(x)
    x = layers.MaxPool2D(2, name='combined_pool')(x)
    x = layers.Dropout(0.25, name='combined_dropout')(x)
    
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dense(256, activation='relu', name='dense1')(x)
    x = layers.Dropout(0.5, name='dropout1')(x)
    x = layers.Dense(embedding_dim, activation='linear', name='embedding_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=x, name='multi_scale_lsm_cnn')
    
    return model

if __name__ == "__main__":
    # Test CNN models
    print("Testing CNN models...")
    
    window_size = 10
    embedding_dim = 128
    num_channels = 1
    
    # Test standard CNN
    print("Creating standard CNN...")
    cnn_model = create_cnn_model(window_size, embedding_dim, num_channels)
    cnn_model = compile_cnn_model(cnn_model)
    cnn_model.summary()
    
    # Test with dummy data
    dummy_input = np.random.random((8, window_size, window_size, num_channels)).astype(np.float32)
    dummy_output = cnn_model(dummy_input)
    print(f"Standard CNN output shape: {dummy_output.shape}")
    
    # Test residual CNN
    print("\nCreating residual CNN...")
    residual_model = create_residual_cnn_model(window_size, embedding_dim, num_channels)
    residual_model = compile_cnn_model(residual_model)
    residual_output = residual_model(dummy_input)
    print(f"Residual CNN output shape: {residual_output.shape}")
    
    # Test multi-scale CNN
    print("\nCreating multi-scale CNN...")
    multiscale_model = create_multi_scale_cnn(window_size, embedding_dim, num_channels)
    multiscale_model = compile_cnn_model(multiscale_model)
    multiscale_output = multiscale_model(dummy_input)
    print(f"Multi-scale CNN output shape: {multiscale_output.shape}")
    
    print("CNN model tests completed successfully!")
