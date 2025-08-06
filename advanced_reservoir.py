import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Dict, Optional, Tuple, Union
import math

from reservoir import SparseDense, ParametricSineActivation, generate_sparse_mask


class HierarchicalReservoir(layers.Layer):
    """
    Hierarchical reservoir with multiple temporal scales processing different aspects of input.
    Each scale has different connection patterns and temporal dynamics.
    """
    
    def __init__(self, scales: List[Dict], global_connectivity: float = 0.05, **kwargs):
        """
        Initialize hierarchical reservoir.
        
        Args:
            scales: List of scale configurations, each with:
                - units: number of units
                - sparsity: connection sparsity
                - time_constant: temporal decay constant
                - frequency_range: (min_freq, max_freq) for sine activations
            global_connectivity: inter-scale connection density
        """
        super(HierarchicalReservoir, self).__init__(**kwargs)
        self.scales = scales
        self.global_connectivity = global_connectivity
        self.num_scales = len(scales)
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Build scale-specific reservoirs
        self.scale_reservoirs = []
        self.scale_connections = []
        
        for i, scale_config in enumerate(self.scales):
            # Main reservoir for this scale
            reservoir_layers = []
            
            # Input layer for this scale
            if i == 0:
                current_input_dim = input_dim
            else:
                # Higher scales receive input from previous scale
                current_input_dim = input_dim + self.scales[i-1]['units']
            
            # Sparse dense layer
            mask = generate_sparse_mask(
                current_input_dim, 
                scale_config['units'], 
                scale_config['sparsity'],
                random_seed=42 + i
            )
            
            sparse_layer = SparseDense(
                units=scale_config['units'],
                mask=mask,
                use_bias=True,
                name=f'scale_{i}_sparse'
            )
            
            # Parametric sine activation with scale-specific frequency
            freq_min, freq_max = scale_config.get('frequency_range', (0.5, 2.0))
            initial_freq = freq_min + (freq_max - freq_min) * np.random.random()
            
            sine_activation = ParametricSineActivation(
                initial_frequency=initial_freq,
                initial_amplitude=1.0,
                initial_decay=scale_config.get('time_constant', 0.1),
                name=f'scale_{i}_sine'
            )
            
            reservoir_layers.append(sparse_layer)
            reservoir_layers.append(sine_activation)
            self.scale_reservoirs.append(reservoir_layers)
            
            # Inter-scale connections
            if i > 0:
                inter_mask = generate_sparse_mask(
                    self.scales[i-1]['units'],
                    scale_config['units'],
                    self.global_connectivity,
                    random_seed=100 + i
                )
                inter_connection = SparseDense(
                    units=scale_config['units'],
                    mask=inter_mask,
                    use_bias=False,
                    name=f'inter_scale_{i-1}_to_{i}'
                )
                self.scale_connections.append(inter_connection)
        
        super(HierarchicalReservoir, self).build(input_shape)
        
    def call(self, inputs):
        scale_outputs = []
        
        for i, (sparse_layer, sine_layer) in enumerate(self.scale_reservoirs):
            if i == 0:
                # First scale processes raw input
                scale_input = inputs
            else:
                # Higher scales get input + previous scale output
                scale_input = tf.concat([inputs, scale_outputs[i-1]], axis=-1)
            
            # Process through this scale
            x = sparse_layer(scale_input)
            x = sine_layer(x)
            scale_outputs.append(x)
        
        # Concatenate all scale outputs
        return tf.concat(scale_outputs, axis=-1)
    
    def get_scale_outputs(self, inputs):
        """Get individual outputs from each scale for analysis."""
        scale_outputs = []
        
        for i, (sparse_layer, sine_layer) in enumerate(self.scale_reservoirs):
            if i == 0:
                scale_input = inputs
            else:
                scale_input = tf.concat([inputs, scale_outputs[i-1]], axis=-1)
                if i-1 < len(self.scale_connections):
                    inter_contribution = self.scale_connections[i-1](scale_outputs[i-1])
                    scale_input = tf.concat([scale_input, inter_contribution], axis=-1)
            
            x = sparse_layer(scale_input)
            x = sine_layer(x)
            scale_outputs.append(x)
        
        return scale_outputs


class AttentiveReservoir(layers.Layer):
    """
    Reservoir with attention mechanism to dynamically weight different reservoir regions.
    """
    
    def __init__(self, units: int, num_heads: int = 4, sparsity: float = 0.1, 
                 attention_dim: int = 64, **kwargs):
        super(AttentiveReservoir, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.sparsity = sparsity
        self.attention_dim = attention_dim
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Main sparse reservoir
        self.reservoir_mask = generate_sparse_mask(input_dim, self.units, self.sparsity)
        self.sparse_layer = SparseDense(
            units=self.units,
            mask=self.reservoir_mask,
            use_bias=True,
            name='attentive_sparse'
        )
        
        self.sine_activation = ParametricSineActivation(
            initial_frequency=1.0,
            initial_amplitude=1.0,
            initial_decay=0.1,
            name='attentive_sine'
        )
        
        # Attention mechanism
        self.attention_query = layers.Dense(self.attention_dim, name='attention_query')
        self.attention_key = layers.Dense(self.attention_dim, name='attention_key')
        self.attention_value = layers.Dense(self.attention_dim, name='attention_value')
        
        # Multi-head attention
        self.multi_head_attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.attention_dim // self.num_heads,
            name='reservoir_attention'
        )
        
        # Output projection
        self.output_projection = layers.Dense(self.units, name='attention_output')
        
        super(AttentiveReservoir, self).build(input_shape)
        
    def call(self, inputs):
        # Standard reservoir processing
        reservoir_out = self.sparse_layer(inputs)
        reservoir_out = self.sine_activation(reservoir_out)
        
        # Reshape for attention (add sequence dimension)
        batch_size = tf.shape(inputs)[0]
        reservoir_reshaped = tf.expand_dims(reservoir_out, axis=1)  # (batch, 1, units)
        
        # Self-attention on reservoir units
        query = self.attention_query(reservoir_reshaped)
        key = self.attention_key(reservoir_reshaped)
        value = self.attention_value(reservoir_reshaped)
        
        attended_output = self.multi_head_attention(query, key, value)
        attended_output = tf.squeeze(attended_output, axis=1)  # Remove sequence dim
        
        # Project back to reservoir dimension
        attended_output = self.output_projection(attended_output)
        
        # Residual connection
        return reservoir_out + attended_output


class EchoStateReservoir(layers.Layer):
    """
    Echo State Network-style reservoir with fixed random weights and trainable output connections.
    """
    
    def __init__(self, units: int, spectral_radius: float = 0.9, sparsity: float = 0.1,
                 input_scaling: float = 1.0, **kwargs):
        super(EchoStateReservoir, self).__init__(**kwargs)
        self.units = units
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scaling = input_scaling
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Fixed random input weights
        self.W_in = self.add_weight(
            name='input_weights',
            shape=(input_dim, self.units),
            initializer='random_uniform',
            trainable=False  # Fixed weights
        )
        
        # Fixed random recurrent weights with spectral radius constraint
        W_rec_dense = np.random.randn(self.units, self.units)
        
        # Apply sparsity
        mask = np.random.random((self.units, self.units)) < self.sparsity
        W_rec_sparse = W_rec_dense * mask
        
        # Scale to desired spectral radius
        eigenvalues = np.linalg.eigvals(W_rec_sparse)
        current_radius = np.max(np.abs(eigenvalues))
        if current_radius > 0:
            W_rec_sparse = W_rec_sparse * (self.spectral_radius / current_radius)
        
        self.W_rec = tf.constant(W_rec_sparse.astype(np.float32))
        
        # Trainable sine activation parameters
        self.sine_activation = ParametricSineActivation(
            initial_frequency=1.0,
            initial_amplitude=1.0,
            initial_decay=0.1,
            name='echo_sine'
        )
        
        # State initialization
        self.initial_state = self.add_weight(
            name='initial_state',
            shape=(1, self.units),
            initializer='zeros',
            trainable=True
        )
        
        super(EchoStateReservoir, self).build(input_shape)
        
    def call(self, inputs, state=None):
        if state is None:
            batch_size = tf.shape(inputs)[0]
            state = tf.tile(self.initial_state, [batch_size, 1])
        
        # Echo state update: x(t+1) = f(W_in * u(t) + W_rec * x(t))
        input_contribution = tf.matmul(inputs * self.input_scaling, self.W_in)
        recurrent_contribution = tf.matmul(state, self.W_rec)
        
        new_state = input_contribution + recurrent_contribution
        new_state = self.sine_activation(new_state)
        
        return new_state


class DeepReservoir(layers.Layer):
    """
    Deep reservoir with multiple stacked layers and skip connections.
    """
    
    def __init__(self, layer_configs: List[Dict], use_skip_connections: bool = True, **kwargs):
        """
        Initialize deep reservoir.
        
        Args:
            layer_configs: List of configurations for each layer
            use_skip_connections: Whether to use skip connections between layers
        """
        super(DeepReservoir, self).__init__(**kwargs)
        self.layer_configs = layer_configs
        self.use_skip_connections = use_skip_connections
        self.num_layers = len(layer_configs)
        
    def build(self, input_shape):
        self.reservoir_layers = []
        self.skip_connections = []
        
        current_dim = input_shape[-1]
        
        for i, config in enumerate(self.layer_configs):
            # Sparse layer
            mask = generate_sparse_mask(
                current_dim, 
                config['units'], 
                config.get('sparsity', 0.1),
                random_seed=42 + i
            )
            
            sparse_layer = SparseDense(
                units=config['units'],
                mask=mask,
                use_bias=True,
                name=f'deep_sparse_{i}'
            )
            
            # Sine activation
            sine_layer = ParametricSineActivation(
                initial_frequency=config.get('frequency', 1.0 + i * 0.5),
                initial_amplitude=config.get('amplitude', 1.0),
                initial_decay=config.get('decay', 0.1),
                name=f'deep_sine_{i}'
            )
            
            self.reservoir_layers.append([sparse_layer, sine_layer])
            
            # Skip connection from input to this layer (if not first layer)
            if self.use_skip_connections and i > 0:
                skip_mask = generate_sparse_mask(
                    input_shape[-1],
                    config['units'],
                    config.get('skip_sparsity', 0.05),
                    random_seed=200 + i
                )
                skip_connection = SparseDense(
                    units=config['units'],
                    mask=skip_mask,
                    use_bias=False,
                    name=f'skip_connection_{i}'
                )
                self.skip_connections.append(skip_connection)
            else:
                self.skip_connections.append(None)
                
            current_dim = config['units']
        
        super(DeepReservoir, self).build(input_shape)
        
    def call(self, inputs):
        layer_outputs = []
        x = inputs
        
        for i, (sparse_layer, sine_layer) in enumerate(self.reservoir_layers):
            # Standard forward pass
            layer_out = sparse_layer(x)
            layer_out = sine_layer(layer_out)
            
            # Add skip connection from input
            if self.skip_connections[i] is not None:
                skip_contribution = self.skip_connections[i](inputs)
                layer_out = layer_out + skip_contribution
            
            layer_outputs.append(layer_out)
            x = layer_out
        
        # Concatenate outputs from all layers
        return tf.concat(layer_outputs, axis=-1)


def create_advanced_reservoir(architecture_type: str, input_dim: int, **kwargs) -> keras.Model:
    """
    Factory function to create different advanced reservoir architectures.
    
    Args:
        architecture_type: Type of reservoir ('hierarchical', 'attentive', 'echo_state', 'deep')
        input_dim: Input dimension
        **kwargs: Architecture-specific parameters
        
    Returns:
        Keras model with the specified reservoir architecture
    """
    inputs = keras.Input(shape=(input_dim,), name='reservoir_input')
    
    if architecture_type == 'hierarchical':
        scales = kwargs.get('scales', [
            {'units': 128, 'sparsity': 0.1, 'time_constant': 0.05, 'frequency_range': (0.5, 1.0)},
            {'units': 96, 'sparsity': 0.08, 'time_constant': 0.1, 'frequency_range': (1.0, 2.0)},
            {'units': 64, 'sparsity': 0.06, 'time_constant': 0.2, 'frequency_range': (2.0, 4.0)}
        ])
        global_connectivity = kwargs.get('global_connectivity', 0.05)
        reservoir = HierarchicalReservoir(scales=scales, global_connectivity=global_connectivity)
        
    elif architecture_type == 'attentive':
        units = kwargs.get('units', 256)
        num_heads = kwargs.get('num_heads', 4)
        sparsity = kwargs.get('sparsity', 0.1)
        attention_dim = kwargs.get('attention_dim', 64)
        reservoir = AttentiveReservoir(units=units, num_heads=num_heads, 
                                     sparsity=sparsity, attention_dim=attention_dim)
        
    elif architecture_type == 'echo_state':
        units = kwargs.get('units', 256)
        spectral_radius = kwargs.get('spectral_radius', 0.9)
        sparsity = kwargs.get('sparsity', 0.1)
        input_scaling = kwargs.get('input_scaling', 1.0)
        reservoir = EchoStateReservoir(units=units, spectral_radius=spectral_radius,
                                     sparsity=sparsity, input_scaling=input_scaling)
        
    elif architecture_type == 'deep':
        layer_configs = kwargs.get('layer_configs', [
            {'units': 256, 'sparsity': 0.1, 'frequency': 1.0},
            {'units': 128, 'sparsity': 0.08, 'frequency': 1.5},
            {'units': 64, 'sparsity': 0.06, 'frequency': 2.0}
        ])
        use_skip_connections = kwargs.get('use_skip_connections', True)
        reservoir = DeepReservoir(layer_configs=layer_configs, use_skip_connections=use_skip_connections)
        
    else:
        raise ValueError(f"Unknown architecture type: {architecture_type}")
    
    outputs = reservoir(inputs)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{architecture_type}_reservoir')
    return model


if __name__ == "__main__":
    # Test advanced reservoir architectures
    input_dim = 128
    test_input = tf.random.normal((16, input_dim))
    
    print("Testing Advanced Reservoir Architectures...")
    
    # Test Hierarchical Reservoir
    print("\n1. Hierarchical Reservoir:")
    hierarchical_model = create_advanced_reservoir('hierarchical', input_dim)
    hierarchical_output = hierarchical_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {hierarchical_output.shape}")
    hierarchical_model.summary()
    
    # Test Attentive Reservoir
    print("\n2. Attentive Reservoir:")
    attentive_model = create_advanced_reservoir('attentive', input_dim, units=256, num_heads=4)
    attentive_output = attentive_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {attentive_output.shape}")
    
    # Test Echo State Reservoir
    print("\n3. Echo State Reservoir:")
    echo_model = create_advanced_reservoir('echo_state', input_dim, units=256, spectral_radius=0.9)
    echo_output = echo_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {echo_output.shape}")
    
    # Test Deep Reservoir
    print("\n4. Deep Reservoir:")
    deep_model = create_advanced_reservoir('deep', input_dim)
    deep_output = deep_model(test_input)
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {deep_output.shape}")
    
    print("\nAll advanced reservoir architectures tested successfully!")