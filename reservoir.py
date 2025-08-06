import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional
import math

class SparseDense(layers.Layer):
    """
    Dense layer with sparse connectivity defined by a binary mask.
    The mask is fixed during training to maintain sparse structure.
    """
    
    def __init__(self, units: int, mask: Optional[np.ndarray] = None, sparsity: float = 0.1, 
                 activation=None, use_bias: bool = True, **kwargs):
        super(SparseDense, self).__init__(**kwargs)
        self.units = units
        self.sparsity = sparsity
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self._mask = mask
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Initialize weights
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Create or use provided mask as a Variable to avoid graph scope issues
        if self._mask is not None:
            if self._mask.shape != (input_dim, self.units):
                raise ValueError(f"Mask shape {self._mask.shape} doesn't match kernel shape {(input_dim, self.units)}")
            mask_array = self._mask.astype(np.float32)
        else:
            # Generate random sparse mask
            mask = np.random.random((input_dim, self.units)) < self.sparsity
            mask_array = mask.astype(np.float32)
            
        # Use add_weight to create mask as a non-trainable variable with proper initialization
        self.mask = self.add_weight(
            name='mask',
            shape=(input_dim, self.units),
            initializer=keras.initializers.Constant(mask_array),
            trainable=False
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.bias = None
            
        super(SparseDense, self).build(input_shape)
        
    def call(self, inputs):
        # Apply sparse mask to kernel
        masked_kernel = self.kernel * self.mask
        
        # Perform matrix multiplication
        output = tf.matmul(inputs, masked_kernel)
        
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
            
        if self.activation is not None:
            output = self.activation(output)
            
        return output
    
    def get_config(self):
        config = super(SparseDense, self).get_config()
        config.update({
            'units': self.units,
            'sparsity': self.sparsity,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias
        })
        return config

class ParametricSineActivation(layers.Layer):
    """
    Parametric sine activation function that learns frequency, amplitude, and decay parameters.
    Computes: output = A * exp(-α * |x|) * sin(ω * x)
    """
    
    def __init__(self, initial_frequency: float = 1.0, initial_amplitude: float = 1.0, 
                 initial_decay: float = 0.1, **kwargs):
        super(ParametricSineActivation, self).__init__(**kwargs)
        self.initial_frequency = initial_frequency
        self.initial_amplitude = initial_amplitude
        self.initial_decay = initial_decay
        
    def build(self, input_shape):
        # Learnable parameters
        self.frequency = self.add_weight(
            name='frequency',
            shape=(),
            initializer=keras.initializers.Constant(self.initial_frequency),
            trainable=True,
            constraint=keras.constraints.NonNeg()  # Ensure frequency is non-negative
        )
        
        self.amplitude = self.add_weight(
            name='amplitude',
            shape=(),
            initializer=keras.initializers.Constant(self.initial_amplitude),
            trainable=True
        )
        
        self.decay = self.add_weight(
            name='decay',
            shape=(),
            initializer=keras.initializers.Constant(self.initial_decay),
            trainable=True,
            constraint=keras.constraints.NonNeg()  # Ensure decay is non-negative
        )
        
        super(ParametricSineActivation, self).build(input_shape)
        
    def call(self, inputs):
        # Compute parametric sine activation: A * exp(-α * |x|) * sin(ω * x)
        abs_inputs = tf.abs(inputs)
        decay_term = tf.exp(-self.decay * abs_inputs)
        sine_term = tf.sin(self.frequency * inputs)
        
        output = self.amplitude * decay_term * sine_term
        
        return output
    
    def get_config(self):
        config = super(ParametricSineActivation, self).get_config()
        config.update({
            'initial_frequency': self.initial_frequency,
            'initial_amplitude': self.initial_amplitude,
            'initial_decay': self.initial_decay
        })
        return config

def generate_sparse_mask(input_dim: int, output_dim: int, sparsity: float = 0.1, 
                        random_seed: int = None) -> np.ndarray:
    """
    Generate a binary sparse connectivity mask.
    
    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        sparsity: Fraction of connections to keep (0.1 = 10% connections)
        random_seed: Random seed for reproducibility
        
    Returns:
        Binary mask of shape (input_dim, output_dim)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    mask = np.random.random((input_dim, output_dim)) < sparsity
    
    # Ensure each output neuron has at least one connection
    for j in range(output_dim):
        if not np.any(mask[:, j]):
            i = np.random.randint(input_dim)
            mask[i, j] = True
    
    return mask.astype(np.float32)

def build_reservoir(input_dim: int, hidden_units: List[int], 
                   masks: Optional[List[np.ndarray]] = None,
                   sparsity: float = 0.1,
                   sine_params: Optional[List[dict]] = None) -> tf.keras.Model:
    """
    Build a reservoir model with sparse connections and sine activations.
    
    Args:
        input_dim: Input dimension
        hidden_units: List of hidden layer sizes
        masks: Optional list of pre-computed sparse masks for each layer
        sparsity: Sparsity level if masks are not provided
        sine_params: Optional list of sine activation parameters for each layer
        
    Returns:
        Keras model representing the reservoir
    """
    if masks is not None and len(masks) != len(hidden_units):
        raise ValueError("Number of masks must match number of hidden layers")
    
    if sine_params is not None and len(sine_params) != len(hidden_units):
        raise ValueError("Number of sine parameter sets must match number of hidden layers")
    
    # Input layer
    inputs = keras.Input(shape=(input_dim,), name='reservoir_input')
    x = inputs
    
    # Build reservoir layers
    for i, units in enumerate(hidden_units):
        # Get mask for this layer
        if masks is not None:
            mask = masks[i]
        else:
            current_input_dim = x.shape[-1]
            mask = generate_sparse_mask(current_input_dim, units, sparsity, random_seed=42+i)
        
        # Sparse dense layer
        x = SparseDense(
            units=units,
            mask=mask,
            use_bias=True,
            name=f'sparse_dense_{i+1}'
        )(x)
        
        # Parametric sine activation
        if sine_params is not None and i < len(sine_params):
            params = sine_params[i]
            x = ParametricSineActivation(
                initial_frequency=params.get('frequency', 1.0),
                initial_amplitude=params.get('amplitude', 1.0),
                initial_decay=params.get('decay', 0.1),
                name=f'sine_activation_{i+1}'
            )(x)
        else:
            x = ParametricSineActivation(
                initial_frequency=1.0 + i * 0.5,  # Vary frequency across layers
                initial_amplitude=1.0,
                initial_decay=0.1,
                name=f'sine_activation_{i+1}'
            )(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=x, name='sparse_sine_reservoir')
    
    return model

class ReservoirLayer(layers.Layer):
    """
    Complete reservoir layer combining sparse dense and sine activation.
    """
    
    def __init__(self, units: int, sparsity: float = 0.1, 
                 frequency: float = 1.0, amplitude: float = 1.0, decay: float = 0.1,
                 **kwargs):
        super(ReservoirLayer, self).__init__(**kwargs)
        self.units = units
        self.sparsity = sparsity
        self.frequency = frequency
        self.amplitude = amplitude
        self.decay = decay
        
    def build(self, input_shape):
        # Build sparse dense layer
        self.sparse_dense = SparseDense(
            units=self.units,
            sparsity=self.sparsity,
            use_bias=True
        )
        self.sparse_dense.build(input_shape)
        
        # Build sine activation
        self.sine_activation = ParametricSineActivation(
            initial_frequency=self.frequency,
            initial_amplitude=self.amplitude,
            initial_decay=self.decay
        )
        self.sine_activation.build((None, self.units))
        
        super(ReservoirLayer, self).build(input_shape)
        
    def call(self, inputs):
        x = self.sparse_dense(inputs)
        x = self.sine_activation(x)
        return x
    
    def get_config(self):
        config = super(ReservoirLayer, self).get_config()
        config.update({
            'units': self.units,
            'sparsity': self.sparsity,
            'frequency': self.frequency,
            'amplitude': self.amplitude,
            'decay': self.decay
        })
        return config

if __name__ == "__main__":
    # Test reservoir components
    print("Testing SparseDense layer...")
    sparse_layer = SparseDense(units=64, sparsity=0.1)
    test_input = tf.random.normal((32, 128))
    output = sparse_layer(test_input)
    print(f"SparseDense output shape: {output.shape}")
    
    print("\nTesting ParametricSineActivation...")
    sine_activation = ParametricSineActivation()
    sine_output = sine_activation(output)
    print(f"Sine activation output shape: {sine_output.shape}")
    
    print("\nTesting full reservoir...")
    reservoir = build_reservoir(input_dim=128, hidden_units=[64, 32, 16])
    reservoir.summary()
    
    test_output = reservoir(test_input)
    print(f"Reservoir output shape: {test_output.shape}")
    
    print("Reservoir component tests completed successfully!")
