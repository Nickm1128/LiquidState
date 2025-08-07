"""
Mock TensorFlow module for testing when TensorFlow installation fails.
This provides minimal functionality to allow tests to run.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

class MockTensor:
    """Mock TensorFlow tensor."""
    def __init__(self, data):
        self.data = np.array(data)
        self.shape = self.data.shape
        self.dtype = self.data.dtype
    
    def numpy(self):
        return self.data

class MockModel:
    """Mock Keras model."""
    def __init__(self):
        self.history = None
        self.weights = []
    
    def compile(self, optimizer=None, loss=None, metrics=None):
        pass
    
    def fit(self, x, y, epochs=1, batch_size=32, validation_data=None, verbose=1):
        # Mock training history
        history = {
            'loss': [0.5, 0.4, 0.3][:epochs],
            'val_loss': [0.6, 0.5, 0.4][:epochs] if validation_data else None
        }
        self.history = type('History', (), {'history': history})()
        return self.history
    
    def predict(self, x):
        # Return mock predictions with same shape as input
        if isinstance(x, np.ndarray):
            return np.random.random((x.shape[0], 1))
        return np.random.random((len(x), 1))
    
    def save(self, filepath):
        # Mock save - just create an empty file
        with open(filepath, 'w') as f:
            f.write("mock_model")
    
    def save_weights(self, filepath):
        # Mock save weights
        with open(filepath, 'w') as f:
            f.write("mock_weights")
    
    def load_weights(self, filepath):
        pass

class MockKeras:
    """Mock Keras module."""
    class layers:
        @staticmethod
        def Dense(units, activation=None, **kwargs):
            return f"Dense({units}, {activation})"
        
        @staticmethod
        def LSTM(units, return_sequences=False, **kwargs):
            return f"LSTM({units}, return_sequences={return_sequences})"
        
        @staticmethod
        def Input(shape=None, **kwargs):
            return f"Input(shape={shape})"
        
        @staticmethod
        def Dropout(rate, **kwargs):
            return f"Dropout({rate})"
    
    class models:
        @staticmethod
        def Sequential(layers=None):
            return MockModel()
        
        @staticmethod
        def Model(inputs=None, outputs=None):
            return MockModel()
        
        @staticmethod
        def load_model(filepath):
            return MockModel()
    
    class optimizers:
        @staticmethod
        def Adam(learning_rate=0.001):
            return "Adam"
        
        @staticmethod
        def RMSprop(learning_rate=0.001):
            return "RMSprop"

class MockTensorFlow:
    """Mock TensorFlow module."""
    keras = MockKeras()
    
    @staticmethod
    def constant(value, dtype=None):
        return MockTensor(value)
    
    @staticmethod
    def random_normal(shape, mean=0.0, stddev=1.0):
        return MockTensor(np.random.normal(mean, stddev, shape))
    
    class random:
        @staticmethod
        def set_seed(seed):
            np.random.seed(seed)
    
    class nn:
        @staticmethod
        def tanh(x):
            if hasattr(x, 'data'):
                return MockTensor(np.tanh(x.data))
            return MockTensor(np.tanh(x))

# Create the mock tensorflow module
tf = MockTensorFlow()
keras = tf.keras

# Export the main components
__all__ = ['tf', 'keras']