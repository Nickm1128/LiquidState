import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from typing import Tuple, Dict, List, Any
import time
from datetime import datetime

from ..data.data_loader import load_data, DialogueTokenizer
from ..core.reservoir import build_reservoir
from ..core.advanced_reservoir import create_advanced_reservoir
from ..core.rolling_wave import RollingWaveBuffer, MultiChannelRollingWaveBuffer
from ..core.cnn_model import create_cnn_model, compile_cnn_model, create_residual_cnn_model
from .model_config import ModelConfiguration, TrainingMetadata
from lsm_exceptions import (
    ModelLoadError, ModelSaveError, TrainingSetupError, TrainingExecutionError,
    ConfigurationError, handle_file_operation_error
)
from lsm_logging import get_logger, log_performance, create_operation_logger

logger = get_logger(__name__)

def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class LSMTrainer:
    """
    Main trainer class that integrates reservoir computing with CNN for next-token prediction.
    """
    
    def __init__(self, window_size: int = 10, embedding_dim: int = 128,
                 reservoir_units: List[int] = None, sparsity: float = 0.1,
                 use_multichannel: bool = True, reservoir_type: str = 'standard',
                 reservoir_config: Dict = None, use_attention: bool = True):
        """
        Initialize the LSM trainer.
        
        Args:
            window_size: Size of temporal window for sequences
            embedding_dim: Dimension of token embeddings
            reservoir_units: List of hidden units for reservoir layers (for standard reservoir)
            sparsity: Sparsity level for reservoir connections
            use_multichannel: Whether to use multi-channel rolling wave buffer
            reservoir_type: Type of reservoir ('standard', 'hierarchical', 'attentive', 'echo_state', 'deep')
            reservoir_config: Configuration dictionary for advanced reservoirs
            use_attention: Whether to use spatial attention in CNN
        """
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.reservoir_units = reservoir_units or [256, 128, 64]
        self.sparsity = sparsity
        self.use_multichannel = use_multichannel
        self.reservoir_type = reservoir_type
        self.reservoir_config = reservoir_config or {}
        self.use_attention = use_attention
        
        # Models
        self.reservoir = None
        self.cnn_model = None
        
        # Training history
        self.history = {
            'train_mse': [],
            'test_mse': [],
            'train_mae': [],
            'test_mae': [],
            'epoch_times': []
        }
        
        # Set random seeds
        set_random_seeds()
        
    def build_models(self):
        """Build reservoir and CNN models."""
        print(f"Building {self.reservoir_type} reservoir model...")
        
        if self.reservoir_type == 'standard':
            self.reservoir = build_reservoir(
                input_dim=self.embedding_dim,
                hidden_units=self.reservoir_units,
                sparsity=self.sparsity
            )
            reservoir_output_dim = self.reservoir_units[-1]
        else:
            # Advanced reservoir architectures
            self.reservoir = create_advanced_reservoir(
                architecture_type=self.reservoir_type,
                input_dim=self.embedding_dim,
                **self.reservoir_config
            )
            # Calculate output dimension based on reservoir type
            reservoir_output_dim = self._calculate_reservoir_output_dim()
        
        print("Building CNN model...")
        num_channels = self._calculate_num_channels(reservoir_output_dim)
        self.cnn_model = create_cnn_model(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim,
            num_channels=num_channels,
            use_attention=self.use_attention
        )
        self.cnn_model = compile_cnn_model(self.cnn_model)
        
        print("Models built successfully!")
        print(f"Reservoir type: {self.reservoir_type}")
        print(f"Reservoir output shape: {self.reservoir.output.shape}")
        print(f"CNN input shape: {self.cnn_model.input.shape}")
        print(f"CNN output shape: {self.cnn_model.output.shape}")
    
    def _calculate_reservoir_output_dim(self) -> int:
        """Calculate the output dimension of the reservoir based on its type."""
        if self.reservoir_type == 'hierarchical':
            scales = self.reservoir_config.get('scales', [
                {'units': 128}, {'units': 96}, {'units': 64}
            ])
            return sum(scale['units'] for scale in scales)
        elif self.reservoir_type == 'attentive':
            return self.reservoir_config.get('units', 256)
        elif self.reservoir_type == 'echo_state':
            return self.reservoir_config.get('units', 256)
        elif self.reservoir_type == 'deep':
            layer_configs = self.reservoir_config.get('layer_configs', [
                {'units': 256}, {'units': 128}, {'units': 64}
            ])
            return sum(config['units'] for config in layer_configs)
        else:
            return 256  # Default fallback
    
    def _calculate_num_channels(self, reservoir_output_dim: int) -> int:
        """Calculate number of channels for rolling wave buffer."""
        if not self.use_multichannel:
            return 1
        
        if self.reservoir_type == 'standard':
            return len(self.reservoir_units)
        elif self.reservoir_type == 'hierarchical':
            return len(self.reservoir_config.get('scales', [{'units': 128}, {'units': 96}, {'units': 64}]))
        elif self.reservoir_type == 'deep':
            return len(self.reservoir_config.get('layer_configs', [{'units': 256}, {'units': 128}, {'units': 64}]))
        else:
            # For attentive and echo_state, use single channel but larger dimension
            return 1
    
    def process_sequence_to_waveform(self, sequence: np.ndarray) -> np.ndarray:
        """
        Process a single sequence through the reservoir to create 2D waveform.
        
        Args:
            sequence: Input sequence of shape (window_size, embedding_dim)
            
        Returns:
            2D waveform of shape (window_size, window_size, channels)
        """
        num_channels = self._calculate_num_channels(self._calculate_reservoir_output_dim())
        
        if self.use_multichannel and num_channels > 1:
            # Multi-channel approach
            buffer = MultiChannelRollingWaveBuffer(
                window_size=self.window_size,
                num_channels=num_channels
            )
            buffer.reset()
            
            # Process each timestep
            for t in range(self.window_size):
                timestep_embedding = sequence[t:t+1]  # Shape: (1, embedding_dim)
                
                # Get outputs based on reservoir type
                layer_outputs = self._extract_layer_outputs(timestep_embedding)
                
                # Append to buffer
                buffer.append_waves(layer_outputs)
            
            return buffer.get_buffer_3d()
        
        else:
            # Single channel approach
            buffer = RollingWaveBuffer(window_size=self.window_size)
            buffer.reset()
            
            # Process each timestep
            for t in range(self.window_size):
                timestep_embedding = sequence[t:t+1]  # Shape: (1, embedding_dim)
                
                # Get reservoir output
                reservoir_output = self.reservoir(timestep_embedding)
                wave = reservoir_output.numpy().flatten()
                
                # Append to buffer
                buffer.append_wave(wave)
            
            return buffer.get_buffer_3d()
    
    def _extract_layer_outputs(self, timestep_embedding: tf.Tensor) -> List[np.ndarray]:
        """Extract layer-wise outputs from reservoir for multi-channel processing."""
        if self.reservoir_type == 'standard':
            # Extract outputs from each layer in standard reservoir
            layer_outputs = []
            x = timestep_embedding
            for i, layer in enumerate(self.reservoir.layers):
                if hasattr(layer, 'units'):  # Dense or SparseDense layer
                    x = layer(x)
                    # Get the output after sine activation (next layer)
                    if i + 1 < len(self.reservoir.layers):
                        x = self.reservoir.layers[i + 1](x)
                        layer_outputs.append(x.numpy().flatten())
                    i += 1  # Skip sine activation layer in next iteration
            
            # Ensure we have the right number of outputs
            if len(layer_outputs) != len(self.reservoir_units):
                final_output = self.reservoir(timestep_embedding).numpy().flatten()
                layer_outputs = [final_output[i::len(self.reservoir_units)] 
                               for i in range(len(self.reservoir_units))]
            
            return layer_outputs
            
        elif self.reservoir_type == 'hierarchical':
            # Extract outputs from each hierarchical scale
            reservoir_output = self.reservoir(timestep_embedding)
            scales = self.reservoir_config.get('scales', [
                {'units': 128}, {'units': 96}, {'units': 64}
            ])
            
            outputs = []
            start_idx = 0
            for scale in scales:
                end_idx = start_idx + scale['units']
                scale_output = reservoir_output[:, start_idx:end_idx].numpy().flatten()
                outputs.append(scale_output)
                start_idx = end_idx
            
            return outputs
            
        elif self.reservoir_type == 'deep':
            # Extract outputs from each deep layer
            layer_configs = self.reservoir_config.get('layer_configs', [
                {'units': 256}, {'units': 128}, {'units': 64}
            ])
            
            # For deep reservoir, we need to access intermediate layer outputs
            # This is a simplified version - in practice, you might want to store 
            # intermediate outputs during forward pass
            reservoir_output = self.reservoir(timestep_embedding).numpy().flatten()
            
            outputs = []
            start_idx = 0
            for config in layer_configs:
                end_idx = start_idx + config['units']
                layer_output = reservoir_output[start_idx:end_idx]
                outputs.append(layer_output)
                start_idx = end_idx
            
            return outputs
            
        else:
            # For attentive and echo_state reservoirs, use single output
            reservoir_output = self.reservoir(timestep_embedding).numpy().flatten()
            return [reservoir_output]
    
    def create_training_data(self, X: np.ndarray, y: np.ndarray, 
                           batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert input sequences to 2D waveforms for CNN training.
        
        Args:
            X: Input sequences of shape (num_samples, window_size, embedding_dim)
            y: Target embeddings of shape (num_samples, embedding_dim)
            batch_size: Batch size for processing (to manage memory)
            
        Returns:
            X_waveforms: 2D waveforms for CNN input
            y: Target embeddings (unchanged)
        """
        print(f"Converting {len(X)} sequences to waveforms...")
        
        num_samples = len(X)
        num_channels = self._calculate_num_channels(self._calculate_reservoir_output_dim())
        waveform_shape = (num_samples, self.window_size, self.window_size, num_channels)
        
        # Use memory-mapped array for large datasets
        if num_samples > 1000:
            import tempfile
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            X_waveforms = np.memmap(temp_file.name, dtype=np.float32, mode='w+', shape=waveform_shape)
        else:
            X_waveforms = np.zeros(waveform_shape, dtype=np.float32)
        
        # Process in batches to manage memory
        for i in range(0, num_samples, batch_size):
            end_i = min(i + batch_size, num_samples)
            batch_X = X[i:end_i]
            
            print(f"Processing batch {i//batch_size + 1}/{(num_samples + batch_size - 1)//batch_size}")
            
            for j, sequence in enumerate(batch_X):
                waveform = self.process_sequence_to_waveform(sequence)
                X_waveforms[i + j] = waveform
        
        return X_waveforms, y
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              epochs: int = 10, batch_size: int = 32,
              validation_split: float = 0.1) -> Dict:
        """
        Train the complete LSM + CNN system.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_test: Test sequences
            y_test: Test targets
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of training data for validation
            
        Returns:
            Training history dictionary
        """
        print("Starting LSM training...")
        
        # Build models if not already built
        if self.reservoir is None or self.cnn_model is None:
            self.build_models()
        
        # Convert sequences to waveforms
        print("Creating training waveforms...")
        X_train_waveforms, y_train = self.create_training_data(X_train, y_train, batch_size)
        
        print("Creating test waveforms...")
        X_test_waveforms, y_test = self.create_training_data(X_test, y_test, batch_size)
        
        # Set up callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1,
                min_lr=1e-6
            )
        ]
        
        # Train CNN
        print("Training CNN on waveforms...")
        start_time = time.time()
        
        history = self.cnn_model.fit(
            X_train_waveforms, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        # Evaluate on test set
        print("Evaluating on test set...")
        test_loss, test_mae, test_mse = self.cnn_model.evaluate(
            X_test_waveforms, y_test, verbose=0
        )
        
        # Update history
        self.history['train_mse'].extend(history.history['mse'])
        self.history['train_mae'].extend(history.history['mae'])
        if 'val_mse' in history.history:
            self.history['test_mse'].extend(history.history['val_mse'])
            self.history['test_mae'].extend(history.history['val_mae'])
        
        # Print final results
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Final test MSE: {test_mse:.6f}")
        print(f"Final test MAE: {test_mae:.6f}")
        
        return {
            'history': history.history,
            'test_mse': test_mse,
            'test_mae': test_mae,
            'training_time': training_time
        }
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions on new sequences."""
        X_waveforms, _ = self.create_training_data(X, np.zeros((len(X), self.embedding_dim)), batch_size)
        return self.cnn_model.predict(X_waveforms, batch_size=batch_size)
    
    def save_complete_model(self, save_dir: str, tokenizer: DialogueTokenizer, 
                           training_results: Dict = None, dataset_info: Dict = None) -> None:
        """
        Save complete model state including reservoir, CNN, tokenizer, and configuration.
        
        Args:
            save_dir: Directory to save all model artifacts
            tokenizer: Fitted tokenizer to save with the model
            training_results: Optional training results for metadata
            dataset_info: Optional dataset information for metadata
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        if self.reservoir is not None:
            reservoir_path = os.path.join(save_dir, "reservoir_model.keras")
            self.reservoir.save(reservoir_path)
            print(f"✓ Reservoir model saved to {reservoir_path}")
        
        if self.cnn_model is not None:
            cnn_path = os.path.join(save_dir, "cnn_model.keras")
            self.cnn_model.save(cnn_path)
            print(f"✓ CNN model saved to {cnn_path}")
        
        # Save tokenizer
        if tokenizer is not None and tokenizer.is_fitted:
            tokenizer_path = os.path.join(save_dir, "tokenizer")
            tokenizer.save(tokenizer_path)
            print(f"✓ Tokenizer saved to {tokenizer_path}")
        
        # Create and save configuration
        config = ModelConfiguration(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim,
            reservoir_type=self.reservoir_type,
            reservoir_config=self.reservoir_config,
            reservoir_units=self.reservoir_units,
            sparsity=self.sparsity,
            use_multichannel=self.use_multichannel,
            tokenizer_max_features=tokenizer.max_features if tokenizer else 10000,
            tokenizer_ngram_range=(1, 2)  # Default from tokenizer
        )
        
        config_path = os.path.join(save_dir, "config.json")
        config.save(config_path)
        print(f"✓ Configuration saved to {config_path}")
        
        # Save training history
        if self.history:
            try:
                # Ensure all arrays have the same length before creating DataFrame
                max_len = max(len(v) for v in self.history.values() if isinstance(v, list))
                cleaned_history = {}
                for key, value in self.history.items():
                    if isinstance(value, list):
                        # Pad shorter lists with None or use only the first max_len items
                        if len(value) < max_len:
                            cleaned_history[key] = value + [None] * (max_len - len(value))
                        else:
                            cleaned_history[key] = value[:max_len]
                    else:
                        cleaned_history[key] = value
                
                history_df = pd.DataFrame(cleaned_history)
                history_path = os.path.join(save_dir, "training_history.csv")
                history_df.to_csv(history_path, index=False)
                print(f"✓ Training history saved to {history_path}")
            except Exception as e:
                print(f"⚠ Warning: Could not save training history: {e}")
        
        # Save training metadata if provided
        if training_results and dataset_info:
            metadata = TrainingMetadata.create_from_training(training_results, dataset_info)
            metadata_path = os.path.join(save_dir, "metadata.json")
            metadata.save(metadata_path)
            print(f"✓ Training metadata saved to {metadata_path}")
        
        print(f"🎉 Complete model saved to {save_dir}")
    
    def load_complete_model(self, save_dir: str) -> Tuple['LSMTrainer', DialogueTokenizer]:
        """
        Load complete model state including configuration and tokenizer.
        
        Args:
            save_dir: Directory containing saved model artifacts
            
        Returns:
            Tuple of (loaded_trainer, loaded_tokenizer)
        """
        if not os.path.exists(save_dir):
            raise ModelLoadError(save_dir, "Model directory does not exist")
        
        # Load configuration
        config_path = os.path.join(save_dir, "config.json")
        if os.path.exists(config_path):
            config = ModelConfiguration.load(config_path)
            print(f"✓ Configuration loaded from {config_path}")
            
            # Update trainer parameters from config
            self.window_size = config.window_size
            self.embedding_dim = config.embedding_dim
            self.reservoir_type = config.reservoir_type
            self.reservoir_config = config.reservoir_config
            self.reservoir_units = config.reservoir_units
            self.sparsity = config.sparsity
            self.use_multichannel = config.use_multichannel
        else:
            print("⚠ No configuration file found, using current trainer settings")
        
        # Load models
        reservoir_path = os.path.join(save_dir, "reservoir_model.keras")
        if os.path.exists(reservoir_path):
            self.reservoir = keras.models.load_model(reservoir_path)
            print(f"✓ Reservoir model loaded from {reservoir_path}")
        else:
            print("⚠ No reservoir model found")
        
        cnn_path = os.path.join(save_dir, "cnn_model.keras")
        if os.path.exists(cnn_path):
            self.cnn_model = keras.models.load_model(cnn_path)
            print(f"✓ CNN model loaded from {cnn_path}")
        else:
            print("⚠ No CNN model found")
        
        # Load tokenizer
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        tokenizer = DialogueTokenizer(embedding_dim=self.embedding_dim)
        
        if os.path.exists(tokenizer_path):
            tokenizer.load(tokenizer_path)
            print(f"✓ Tokenizer loaded from {tokenizer_path}")
        else:
            print("⚠ No tokenizer found - you'll need to fit it on your data")
        
        # Load training history if exists
        history_path = os.path.join(save_dir, "training_history.csv")
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            for col in history_df.columns:
                if col in self.history:
                    self.history[col] = history_df[col].tolist()
            print(f"✓ Training history loaded from {history_path}")
        
        return self, tokenizer
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current model state.
        
        Returns:
            Dictionary containing model information
        """
        info = {
            'architecture': {
                'window_size': self.window_size,
                'embedding_dim': self.embedding_dim,
                'reservoir_type': self.reservoir_type,
                'reservoir_config': self.reservoir_config,
                'reservoir_units': self.reservoir_units,
                'sparsity': self.sparsity,
                'use_multichannel': self.use_multichannel
            },
            'model_state': {
                'reservoir_loaded': self.reservoir is not None,
                'cnn_loaded': self.cnn_model is not None,
                'has_training_history': len(self.history.get('train_mse', [])) > 0
            }
        }
        
        # Add model shapes if available
        if self.reservoir is not None:
            info['model_state']['reservoir_input_shape'] = str(self.reservoir.input.shape)
            info['model_state']['reservoir_output_shape'] = str(self.reservoir.output.shape)
        
        if self.cnn_model is not None:
            info['model_state']['cnn_input_shape'] = str(self.cnn_model.input.shape)
            info['model_state']['cnn_output_shape'] = str(self.cnn_model.output.shape)
        
        # Add training history summary if available
        if self.history.get('train_mse'):
            info['training_summary'] = {
                'epochs_trained': len(self.history['train_mse']),
                'final_train_mse': float(self.history['train_mse'][-1]),
                'final_train_mae': float(self.history['train_mae'][-1]) if self.history.get('train_mae') else None,
                'best_train_mse': float(min(self.history['train_mse']))
            }
            
            if self.history.get('test_mse'):
                info['training_summary'].update({
                    'final_test_mse': float(self.history['test_mse'][-1]),
                    'final_test_mae': float(self.history['test_mae'][-1]) if self.history.get('test_mae') else None,
                    'best_test_mse': float(min(self.history['test_mse']))
                })
        
        return info
    
    def save_models(self, save_dir: str = "saved_models"):
        """Save trained models (legacy method - use save_complete_model for new code)."""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.reservoir is not None:
            self.reservoir.save(os.path.join(save_dir, "reservoir_model.keras"))
        
        if self.cnn_model is not None:
            self.cnn_model.save(os.path.join(save_dir, "cnn_model.keras"))
        
        # Save training history  
        try:
            # Ensure all arrays have the same length before creating DataFrame
            if self.history:
                max_len = max(len(v) for v in self.history.values() if isinstance(v, list))
                cleaned_history = {}
                for key, value in self.history.items():
                    if isinstance(value, list):
                        if len(value) < max_len:
                            cleaned_history[key] = value + [None] * (max_len - len(value))
                        else:
                            cleaned_history[key] = value[:max_len]
                    else:
                        cleaned_history[key] = value
                
                history_df = pd.DataFrame(cleaned_history)
                history_df.to_csv(os.path.join(save_dir, "training_history.csv"), index=False)
        except Exception as e:
            print(f"Warning: Could not save training history: {e}")
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str = "saved_models"):
        """Load pre-trained models (legacy method - use load_complete_model for new code)."""
        reservoir_path = os.path.join(save_dir, "reservoir_model.keras")
        cnn_path = os.path.join(save_dir, "cnn_model.keras")
        
        if os.path.exists(reservoir_path):
            self.reservoir = keras.models.load_model(reservoir_path)
            print("Reservoir model loaded")
        
        if os.path.exists(cnn_path):
            self.cnn_model = keras.models.load_model(cnn_path)
            print("CNN model loaded")
        
        # Load training history if exists
        history_path = os.path.join(save_dir, "training_history.csv")
        if os.path.exists(history_path):
            history_df = pd.read_csv(history_path)
            for col in history_df.columns:
                if col in self.history:
                    self.history[col] = history_df[col].tolist()

@log_performance("LSM training")
def run_training(window_size: int = 10, batch_size: int = 32, epochs: int = 20,
                test_size: float = 0.2, embedding_dim: int = 128,
                reservoir_type: str = 'standard', reservoir_config: Dict = None,
                use_attention: bool = True) -> Dict:
    """
    Main training function that ties everything together.
    
    Args:
        window_size: Size of sequence windows
        batch_size: Training batch size
        epochs: Number of training epochs
        test_size: Fraction of data for testing
        embedding_dim: Token embedding dimension
        reservoir_type: Type of reservoir architecture
        reservoir_config: Configuration for advanced reservoirs
        use_attention: Whether to use spatial attention in CNN
        
    Returns:
        Training results dictionary
    """
    # Validate input parameters
    from input_validation import (
        validate_positive_integer, validate_positive_float, 
        validate_training_parameters, create_helpful_error_message
    )
    
    try:
        window_size = validate_positive_integer(window_size, "window_size", min_value=1, max_value=100)
        batch_size = validate_positive_integer(batch_size, "batch_size", min_value=1, max_value=1024)
        epochs = validate_positive_integer(epochs, "epochs", min_value=1, max_value=1000)
        test_size = validate_positive_float(test_size, "test_size", min_value=0.01, max_value=0.9)
        embedding_dim = validate_positive_integer(embedding_dim, "embedding_dim", min_value=1, max_value=2048)
        
        valid_reservoir_types = ['standard', 'hierarchical', 'attentive', 'echo_state', 'deep']
        if reservoir_type not in valid_reservoir_types:
            raise ConfigurationError(f"Invalid reservoir_type: {reservoir_type}. Valid types: {valid_reservoir_types}")
            
    except Exception as e:
        error_msg = create_helpful_error_message(
            "Training parameter validation",
            e,
            [
                "Check that all numeric parameters are positive",
                "Ensure test_size is between 0.01 and 0.9",
                f"Use one of these reservoir types: {valid_reservoir_types}"
            ]
        )
        logger.error(error_msg)
        raise TrainingSetupError(str(e))
    
    logger.info("Starting LSM training", 
                window_size=window_size, 
                batch_size=batch_size, 
                epochs=epochs,
                reservoir_type=reservoir_type)
    
    print("="*80)
    print(f"LIQUID STATE MACHINE TRAINING - {reservoir_type.upper()} RESERVOIR")
    print("="*80)
    
    # Load and prepare data
    print("Loading data...")
    try:
        X_train, y_train, X_test, y_test, tokenizer = load_data(
            window_size=window_size,
            test_size=test_size,
            embedding_dim=embedding_dim
        )
        
        print(f"Data loaded: Train={X_train.shape}, Test={X_test.shape}")
        logger.info("Data loading completed successfully", 
                   train_shape=X_train.shape, 
                   test_shape=X_test.shape)
        
    except Exception as e:
        logger.exception("Data loading failed")
        error_msg = create_helpful_error_message(
            "Data loading",
            e,
            [
                "Check that the dataset file exists and is readable",
                "Ensure sufficient memory is available for data processing",
                "Verify that the data format is correct"
            ]
        )
        print(error_msg)
        raise TrainingSetupError(f"Data loading failed: {e}")
    
    # Initialize trainer with advanced reservoir support
    trainer = LSMTrainer(
        window_size=window_size,
        embedding_dim=embedding_dim,
        reservoir_units=[256, 128, 64],
        sparsity=0.1,
        use_multichannel=True,
        reservoir_type=reservoir_type,
        reservoir_config=reservoir_config or {},
        use_attention=use_attention
    )
    
    # Train the model
    results = trainer.train(
        X_train, y_train, X_test, y_test,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1
    )
    
    # Save models with complete state
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models_{timestamp}"
    
    # Create dataset info for metadata
    dataset_info = {
        'source': 'Synthetic-Persona-Chat',
        'num_sequences': len(X_train) + len(X_test),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'window_size': window_size,
        'embedding_dim': embedding_dim
    }
    
    trainer.save_complete_model(save_dir, tokenizer, results, dataset_info)
    
    return results

if __name__ == "__main__":
    # Run training with default parameters
    results = run_training(
        window_size=8,
        batch_size=16,
        epochs=15,
        test_size=0.2,
        embedding_dim=128
    )
    
    print("\nTraining completed!")
    print(f"Final test MSE: {results['test_mse']:.6f}")
    print(f"Final test MAE: {results['test_mae']:.6f}")
