import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from typing import Tuple, Dict, List, Any, Optional
import time
from datetime import datetime

# Updated imports for new tokenization and dataset integration
from ..data.data_loader import load_data, DialogueTokenizer  # Keep for backward compatibility
from ..data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from ..data.huggingface_loader import HuggingFaceDatasetLoader, ConversationSplitter, DatasetProcessor
from ..core.reservoir import build_reservoir
from ..core.advanced_reservoir import create_advanced_reservoir
from ..core.rolling_wave import RollingWaveBuffer, MultiChannelRollingWaveBuffer
from ..core.cnn_model import create_cnn_model, compile_cnn_model, create_residual_cnn_model
from ..core.cnn_architecture_factory import CNNArchitectureFactory
from ..core.cnn_3d_processor import CNN3DProcessor, SystemContext
from ..core.loss_functions import (
    CosineSimilarityLoss, ResponseLevelCosineLoss, CNNLossCalculator,
    create_cosine_similarity_loss, create_response_level_loss
)
from ..core.system_message_processor import SystemMessageProcessor
from ..inference.response_generator import ResponseGenerator, TokenEmbeddingSequence
from ..inference.response_inference_model import ResponseInferenceModel, TrainingConfig
from .model_config import ModelConfiguration, TrainingMetadata
from ..utils.lsm_exceptions import (
    ModelLoadError, ModelSaveError, TrainingSetupError, TrainingExecutionError,
    ConfigurationError, handle_file_operation_error
)
from ..utils.lsm_logging import get_logger, log_performance, create_operation_logger

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
                 reservoir_config: Dict = None, use_attention: bool = True,
                 tokenizer_name: str = 'gpt2', use_huggingface_data: bool = True,
                 cache_dir: str = "data/huggingface_cache"):
        """
        Initialize the LSM trainer with enhanced tokenization and dataset support.
        
        Args:
            window_size: Size of temporal window for sequences
            embedding_dim: Dimension of token embeddings
            reservoir_units: List of hidden units for reservoir layers (for standard reservoir)
            sparsity: Sparsity level for reservoir connections
            use_multichannel: Whether to use multi-channel rolling wave buffer
            reservoir_type: Type of reservoir ('standard', 'hierarchical', 'attentive', 'echo_state', 'deep')
            reservoir_config: Configuration dictionary for advanced reservoirs
            use_attention: Whether to use spatial attention in CNN
            tokenizer_name: Name of standard tokenizer to use ('gpt2', 'bert-base-uncased', etc.)
            use_huggingface_data: Whether to use HuggingFace dataset integration
            cache_dir: Directory for caching HuggingFace datasets
        """
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.reservoir_units = reservoir_units or [256, 128, 64]
        self.sparsity = sparsity
        self.use_multichannel = use_multichannel
        self.reservoir_type = reservoir_type
        self.reservoir_config = reservoir_config or {}
        self.use_attention = use_attention
        
        # New tokenization and dataset parameters
        self.tokenizer_name = tokenizer_name
        self.use_huggingface_data = use_huggingface_data
        self.cache_dir = cache_dir
        
        # Models and components
        self.reservoir = None
        self.cnn_model = None
        self.tokenizer = None  # Will be StandardTokenizerWrapper or DialogueTokenizer
        self.embedder = None   # SinusoidalEmbedder
        
        # Dataset components
        self.dataset_loader = None
        self.conversation_splitter = None
        self.dataset_processor = None
        
        # Training history
        self.history = {
            'train_mse': [],
            'test_mse': [],
            'train_mae': [],
            'test_mae': [],
            'epoch_times': [],
            # Response-level training metrics
            'train_cosine_loss': [],
            'test_cosine_loss': [],
            'response_coherence': [],
            'system_influence': []
        }
        
        # Response-level training components
        self.cnn_architecture_factory = None
        self.cnn_3d_processor = None
        self.system_message_processor = None
        self.response_generator = None
        self.response_inference_model = None
        self.response_level_mode = False
        
        # 3D CNN and system message training components
        self.embedding_modifier_generator = None
        self.system_training_enabled = False
        self.use_3d_cnn_training = False
        
        # Set random seeds
        set_random_seeds()
    
    def initialize_tokenization_system(self, max_length: int = 512):
        """
        Initialize the new tokenization and embedding system.
        
        Args:
            max_length: Maximum sequence length for tokenization
        """
        logger.info(f"Initializing tokenization system with {self.tokenizer_name}")
        
        try:
            # Initialize standard tokenizer wrapper
            self.tokenizer = StandardTokenizerWrapper(
                tokenizer_name=self.tokenizer_name,
                max_length=max_length
            )
            
            # Initialize sinusoidal embedder
            vocab_size = self.tokenizer.get_vocab_size()
            self.embedder = SinusoidalEmbedder(
                vocab_size=vocab_size,
                embedding_dim=self.embedding_dim
            )
            
            logger.info(f"Tokenization system initialized with vocab_size={vocab_size}")
            
        except Exception as e:
            logger.error(f"Failed to initialize tokenization system: {e}")
            # Fallback to legacy tokenizer
            logger.warning("Falling back to legacy DialogueTokenizer")
            self.tokenizer = DialogueTokenizer(embedding_dim=self.embedding_dim)
            self.embedder = None
    
    def initialize_dataset_components(self):
        """Initialize HuggingFace dataset integration components."""
        if self.use_huggingface_data:
            logger.info("Initializing HuggingFace dataset components")
            
            self.dataset_loader = HuggingFaceDatasetLoader(cache_dir=self.cache_dir)
            self.conversation_splitter = ConversationSplitter()
            self.dataset_processor = DatasetProcessor()
            
            logger.info("HuggingFace dataset components initialized")
        else:
            logger.info("Using legacy data loading system")
    
    def load_and_prepare_data(self, test_ratio: float = 0.2, force_download: bool = False,
                            fit_embedder: bool = True, embedding_epochs: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and prepare data using the new HuggingFace integration or legacy system.
        
        Args:
            test_ratio: Ratio of data to use for testing
            force_download: Whether to force re-download of HuggingFace data
            fit_embedder: Whether to fit the sinusoidal embedder
            embedding_epochs: Number of epochs for embedding optimization
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        # Initialize components if not already done
        if self.tokenizer is None:
            self.initialize_tokenization_system()
        
        if self.use_huggingface_data and self.dataset_loader is None:
            self.initialize_dataset_components()
        
        if self.use_huggingface_data:
            return self._load_huggingface_data(test_ratio, force_download, fit_embedder, embedding_epochs)
        else:
            return self._load_legacy_data(test_ratio)
    
    def _load_huggingface_data(self, test_ratio: float, force_download: bool, 
                              fit_embedder: bool, embedding_epochs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data using HuggingFace dataset integration."""
        logger.info("Loading data using HuggingFace dataset integration")
        
        try:
            # Download and load dataset
            csv_files = self.dataset_loader.download_cosmopedia_csvs(force_download=force_download)
            df = self.dataset_loader.load_cached_datasets()
            
            # Validate dataset
            self.dataset_loader.validate_dataset_integrity(df)
            
            # Process dataset for training
            processed_df = self.dataset_processor.prepare_for_training(df)
            
            # Split by conversations
            train_df, test_df = self.conversation_splitter.split_by_conversation(
                processed_df, test_ratio=test_ratio
            )
            
            # Verify conversation integrity
            self.conversation_splitter.ensure_conversation_integrity(train_df, test_df)
            
            # Extract text data
            train_texts = train_df['text'].tolist() if 'text' in train_df.columns else []
            test_texts = test_df['text'].tolist() if 'text' in test_df.columns else []
            
            if not train_texts or not test_texts:
                raise TrainingSetupError("No text data found in processed dataset")
            
            # Tokenize texts
            logger.info(f"Tokenizing {len(train_texts)} training texts and {len(test_texts)} test texts")
            
            if isinstance(self.tokenizer, StandardTokenizerWrapper):
                # Use new tokenization system
                train_token_ids = self.tokenizer.tokenize(train_texts, padding=True, truncation=True)
                test_token_ids = self.tokenizer.tokenize(test_texts, padding=True, truncation=True)
                
                # Convert to numpy arrays
                train_token_ids = np.array(train_token_ids)
                test_token_ids = np.array(test_token_ids)
                
                # Fit sinusoidal embedder if requested
                if fit_embedder and self.embedder is not None:
                    logger.info(f"Fitting sinusoidal embedder for {embedding_epochs} epochs")
                    self.embedder.fit(train_token_ids, epochs=embedding_epochs)
                
                # Convert to embeddings
                if self.embedder is not None and self.embedder._is_fitted:
                    train_embeddings = self.embedder.embed(train_token_ids)
                    test_embeddings = self.embedder.embed(test_token_ids)
                else:
                    # Fallback: create simple embeddings from token IDs
                    logger.warning("Using simple token ID embeddings (embedder not fitted)")
                    train_embeddings = self._token_ids_to_simple_embeddings(train_token_ids)
                    test_embeddings = self._token_ids_to_simple_embeddings(test_token_ids)
            else:
                # Use legacy tokenization system
                logger.info("Using legacy tokenization system")
                all_texts = train_texts + test_texts
                self.tokenizer.fit(all_texts)
                
                train_embeddings = self.tokenizer.encode(train_texts)
                test_embeddings = self.tokenizer.encode(test_texts)
            
            # Create sequences for training
            X_train, y_train = self._create_sequences_from_embeddings(train_embeddings)
            X_test, y_test = self._create_sequences_from_embeddings(test_embeddings)
            
            logger.info(f"Data preparation completed: train={len(X_train)}, test={len(X_test)}")
            logger.info(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
            
            return X_train, y_train, X_test, y_test
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace data: {e}")
            logger.warning("Falling back to legacy data loading")
            return self._load_legacy_data(test_ratio)
    
    def _load_legacy_data(self, test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Load data using the legacy system."""
        logger.info("Loading data using legacy system")
        
        X_train, y_train, X_test, y_test, legacy_tokenizer = load_data(
            window_size=self.window_size,
            test_size=test_ratio,
            embedding_dim=self.embedding_dim
        )
        
        # Store the legacy tokenizer if we don't have a new one
        if self.tokenizer is None:
            self.tokenizer = legacy_tokenizer
        
        return X_train, y_train, X_test, y_test
    
    def _token_ids_to_simple_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        """Convert token IDs to simple embeddings when sinusoidal embedder is not available."""
        # Create simple random embeddings for each unique token ID
        vocab_size = self.tokenizer.get_vocab_size()
        
        # Initialize embedding matrix
        embedding_matrix = np.random.normal(0, 0.1, (vocab_size, self.embedding_dim)).astype(np.float32)
        
        # Convert token sequences to embeddings
        if token_ids.ndim == 2:
            batch_size, seq_len = token_ids.shape
            embeddings = np.zeros((batch_size, seq_len, self.embedding_dim), dtype=np.float32)
            
            for i in range(batch_size):
                for j in range(seq_len):
                    token_id = token_ids[i, j]
                    if token_id < vocab_size:
                        embeddings[i, j] = embedding_matrix[token_id]
        else:
            # Single sequence
            seq_len = len(token_ids)
            embeddings = np.zeros((seq_len, self.embedding_dim), dtype=np.float32)
            
            for j in range(seq_len):
                token_id = token_ids[j]
                if token_id < vocab_size:
                    embeddings[j] = embedding_matrix[token_id]
        
        return embeddings
    
    def _create_sequences_from_embeddings(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create training sequences from embeddings."""
        if embeddings.ndim == 3:
            # Batch of sequences: (batch_size, seq_len, embedding_dim)
            X, y = [], []
            
            for sequence in embeddings:
                # Create sliding windows within each sequence
                for i in range(len(sequence) - self.window_size):
                    input_seq = sequence[i:i + self.window_size]
                    target = sequence[i + self.window_size]
                    
                    X.append(input_seq)
                    y.append(target)
            
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
        elif embeddings.ndim == 2:
            # Single sequence: (seq_len, embedding_dim)
            X, y = [], []
            
            for i in range(len(embeddings) - self.window_size):
                input_seq = embeddings[i:i + self.window_size]
                target = embeddings[i + self.window_size]
                
                X.append(input_seq)
                y.append(target)
            
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        
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
    
    def train(self, X_train: Optional[np.ndarray] = None, y_train: Optional[np.ndarray] = None,
              X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
              epochs: int = 10, batch_size: int = 32,
              validation_split: float = 0.1, test_ratio: float = 0.2,
              force_download: bool = False, fit_embedder: bool = True,
              embedding_epochs: int = 100) -> Dict:
        """
        Train the complete LSM + CNN system with enhanced data loading.
        
        Args:
            X_train: Optional training sequences (if None, will load data automatically)
            y_train: Optional training targets
            X_test: Optional test sequences
            y_test: Optional test targets
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Fraction of training data for validation
            test_ratio: Ratio of data for testing (used when loading data automatically)
            force_download: Whether to force re-download of HuggingFace data
            fit_embedder: Whether to fit the sinusoidal embedder
            embedding_epochs: Number of epochs for embedding optimization
            
        Returns:
            Training history dictionary
        """
        print("Starting enhanced LSM training...")
        
        # Load data if not provided
        if X_train is None or y_train is None or X_test is None or y_test is None:
            print("Loading data using enhanced system...")
            X_train, y_train, X_test, y_test = self.load_and_prepare_data(
                test_ratio=test_ratio,
                force_download=force_download,
                fit_embedder=fit_embedder,
                embedding_epochs=embedding_epochs
            )
        
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
    
    def initialize_response_level_training(self,
                                         use_3d_cnn: bool = False,
                                         system_message_support: bool = False,
                                         response_inference_architecture: str = "transformer"):
        """
        Initialize components for response-level training.
        
        Args:
            use_3d_cnn: Whether to use 3D CNN for system message integration
            system_message_support: Whether to enable system message processing
            response_inference_architecture: Architecture for response inference model
        """
        try:
            logger.info("Initializing response-level training components...")
            
            # Initialize CNN architecture factory
            if self.cnn_architecture_factory is None:
                self.cnn_architecture_factory = CNNArchitectureFactory()
            
            # Initialize system message processor if needed
            if system_message_support and self.system_message_processor is None:
                from ..core.system_message_processor import SystemMessageProcessor
                self.system_message_processor = SystemMessageProcessor(
                    tokenizer=self.tokenizer
                )
            
            # Initialize 3D CNN processor if needed
            if use_3d_cnn and self.cnn_3d_processor is None:
                # Determine reservoir output shape for 3D CNN
                reservoir_output_dim = self._calculate_reservoir_output_dim()
                reservoir_shape = (self.window_size, self.window_size, self.window_size, 1)
                
                self.cnn_3d_processor = CNN3DProcessor(
                    reservoir_shape=reservoir_shape,
                    system_embedding_dim=256,
                    output_embedding_dim=self.embedding_dim
                )
            
            # Initialize response inference model
            if self.response_inference_model is None:
                vocab_size = self.tokenizer.get_vocab_size() if hasattr(self.tokenizer, 'get_vocab_size') else 50000
                
                self.response_inference_model = ResponseInferenceModel(
                    input_embedding_dim=self.embedding_dim,
                    max_sequence_length=self.window_size * 2,  # Allow longer sequences for response-level
                    vocab_size=vocab_size,
                    tokenizer=self.tokenizer if isinstance(self.tokenizer, StandardTokenizerWrapper) else None,
                    architecture=response_inference_architecture
                )
            
            # Initialize response generator
            if self.response_generator is None and isinstance(self.tokenizer, StandardTokenizerWrapper):
                self.response_generator = ResponseGenerator(
                    tokenizer=self.tokenizer,
                    embedder=self.embedder,
                    reservoir_model=self.reservoir,
                    cnn_architecture_factory=self.cnn_architecture_factory,
                    cnn_3d_processor=self.cnn_3d_processor if use_3d_cnn else None
                )
            
            self.response_level_mode = True
            logger.info("Response-level training components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize response-level training: {e}")
            raise TrainingSetupError(f"Response-level training initialization failed: {str(e)}")
    
    def prepare_response_level_data(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  system_messages: Optional[List[str]] = None) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
        """
        Prepare data for response-level training by converting token-level data to response-level.
        
        Args:
            X_train: Training input sequences (num_samples, window_size, embedding_dim)
            y_train: Training target embeddings (num_samples, embedding_dim)
            X_test: Test input sequences
            y_test: Test target embeddings
            system_messages: Optional list of system messages for training
            
        Returns:
            Tuple of (train_embeddings, train_responses, test_embeddings, test_responses)
        """
        try:
            logger.info("Preparing data for response-level training...")
            
            # Convert embeddings to response sequences
            train_embedding_sequences = []
            train_response_texts = []
            
            for i in range(len(X_train)):
                # Create embedding sequence from input + target
                input_seq = X_train[i]  # (window_size, embedding_dim)
                target_emb = y_train[i]  # (embedding_dim,)
                
                # Combine input sequence with target as complete sequence
                full_sequence = np.vstack([input_seq, target_emb.reshape(1, -1)])
                train_embedding_sequences.append(full_sequence)
                
                # Generate response text from target embedding (simplified)
                response_text = self._embedding_to_response_text(target_emb)
                train_response_texts.append(response_text)
            
            # Same for test data
            test_embedding_sequences = []
            test_response_texts = []
            
            for i in range(len(X_test)):
                input_seq = X_test[i]
                target_emb = y_test[i]
                
                full_sequence = np.vstack([input_seq, target_emb.reshape(1, -1)])
                test_embedding_sequences.append(full_sequence)
                
                response_text = self._embedding_to_response_text(target_emb)
                test_response_texts.append(response_text)
            
            # Add system message context if provided
            if system_messages and self.system_message_processor:
                logger.info(f"Processing {len(system_messages)} system messages")
                # For now, we'll use the first system message for all samples
                # In practice, you might want to pair specific system messages with specific samples
                system_msg = system_messages[0] if system_messages else None
                
                if system_msg:
                    system_context = self.system_message_processor.process_system_message(system_msg)
                    logger.info(f"System message processed: {system_context.parsed_content['format']}")
            
            logger.info(f"Prepared {len(train_embedding_sequences)} training and {len(test_embedding_sequences)} test response sequences")
            
            return train_embedding_sequences, train_response_texts, test_embedding_sequences, test_response_texts
            
        except Exception as e:
            logger.error(f"Failed to prepare response-level data: {e}")
            raise TrainingSetupError(f"Response-level data preparation failed: {str(e)}")
    
    def train_response_level(self,
                           X_train: Optional[np.ndarray] = None,
                           y_train: Optional[np.ndarray] = None,
                           X_test: Optional[np.ndarray] = None,
                           y_test: Optional[np.ndarray] = None,
                           system_messages: Optional[List[str]] = None,
                           training_config: Optional[TrainingConfig] = None,
                           use_3d_cnn: bool = False,
                           epochs: int = 50,
                           batch_size: int = 16,
                           validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model for response-level generation using cosine similarity loss.
        
        Args:
            X_train: Training input sequences
            y_train: Training target embeddings
            X_test: Test input sequences
            y_test: Test target embeddings
            system_messages: Optional system messages for training
            training_config: Configuration for response-level training
            use_3d_cnn: Whether to use 3D CNN architecture
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation split ratio
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting response-level training...")
            
            # Load data if not provided
            if X_train is None or y_train is None or X_test is None or y_test is None:
                logger.info("Loading data for response-level training...")
                X_train, y_train, X_test, y_test = self.load_and_prepare_data()
            
            # Initialize response-level training components
            self.initialize_response_level_training(
                use_3d_cnn=use_3d_cnn,
                system_message_support=system_messages is not None
            )
            
            # Prepare response-level data
            train_embeddings, train_responses, test_embeddings, test_responses = self.prepare_response_level_data(
                X_train, y_train, X_test, y_test, system_messages
            )
            
            # Configure training
            if training_config is None:
                training_config = TrainingConfig(
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=0.001,
                    validation_split=validation_split,
                    loss_type="response_level_cosine",
                    loss_config={
                        "sequence_weight": 1.0,
                        "coherence_weight": 0.1,
                        "diversity_weight": 0.05
                    }
                )
            
            # Train response inference model
            logger.info("Training response inference model...")
            response_training_results = self.response_inference_model.train_on_response_pairs(
                input_embeddings=train_embeddings,
                target_responses=train_responses,
                training_config=training_config,
                validation_data=(test_embeddings, test_responses)
            )
            
            # Train enhanced CNN models with cosine similarity loss
            logger.info("Training enhanced CNN models...")
            cnn_training_results = self._train_enhanced_cnn_models(
                X_train, y_train, X_test, y_test,
                use_3d_cnn=use_3d_cnn,
                system_messages=system_messages,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Update training history with response-level metrics
            self.history['train_cosine_loss'].extend(
                response_training_results['history'].get('loss', [])
            )
            self.history['test_cosine_loss'].extend(
                response_training_results['history'].get('val_loss', [])
            )
            
            # Calculate response coherence metrics
            coherence_metrics = self._calculate_response_coherence(test_embeddings, test_responses)
            self.history['response_coherence'].append(coherence_metrics['average_coherence'])
            
            # Calculate system influence if system messages were used
            system_influence = None
            if system_messages:
                system_influence = self._calculate_system_influence(test_embeddings, system_messages)
                self.history['system_influence'].append(system_influence)
            
            logger.info("Response-level training completed successfully")
            
            return {
                'response_training_results': response_training_results,
                'cnn_training_results': cnn_training_results,
                'coherence_metrics': coherence_metrics,
                'system_influence': system_influence,
                'training_config': training_config.__dict__,
                'history': self.history
            }
            
        except Exception as e:
            logger.error(f"Response-level training failed: {e}")
            raise TrainingExecutionError(f"Response-level training failed: {str(e)}")
    
    def _train_enhanced_cnn_models(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 use_3d_cnn: bool = False,
                                 system_messages: Optional[List[str]] = None,
                                 epochs: int = 50,
                                 batch_size: int = 16) -> Dict[str, Any]:
        """
        Train enhanced CNN models (2D/3D) with cosine similarity loss.
        
        Args:
            X_train: Training input sequences
            y_train: Training target embeddings
            X_test: Test input sequences
            y_test: Test target embeddings
            use_3d_cnn: Whether to use 3D CNN
            system_messages: Optional system messages
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training results for CNN models
        """
        try:
            results = {}
            
            # Convert sequences to waveforms for CNN training
            logger.info("Converting sequences to waveforms for CNN training...")
            X_train_waveforms, y_train = self.create_training_data(X_train, y_train, batch_size)
            X_test_waveforms, y_test = self.create_training_data(X_test, y_test, batch_size)
            
            if use_3d_cnn and system_messages:
                # Train 3D CNN with system message integration
                logger.info("Training 3D CNN with system message integration...")
                
                # Process system messages
                system_embeddings = []
                if self.system_message_processor:
                    for msg in system_messages[:len(X_train)]:  # Match training data size
                        context = self.system_message_processor.process_system_message(msg)
                        system_embeddings.append(context.embeddings)
                
                # Pad or truncate system embeddings to match training data
                while len(system_embeddings) < len(X_train):
                    system_embeddings.append(system_embeddings[0] if system_embeddings else np.zeros(256))
                system_embeddings = system_embeddings[:len(X_train)]
                
                # Create 3D CNN model
                input_shape_3d = X_train_waveforms.shape[1:] + (1,)  # Add depth dimension
                cnn_3d_model = self.cnn_architecture_factory.create_3d_cnn(
                    input_shape=input_shape_3d,
                    output_dim=self.embedding_dim,
                    system_dim=256
                )
                
                # Compile with cosine similarity loss
                cnn_3d_model = self.cnn_architecture_factory.compile_model(
                    cnn_3d_model,
                    loss_type="cosine_similarity",
                    learning_rate=0.001,
                    loss_config={"temperature": 0.5, "weight_factor": 1.5}
                )
                
                # Prepare 3D input data
                X_train_3d = np.expand_dims(X_train_waveforms, axis=-1)
                X_test_3d = np.expand_dims(X_test_waveforms, axis=-1)
                
                system_train = np.array(system_embeddings)
                system_test = np.tile(system_train[:len(X_test)], (1, 1))  # Reuse for test
                
                # Train 3D CNN
                history_3d = cnn_3d_model.fit(
                    [X_train_3d, system_train], y_train,
                    validation_data=([X_test_3d, system_test], y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
                
                results['3d_cnn'] = {
                    'model': cnn_3d_model,
                    'history': history_3d.history,
                    'final_loss': history_3d.history['loss'][-1],
                    'final_val_loss': history_3d.history['val_loss'][-1]
                }
                
            else:
                # Train 2D CNN with enhanced loss functions
                logger.info("Training 2D CNN with cosine similarity loss...")
                
                # Create or update existing CNN model with enhanced loss
                if self.cnn_model is None:
                    self.build_models()
                
                # Recompile with cosine similarity loss
                self.cnn_model = self.cnn_architecture_factory.compile_model(
                    self.cnn_model,
                    loss_type="cosine_similarity",
                    learning_rate=0.001,
                    loss_config={"temperature": 1.0, "weight_factor": 1.0}
                )
                
                # Set up callbacks for enhanced training
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        verbose=1,
                        min_lr=1e-6
                    )
                ]
                
                # Train 2D CNN
                history_2d = self.cnn_model.fit(
                    X_train_waveforms, y_train,
                    validation_data=(X_test_waveforms, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
                
                results['2d_cnn'] = {
                    'model': self.cnn_model,
                    'history': history_2d.history,
                    'final_loss': history_2d.history['loss'][-1],
                    'final_val_loss': history_2d.history['val_loss'][-1]
                }
            
            logger.info("Enhanced CNN training completed")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced CNN training failed: {e}")
            raise TrainingExecutionError(f"Enhanced CNN training failed: {str(e)}")
    
    def _embedding_to_response_text(self, embedding: np.ndarray) -> str:
        """
        Convert embedding to response text (simplified implementation).
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Generated response text
        """
        # This is a simplified implementation for demonstration
        # In practice, you would use proper embedding-to-text decoding
        
        embedding_norm = np.linalg.norm(embedding)
        embedding_mean = np.mean(embedding)
        embedding_std = np.std(embedding)
        
        # Generate response based on embedding characteristics
        if embedding_norm > 10.0:
            if embedding_mean > 0:
                return "I understand your request and here's my detailed positive response."
            else:
                return "I acknowledge your input and provide this comprehensive response."
        elif embedding_norm > 5.0:
            if embedding_std > 2.0:
                return "Thank you for your message. Here's my varied response."
            else:
                return "I received your input and here's my consistent response."
        else:
            return "Brief acknowledgment of your message."
    
    def _calculate_response_coherence(self, test_embeddings: List[np.ndarray], test_responses: List[str]) -> Dict[str, float]:
        """
        Calculate response coherence metrics.
        
        Args:
            test_embeddings: List of test embedding sequences
            test_responses: List of test response texts
            
        Returns:
            Dictionary with coherence metrics
        """
        try:
            coherence_scores = []
            
            for i, (embeddings, response) in enumerate(zip(test_embeddings, test_responses)):
                # Simple coherence calculation based on embedding consistency
                if len(embeddings) > 1:
                    # Calculate variance in embeddings (lower = more coherent)
                    embedding_variance = np.mean(np.var(embeddings, axis=0))
                    coherence_score = 1.0 / (1.0 + embedding_variance)
                else:
                    coherence_score = 1.0
                
                coherence_scores.append(coherence_score)
            
            return {
                'average_coherence': np.mean(coherence_scores),
                'coherence_std': np.std(coherence_scores),
                'min_coherence': np.min(coherence_scores),
                'max_coherence': np.max(coherence_scores)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate response coherence: {e}")
            return {'average_coherence': 0.0, 'coherence_std': 0.0, 'min_coherence': 0.0, 'max_coherence': 0.0}
    
    def _calculate_system_influence(self, test_embeddings: List[np.ndarray], system_messages: List[str]) -> float:
        """
        Calculate the influence of system messages on responses.
        
        Args:
            test_embeddings: List of test embedding sequences
            system_messages: List of system messages
            
        Returns:
            Average system influence score
        """
        try:
            if not self.system_message_processor:
                return 0.0
            
            influence_scores = []
            
            for i, (embeddings, sys_msg) in enumerate(zip(test_embeddings, system_messages[:len(test_embeddings)])):
                # Process system message
                sys_context = self.system_message_processor.process_system_message(sys_msg)
                sys_embedding = sys_context.embeddings
                
                # Calculate influence as similarity between response and system embeddings
                response_embedding = np.mean(embeddings, axis=0)
                
                # Normalize embeddings
                response_norm = response_embedding / (np.linalg.norm(response_embedding) + 1e-8)
                system_norm = sys_embedding / (np.linalg.norm(sys_embedding) + 1e-8)
                
                # Calculate cosine similarity
                influence = np.dot(response_norm, system_norm)
                influence_scores.append(max(0.0, influence))  # Ensure non-negative
            
            return np.mean(influence_scores) if influence_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate system influence: {e}")
            return 0.0
    
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Make predictions on new sequences."""
        X_waveforms, _ = self.create_training_data(X, np.zeros((len(X), self.embedding_dim)), batch_size)
        return self.cnn_model.predict(X_waveforms, batch_size=batch_size)
    
    def save_complete_model(self, save_dir: str, tokenizer: Optional[Any] = None, 
                           training_results: Dict = None, dataset_info: Dict = None) -> None:
        """
        Save complete model state including reservoir, CNN, tokenizer, and configuration.
        
        Args:
            save_dir: Directory to save all model artifacts
            tokenizer: Optional tokenizer to save (uses self.tokenizer if None)
            training_results: Optional training results for metadata
            dataset_info: Optional dataset information for metadata
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Use provided tokenizer or self.tokenizer
        tokenizer_to_save = tokenizer if tokenizer is not None else self.tokenizer
        
        # Save models
        if self.reservoir is not None:
            reservoir_path = os.path.join(save_dir, "reservoir_model.keras")
            self.reservoir.save(reservoir_path)
            print(f"✓ Reservoir model saved to {reservoir_path}")
        
        if self.cnn_model is not None:
            cnn_path = os.path.join(save_dir, "cnn_model.keras")
            self.cnn_model.save(cnn_path)
            print(f"✓ CNN model saved to {cnn_path}")
        
        # Save tokenizer (new system)
        if tokenizer_to_save is not None:
            tokenizer_path = os.path.join(save_dir, "tokenizer")
            
            if isinstance(tokenizer_to_save, StandardTokenizerWrapper):
                tokenizer_to_save.save(tokenizer_path)
                print(f"✓ StandardTokenizerWrapper saved to {tokenizer_path}")
            elif hasattr(tokenizer_to_save, 'is_fitted') and tokenizer_to_save.is_fitted:
                # Legacy DialogueTokenizer
                tokenizer_to_save.save(tokenizer_path)
                print(f"✓ DialogueTokenizer saved to {tokenizer_path}")
            else:
                print("⚠ Warning: Tokenizer not fitted or unsupported type")
        
        # Save sinusoidal embedder
        if self.embedder is not None and hasattr(self.embedder, '_is_fitted') and self.embedder._is_fitted:
            embedder_path = os.path.join(save_dir, "sinusoidal_embedder")
            self.embedder.save(embedder_path)
            print(f"✓ SinusoidalEmbedder saved to {embedder_path}")
        
        # Create and save enhanced configuration
        config = ModelConfiguration(
            window_size=self.window_size,
            embedding_dim=self.embedding_dim,
            reservoir_type=self.reservoir_type,
            reservoir_config=self.reservoir_config,
            reservoir_units=self.reservoir_units,
            sparsity=self.sparsity,
            use_multichannel=self.use_multichannel,
            tokenizer_max_features=getattr(tokenizer_to_save, 'max_features', 10000),
            tokenizer_ngram_range=(1, 2)  # Default from tokenizer
        )
        
        # Add new configuration fields
        config_dict = config.__dict__.copy()
        config_dict.update({
            'tokenizer_name': self.tokenizer_name,
            'use_huggingface_data': self.use_huggingface_data,
            'cache_dir': self.cache_dir,
            'tokenizer_type': type(tokenizer_to_save).__name__ if tokenizer_to_save else None,
            'has_sinusoidal_embedder': self.embedder is not None and hasattr(self.embedder, '_is_fitted') and self.embedder._is_fitted
        })
        
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, 'w') as f:
            import json
            json.dump(config_dict, f, indent=2)
        print(f"✓ Enhanced configuration saved to {config_path}")
        
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
    
    def load_complete_model(self, save_dir: str) -> Tuple['LSMTrainer', Any]:
        """
        Load complete model state including configuration and tokenizer.
        
        Args:
            save_dir: Directory containing saved model artifacts
            
        Returns:
            Tuple of (loaded_trainer, loaded_tokenizer)
        """
        if not os.path.exists(save_dir):
            raise ModelLoadError(save_dir, "Model directory does not exist")
        
        # Load enhanced configuration
        config_path = os.path.join(save_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                import json
                config_dict = json.load(f)
            
            print(f"✓ Enhanced configuration loaded from {config_path}")
            
            # Update trainer parameters from config
            self.window_size = config_dict.get('window_size', self.window_size)
            self.embedding_dim = config_dict.get('embedding_dim', self.embedding_dim)
            self.reservoir_type = config_dict.get('reservoir_type', self.reservoir_type)
            self.reservoir_config = config_dict.get('reservoir_config', self.reservoir_config)
            self.reservoir_units = config_dict.get('reservoir_units', self.reservoir_units)
            self.sparsity = config_dict.get('sparsity', self.sparsity)
            self.use_multichannel = config_dict.get('use_multichannel', self.use_multichannel)
            
            # Update new parameters
            self.tokenizer_name = config_dict.get('tokenizer_name', self.tokenizer_name)
            self.use_huggingface_data = config_dict.get('use_huggingface_data', self.use_huggingface_data)
            self.cache_dir = config_dict.get('cache_dir', self.cache_dir)
            
            tokenizer_type = config_dict.get('tokenizer_type')
            has_sinusoidal_embedder = config_dict.get('has_sinusoidal_embedder', False)
        else:
            print("⚠ No configuration file found, using current trainer settings")
            tokenizer_type = None
            has_sinusoidal_embedder = False
        
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
        
        # Load tokenizer (enhanced system)
        tokenizer_path = os.path.join(save_dir, "tokenizer")
        tokenizer = None
        
        if os.path.exists(tokenizer_path):
            if tokenizer_type == 'StandardTokenizerWrapper':
                try:
                    tokenizer = StandardTokenizerWrapper.load(tokenizer_path)
                    self.tokenizer = tokenizer
                    print(f"✓ StandardTokenizerWrapper loaded from {tokenizer_path}")
                except Exception as e:
                    print(f"⚠ Failed to load StandardTokenizerWrapper: {e}")
            
            if tokenizer is None:
                # Try loading as legacy DialogueTokenizer
                try:
                    tokenizer = DialogueTokenizer(embedding_dim=self.embedding_dim)
                    tokenizer.load(tokenizer_path)
                    self.tokenizer = tokenizer
                    print(f"✓ DialogueTokenizer loaded from {tokenizer_path}")
                except Exception as e:
                    print(f"⚠ Failed to load DialogueTokenizer: {e}")
        else:
            print("⚠ No tokenizer found - you'll need to fit it on your data")
        
        # Load sinusoidal embedder if available
        embedder_path = os.path.join(save_dir, "sinusoidal_embedder")
        if has_sinusoidal_embedder and os.path.exists(embedder_path):
            try:
                self.embedder = SinusoidalEmbedder.load(embedder_path)
                print(f"✓ SinusoidalEmbedder loaded from {embedder_path}")
            except Exception as e:
                print(f"⚠ Failed to load SinusoidalEmbedder: {e}")
                self.embedder = None
        
        # Initialize dataset components if using HuggingFace data
        if self.use_huggingface_data:
            self.initialize_dataset_components()
        
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
    
    def initialize_response_level_training(self,
                                         use_3d_cnn: bool = False,
                                         system_message_support: bool = False,
                                         response_inference_architecture: str = "transformer"):
        """
        Initialize components for response-level training.
        
        Args:
            use_3d_cnn: Whether to use 3D CNN for system message integration
            system_message_support: Whether to enable system message processing
            response_inference_architecture: Architecture for response inference model
        """
        try:
            logger.info("Initializing response-level training components...")
            
            # Initialize CNN architecture factory
            if self.cnn_architecture_factory is None:
                self.cnn_architecture_factory = CNNArchitectureFactory()
            
            # Initialize system message processor if needed
            if system_message_support and self.system_message_processor is None:
                from ..core.system_message_processor import SystemMessageProcessor
                self.system_message_processor = SystemMessageProcessor(
                    tokenizer=self.tokenizer
                )
            
            # Initialize 3D CNN processor if needed
            if use_3d_cnn and self.cnn_3d_processor is None:
                # Determine reservoir output shape for 3D CNN
                reservoir_output_dim = self._calculate_reservoir_output_dim()
                reservoir_shape = (self.window_size, self.window_size, self.window_size, 1)
                
                self.cnn_3d_processor = CNN3DProcessor(
                    reservoir_shape=reservoir_shape,
                    system_embedding_dim=256,
                    output_embedding_dim=self.embedding_dim
                )
            
            # Initialize response inference model
            if self.response_inference_model is None:
                vocab_size = self.tokenizer.get_vocab_size() if hasattr(self.tokenizer, 'get_vocab_size') else 50000
                
                self.response_inference_model = ResponseInferenceModel(
                    input_embedding_dim=self.embedding_dim,
                    max_sequence_length=self.window_size * 2,  # Allow longer sequences for response-level
                    vocab_size=vocab_size,
                    tokenizer=self.tokenizer if isinstance(self.tokenizer, StandardTokenizerWrapper) else None,
                    architecture=response_inference_architecture
                )
            
            # Initialize response generator
            if self.response_generator is None and isinstance(self.tokenizer, StandardTokenizerWrapper):
                self.response_generator = ResponseGenerator(
                    tokenizer=self.tokenizer,
                    embedder=self.embedder,
                    reservoir_model=self.reservoir,
                    cnn_architecture_factory=self.cnn_architecture_factory,
                    cnn_3d_processor=self.cnn_3d_processor if use_3d_cnn else None
                )
            
            self.response_level_mode = True
            logger.info("Response-level training components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize response-level training: {e}")
            raise TrainingSetupError(f"Response-level training initialization failed: {str(e)}")
    
    def prepare_response_level_data(self,
                                  X_train: np.ndarray,
                                  y_train: np.ndarray,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  system_messages: Optional[List[str]] = None) -> Tuple[List[np.ndarray], List[str], List[np.ndarray], List[str]]:
        """
        Prepare data for response-level training by converting token-level data to response-level.
        
        Args:
            X_train: Training input sequences (num_samples, window_size, embedding_dim)
            y_train: Training target embeddings (num_samples, embedding_dim)
            X_test: Test input sequences
            y_test: Test target embeddings
            system_messages: Optional list of system messages for training
            
        Returns:
            Tuple of (train_embeddings, train_responses, test_embeddings, test_responses)
        """
        try:
            logger.info("Preparing data for response-level training...")
            
            # Convert embeddings to response sequences
            train_embedding_sequences = []
            train_response_texts = []
            
            for i in range(len(X_train)):
                # Create embedding sequence from input + target
                input_seq = X_train[i]  # (window_size, embedding_dim)
                target_emb = y_train[i]  # (embedding_dim,)
                
                # Combine input sequence with target as complete sequence
                full_sequence = np.vstack([input_seq, target_emb.reshape(1, -1)])
                train_embedding_sequences.append(full_sequence)
                
                # Generate response text from target embedding (simplified)
                response_text = self._embedding_to_response_text(target_emb)
                train_response_texts.append(response_text)
            
            # Same for test data
            test_embedding_sequences = []
            test_response_texts = []
            
            for i in range(len(X_test)):
                input_seq = X_test[i]
                target_emb = y_test[i]
                
                full_sequence = np.vstack([input_seq, target_emb.reshape(1, -1)])
                test_embedding_sequences.append(full_sequence)
                
                response_text = self._embedding_to_response_text(target_emb)
                test_response_texts.append(response_text)
            
            # Add system message context if provided
            if system_messages and self.system_message_processor:
                logger.info(f"Processing {len(system_messages)} system messages")
                # For now, we'll use the first system message for all samples
                # In practice, you might want to pair specific system messages with specific samples
                system_msg = system_messages[0] if system_messages else None
                
                if system_msg:
                    system_context = self.system_message_processor.process_system_message(system_msg)
                    logger.info(f"System message processed: {system_context.parsed_content['format']}")
            
            logger.info(f"Prepared {len(train_embedding_sequences)} training and {len(test_embedding_sequences)} test response sequences")
            
            return train_embedding_sequences, train_response_texts, test_embedding_sequences, test_response_texts
            
        except Exception as e:
            logger.error(f"Failed to prepare response-level data: {e}")
            raise TrainingSetupError(f"Response-level data preparation failed: {str(e)}")
    
    def _train_enhanced_cnn_models(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_test: np.ndarray,
                                 y_test: np.ndarray,
                                 use_3d_cnn: bool = False,
                                 system_messages: Optional[List[str]] = None,
                                 epochs: int = 50,
                                 batch_size: int = 16) -> Dict[str, Any]:
        """
        Train enhanced CNN models (2D/3D) with cosine similarity loss.
        
        Args:
            X_train: Training input sequences
            y_train: Training target embeddings
            X_test: Test input sequences
            y_test: Test target embeddings
            use_3d_cnn: Whether to use 3D CNN
            system_messages: Optional system messages
            epochs: Number of epochs
            batch_size: Batch size
            
        Returns:
            Training results for CNN models
        """
        try:
            results = {}
            
            # Convert sequences to waveforms for CNN training
            logger.info("Converting sequences to waveforms for CNN training...")
            X_train_waveforms, y_train = self.create_training_data(X_train, y_train, batch_size)
            X_test_waveforms, y_test = self.create_training_data(X_test, y_test, batch_size)
            
            if use_3d_cnn and system_messages:
                # Train 3D CNN with system message integration
                logger.info("Training 3D CNN with system message integration...")
                
                # Process system messages
                system_embeddings = []
                if self.system_message_processor:
                    for msg in system_messages[:len(X_train)]:  # Match training data size
                        context = self.system_message_processor.process_system_message(msg)
                        system_embeddings.append(context.embeddings)
                
                # Pad or truncate system embeddings to match training data
                while len(system_embeddings) < len(X_train):
                    system_embeddings.append(system_embeddings[0] if system_embeddings else np.zeros(256))
                system_embeddings = system_embeddings[:len(X_train)]
                
                # Create 3D CNN model
                input_shape_3d = X_train_waveforms.shape[1:] + (1,)  # Add depth dimension
                cnn_3d_model = self.cnn_architecture_factory.create_3d_cnn(
                    input_shape=input_shape_3d,
                    output_dim=self.embedding_dim,
                    system_dim=256
                )
                
                # Compile with cosine similarity loss
                cnn_3d_model = self.cnn_architecture_factory.compile_model(
                    cnn_3d_model,
                    loss_type="cosine_similarity",
                    learning_rate=0.001,
                    loss_config={"temperature": 0.5, "weight_factor": 1.5}
                )
                
                # Prepare 3D input data
                X_train_3d = np.expand_dims(X_train_waveforms, axis=-1)
                X_test_3d = np.expand_dims(X_test_waveforms, axis=-1)
                
                system_train = np.array(system_embeddings)
                system_test = np.tile(system_train[:len(X_test)], (1, 1))  # Reuse for test
                
                # Train 3D CNN
                history_3d = cnn_3d_model.fit(
                    [X_train_3d, system_train], y_train,
                    validation_data=([X_test_3d, system_test], y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=1
                )
                
                results['3d_cnn'] = {
                    'model': cnn_3d_model,
                    'history': history_3d.history,
                    'final_loss': history_3d.history['loss'][-1],
                    'final_val_loss': history_3d.history['val_loss'][-1]
                }
                
            else:
                # Train 2D CNN with enhanced loss functions
                logger.info("Training 2D CNN with cosine similarity loss...")
                
                # Create or update existing CNN model with enhanced loss
                if self.cnn_model is None:
                    self.build_models()
                
                # Recompile with cosine similarity loss
                self.cnn_model = self.cnn_architecture_factory.compile_model(
                    self.cnn_model,
                    loss_type="cosine_similarity",
                    learning_rate=0.001,
                    loss_config={"temperature": 1.0, "weight_factor": 1.0}
                )
                
                # Set up callbacks for enhanced training
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor='val_loss',
                        patience=10,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        verbose=1,
                        min_lr=1e-6
                    )
                ]
                
                # Train 2D CNN
                history_2d = self.cnn_model.fit(
                    X_train_waveforms, y_train,
                    validation_data=(X_test_waveforms, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
                
                results['2d_cnn'] = {
                    'model': self.cnn_model,
                    'history': history_2d.history,
                    'final_loss': history_2d.history['loss'][-1],
                    'final_val_loss': history_2d.history['val_loss'][-1]
                }
            
            logger.info("Enhanced CNN training completed")
            return results
            
        except Exception as e:
            logger.error(f"Enhanced CNN training failed: {e}")
            raise TrainingExecutionError(f"Enhanced CNN training failed: {str(e)}")
    
    def train_response_level(self,
                           X_train: Optional[np.ndarray] = None,
                           y_train: Optional[np.ndarray] = None,
                           X_test: Optional[np.ndarray] = None,
                           y_test: Optional[np.ndarray] = None,
                           system_messages: Optional[List[str]] = None,
                           training_config: Optional[TrainingConfig] = None,
                           use_3d_cnn: bool = False,
                           epochs: int = 50,
                           batch_size: int = 16,
                           validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train the model for response-level generation using cosine similarity loss.
        
        Args:
            X_train: Training input sequences
            y_train: Training target embeddings
            X_test: Test input sequences
            y_test: Test target embeddings
            system_messages: Optional system messages for training
            training_config: Configuration for response-level training
            use_3d_cnn: Whether to use 3D CNN architecture
            epochs: Number of training epochs
            batch_size: Training batch size
            validation_split: Validation split ratio
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting response-level training...")
            
            # Load data if not provided
            if X_train is None or y_train is None or X_test is None or y_test is None:
                logger.info("Loading data for response-level training...")
                X_train, y_train, X_test, y_test = self.load_and_prepare_data()
            
            # Initialize response-level training components
            self.initialize_response_level_training(
                use_3d_cnn=use_3d_cnn,
                system_message_support=system_messages is not None
            )
            
            # Prepare response-level data
            train_embeddings, train_responses, test_embeddings, test_responses = self.prepare_response_level_data(
                X_train, y_train, X_test, y_test, system_messages
            )
            
            # Configure training
            if training_config is None:
                training_config = TrainingConfig(
                    batch_size=batch_size,
                    epochs=epochs,
                    learning_rate=0.001,
                    validation_split=validation_split,
                    loss_type="response_level_cosine",
                    loss_config={
                        "sequence_weight": 1.0,
                        "coherence_weight": 0.1,
                        "diversity_weight": 0.05
                    }
                )
            
            # Train response inference model
            logger.info("Training response inference model...")
            response_training_results = self.response_inference_model.train_on_response_pairs(
                input_embeddings=train_embeddings,
                target_responses=train_responses,
                training_config=training_config,
                validation_data=(test_embeddings, test_responses)
            )
            
            # Train enhanced CNN models with cosine similarity loss
            logger.info("Training enhanced CNN models...")
            cnn_training_results = self._train_enhanced_cnn_models(
                X_train, y_train, X_test, y_test,
                use_3d_cnn=use_3d_cnn,
                system_messages=system_messages,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Update training history with response-level metrics
            self.history['train_cosine_loss'].extend(
                response_training_results['history'].get('loss', [])
            )
            self.history['test_cosine_loss'].extend(
                response_training_results['history'].get('val_loss', [])
            )
            
            # Calculate response coherence metrics
            coherence_metrics = self._calculate_response_coherence(test_embeddings, test_responses)
            self.history['response_coherence'].append(coherence_metrics['average_coherence'])
            
            # Calculate system influence if system messages were used
            system_influence = None
            if system_messages:
                system_influence = self._calculate_system_influence(test_embeddings, system_messages)
                self.history['system_influence'].append(system_influence)
            
            logger.info("Response-level training completed successfully")
            
            return {
                'response_training_results': response_training_results,
                'cnn_training_results': cnn_training_results,
                'coherence_metrics': coherence_metrics,
                'system_influence': system_influence,
                'training_config': training_config.__dict__,
                'history': self.history
            }
            
        except Exception as e:
            logger.error(f"Response-level training failed: {e}")
            raise TrainingExecutionError(f"Response-level training failed: {str(e)}")
    
    def _embedding_to_response_text(self, embedding: np.ndarray) -> str:
        """
        Convert embedding to response text (simplified implementation).
        
        Args:
            embedding: Embedding vector
            
        Returns:
            Generated response text
        """
        # This is a simplified implementation for demonstration
        # In practice, you would use proper embedding-to-text decoding
        
        embedding_norm = np.linalg.norm(embedding)
        embedding_mean = np.mean(embedding)
        embedding_std = np.std(embedding)
        
        # Generate response based on embedding characteristics
        if embedding_norm > 10.0:
            if embedding_mean > 0:
                return "I understand your request and here's my detailed positive response."
            else:
                return "I acknowledge your input and provide this comprehensive response."
        elif embedding_norm > 5.0:
            if embedding_std > 2.0:
                return "Thank you for your message. Here's my varied response."
            else:
                return "I received your input and here's my consistent response."
        else:
            return "Brief acknowledgment of your message."
    
    def _calculate_response_coherence(self, test_embeddings: List[np.ndarray], test_responses: List[str]) -> Dict[str, float]:
        """
        Calculate response coherence metrics.
        
        Args:
            test_embeddings: List of test embedding sequences
            test_responses: List of test response texts
            
        Returns:
            Dictionary with coherence metrics
        """
        try:
            coherence_scores = []
            
            for i, (embeddings, response) in enumerate(zip(test_embeddings, test_responses)):
                # Simple coherence calculation based on embedding consistency
                if len(embeddings) > 1:
                    # Calculate variance in embeddings (lower = more coherent)
                    embedding_variance = np.mean(np.var(embeddings, axis=0))
                    coherence_score = 1.0 / (1.0 + embedding_variance)
                else:
                    coherence_score = 1.0
                
                coherence_scores.append(coherence_score)
            
            return {
                'average_coherence': np.mean(coherence_scores),
                'coherence_std': np.std(coherence_scores),
                'min_coherence': np.min(coherence_scores),
                'max_coherence': np.max(coherence_scores)
            }
            
        except Exception as e:
            logger.warning(f"Failed to calculate response coherence: {e}")
            return {'average_coherence': 0.0, 'coherence_std': 0.0, 'min_coherence': 0.0, 'max_coherence': 0.0}
    
    def _calculate_system_influence(self, test_embeddings: List[np.ndarray], system_messages: List[str]) -> float:
        """
        Calculate the influence of system messages on responses.
        
        Args:
            test_embeddings: List of test embedding sequences
            system_messages: List of system messages
            
        Returns:
            Average system influence score
        """
        try:
            if not self.system_message_processor:
                return 0.0
            
            influence_scores = []
            
            for i, (embeddings, sys_msg) in enumerate(zip(test_embeddings, system_messages[:len(test_embeddings)])):
                # Process system message
                sys_context = self.system_message_processor.process_system_message(sys_msg)
                sys_embedding = sys_context.embeddings
                
                # Calculate influence as similarity between response and system embeddings
                response_embedding = np.mean(embeddings, axis=0)
                
                # Normalize embeddings
                response_norm = response_embedding / (np.linalg.norm(response_embedding) + 1e-8)
                system_norm = sys_embedding / (np.linalg.norm(sys_embedding) + 1e-8)
                
                # Calculate cosine similarity
                influence = np.dot(response_norm, system_norm)
                influence_scores.append(max(0.0, influence))  # Ensure non-negative
            
            return np.mean(influence_scores) if influence_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate system influence: {e}")
            return 0.0


@log_performance("LSM training")
def run_training(window_size: int = 10, batch_size: int = 32, epochs: int = 20,
                test_size: float = 0.2, embedding_dim: int = 128,
                reservoir_type: str = 'standard', reservoir_config: Dict = None,
                use_attention: bool = True, tokenizer_name: str = 'gpt2',
                use_huggingface_data: bool = True, cache_dir: str = "data/huggingface_cache",
                force_download: bool = False, fit_embedder: bool = True,
                embedding_epochs: int = 100) -> Dict:
    """
    Main training function with enhanced tokenization and dataset integration.
    
    Args:
        window_size: Size of sequence windows
        batch_size: Training batch size
        epochs: Number of training epochs
        test_size: Fraction of data for testing
        embedding_dim: Token embedding dimension
        reservoir_type: Type of reservoir architecture
        reservoir_config: Configuration for advanced reservoirs
        use_attention: Whether to use spatial attention in CNN
        tokenizer_name: Name of standard tokenizer to use
        use_huggingface_data: Whether to use HuggingFace dataset integration
        cache_dir: Directory for caching HuggingFace datasets
        force_download: Whether to force re-download of HuggingFace data
        fit_embedder: Whether to fit the sinusoidal embedder
        embedding_epochs: Number of epochs for embedding optimization
        
    Returns:
        Training results dictionary
    """
    # Validate input parameters
    from ..utils.input_validation import (
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
    
    # Initialize enhanced trainer
    print("Initializing enhanced LSM trainer...")
    try:
        trainer = LSMTrainer(
            window_size=window_size,
            embedding_dim=embedding_dim,
            reservoir_units=[256, 128, 64],
            sparsity=0.1,
            use_multichannel=True,
            reservoir_type=reservoir_type,
            reservoir_config=reservoir_config or {},
            use_attention=use_attention,
            tokenizer_name=tokenizer_name,
            use_huggingface_data=use_huggingface_data,
            cache_dir=cache_dir
        )
        
        logger.info("Enhanced trainer initialized successfully")
        
    except Exception as e:
        logger.exception("Trainer initialization failed")
        error_msg = create_helpful_error_message(
            "Trainer initialization",
            e,
            [
                "Check that the tokenizer name is valid",
                "Ensure transformers library is installed for new tokenization",
                "Verify cache directory permissions"
            ]
        )
        print(error_msg)
        raise TrainingSetupError(f"Trainer initialization failed: {e}")
    
    # Train the model with enhanced data loading
    print("Starting training with enhanced data loading...")
    try:
        results = trainer.train(
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            test_ratio=test_size,
            force_download=force_download,
            fit_embedder=fit_embedder,
            embedding_epochs=embedding_epochs
        )
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.exception("Training failed")
        error_msg = create_helpful_error_message(
            "Training execution",
            e,
            [
                "Check available memory for large datasets",
                "Verify HuggingFace dataset access",
                "Ensure GPU/CPU resources are sufficient"
            ]
        )
        print(error_msg)
        raise TrainingExecutionError(f"Training failed: {e}")
    
    # Save models with complete state
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"models_{timestamp}"
    
    # Create enhanced dataset info for metadata
    dataset_info = {
        'source': 'HuggingFace cosmopedia-v2' if use_huggingface_data else 'Synthetic-Persona-Chat',
        'tokenizer_name': tokenizer_name,
        'use_huggingface_data': use_huggingface_data,
        'window_size': window_size,
        'embedding_dim': embedding_dim,
        'fit_embedder': fit_embedder,
        'embedding_epochs': embedding_epochs
    }
    
    print(f"Saving enhanced model to {save_dir}...")
    trainer.save_complete_model(save_dir, training_results=results, dataset_info=dataset_info)
    
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
