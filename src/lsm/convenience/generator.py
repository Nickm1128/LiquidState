"""
LSMGenerator class for text generation using the LSM convenience API.

This module provides a simple, scikit-learn-like interface for training LSM models
and generating text responses, abstracting away the complexity of the underlying
multi-component architecture.
"""

import time
import warnings
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
from pathlib import Path

from .base import LSMBase
from .config import ConvenienceConfig, ConvenienceValidationError
from ..utils.lsm_exceptions import (
    LSMError, TrainingSetupError, TrainingExecutionError, 
    InvalidInputError, ModelLoadError
)
from ..utils.input_validation import (
    validate_positive_integer, validate_positive_float,
    validate_string_list, create_helpful_error_message
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)

# Lazy import flag - components will be imported when needed
_TRAINING_AVAILABLE = None
LSMTrainer = None
ResponseGenerator = None
SystemMessageProcessor = None
StandardTokenizerWrapper = None
SinusoidalEmbedder = None

def _check_training_components():
    """Lazy import of training components to avoid circular imports."""
    global _TRAINING_AVAILABLE, LSMTrainer, ResponseGenerator, SystemMessageProcessor
    global StandardTokenizerWrapper, SinusoidalEmbedder
    
    if _TRAINING_AVAILABLE is not None:
        return _TRAINING_AVAILABLE
    
    try:
        from ..training.train import LSMTrainer as _LSMTrainer
        from ..inference.response_generator import ResponseGenerator as _ResponseGenerator
        from ..core.system_message_processor import SystemMessageProcessor as _SystemMessageProcessor
        from ..data.tokenization import StandardTokenizerWrapper as _StandardTokenizerWrapper
        from ..data.tokenization import SinusoidalEmbedder as _SinusoidalEmbedder
        
        # Assign to module-level variables
        LSMTrainer = _LSMTrainer
        ResponseGenerator = _ResponseGenerator
        SystemMessageProcessor = _SystemMessageProcessor
        StandardTokenizerWrapper = _StandardTokenizerWrapper
        SinusoidalEmbedder = _SinusoidalEmbedder
        
        _TRAINING_AVAILABLE = True
        logger.info("Training components loaded successfully")
        
    except ImportError as e:
        logger.warning(f"Training components not available: {e}")
        _TRAINING_AVAILABLE = False
    
    return _TRAINING_AVAILABLE


class LSMGenerator(LSMBase):
    """
    LSM-based text generator with scikit-learn-like interface.
    
    This class provides a simple interface for training LSM models on conversational
    data and generating text responses. It wraps the complex LSMTrainer and 
    ResponseGenerator components with intelligent defaults and automatic configuration.
    
    Parameters
    ----------
    window_size : int, default=10
        Size of the sliding window for sequence processing
    embedding_dim : int, default=128
        Dimension of the embedding space
    reservoir_type : str, default='hierarchical'
        Type of reservoir ('standard', 'hierarchical', 'attentive', 'echo_state', 'deep')
    reservoir_config : dict, optional
        Additional configuration for the reservoir
    system_message_support : bool, default=True
        Whether to enable system message processing
    response_level : bool, default=True
        Whether to use response-level training and generation
    tokenizer : str, default='gpt2'
        Name of the tokenizer to use ('gpt2', 'bert-base-uncased', etc.)
    max_length : int, default=512
        Maximum sequence length for tokenization
    temperature : float, default=1.0
        Default temperature for text generation
    random_state : int, optional
        Random seed for reproducibility
    **kwargs : dict
        Additional parameters passed to the base class
        
    Attributes
    ----------
    is_fitted_ : bool
        Whether the model has been fitted
    classes_ : list
        Not applicable for generation (maintained for sklearn compatibility)
    feature_names_in_ : list
        Input feature names (conversation keys)
    n_features_in_ : int
        Number of input features
        
    Examples
    --------
    >>> from lsm import LSMGenerator
    >>> 
    >>> # Simple usage
    >>> generator = LSMGenerator()
    >>> conversations = ["Hello", "Hi there", "How are you?", "I'm doing well"]
    >>> generator.fit(conversations)
    >>> response = generator.generate("Hello, how are you?")
    >>> 
    >>> # With system messages
    >>> generator = LSMGenerator(system_message_support=True)
    >>> conversations = [
    ...     {"messages": ["Hello", "Hi there"], "system": "Be friendly"},
    ...     {"messages": ["Help me", "Sure!"], "system": "Be helpful"}
    ... ]
    >>> generator.fit(conversations)
    >>> response = generator.generate("I need help", system_message="Be helpful")
    >>> 
    >>> # Using presets
    >>> generator = LSMGenerator.from_preset('fast')
    >>> generator.fit(conversations)
    """
    
    def __init__(self,
                 window_size: int = 10,
                 embedding_dim: int = 128,
                 reservoir_type: str = 'hierarchical',
                 reservoir_config: Optional[Dict[str, Any]] = None,
                 system_message_support: bool = True,
                 response_level: bool = True,
                 tokenizer: str = 'gpt2',
                 max_length: int = 512,
                 temperature: float = 1.0,
                 random_state: Optional[int] = None,
                 **kwargs):
        
        # Check if training components are available (lazy import)
        if not _check_training_components():
            raise ImportError(
                "LSM training components are not available. "
                "Please ensure TensorFlow and all dependencies are installed."
            )
        
        # Set defaults optimized for text generation
        if reservoir_config is None:
            reservoir_config = {
                'reservoir_units': [100, 50],
                'sparsity': 0.1,
                'spectral_radius': 0.9
            }
            if reservoir_type == 'hierarchical':
                reservoir_config['hierarchy_levels'] = 2
            elif reservoir_type == 'attentive':
                reservoir_config['attention_heads'] = 4
        
        # Store generation-specific parameters
        self.system_message_support = system_message_support
        self.response_level = response_level
        self.tokenizer_name = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        
        # Initialize base class
        super().__init__(
            window_size=window_size,
            embedding_dim=embedding_dim,
            reservoir_type=reservoir_type,
            reservoir_config=reservoir_config,
            random_state=random_state,
            **kwargs
        )
        
        # Generation components (initialized during fit)
        self._response_generator = None
        self._system_processor = None
        
        # sklearn-compatible attributes
        self.classes_ = None  # Not applicable for generation
        self.feature_names_in_ = None
        self.n_features_in_ = None
        
        # Validate generation-specific parameters
        self._validate_generation_parameters()
    
    def _validate_generation_parameters(self) -> None:
        """Validate generation-specific parameters."""
        try:
            self.max_length = validate_positive_integer(
                self.max_length, 'max_length', min_value=1, max_value=2048
            )
            
            self.temperature = validate_positive_float(
                self.temperature, 'temperature', min_value=0.1, max_value=5.0
            )
            
            if not isinstance(self.system_message_support, bool):
                raise ConvenienceValidationError(
                    f"system_message_support must be boolean, got {type(self.system_message_support).__name__}",
                    suggestion="Use True to enable system message processing, False to disable"
                )
            
            if not isinstance(self.response_level, bool):
                raise ConvenienceValidationError(
                    f"response_level must be boolean, got {type(self.response_level).__name__}",
                    suggestion="Use True for response-level training (recommended), False for token-level"
                )
            
            # Validate tokenizer name
            valid_tokenizers = ['gpt2', 'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']
            if self.tokenizer_name not in valid_tokenizers:
                logger.warning(
                    f"Tokenizer '{self.tokenizer_name}' not in recommended list. "
                    f"Recommended: {valid_tokenizers}"
                )
        
        except Exception as e:
            logger.error(f"Generation parameter validation failed: {e}")
            raise
    
    @classmethod
    def from_preset(cls, preset_name: str, **overrides) -> 'LSMGenerator':
        """
        Create LSMGenerator from a preset configuration.
        
        Parameters
        ----------
        preset_name : str
            Name of the preset ('fast', 'balanced', 'quality', 'text_generation')
        **overrides : dict
            Parameters to override in the preset
            
        Returns
        -------
        generator : LSMGenerator
            Configured generator instance
            
        Examples
        --------
        >>> generator = LSMGenerator.from_preset('fast')
        >>> generator = LSMGenerator.from_preset('quality', temperature=0.8)
        """
        config = ConvenienceConfig.create_config(
            preset=preset_name, 
            task_type='text_generation',
            **overrides
        )
        
        # Extract generation-specific parameters
        gen_params = {}
        for param in ['system_message_support', 'response_level', 'temperature']:
            if param in config:
                gen_params[param] = config.pop(param)
        
        return cls(**config, **gen_params)
    
    def fit(self, 
            X: Union[List[str], List[Dict[str, Any]], str],
            y: Optional[Any] = None,
            system_messages: Optional[Union[List[str], str]] = None,
            validation_split: float = 0.2,
            epochs: int = 50,
            batch_size: int = 32,
            verbose: bool = True,
            **fit_params) -> 'LSMGenerator':
        """
        Train the LSM model on conversation data.
        
        Parameters
        ----------
        X : list or str
            Training conversations. Can be:
            - List of strings: ["Hello", "Hi there", "How are you?"]
            - List of dicts: [{"messages": [...], "system": "..."}, ...]
            - Single string: "User: Hello\\nAssistant: Hi there"
        y : ignored
            Not used for text generation (maintained for sklearn compatibility)
        system_messages : list or str, optional
            System messages for training (if not included in X)
        validation_split : float, default=0.2
            Fraction of data to use for validation
        epochs : int, default=50
            Number of training epochs
        batch_size : int, default=32
            Training batch size
        verbose : bool, default=True
            Whether to show training progress
        **fit_params : dict
            Additional parameters passed to the trainer
            
        Returns
        -------
        self : LSMGenerator
            Returns self for method chaining
            
        Raises
        ------
        ConvenienceValidationError
            If input data format is invalid
        TrainingSetupError
            If training setup fails
        TrainingExecutionError
            If training execution fails
        """
        try:
            logger.info("Starting LSM text generation training...")
            
            # Validate and preprocess input data
            processed_data = self._preprocess_training_data(X, system_messages)
            
            # Store feature information for sklearn compatibility
            self.feature_names_in_ = list(processed_data.keys()) if isinstance(processed_data, dict) else ['conversations']
            self.n_features_in_ = len(self.feature_names_in_)
            
            # Initialize trainer with configuration
            trainer_config = self._create_trainer_config(epochs, batch_size, validation_split)
            self._trainer = LSMTrainer(**trainer_config)
            
            # Initialize tokenization system
            self._trainer.initialize_tokenization_system(max_length=self.max_length)
            
            # Prepare training data
            if verbose:
                logger.info("Preparing training data...")
            
            X_train, y_train, X_test, y_test = self._prepare_training_data(
                processed_data, validation_split
            )
            
            # Train the model
            if verbose:
                logger.info(f"Training LSM model for {epochs} epochs...")
            
            start_time = time.time()
            
            # Use response-level training if enabled
            if self.response_level:
                history = self._trainer.train_response_level(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    **fit_params
                )
            else:
                history = self._trainer.train(
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    epochs=epochs,
                    batch_size=batch_size,
                    **fit_params
                )
            
            training_time = time.time() - start_time
            
            # Initialize generation components
            self._initialize_generation_components()
            
            # Store training metadata
            self._training_metadata = {
                'training_time': training_time,
                'epochs': epochs,
                'batch_size': batch_size,
                'validation_split': validation_split,
                'data_size': len(X) if isinstance(X, list) else 1,
                'history': history,
                'response_level': self.response_level,
                'system_message_support': self.system_message_support
            }
            
            self._is_fitted = True
            
            if verbose:
                logger.info(f"Training completed in {training_time:.2f} seconds")
                if history and 'train_mse' in history:
                    final_loss = history['train_mse'][-1] if history['train_mse'] else 'N/A'
                    logger.info(f"Final training loss: {final_loss}")
            
            return self
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            suggestions = self._create_error_suggestions(e, "fit")
            
            if isinstance(e, (ConvenienceValidationError, TrainingSetupError, TrainingExecutionError)):
                raise
            else:
                raise TrainingExecutionError(
                    epoch=None,
                    reason=f"LSM training failed: {e}"
                )
    
    def generate(self,
                 prompt: Union[str, List[str]],
                 system_message: Optional[str] = None,
                 max_length: Optional[int] = None,
                 temperature: Optional[float] = None,
                 return_confidence: bool = False) -> Union[str, Tuple[str, float]]:
        """
        Generate a text response to the given prompt.
        
        Parameters
        ----------
        prompt : str or list
            Input prompt or conversation history
        system_message : str, optional
            System message to guide generation
        max_length : int, optional
            Maximum length of generated response (overrides default)
        temperature : float, optional
            Temperature for generation randomness (overrides default)
        return_confidence : bool, default=False
            Whether to return confidence score along with response
            
        Returns
        -------
        response : str or tuple
            Generated response, optionally with confidence score
            
        Raises
        ------
        InvalidInputError
            If the model is not fitted or input is invalid
        """
        self._check_is_fitted()
        
        try:
            # Use provided parameters or defaults
            gen_max_length = max_length or self.max_length
            gen_temperature = temperature or self.temperature
            
            # Validate inputs
            if isinstance(prompt, list):
                prompt = validate_string_list(prompt, 'prompt')
                prompt_text = " ".join(prompt)
            else:
                if not isinstance(prompt, str):
                    raise InvalidInputError(
                        "prompt",
                        "string or list of strings",
                        f"{type(prompt).__name__}"
                    )
                prompt_text = prompt
            
            if system_message is not None and not isinstance(system_message, str):
                raise InvalidInputError(
                    "system_message",
                    "string or None",
                    f"{type(system_message).__name__}"
                )
            
            # Generate response using the response generator
            if self._response_generator is None:
                raise InvalidInputError(
                    "model state",
                    "initialized response generator",
                    "uninitialized generator (training may have failed)"
                )
            
            # Prepare input for generation
            generation_input = self._prepare_generation_input(prompt_text, system_message)
            
            # Generate response
            result = self._response_generator.generate_complete_response(
                input_sequence=generation_input,
                system_context=self._create_system_context(system_message) if system_message else None,
                return_intermediate=return_confidence
            )
            
            # Extract response and confidence
            response = result.response_text if hasattr(result, 'response_text') else str(result)
            confidence = result.confidence if hasattr(result, 'confidence') else 1.0
            
            # Apply temperature if different from training
            if gen_temperature != self.temperature:
                response = self._apply_temperature(response, gen_temperature)
            
            # Truncate to max_length if needed
            if len(response.split()) > gen_max_length:
                words = response.split()[:gen_max_length]
                response = " ".join(words)
            
            logger.debug(f"Generated response: {response[:100]}...")
            
            if return_confidence:
                return response, confidence
            else:
                return response
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            suggestions = self._create_error_suggestions(e, "generate")
            
            if isinstance(e, InvalidInputError):
                raise
            else:
                raise InvalidInputError(
                    "generation process",
                    "successful text generation",
                    f"generation failed: {e}"
                )
    
    def batch_generate(self,
                      prompts: List[str],
                      system_messages: Optional[List[str]] = None,
                      max_length: Optional[int] = None,
                      temperature: Optional[float] = None,
                      return_confidence: bool = False,
                      batch_size: int = 8) -> List[Union[str, Tuple[str, float]]]:
        """
        Generate responses for multiple prompts efficiently.
        
        Parameters
        ----------
        prompts : list of str
            List of input prompts
        system_messages : list of str, optional
            System messages for each prompt (must match length of prompts)
        max_length : int, optional
            Maximum length of generated responses
        temperature : float, optional
            Temperature for generation randomness
        return_confidence : bool, default=False
            Whether to return confidence scores
        batch_size : int, default=8
            Number of prompts to process in each batch
            
        Returns
        -------
        responses : list
            List of generated responses, optionally with confidence scores
        """
        self._check_is_fitted()
        
        # Validate inputs
        prompts = validate_string_list(prompts, 'prompts')
        
        if system_messages is not None:
            system_messages = validate_string_list(system_messages, 'system_messages')
            if len(system_messages) != len(prompts):
                raise ConvenienceValidationError(
                    f"Length mismatch: {len(prompts)} prompts but {len(system_messages)} system messages",
                    suggestion="Provide one system message per prompt, or use None for no system messages"
                )
        else:
            system_messages = [None] * len(prompts)
        
        # Process in batches
        responses = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_systems = system_messages[i:i + batch_size]
            
            batch_responses = []
            for prompt, system_msg in zip(batch_prompts, batch_systems):
                try:
                    response = self.generate(
                        prompt=prompt,
                        system_message=system_msg,
                        max_length=max_length,
                        temperature=temperature,
                        return_confidence=return_confidence
                    )
                    batch_responses.append(response)
                except Exception as e:
                    logger.warning(f"Failed to generate response for prompt: {prompt[:50]}... Error: {e}")
                    # Return empty response or error indicator
                    if return_confidence:
                        batch_responses.append(("", 0.0))
                    else:
                        batch_responses.append("")
            
            responses.extend(batch_responses)
        
        return responses
    
    def chat(self, 
             system_message: Optional[str] = None,
             max_turns: int = 100,
             temperature: float = None) -> None:
        """
        Start an interactive chat session.
        
        Parameters
        ----------
        system_message : str, optional
            System message to guide the conversation
        max_turns : int, default=100
            Maximum number of conversation turns
        temperature : float, optional
            Temperature for generation randomness
        """
        self._check_is_fitted()
        
        print("🤖 LSM Chat Session Started")
        print("Type 'quit', 'exit', or 'bye' to end the session")
        if system_message:
            print(f"System: {system_message}")
        print("-" * 50)
        
        conversation_history = []
        turn_count = 0
        
        try:
            while turn_count < max_turns:
                # Get user input
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Chat session ended")
                    break
                
                if not user_input:
                    continue
                
                # Add to conversation history
                conversation_history.append(f"User: {user_input}")
                
                # Generate response
                try:
                    prompt = " ".join(conversation_history[-10:])  # Use last 10 turns
                    response = self.generate(
                        prompt=prompt,
                        system_message=system_message,
                        temperature=temperature
                    )
                    
                    print(f"Bot: {response}")
                    conversation_history.append(f"Bot: {response}")
                    
                except Exception as e:
                    print(f"❌ Error generating response: {e}")
                    continue
                
                turn_count += 1
        
        except KeyboardInterrupt:
            print("\n👋 Chat session interrupted")
        except Exception as e:
            print(f"❌ Chat session error: {e}")
    
    def predict(self, X) -> List[str]:
        """
        Generate predictions (responses) for input data.
        
        This method provides sklearn-compatible interface for text generation.
        
        Parameters
        ----------
        X : list of str
            Input prompts
            
        Returns
        -------
        predictions : list of str
            Generated responses
        """
        if isinstance(X, str):
            X = [X]
        
        return self.batch_generate(X, return_confidence=False)
    
    def score(self, X, y=None) -> float:
        """
        Return a dummy score for sklearn compatibility.
        
        For text generation, scoring is not straightforward, so this returns 1.0
        to indicate the model is functional.
        
        Parameters
        ----------
        X : array-like
            Input data
        y : ignored
            Not used for text generation
            
        Returns
        -------
        score : float
            Always returns 1.0 for compatibility
        """
        self._check_is_fitted()
        return 1.0
    
    def _preprocess_training_data(self, 
                                 X: Union[List[str], List[Dict], str],
                                 system_messages: Optional[Union[List[str], str]]) -> Dict[str, Any]:
        """Preprocess and validate training data."""
        if isinstance(X, str):
            # Single string - split into conversations
            conversations = self._split_conversation_string(X)
        elif isinstance(X, list):
            if all(isinstance(item, str) for item in X):
                # List of strings
                conversations = X
            elif all(isinstance(item, dict) for item in X):
                # List of dictionaries with structured data
                conversations = []
                extracted_systems = []
                for item in X:
                    if 'messages' in item:
                        conversations.extend(item['messages'])
                    if 'system' in item:
                        extracted_systems.append(item['system'])
                
                # Use extracted system messages if not provided separately
                if system_messages is None and extracted_systems:
                    system_messages = extracted_systems
            else:
                raise ConvenienceValidationError(
                    "Mixed data types in conversation list",
                    suggestion="Use either all strings or all dictionaries with 'messages' key"
                )
        else:
            raise ConvenienceValidationError(
                f"Invalid data type: {type(X).__name__}",
                suggestion="Use list of strings, list of dicts, or single string",
                valid_options=["list of strings", "list of dicts", "string"]
            )
        
        # Validate conversations
        if not conversations:
            raise ConvenienceValidationError(
                "No conversations found in training data",
                suggestion="Ensure your data contains actual conversation text"
            )
        
        # Process system messages
        processed_systems = None
        if system_messages is not None:
            if isinstance(system_messages, str):
                processed_systems = [system_messages] * len(conversations)
            elif isinstance(system_messages, list):
                processed_systems = system_messages
                if len(processed_systems) != len(conversations):
                    logger.warning(
                        f"System messages length ({len(processed_systems)}) doesn't match "
                        f"conversations length ({len(conversations)}). Cycling system messages."
                    )
                    # Cycle system messages to match conversation length
                    processed_systems = [
                        processed_systems[i % len(processed_systems)] 
                        for i in range(len(conversations))
                    ]
        
        return {
            'conversations': conversations,
            'system_messages': processed_systems
        }
    
    def _split_conversation_string(self, text: str) -> List[str]:
        """Split a conversation string into individual messages."""
        # Simple splitting by common patterns
        lines = text.strip().split('\n')
        conversations = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove common prefixes
            for prefix in ['User:', 'Assistant:', 'Bot:', 'Human:', 'AI:']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            if line:
                conversations.append(line)
        
        return conversations
    
    def _create_trainer_config(self, epochs: int, batch_size: int, validation_split: float) -> Dict[str, Any]:
        """Create configuration for the LSMTrainer."""
        config = {
            'window_size': self.window_size,
            'embedding_dim': self.embedding_dim,
            'reservoir_type': self.reservoir_type,
            'reservoir_config': self.reservoir_config.copy(),
            'tokenizer_name': self.tokenizer_name,
            'use_huggingface_data': False,  # We're providing our own data
        }
        
        # Add reservoir-specific configuration
        if self.reservoir_type == 'standard':
            config['reservoir_units'] = self.reservoir_config.get('reservoir_units', [100, 50])
        
        config['sparsity'] = self.reservoir_config.get('sparsity', 0.1)
        config['use_attention'] = self.reservoir_type in ['attentive', 'hierarchical']
        
        # Set random seed if provided
        if self.random_state is not None:
            config['random_state'] = self.random_state
        
        return config
    
    def _prepare_training_data(self, processed_data: Dict[str, Any], validation_split: float) -> Tuple[Any, Any, Any, Any]:
        """Prepare training data for the LSMTrainer."""
        conversations = processed_data['conversations']
        system_messages = processed_data.get('system_messages')
        
        # Validate we have enough data
        if len(conversations) < 2:
            raise ConvenienceValidationError(
                f"Need at least 2 conversations for training, got {len(conversations)}",
                suggestion="Provide more conversation examples or use a single string with multiple exchanges"
            )
        
        # Split data for validation
        split_idx = int(len(conversations) * (1 - validation_split))
        
        X_train = conversations[:split_idx]
        X_test = conversations[split_idx:]
        
        # For text generation, y is typically the next token/response
        # Let the trainer handle this automatically
        y_train = None
        y_test = None
        
        return X_train, y_train, X_test, y_test
    
    def _initialize_generation_components(self) -> None:
        """Initialize components needed for text generation."""
        try:
            # Get components from the trained model
            if self._trainer is None:
                raise TrainingSetupError("Trainer not initialized")
            
            # Initialize response generator
            if hasattr(self._trainer, 'response_generator') and self._trainer.response_generator:
                self._response_generator = self._trainer.response_generator
            else:
                # Create response generator from trainer components
                self._response_generator = ResponseGenerator(
                    tokenizer=self._trainer.tokenizer,
                    embedder=self._trainer.embedder,
                    reservoir_model=self._trainer.reservoir,
                    max_response_length=self.max_length
                )
            
            # Initialize system message processor if enabled
            if self.system_message_support:
                if hasattr(self._trainer, 'system_message_processor') and self._trainer.system_message_processor:
                    self._system_processor = self._trainer.system_message_processor
                else:
                    self._system_processor = SystemMessageProcessor(
                        tokenizer=self._trainer.tokenizer,
                        embedder=self._trainer.embedder
                    )
            
            logger.info("Generation components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize generation components: {e}")
            raise TrainingSetupError(f"Generation component initialization failed: {e}")
    
    def _prepare_generation_input(self, prompt: str, system_message: Optional[str]) -> Any:
        """Prepare input for the response generator."""
        # This would typically involve tokenizing and embedding the prompt
        # For now, return the prompt as-is and let the response generator handle it
        return prompt
    
    def _create_system_context(self, system_message: str) -> Any:
        """Create system context for generation."""
        if self._system_processor is None:
            logger.warning("System message processor not available, ignoring system message")
            return None
        
        # Process system message into the format expected by the response generator
        # This is a placeholder - actual implementation would depend on SystemMessageProcessor interface
        return {"system_message": system_message}
    
    def _apply_temperature(self, response: str, temperature: float) -> str:
        """Apply temperature scaling to response (placeholder implementation)."""
        # This is a simplified placeholder - actual temperature application
        # would happen during the generation process in the neural network
        return response
    
    def _load_trainer(self, model_path: str) -> None:
        """Load the underlying trainer from saved model."""
        try:
            # Initialize trainer
            self._trainer = LSMTrainer()
            
            # Load the complete model
            self._trainer, _ = self._trainer.load_complete_model(model_path)
            
            # Initialize generation components
            self._initialize_generation_components()
            
            logger.info(f"Trainer loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load trainer: {e}")
            raise ModelLoadError(model_path, f"Trainer loading failed: {e}")
    
    def __sklearn_tags__(self):
        """Return sklearn tags for this estimator."""
        tags = super().__sklearn_tags__()
        tags.update({
            'requires_y': False,
            'multioutput': True,  # Can generate multiple responses
            'text': True,  # Works with text data
        })
        return tags