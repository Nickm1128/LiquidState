#!/usr/bin/env python3
"""
OpenAI tiktoken adapter for enhanced tokenization system.

This module provides an adapter for OpenAI's tiktoken library,
allowing it to work with the enhanced tokenizer system.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from ..enhanced_tokenization import TokenizerAdapter, TokenizerConfig, TokenizerRegistry
from ...utils.lsm_exceptions import TokenizerError, TokenizerLoadError
from ...utils.lsm_logging import get_logger

logger = get_logger(__name__)


class TiktokenAdapter(TokenizerAdapter):
    """
    Adapter for OpenAI's tiktoken tokenizers.
    
    This adapter wraps tiktoken encoders to work with the enhanced
    tokenizer system, providing vocabulary extraction, token mapping,
    and special token handling for different OpenAI models.
    """
    
    # Supported OpenAI models and their encodings
    SUPPORTED_MODELS = {
        # GPT-3.5 models
        'gpt-3.5-turbo': 'cl100k_base',
        'gpt-3.5-turbo-0301': 'cl100k_base',
        'gpt-3.5-turbo-0613': 'cl100k_base',
        'gpt-3.5-turbo-1106': 'cl100k_base',
        'gpt-3.5-turbo-0125': 'cl100k_base',
        'gpt-3.5-turbo-16k': 'cl100k_base',
        'gpt-3.5-turbo-16k-0613': 'cl100k_base',
        'gpt-3.5-turbo-instruct': 'cl100k_base',
        
        # GPT-4 models
        'gpt-4': 'cl100k_base',
        'gpt-4-0314': 'cl100k_base',
        'gpt-4-0613': 'cl100k_base',
        'gpt-4-1106-preview': 'cl100k_base',
        'gpt-4-0125-preview': 'cl100k_base',
        'gpt-4-turbo-preview': 'cl100k_base',
        'gpt-4-32k': 'cl100k_base',
        'gpt-4-32k-0314': 'cl100k_base',
        'gpt-4-32k-0613': 'cl100k_base',
        
        # Text models
        'text-davinci-003': 'p50k_base',
        'text-davinci-002': 'p50k_base',
        'text-davinci-001': 'r50k_base',
        'text-curie-001': 'r50k_base',
        'text-babbage-001': 'r50k_base',
        'text-ada-001': 'r50k_base',
        'davinci': 'r50k_base',
        'curie': 'r50k_base',
        'babbage': 'r50k_base',
        'ada': 'r50k_base',
        
        # Code models
        'code-davinci-002': 'p50k_base',
        'code-davinci-001': 'p50k_base',
        'code-cushman-002': 'p50k_base',
        'code-cushman-001': 'p50k_base',
        
        # Direct encoding names
        'cl100k_base': 'cl100k_base',
        'p50k_base': 'p50k_base',
        'r50k_base': 'r50k_base',
        'gpt2': 'gpt2',
        'p50k_edit': 'p50k_edit'
    }
    
    # Special tokens for different encodings
    ENCODING_SPECIAL_TOKENS = {
        'cl100k_base': {
            'eos_token_id': 100257,  # <|endoftext|>
            'pad_token_id': 100257,  # Use EOS as pad token
        },
        'p50k_base': {
            'eos_token_id': 50256,   # <|endoftext|>
            'pad_token_id': 50256,   # Use EOS as pad token
        },
        'r50k_base': {
            'eos_token_id': 50256,   # <|endoftext|>
            'pad_token_id': 50256,   # Use EOS as pad token
        },
        'gpt2': {
            'eos_token_id': 50256,   # <|endoftext|>
            'pad_token_id': 50256,   # Use EOS as pad token
        }
    }
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialize tiktoken adapter.
        
        Args:
            config: Tokenizer configuration
        """
        super().__init__(config)
        
        if not TIKTOKEN_AVAILABLE:
            raise TokenizerError(
                "tiktoken library not available. Install with: pip install tiktoken"
            )
        
        self._encoding_name = None
        self._special_tokens = {}
    
    def initialize(self) -> None:
        """Initialize the tiktoken encoder."""
        try:
            model_name = self.config.model_name
            
            # Determine encoding name
            if model_name in self.SUPPORTED_MODELS:
                self._encoding_name = self.SUPPORTED_MODELS[model_name]
            else:
                # Try to use model_name directly as encoding name
                self._encoding_name = model_name
            
            # Load the encoder
            try:
                self._tokenizer = tiktoken.get_encoding(self._encoding_name)
            except ValueError:
                # If encoding name fails, try as model name
                try:
                    self._tokenizer = tiktoken.encoding_for_model(model_name)
                    self._encoding_name = self._tokenizer.name
                except KeyError:
                    raise TokenizerError(
                        f"Unsupported tiktoken model/encoding: {model_name}. "
                        f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
                    )
            
            # Set up special tokens
            self._setup_special_tokens()
            
            # Get vocabulary size
            self._vocab_size = self._tokenizer.n_vocab
            self._is_initialized = True
            
            logger.info(f"Initialized tiktoken encoder '{self._encoding_name}' with vocab size {self._vocab_size}")
            
        except Exception as e:
            raise TokenizerError(f"Failed to initialize tiktoken encoder '{self.config.model_name}': {str(e)}")
    
    def _setup_special_tokens(self) -> None:
        """Set up special tokens for the encoder."""
        # Get default special tokens for this encoding
        if self._encoding_name in self.ENCODING_SPECIAL_TOKENS:
            self._special_tokens = self.ENCODING_SPECIAL_TOKENS[self._encoding_name].copy()
        else:
            # Default fallback
            self._special_tokens = {
                'eos_token_id': 50256,  # Common <|endoftext|> token
                'pad_token_id': 50256,  # Use EOS as pad token
            }
        
        # Apply custom special tokens if provided
        if self.config.special_tokens:
            for token_type, token_value in self.config.special_tokens.items():
                if token_type.endswith('_token'):
                    # Convert token string to ID
                    try:
                        token_ids = self._tokenizer.encode(token_value)
                        if len(token_ids) == 1:
                            self._special_tokens[f'{token_type}_id'] = token_ids[0]
                        else:
                            logger.warning(f"Special token '{token_value}' encodes to multiple tokens, using first")
                            self._special_tokens[f'{token_type}_id'] = token_ids[0]
                    except Exception as e:
                        logger.warning(f"Failed to encode special token '{token_value}': {e}")
                elif token_type.endswith('_token_id'):
                    # Direct token ID
                    self._special_tokens[token_type] = token_value
    
    def tokenize(self, texts: Union[str, List[str]], 
                 add_special_tokens: bool = True,
                 padding: bool = True, 
                 truncation: bool = True) -> List[List[int]]:
        """
        Tokenize texts to token IDs.
        
        Args:
            texts: Single text or list of texts to tokenize
            add_special_tokens: Whether to add special tokens (EOS)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            
        Returns:
            List of token ID sequences
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            tokenized_sequences = []
            max_length = 0
            
            for text in texts:
                # Encode the text
                token_ids = self._tokenizer.encode(text)
                
                # Truncate if needed
                if truncation and self.config.max_length:
                    if add_special_tokens and len(token_ids) >= self.config.max_length:
                        # Leave space for EOS token
                        token_ids = token_ids[:self.config.max_length - 1]
                    elif len(token_ids) > self.config.max_length:
                        token_ids = token_ids[:self.config.max_length]
                
                # Add special tokens
                if add_special_tokens and 'eos_token_id' in self._special_tokens:
                    token_ids.append(self._special_tokens['eos_token_id'])
                
                tokenized_sequences.append(token_ids)
                max_length = max(max_length, len(token_ids))
            
            # Apply padding if requested
            if padding:
                pad_token_id = self._special_tokens.get('pad_token_id', 0)
                target_length = min(max_length, self.config.max_length) if self.config.max_length else max_length
                
                for i, seq in enumerate(tokenized_sequences):
                    if len(seq) < target_length:
                        # Pad to target length
                        tokenized_sequences[i] = seq + [pad_token_id] * (target_length - len(seq))
            
            return tokenized_sequences
            
        except Exception as e:
            raise TokenizerError(f"Tiktoken tokenization failed: {str(e)}")
    
    def decode(self, token_ids: Union[List[int], List[List[int]]], 
               skip_special_tokens: bool = True) -> Union[str, List[str]]:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text(s)
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        try:
            # Handle single sequence
            if isinstance(token_ids[0], int):
                return self._decode_single(token_ids, skip_special_tokens)
            
            # Handle batch of sequences
            return [
                self._decode_single(seq, skip_special_tokens) 
                for seq in token_ids
            ]
            
        except Exception as e:
            raise TokenizerError(f"Tiktoken decoding failed: {str(e)}")
    
    def _decode_single(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a single sequence of token IDs."""
        if skip_special_tokens:
            # Filter out special tokens
            special_token_ids = set(self._special_tokens.values())
            filtered_ids = [tid for tid in token_ids if tid not in special_token_ids]
            return self._tokenizer.decode(filtered_ids)
        else:
            return self._tokenizer.decode(token_ids)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """
        Get vocabulary mapping.
        
        Note: tiktoken doesn't provide direct access to the full vocabulary,
        so this returns a partial mapping based on common tokens.
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        # tiktoken doesn't expose the full vocabulary directly
        # We can only provide a partial mapping for common tokens
        vocab = {}
        
        # Add some common tokens by trying to decode token IDs
        sample_size = min(1000, self._vocab_size)  # Sample first 1000 tokens
        for token_id in range(sample_size):
            try:
                token_str = self._tokenizer.decode([token_id])
                if token_str and len(token_str.strip()) > 0:
                    vocab[token_str] = token_id
            except Exception:
                # Skip tokens that can't be decoded individually
                continue
        
        logger.warning(f"tiktoken vocabulary mapping is partial ({len(vocab)} tokens). "
                      "Full vocabulary not available through tiktoken API.")
        
        return vocab
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        return self._special_tokens.copy()
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """
        Get detailed information about the tokenizer.
        
        Returns:
            Dictionary with tokenizer information
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        info = {
            'model_name': self.config.model_name,
            'encoding_name': self._encoding_name,
            'vocab_size': self._vocab_size,
            'max_length': self.config.max_length,
            'special_tokens': self.get_special_tokens(),
            'tokenizer_class': 'tiktoken.Encoding',
            'backend': 'tiktoken'
        }
        
        return info
    
    @classmethod
    def load_adapter_config(cls, load_path: str) -> 'TiktokenAdapter':
        """
        Load adapter from saved configuration.
        
        Args:
            load_path: Directory path to load config from
            
        Returns:
            Loaded TiktokenAdapter instance
        """
        try:
            config_path = os.path.join(load_path, 'tiktoken_adapter_config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Tiktoken adapter config not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            config = TokenizerConfig(
                backend=config_dict['backend'],
                model_name=config_dict['model_name'],
                max_length=config_dict['max_length'],
                special_tokens=config_dict.get('special_tokens'),
                backend_specific_config=config_dict.get('backend_specific_config')
            )
            
            adapter = cls(config)
            adapter.initialize()
            
            logger.info(f"Loaded tiktoken adapter from {load_path}")
            return adapter
            
        except Exception as e:
            raise TokenizerLoadError(load_path, str(e))
    
    def save_adapter_config(self, save_path: str) -> None:
        """
        Save adapter configuration.
        
        Args:
            save_path: Directory path to save config to
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            config_dict = {
                'backend': self.config.backend,
                'model_name': self.config.model_name,
                'encoding_name': self._encoding_name,
                'max_length': self.config.max_length,
                'special_tokens': self.config.special_tokens,
                'backend_specific_config': self.config.backend_specific_config
            }
            
            config_path = os.path.join(save_path, 'tiktoken_adapter_config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Saved tiktoken adapter config to {save_path}")
            
        except Exception as e:
            raise TokenizerError(f"Failed to save tiktoken adapter config: {str(e)}")
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Get list of supported model names."""
        return list(cls.SUPPORTED_MODELS.keys())
    
    def __repr__(self) -> str:
        return (f"TiktokenAdapter(model={self.config.model_name}, "
                f"encoding={self._encoding_name}, vocab_size={self._vocab_size}, "
                f"initialized={self._is_initialized})")


# Register the tiktoken adapter
def register_tiktoken_adapter():
    """Register the tiktoken adapter with the tokenizer registry."""
    model_patterns = list(TiktokenAdapter.SUPPORTED_MODELS.keys())
    
    # Add common OpenAI model prefixes
    model_patterns.extend([
        'gpt-3.5', 'gpt-4', 'text-davinci', 'text-curie', 'text-babbage', 'text-ada',
        'code-davinci', 'code-cushman', 'davinci', 'curie', 'babbage', 'ada'
    ])
    
    TokenizerRegistry.register_adapter(
        'tiktoken', 
        TiktokenAdapter, 
        model_patterns
    )
    
    logger.info(f"Registered tiktoken adapter with {len(model_patterns)} model patterns")


# Auto-register when module is imported
register_tiktoken_adapter()