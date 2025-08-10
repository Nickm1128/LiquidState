#!/usr/bin/env python3
"""
HuggingFace tokenizer adapter for enhanced tokenization system.

This module provides an adapter for HuggingFace transformers tokenizers,
allowing them to work with the enhanced tokenizer system.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..enhanced_tokenization import TokenizerAdapter, TokenizerConfig, TokenizerRegistry
from ...utils.lsm_exceptions import TokenizerError, TokenizerLoadError
from ...utils.lsm_logging import get_logger

logger = get_logger(__name__)


class HuggingFaceAdapter(TokenizerAdapter):
    """
    Adapter for HuggingFace transformers tokenizers.
    
    This adapter wraps HuggingFace tokenizers to work with the enhanced
    tokenizer system, providing vocabulary extraction, token mapping,
    and special token handling.
    """
    
    # Supported HuggingFace models
    SUPPORTED_MODELS = {
        'gpt2': 'gpt2',
        'gpt2-medium': 'gpt2-medium',
        'gpt2-large': 'gpt2-large',
        'gpt2-xl': 'gpt2-xl',
        'bert-base-uncased': 'bert-base-uncased',
        'bert-base-cased': 'bert-base-cased',
        'bert-large-uncased': 'bert-large-uncased',
        'bert-large-cased': 'bert-large-cased',
        'distilbert-base-uncased': 'distilbert-base-uncased',
        'distilbert-base-cased': 'distilbert-base-cased',
        'roberta-base': 'roberta-base',
        'roberta-large': 'roberta-large',
        'albert-base-v2': 'albert-base-v2',
        'albert-large-v2': 'albert-large-v2',
        't5-small': 't5-small',
        't5-base': 't5-base',
        't5-large': 't5-large'
    }
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialize HuggingFace adapter.
        
        Args:
            config: Tokenizer configuration
        """
        super().__init__(config)
        
        if not TRANSFORMERS_AVAILABLE:
            raise TokenizerError(
                "transformers library not available. Install with: pip install transformers"
            )
        
        self._pad_token_id = None
        self._eos_token_id = None
        self._bos_token_id = None
        self._unk_token_id = None
        self._cls_token_id = None
        self._sep_token_id = None
        self._mask_token_id = None
    
    def initialize(self) -> None:
        """Initialize the HuggingFace tokenizer."""
        try:
            model_name = self.config.model_name
            
            # Check if model is in supported list or use as-is
            if model_name in self.SUPPORTED_MODELS:
                model_name = self.SUPPORTED_MODELS[model_name]
            
            # Load tokenizer with backend-specific config
            tokenizer_kwargs = self.config.backend_specific_config or {}
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
            
            # Set up special tokens
            self._setup_special_tokens()
            
            # Get vocabulary size
            self._vocab_size = len(self._tokenizer)
            self._is_initialized = True
            
            logger.info(f"Initialized HuggingFace tokenizer '{model_name}' with vocab size {self._vocab_size}")
            
        except Exception as e:
            raise TokenizerError(f"Failed to initialize HuggingFace tokenizer '{self.config.model_name}': {str(e)}")
    
    def _setup_special_tokens(self) -> None:
        """Set up special tokens for the tokenizer."""
        # Handle pad token
        if self._tokenizer.pad_token is None:
            if self._tokenizer.eos_token is not None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            elif self._tokenizer.unk_token is not None:
                self._tokenizer.pad_token = self._tokenizer.unk_token
            else:
                # Add a pad token if none exists
                self._tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Store special token IDs
        self._pad_token_id = getattr(self._tokenizer, 'pad_token_id', None)
        self._eos_token_id = getattr(self._tokenizer, 'eos_token_id', None)
        self._bos_token_id = getattr(self._tokenizer, 'bos_token_id', None)
        self._unk_token_id = getattr(self._tokenizer, 'unk_token_id', None)
        self._cls_token_id = getattr(self._tokenizer, 'cls_token_id', None)
        self._sep_token_id = getattr(self._tokenizer, 'sep_token_id', None)
        self._mask_token_id = getattr(self._tokenizer, 'mask_token_id', None)
        
        # Apply custom special tokens if provided
        if self.config.special_tokens:
            special_tokens_dict = {}
            for token_type, token_value in self.config.special_tokens.items():
                if hasattr(self._tokenizer, f'{token_type}_token'):
                    special_tokens_dict[f'{token_type}_token'] = token_value
            
            if special_tokens_dict:
                self._tokenizer.add_special_tokens(special_tokens_dict)
                # Update vocab size after adding tokens
                self._vocab_size = len(self._tokenizer)
    
    def tokenize(self, texts: Union[str, List[str]], 
                 add_special_tokens: bool = True,
                 padding: bool = True, 
                 truncation: bool = True) -> List[List[int]]:
        """
        Tokenize texts to token IDs.
        
        Args:
            texts: Single text or list of texts to tokenize
            add_special_tokens: Whether to add special tokens
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
            # Determine padding strategy
            if padding:
                padding_strategy = 'max_length' if self.config.max_length else 'longest'
            else:
                padding_strategy = False
            
            encoded = self._tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
                padding=padding_strategy,
                truncation=truncation,
                max_length=self.config.max_length if truncation else None,
                return_tensors=None
            )
            
            return encoded['input_ids']
            
        except Exception as e:
            raise TokenizerError(f"HuggingFace tokenization failed: {str(e)}")
    
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
                return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
            
            # Handle batch of sequences
            return [
                self._tokenizer.decode(seq, skip_special_tokens=skip_special_tokens) 
                for seq in token_ids
            ]
            
        except Exception as e:
            raise TokenizerError(f"HuggingFace decoding failed: {str(e)}")
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._tokenizer.get_vocab()
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        special_tokens = {}
        
        # Add all available special tokens
        if self._pad_token_id is not None:
            special_tokens['pad_token_id'] = self._pad_token_id
        if self._eos_token_id is not None:
            special_tokens['eos_token_id'] = self._eos_token_id
        if self._bos_token_id is not None:
            special_tokens['bos_token_id'] = self._bos_token_id
        if self._unk_token_id is not None:
            special_tokens['unk_token_id'] = self._unk_token_id
        if self._cls_token_id is not None:
            special_tokens['cls_token_id'] = self._cls_token_id
        if self._sep_token_id is not None:
            special_tokens['sep_token_id'] = self._sep_token_id
        if self._mask_token_id is not None:
            special_tokens['mask_token_id'] = self._mask_token_id
        
        return special_tokens
    
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
            'vocab_size': self._vocab_size,
            'max_length': self.config.max_length,
            'special_tokens': self.get_special_tokens(),
            'tokenizer_class': self._tokenizer.__class__.__name__,
            'model_max_length': getattr(self._tokenizer, 'model_max_length', None),
            'do_lower_case': getattr(self._tokenizer, 'do_lower_case', None),
            'supports_fast': getattr(self._tokenizer, 'is_fast', False)
        }
        
        return info
    
    @classmethod
    def load_adapter_config(cls, load_path: str) -> 'HuggingFaceAdapter':
        """
        Load adapter from saved configuration.
        
        Args:
            load_path: Directory path to load config from
            
        Returns:
            Loaded HuggingFaceAdapter instance
        """
        try:
            config_path = os.path.join(load_path, 'huggingface_adapter_config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"HuggingFace adapter config not found: {config_path}")
            
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
            
            logger.info(f"Loaded HuggingFace adapter from {load_path}")
            return adapter
            
        except Exception as e:
            raise TokenizerLoadError(load_path, str(e))
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Get list of supported model names."""
        return list(cls.SUPPORTED_MODELS.keys())
    
    def __repr__(self) -> str:
        return (f"HuggingFaceAdapter(model={self.config.model_name}, "
                f"vocab_size={self._vocab_size}, initialized={self._is_initialized})")


# Register the HuggingFace adapter
def register_huggingface_adapter():
    """Register the HuggingFace adapter with the tokenizer registry."""
    model_patterns = list(HuggingFaceAdapter.SUPPORTED_MODELS.keys())
    
    # Add common model prefixes
    model_patterns.extend([
        'gpt', 'bert', 'distilbert', 'roberta', 'albert', 't5',
        'microsoft/', 'google/', 'facebook/', 'huggingface/'
    ])
    
    TokenizerRegistry.register_adapter(
        'huggingface', 
        HuggingFaceAdapter, 
        model_patterns
    )
    
    logger.info(f"Registered HuggingFace adapter with {len(model_patterns)} model patterns")


# Auto-register when module is imported
register_huggingface_adapter()