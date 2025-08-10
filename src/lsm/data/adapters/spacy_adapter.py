#!/usr/bin/env python3
"""
spaCy tokenizer adapter for enhanced tokenization system.

This module provides an adapter for spaCy tokenizers with linguistic features,
allowing them to work with the enhanced tokenizer system with language-specific
tokenization rules and Unicode handling.
"""

import os
import json
import unicodedata
from typing import List, Dict, Any, Optional, Union, Set

try:
    import spacy
    from spacy.lang import LANGUAGES
    from spacy.tokens import Doc
    SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    LANGUAGES = {}
    Doc = None
    SPACY_AVAILABLE = False

from ..enhanced_tokenization import TokenizerAdapter, TokenizerConfig, TokenizerRegistry
from ...utils.lsm_exceptions import TokenizerError, TokenizerLoadError
from ...utils.lsm_logging import get_logger

logger = get_logger(__name__)


class SpacyAdapter(TokenizerAdapter):
    """
    Adapter for spaCy tokenizers with linguistic features.
    
    This adapter wraps spaCy language models to work with the enhanced
    tokenizer system, providing vocabulary extraction, token mapping,
    linguistic features, and language-specific tokenization rules.
    """
    
    # Supported spaCy models and their language codes
    SUPPORTED_MODELS = {
        # English models
        'en_core_web_sm': 'en',
        'en_core_web_md': 'en', 
        'en_core_web_lg': 'en',
        'en_core_web_trf': 'en',
        
        # German models
        'de_core_news_sm': 'de',
        'de_core_news_md': 'de',
        'de_core_news_lg': 'de',
        'de_dep_news_trf': 'de',
        
        # French models
        'fr_core_news_sm': 'fr',
        'fr_core_news_md': 'fr',
        'fr_core_news_lg': 'fr',
        'fr_dep_news_trf': 'fr',
        
        # Spanish models
        'es_core_news_sm': 'es',
        'es_core_news_md': 'es',
        'es_core_news_lg': 'es',
        'es_dep_news_trf': 'es',
        
        # Italian models
        'it_core_news_sm': 'it',
        'it_core_news_md': 'it',
        'it_core_news_lg': 'it',
        
        # Portuguese models
        'pt_core_news_sm': 'pt',
        'pt_core_news_md': 'pt',
        'pt_core_news_lg': 'pt',
        
        # Dutch models
        'nl_core_news_sm': 'nl',
        'nl_core_news_md': 'nl',
        'nl_core_news_lg': 'nl',
        
        # Chinese models
        'zh_core_web_sm': 'zh',
        'zh_core_web_md': 'zh',
        'zh_core_web_lg': 'zh',
        'zh_core_web_trf': 'zh',
        
        # Japanese models
        'ja_core_news_sm': 'ja',
        'ja_core_news_md': 'ja',
        'ja_core_news_lg': 'ja',
        'ja_core_news_trf': 'ja',
        
        # Multi-language models
        'xx_ent_wiki_sm': 'xx',
        'xx_sent_ud_sm': 'xx',
        
        # Language codes for blank models
        'en': 'en', 'de': 'de', 'fr': 'fr', 'es': 'es', 'it': 'it',
        'pt': 'pt', 'nl': 'nl', 'zh': 'zh', 'ja': 'ja', 'ru': 'ru',
        'ar': 'ar', 'hi': 'hi', 'ko': 'ko', 'th': 'th', 'vi': 'vi'
    }
    
    # Unicode normalization forms
    UNICODE_NORMALIZATIONS = {
        'NFC': unicodedata.normalize,
        'NFD': unicodedata.normalize,
        'NFKC': unicodedata.normalize,
        'NFKD': unicodedata.normalize
    }
    
    def __init__(self, config: TokenizerConfig):
        """
        Initialize spaCy adapter.
        
        Args:
            config: Tokenizer configuration
        """
        super().__init__(config)
        
        if not SPACY_AVAILABLE:
            raise TokenizerError(
                "spaCy library not available. Install with: pip install spacy"
            )
        
        self._language_code = None
        self._vocab_to_id = {}
        self._id_to_vocab = {}
        self._special_tokens = {}
        self._unicode_normalization = None
        self._use_linguistic_features = True
        self._custom_pipeline_components = []
    
    def initialize(self) -> None:
        """Initialize the spaCy language model."""
        try:
            model_name = self.config.model_name
            
            # Get backend-specific configuration
            backend_config = self.config.backend_specific_config or {}
            self._use_linguistic_features = backend_config.get('use_linguistic_features', True)
            self._unicode_normalization = backend_config.get('unicode_normalization', 'NFC')
            self._custom_pipeline_components = backend_config.get('custom_components', [])
            
            # Determine language code
            if model_name in self.SUPPORTED_MODELS:
                self._language_code = self.SUPPORTED_MODELS[model_name]
            else:
                # Assume it's a language code or custom model
                self._language_code = model_name
            
            # Load spaCy model
            try:
                if model_name in self.SUPPORTED_MODELS and model_name not in LANGUAGES:
                    # Try to load a trained model
                    self._tokenizer = spacy.load(model_name)
                    logger.info(f"Loaded spaCy trained model: {model_name}")
                else:
                    # Create blank model for language
                    if self._language_code in LANGUAGES:
                        self._tokenizer = spacy.blank(self._language_code)
                        logger.info(f"Created spaCy blank model for language: {self._language_code}")
                    else:
                        # Fallback to English
                        logger.warning(f"Unknown language '{self._language_code}', falling back to English")
                        self._tokenizer = spacy.blank('en')
                        self._language_code = 'en'
                        
            except OSError as e:
                if "Can't find model" in str(e):
                    # Model not installed, try blank model
                    logger.warning(f"Model '{model_name}' not found, creating blank model for language '{self._language_code}'")
                    if self._language_code in LANGUAGES:
                        self._tokenizer = spacy.blank(self._language_code)
                    else:
                        logger.warning(f"Language '{self._language_code}' not supported, using English")
                        self._tokenizer = spacy.blank('en')
                        self._language_code = 'en'
                else:
                    raise
            
            # Add custom pipeline components if specified
            for component_config in self._custom_pipeline_components:
                component_name = component_config.get('name')
                component_factory = component_config.get('factory')
                component_config_dict = component_config.get('config', {})
                
                if component_name and component_factory:
                    try:
                        self._tokenizer.add_pipe(component_factory, name=component_name, config=component_config_dict)
                        logger.info(f"Added custom pipeline component: {component_name}")
                    except Exception as e:
                        logger.warning(f"Failed to add pipeline component '{component_name}': {e}")
            
            # Build vocabulary mapping
            self._build_vocabulary()
            
            # Set up special tokens
            self._setup_special_tokens()
            
            self._is_initialized = True
            
            logger.info(f"Initialized spaCy adapter for '{model_name}' with vocab size {self._vocab_size}")
            
        except Exception as e:
            raise TokenizerError(f"Failed to initialize spaCy adapter '{self.config.model_name}': {str(e)}")
    
    def _build_vocabulary(self) -> None:
        """Build vocabulary mapping from spaCy vocab."""
        # Get all tokens from spaCy vocabulary
        vocab_items = []
        
        # Add tokens from spaCy vocab
        for token_text, lexeme in self._tokenizer.vocab:
            if lexeme.orth != lexeme.norm:  # Skip if same as normalized form
                continue
            vocab_items.append((token_text, lexeme.orth))
        
        # Sort by frequency if available, otherwise by orth ID
        vocab_items.sort(key=lambda x: x[1])
        
        # Create mappings
        self._vocab_to_id = {}
        self._id_to_vocab = {}
        
        for idx, (token_text, orth_id) in enumerate(vocab_items):
            self._vocab_to_id[token_text] = idx
            self._id_to_vocab[idx] = token_text
        
        # Add special tokens to vocabulary if not present
        special_token_texts = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        for token_text in special_token_texts:
            if token_text not in self._vocab_to_id:
                new_id = len(self._vocab_to_id)
                self._vocab_to_id[token_text] = new_id
                self._id_to_vocab[new_id] = token_text
        
        self._vocab_size = len(self._vocab_to_id)
        
        logger.info(f"Built spaCy vocabulary with {self._vocab_size} tokens")
    
    def _setup_special_tokens(self) -> None:
        """Set up special tokens for the tokenizer."""
        # Default special tokens
        self._special_tokens = {
            'pad_token_id': self._vocab_to_id.get('<PAD>', 0),
            'unk_token_id': self._vocab_to_id.get('<UNK>', 1),
            'bos_token_id': self._vocab_to_id.get('<BOS>', 2),
            'eos_token_id': self._vocab_to_id.get('<EOS>', 3)
        }
        
        # Apply custom special tokens if provided
        if self.config.special_tokens:
            for token_type, token_value in self.config.special_tokens.items():
                if token_type.endswith('_token'):
                    # Add token to vocabulary if not present
                    if token_value not in self._vocab_to_id:
                        new_id = len(self._vocab_to_id)
                        self._vocab_to_id[token_value] = new_id
                        self._id_to_vocab[new_id] = token_value
                        self._vocab_size += 1
                    
                    self._special_tokens[f'{token_type}_id'] = self._vocab_to_id[token_value]
                elif token_type.endswith('_token_id'):
                    # Direct token ID
                    self._special_tokens[token_type] = token_value
    
    def _normalize_unicode(self, text: str) -> str:
        """Apply Unicode normalization to text."""
        if self._unicode_normalization and self._unicode_normalization in self.UNICODE_NORMALIZATIONS:
            return unicodedata.normalize(self._unicode_normalization, text)
        return text
    
    def _tokenize_with_spacy(self, text: str) -> List[str]:
        """Tokenize text using spaCy and extract tokens."""
        # Apply Unicode normalization
        normalized_text = self._normalize_unicode(text)
        
        # Process with spaCy
        doc = self._tokenizer(normalized_text)
        
        # Extract tokens based on configuration
        tokens = []
        for token in doc:
            if self._use_linguistic_features:
                # Use lemmatized form if available and different from text
                if token.lemma_ and token.lemma_ != token.text and token.lemma_ != '-PRON-':
                    token_text = token.lemma_.lower()
                else:
                    token_text = token.text.lower()
            else:
                # Use raw token text
                token_text = token.text
            
            # Skip whitespace-only tokens
            if token_text.strip():
                tokens.append(token_text)
        
        return tokens
    
    def tokenize(self, texts: Union[str, List[str]], 
                 add_special_tokens: bool = True,
                 padding: bool = True, 
                 truncation: bool = True) -> List[List[int]]:
        """
        Tokenize texts to token IDs.
        
        Args:
            texts: Single text or list of texts to tokenize
            add_special_tokens: Whether to add special tokens (BOS/EOS)
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
                # Tokenize with spaCy
                tokens = self._tokenize_with_spacy(text)
                
                # Convert tokens to IDs
                token_ids = []
                for token in tokens:
                    if token in self._vocab_to_id:
                        token_ids.append(self._vocab_to_id[token])
                    else:
                        # Use UNK token for unknown tokens
                        token_ids.append(self._special_tokens['unk_token_id'])
                
                # Truncate if needed
                if truncation and self.config.max_length:
                    available_length = self.config.max_length
                    if add_special_tokens:
                        available_length -= 2  # Reserve space for BOS/EOS
                    
                    if len(token_ids) > available_length:
                        token_ids = token_ids[:available_length]
                
                # Add special tokens
                if add_special_tokens:
                    if 'bos_token_id' in self._special_tokens:
                        token_ids.insert(0, self._special_tokens['bos_token_id'])
                    if 'eos_token_id' in self._special_tokens:
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
            raise TokenizerError(f"spaCy tokenization failed: {str(e)}")
    
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
            raise TokenizerError(f"spaCy decoding failed: {str(e)}")
    
    def _decode_single(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode a single sequence of token IDs."""
        tokens = []
        special_token_ids = set(self._special_tokens.values()) if skip_special_tokens else set()
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_token_ids:
                continue
            
            if token_id in self._id_to_vocab:
                tokens.append(self._id_to_vocab[token_id])
            else:
                # Unknown token ID
                tokens.append('<UNK>')
        
        # Join tokens with spaces (simple reconstruction)
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        return self._vocab_to_id.copy()
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        return self._special_tokens.copy()
    
    def get_linguistic_features(self, text: str) -> Dict[str, Any]:
        """
        Extract linguistic features from text using spaCy.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with linguistic features
        """
        if not self._is_initialized:
            raise TokenizerError("Adapter not initialized. Call initialize() first.")
        
        if not self._use_linguistic_features:
            return {}
        
        try:
            doc = self._tokenizer(self._normalize_unicode(text))
            
            features = {
                'tokens': [],
                'pos_tags': [],
                'lemmas': [],
                'entities': [],
                'sentences': [],
                'dependencies': []
            }
            
            # Token-level features
            for token in doc:
                features['tokens'].append(token.text)
                features['pos_tags'].append(token.pos_)
                features['lemmas'].append(token.lemma_)
                features['dependencies'].append({
                    'text': token.text,
                    'dep': token.dep_,
                    'head': token.head.text if token.head != token else 'ROOT'
                })
            
            # Named entities
            for ent in doc.ents:
                features['entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            # Sentences
            for sent in doc.sents:
                features['sentences'].append(sent.text.strip())
            
            return features
            
        except Exception as e:
            logger.warning(f"Failed to extract linguistic features: {e}")
            return {}
    
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
            'language_code': self._language_code,
            'vocab_size': self._vocab_size,
            'max_length': self.config.max_length,
            'special_tokens': self.get_special_tokens(),
            'tokenizer_class': 'spacy.Language',
            'backend': 'spacy',
            'use_linguistic_features': self._use_linguistic_features,
            'unicode_normalization': self._unicode_normalization,
            'pipeline_components': [comp for comp in self._tokenizer.pipe_names],
            'language_info': {
                'lang': self._tokenizer.lang,
                'lang_': self._tokenizer.lang_,
                'vocab_size': len(self._tokenizer.vocab)
            }
        }
        
        return info
    
    @classmethod
    def load_adapter_config(cls, load_path: str) -> 'SpacyAdapter':
        """
        Load adapter from saved configuration.
        
        Args:
            load_path: Directory path to load config from
            
        Returns:
            Loaded SpacyAdapter instance
        """
        try:
            config_path = os.path.join(load_path, 'spacy_adapter_config.json')
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"spaCy adapter config not found: {config_path}")
            
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
            
            logger.info(f"Loaded spaCy adapter from {load_path}")
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
                'language_code': self._language_code,
                'max_length': self.config.max_length,
                'special_tokens': self.config.special_tokens,
                'backend_specific_config': self.config.backend_specific_config
            }
            
            config_path = os.path.join(save_path, 'spacy_adapter_config.json')
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            # Also save vocabulary mapping
            vocab_path = os.path.join(save_path, 'spacy_vocab.json')
            with open(vocab_path, 'w') as f:
                json.dump(self._vocab_to_id, f, indent=2)
            
            logger.info(f"Saved spaCy adapter config to {save_path}")
            
        except Exception as e:
            raise TokenizerError(f"Failed to save spaCy adapter config: {str(e)}")
    
    @classmethod
    def list_supported_models(cls) -> List[str]:
        """Get list of supported model names."""
        return list(cls.SUPPORTED_MODELS.keys())
    
    @classmethod
    def list_supported_languages(cls) -> List[str]:
        """Get list of supported language codes."""
        if SPACY_AVAILABLE:
            return list(LANGUAGES.keys())
        return []
    
    def __repr__(self) -> str:
        return (f"SpacyAdapter(model={self.config.model_name}, "
                f"language={self._language_code}, vocab_size={self._vocab_size}, "
                f"initialized={self._is_initialized})")


# Register the spaCy adapter
def register_spacy_adapter():
    """Register the spaCy adapter with the tokenizer registry."""
    model_patterns = list(SpacyAdapter.SUPPORTED_MODELS.keys())
    
    # Add language codes as patterns
    if SPACY_AVAILABLE:
        model_patterns.extend(list(LANGUAGES.keys()))
    
    TokenizerRegistry.register_adapter(
        'spacy', 
        SpacyAdapter, 
        model_patterns
    )
    
    logger.info(f"Registered spaCy adapter with {len(model_patterns)} model patterns")


# Auto-register when module is imported
register_spacy_adapter()