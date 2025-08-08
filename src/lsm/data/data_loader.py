#!/usr/bin/env python3
"""
Data loading and preprocessing for Sparse Sine-Activated LSM.

This module handles dialogue dataset loading, tokenization with TF-IDF,
and sequence preparation for next-token prediction training.
"""

import os
import requests
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import json

from ..utils.lsm_exceptions import (
    DataLoadError, TokenizerError, TokenizerNotFittedError, 
    TokenizerLoadError, TokenizerSaveError, InvalidInputError
)
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)

class DialogueTokenizer:
    """
    Custom tokenizer using TF-IDF for dialogue text to embedding conversion.
    Supports save/load functionality and decoding capabilities.
    """
    
    def __init__(self, max_features: int = 10000, embedding_dim: int = 128,
                 ngram_range: Tuple[int, int] = (1, 2), min_df: int = 2):
        """
        Initialize DialogueTokenizer.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            embedding_dim: Target embedding dimension
            ngram_range: N-gram range for TF-IDF
            min_df: Minimum document frequency for TF-IDF
        """
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.ngram_range = ngram_range
        self.min_df = min_df
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words='english'
        )
        
        self._vocabulary_texts = []
        self._vocabulary_embeddings = None
        self._text_to_embedding = {}
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        """
        Fit the tokenizer on dialogue texts.
        
        Args:
            texts: List of dialogue texts for training
        """
        logger.info(f"Fitting tokenizer on {len(texts)} texts")
        
        # Fit TF-IDF vectorizer
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        
        # Create embeddings by reducing TF-IDF dimensions
        if tfidf_matrix.shape[1] > self.embedding_dim:
            # Use SVD to reduce dimensions
            from sklearn.decomposition import TruncatedSVD
            svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
            embeddings = svd.fit_transform(tfidf_matrix.toarray())
        else:
            # Pad with zeros if TF-IDF features < embedding_dim
            embeddings = tfidf_matrix.toarray()
            if embeddings.shape[1] < self.embedding_dim:
                padding = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]))
                embeddings = np.concatenate([embeddings, padding], axis=1)
        
        # Store vocabulary and embeddings
        self._vocabulary_texts = texts.copy()
        self._vocabulary_embeddings = embeddings.astype(np.float32)
        
        # Create text-to-embedding mapping
        self._text_to_embedding = {
            text: embeddings[i] for i, text in enumerate(texts)
        }
        
        self.is_fitted = True
        logger.info(f"Tokenizer fitted with {len(self._vocabulary_texts)} vocabulary items")
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.is_fitted:
            raise TokenizerNotFittedError("encode")
        
        embeddings = []
        
        for text in texts:
            if text in self._text_to_embedding:
                # Use cached embedding
                embedding = self._text_to_embedding[text]
            else:
                # Transform new text using TF-IDF
                tfidf_vec = self.vectorizer.transform([text])
                
                # Reduce to embedding_dim if necessary
                if tfidf_vec.shape[1] > self.embedding_dim:
                    # Simple average pooling for new texts
                    embedding = np.mean(tfidf_vec.toarray().reshape(-1, self.embedding_dim), axis=0)
                else:
                    embedding = tfidf_vec.toarray().flatten()
                    if len(embedding) < self.embedding_dim:
                        # Pad with zeros
                        padding = np.zeros(self.embedding_dim - len(embedding))
                        embedding = np.concatenate([embedding, padding])
                
                embedding = embedding.astype(np.float32)
            
            embeddings.append(embedding)
        
        return np.array(embeddings, dtype=np.float32)
    
    def decode_embedding(self, embedding: np.ndarray, top_k: int = 3) -> str:
        """
        Decode an embedding back to text using similarity search.
        
        Args:
            embedding: Single embedding vector
            top_k: Number of top candidates to consider
            
        Returns:
            Decoded text string
        """
        if not self.is_fitted:
            raise TokenizerNotFittedError("decode_embedding")
        
        try:
            # Compute cosine similarities
            embedding = embedding.reshape(1, -1)
            similarities = np.dot(self._vocabulary_embeddings, embedding.T).flatten()
            
            # Get top-k most similar
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            if len(top_indices) > 0 and similarities[top_indices[0]] > 0.1:
                return self._vocabulary_texts[top_indices[0]]
            else:
                return "[UNKNOWN]"
                
        except Exception as e:
            logger.warning(f"Error decoding embedding: {e}")
            return "[ERROR]"
    
    def decode_embeddings(self, embeddings: np.ndarray) -> List[str]:
        """
        Decode multiple embeddings to texts.
        
        Args:
            embeddings: Array of embeddings (batch_size, embedding_dim)
            
        Returns:
            List of decoded text strings
        """
        return [self.decode_embedding(emb) for emb in embeddings]
    
    def save(self, save_path: str):
        """
        Save tokenizer to disk.
        
        Args:
            save_path: Directory path to save tokenizer files
        """
        if not self.is_fitted:
            raise TokenizerNotFittedError("save")
        
        try:
            os.makedirs(save_path, exist_ok=True)
            
            # Save vectorizer
            with open(os.path.join(save_path, "vectorizer.pkl"), "wb") as f:
                pickle.dump(self.vectorizer, f)
            
            # Save vocabulary and embeddings
            np.save(os.path.join(save_path, "vocabulary_embeddings.npy"), 
                   self._vocabulary_embeddings)
            
            with open(os.path.join(save_path, "vocabulary_texts.pkl"), "wb") as f:
                pickle.dump(self._vocabulary_texts, f)
            
            # Save configuration
            config = {
                'max_features': self.max_features,
                'embedding_dim': self.embedding_dim,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'is_fitted': self.is_fitted
            }
            
            with open(os.path.join(save_path, "tokenizer_config.json"), "w") as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Tokenizer saved to {save_path}")
            
        except Exception as e:
            raise TokenizerSaveError(save_path, str(e))
    
    def load(self, load_path: str):
        """
        Load tokenizer from disk.
        
        Args:
            load_path: Directory path to load tokenizer files from
        """
        try:
            # Load configuration
            config_path = os.path.join(load_path, "tokenizer_config.json")
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Set configuration
            self.max_features = config['max_features']
            self.embedding_dim = config['embedding_dim']
            self.ngram_range = tuple(config['ngram_range'])
            self.min_df = config['min_df']
            
            # Load vectorizer
            with open(os.path.join(load_path, "vectorizer.pkl"), "rb") as f:
                self.vectorizer = pickle.load(f)
            
            # Load vocabulary and embeddings
            self._vocabulary_embeddings = np.load(
                os.path.join(load_path, "vocabulary_embeddings.npy")
            )
            
            with open(os.path.join(load_path, "vocabulary_texts.pkl"), "rb") as f:
                self._vocabulary_texts = pickle.load(f)
            
            # Recreate text-to-embedding mapping
            self._text_to_embedding = {
                text: self._vocabulary_embeddings[i] 
                for i, text in enumerate(self._vocabulary_texts)
            }
            
            self.is_fitted = config['is_fitted']
            logger.info(f"Tokenizer loaded from {load_path}")
            
        except Exception as e:
            raise TokenizerLoadError(load_path, str(e))


def download_dataset(cache_file: str = "dataset_cache.csv") -> pd.DataFrame:
    """
    Download dialogue dataset from HuggingFace.
    
    Args:
        cache_file: Local cache file path
        
    Returns:
        DataFrame with dialogue data
    """
    if os.path.exists(cache_file):
        logger.info(f"Loading cached dataset from {cache_file}")
        return pd.read_csv(cache_file)
    
    logger.info("Downloading dialogue dataset from HuggingFace...")
    
    try:
        # Download the Synthetic-Persona-Chat dataset
        url = "https://huggingface.co/datasets/google/Synthetic-Persona-Chat/resolve/main/train.csv"
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse CSV content
        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        
        # Cache the dataset
        df.to_csv(cache_file, index=False)
        logger.info(f"Dataset downloaded and cached to {cache_file}")
        
        return df
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        
        # Create minimal synthetic data for testing
        logger.warning("Creating minimal synthetic dataset for testing")
        synthetic_data = {
            'User': [
                "Hello, how are you today?",
                "What's your favorite hobby?",
                "Tell me about yourself.",
                "What do you like to do for fun?",
                "How's the weather where you are?",
            ] * 20,
            'Assistant': [
                "I'm doing well, thank you for asking!",
                "I enjoy reading and learning new things.",
                "I'm an AI assistant here to help you.",
                "I like helping people solve problems.",
                "I don't experience weather, but I hope it's nice!",
            ] * 20
        }
        
        df = pd.DataFrame(synthetic_data)
        df.to_csv(cache_file, index=False)
        return df


def prepare_dialogue_sequences(df: pd.DataFrame, window_size: int = 10) -> List[List[str]]:
    """
    Convert dialogue DataFrame to sequences for training.
    
    Args:
        df: DataFrame with conversations
        window_size: Size of sequence windows
        
    Returns:
        List of dialogue sequences
    """
    sequences = []
    all_turns = []
    
    # Check for the actual column names in the dataset
    if 'Best Generated Conversation' in df.columns:
        # Parse the conversation strings
        for idx, row in df.iterrows():
            if pd.notna(row['Best Generated Conversation']):
                conversation = str(row['Best Generated Conversation'])
                # Split conversation by lines and extract turns
                lines = conversation.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('User 1:') or line.startswith('User 2:'):
                        # Extract the actual dialogue content
                        text = line.split(':', 1)
                        if len(text) > 1:
                            all_turns.append(text[1].strip())
    elif 'User' in df.columns and 'Assistant' in df.columns:
        # Original format
        for idx, row in df.iterrows():
            if 'User' in row and pd.notna(row['User']):
                all_turns.append(str(row['User']))
            if 'Assistant' in row and pd.notna(row['Assistant']):
                all_turns.append(str(row['Assistant']))
    else:
        # Create synthetic data if columns don't match expected format
        logger.warning("Creating synthetic dialogue data for training")
        synthetic_turns = [
            "Hello, how are you today?",
            "I'm doing well, thank you for asking!",
            "What's your favorite hobby?",
            "I enjoy reading and learning new things.",
            "That sounds interesting. What kind of books do you like?",
            "I like science fiction and fantasy novels.",
            "Have you read any good books lately?",
            "Yes, I just finished a great sci-fi novel.",
            "What was it about?",
            "It was about space exploration and alien encounters.",
            "That sounds fascinating. Would you recommend it?",
            "Definitely! It's a great read for sci-fi fans."
        ] * 20  # Repeat to have enough data
        all_turns.extend(synthetic_turns)
    
    # Create sliding windows
    for i in range(len(all_turns) - window_size):
        sequence = all_turns[i:i + window_size + 1]
        sequences.append(sequence)
    
    logger.info(f"Created {len(sequences)} dialogue sequences with window size {window_size}")
    return sequences


def load_data(window_size: int = 10, test_size: float = 0.2, 
              embedding_dim: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DialogueTokenizer]:
    """
    Load and prepare dialogue data for LSM training.
    
    Args:
        window_size: Size of sequence windows
        test_size: Fraction of data for testing
        embedding_dim: Embedding dimension
        
    Returns:
        Tuple of (X_train, y_train, X_test, y_test, tokenizer)
    """
    logger.info(f"Loading data with window_size={window_size}, embedding_dim={embedding_dim}")
    
    # Download dataset
    df = download_dataset()
    
    # Prepare sequences
    sequences = prepare_dialogue_sequences(df, window_size)
    
    if len(sequences) < 10:
        raise DataLoadError("dataset", "Insufficient dialogue sequences generated")
    
    # Extract all unique texts for tokenizer training
    all_texts = []
    for seq in sequences:
        all_texts.extend(seq)
    unique_texts = list(set(all_texts))
    
    # Initialize and fit tokenizer
    tokenizer = DialogueTokenizer(embedding_dim=embedding_dim)
    tokenizer.fit(unique_texts)
    
    # Create training data
    X, y = [], []
    for sequence in sequences:
        input_sequence = sequence[:-1]  # All but last
        target = sequence[-1]  # Last item
        
        # Convert to embeddings
        input_embeddings = tokenizer.encode(input_sequence)
        target_embedding = tokenizer.encode([target])[0]
        
        X.append(input_embeddings)
        y.append(target_embedding)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    logger.info(f"Data loaded: train={len(X_train)}, test={len(X_test)}")
    logger.info(f"Input shape: {X_train.shape}, Output shape: {y_train.shape}")
    
    return X_train, y_train, X_test, y_test, tokenizer