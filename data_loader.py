import os
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from typing import Tuple, List
import pickle
import re

class DialogueTokenizer:
    """Simple tokenizer that converts dialogue text to embeddings using TF-IDF."""
    
    def __init__(self, max_features: int = 10000, embedding_dim: int = 128):
        self.max_features = max_features
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.is_fitted = False
        
    def fit(self, texts: List[str]):
        """Fit the tokenizer on the given texts."""
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        self.vectorizer.fit(cleaned_texts)
        self.is_fitted = True
        
    def encode(self, texts: List[str]) -> np.ndarray:
        """Convert texts to fixed-size embeddings."""
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted before encoding")
            
        cleaned_texts = [self._clean_text(text) for text in texts]
        tfidf_matrix = self.vectorizer.transform(cleaned_texts).toarray()
        
        # Pad or truncate to fixed embedding dimension
        if tfidf_matrix.shape[1] > self.embedding_dim:
            embeddings = tfidf_matrix[:, :self.embedding_dim]
        else:
            embeddings = np.zeros((tfidf_matrix.shape[0], self.embedding_dim))
            embeddings[:, :tfidf_matrix.shape[1]] = tfidf_matrix
            
        return embeddings.astype(np.float32)
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing special characters and normalizing."""
        if pd.isna(text):
            return ""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', '', str(text))
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()

def download_dataset(url: str, cache_path: str = "dataset_cache.csv") -> str:
    """Download dataset from URL and cache locally."""
    if os.path.exists(cache_path):
        print(f"Using cached dataset from {cache_path}")
        return cache_path
    
    print(f"Downloading dataset from {url}")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(cache_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Dataset downloaded and cached to {cache_path}")
        return cache_path
    
    except requests.RequestException as e:
        print(f"Error downloading dataset: {e}")
        raise

def parse_dialogue_sequences(df: pd.DataFrame, window_size: int) -> Tuple[List[List[str]], List[str]]:
    """Parse dialogue turns into sequences for next-token prediction."""
    sequences = []
    next_tokens = []
    
    # Group by conversation/persona to maintain dialogue context
    if 'conversation_id' in df.columns:
        groups = df.groupby('conversation_id')
    else:
        # If no conversation_id, treat each row as separate dialogue
        groups = [(i, pd.DataFrame([row])) for i, (_, row) in enumerate(df.iterrows())]
    
    for conv_id, conv_df in groups:
        # Extract dialogue turns - assume there are columns like 'user_message', 'assistant_message'
        dialogue_turns = []
        
        for _, row in conv_df.iterrows():
            # Try different possible column names for dialogue content
            text_cols = ['Best Generated Conversation', 'text', 'message', 'user_message', 'assistant_message', 'dialogue', 'content']
            row_text = None
            
            for col in text_cols:
                if col in row and pd.notna(row[col]):
                    row_text = str(row[col])
                    break
            
            # If found dialogue text, split it into turns
            if row_text and len(row_text.strip()) > 0:
                # For the Synthetic-Persona-Chat dataset, the conversation might be in one field
                # Let's split it by common dialogue patterns
                conversation_text = row_text.strip()
                
                # Try to split by common patterns like "User:" or "Assistant:" or line breaks
                turns = []
                if 'User:' in conversation_text and 'Assistant:' in conversation_text:
                    # Split by user/assistant markers
                    parts = conversation_text.replace('User:', '\nUser:').replace('Assistant:', '\nAssistant:').split('\n')
                    for part in parts:
                        part = part.strip()
                        if part and len(part) > 5:  # Skip very short parts
                            turns.append(part)
                elif '\n' in conversation_text:
                    # Split by line breaks
                    turns = [line.strip() for line in conversation_text.split('\n') if line.strip() and len(line.strip()) > 10]
                else:
                    # If no clear structure, treat as single turn but split into sentences
                    import re
                    sentences = re.split(r'[.!?]+', conversation_text)
                    turns = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
                
                dialogue_turns.extend(turns)
        
        # Create sliding windows for sequence prediction
        if len(dialogue_turns) > window_size:
            for i in range(len(dialogue_turns) - window_size):
                sequence = dialogue_turns[i:i + window_size]
                next_token = dialogue_turns[i + window_size]
                sequences.append(sequence)
                next_tokens.append(next_token)
    
    return sequences, next_tokens

def load_data(window_size: int = 10, test_size: float = 0.2, 
              embedding_dim: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess dialogue data for next-token prediction.
    
    Args:
        window_size: Number of previous tokens to use for prediction
        test_size: Fraction of data to use for testing
        embedding_dim: Dimension of token embeddings
        
    Returns:
        X_train, y_train, X_test, y_test as numpy arrays
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Download dataset
    url = "https://huggingface.co/datasets/google/Synthetic-Persona-Chat/resolve/main/data/Synthetic-Persona-Chat_train.csv"
    dataset_path = download_dataset(url)
    
    # Load CSV
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        raise
    
    if len(df) == 0:
        raise ValueError("Dataset is empty")
    
    # Parse dialogue sequences
    sequences, next_tokens = parse_dialogue_sequences(df, window_size)
    
    if len(sequences) == 0:
        raise ValueError("No dialogue sequences could be parsed from the dataset")
    
    print(f"Parsed {len(sequences)} dialogue sequences")
    
    # Initialize tokenizer
    tokenizer = DialogueTokenizer(embedding_dim=embedding_dim)
    
    # Fit tokenizer on all text
    all_texts = []
    for seq in sequences:
        all_texts.extend(seq)
    all_texts.extend(next_tokens)
    
    print("Fitting tokenizer...")
    tokenizer.fit(all_texts)
    
    # Convert sequences to embeddings
    print("Converting sequences to embeddings...")
    X = []
    for seq in sequences:
        seq_embeddings = tokenizer.encode(seq)
        X.append(seq_embeddings)
    
    X = np.array(X)  # Shape: (num_sequences, window_size, embedding_dim)
    
    # Convert next tokens to embeddings
    y = tokenizer.encode(next_tokens)  # Shape: (num_sequences, embedding_dim)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    # Test data loading
    X_train, y_train, X_test, y_test = load_data(window_size=5, test_size=0.2)
    print("Data loading test completed successfully!")
