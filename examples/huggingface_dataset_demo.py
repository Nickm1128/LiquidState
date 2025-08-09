#!/usr/bin/env python3
"""
Demonstration of HuggingFace dataset integration for LSM training.

This script shows how to use the new HuggingFace dataset loader to download
and process the cosmopedia-v2 dataset with conversation-aware splitting.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from lsm.data.huggingface_loader import (
    HuggingFaceDatasetLoader, 
    ConversationSplitter, 
    DatasetProcessor
)
from lsm.utils.lsm_logging import get_logger

logger = get_logger(__name__)

def main():
    """Demonstrate HuggingFace dataset integration."""
    print("=" * 60)
    print("HuggingFace Dataset Integration Demo")
    print("=" * 60)
    
    # HuggingFace API token (replace with your own or set as environment variable)
    api_token = os.getenv('HUGGINGFACE_TOKEN', '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')
    
    try:
        # Initialize the dataset loader
        print("1. Initializing HuggingFace dataset loader...")
        loader = HuggingFaceDatasetLoader(
            cache_dir="data/huggingface_cache",
            api_token=api_token
        )
        
        # Check if we have cached data first
        print("\n2. Checking for cached dataset...")
        dataset_info = loader.get_dataset_info()
        print(f"Dataset info: {dataset_info}")
        
        # Try to load cached data, or download if not available
        try:
            print("\n3. Loading cached dataset...")
            df = loader.load_cached_datasets()
            print(f"Loaded cached dataset with {len(df)} rows")
        except Exception as e:
            print(f"No cached data found: {e}")
            print("\n3. Downloading cosmopedia-v2 dataset...")
            print("Note: This may take several minutes for the full dataset...")
            
            # For demo purposes, we'll skip the actual download
            # Uncomment the next line to actually download the dataset
            # csv_files = loader.download_cosmopedia_csvs()
            
            print("Skipping actual download for demo. In real usage, uncomment the download line.")
            return
        
        # Validate the dataset
        print("\n4. Validating dataset integrity...")
        is_valid = loader.validate_dataset_integrity(df)
        print(f"Dataset validation: {'✓ Passed' if is_valid else '✗ Failed'}")
        
        # Process the dataset
        print("\n5. Processing dataset...")
        processor = DatasetProcessor()
        
        # Get validation results and statistics
        validation_results = processor.validate_dataset_structure(df)
        print(f"Dataset statistics:")
        for key, value in validation_results['statistics'].items():
            print(f"  {key}: {value}")
        
        # Extract conversation metadata
        metadata = processor.extract_conversation_metadata(df)
        print(f"\nConversation metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        # Prepare dataset for training
        prepared_df = processor.prepare_for_training(df)
        print(f"\nPrepared dataset: {len(prepared_df)} rows ready for training")
        
        # Split by conversations
        print("\n6. Splitting dataset by conversations...")
        splitter = ConversationSplitter()
        
        # Identify conversation boundaries
        boundaries = splitter.identify_conversation_boundaries(prepared_df)
        print(f"Found {len(boundaries)} conversation boundaries")
        
        # Split into train and test sets
        train_df, test_df = splitter.split_by_conversation(prepared_df, test_ratio=0.2)
        print(f"Train set: {len(train_df)} rows")
        print(f"Test set: {len(test_df)} rows")
        
        # Verify conversation integrity
        integrity_ok = splitter.ensure_conversation_integrity(train_df, test_df)
        print(f"Conversation integrity: {'✓ Maintained' if integrity_ok else '✗ Violated'}")
        
        # Show sample data
        print("\n7. Sample data from train set:")
        if not train_df.empty and 'text' in train_df.columns:
            for i, (_, row) in enumerate(train_df.head(3).iterrows()):
                print(f"  Row {i+1}: {row['text'][:100]}...")
        
        print("\n" + "=" * 60)
        print("✅ Demo completed successfully!")
        print("=" * 60)
        
        # Usage summary
        print("\nUsage Summary:")
        print("1. HuggingFaceDatasetLoader: Downloads and caches cosmopedia-v2 dataset")
        print("2. DatasetProcessor: Validates and processes the dataset")
        print("3. ConversationSplitter: Splits data while keeping conversations intact")
        print("\nNext steps:")
        print("- Use the train_df and test_df for LSM training")
        print("- Apply tokenization and embedding optimization")
        print("- Train the enhanced LSM model with conversation-aware data")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())