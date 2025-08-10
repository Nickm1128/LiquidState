# Design Document

## Overview

This design enhances the LSM tokenizer/embedder system to support flexible tokenizer backends with sinusoidal embeddings and streaming data processing. The solution builds upon the existing `StandardTokenizerWrapper` and `SinusoidalEmbedder` classes while adding new capabilities for any tokenizer integration, streaming data support, and enhanced convenience API integration.

The design addresses three main goals:
1. **Flexible Tokenizer Support**: Enable any tokenizer (HuggingFace, OpenAI, spaCy, custom) with automatic sinusoidal embedding adaptation
2. **Streaming Data Processing**: Handle large datasets that don't fit in memory with efficient batch processing
3. **Seamless Integration**: Maintain backward compatibility while enhancing the convenience API

## Architecture

### High-Level Design

The enhanced tokenizer system follows a modular architecture with clear separation of concerns:

```python
from lsm import LSMGenerator

# Simple usage with enhanced tokenizer
generator = LSMGenerator(
    tokenizer='gpt2',  # Any supported tokenizer
    embedding_type='sinusoidal',  # Enhanced sinusoidal embeddings
    streaming=True,  # Enable streaming for large datasets
    batch_size=1000  # Configurable batch si