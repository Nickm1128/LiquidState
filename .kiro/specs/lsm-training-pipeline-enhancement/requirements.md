# Requirements Document

## Introduction

This feature enhances the LSM (Liquid State Machine) project with a comprehensive training pipeline overhaul. The enhancement focuses on integrating real-world conversational datasets from HuggingFace, implementing proper tokenization and embedding strategies optimized for sinusoidal patterns, and extending the CNN architecture to support both 2D and 3D models. The system will transition from token-level to response-level inference with support for system messages and conversation-aware training splits.

## Requirements

### Requirement 1: HuggingFace Dataset Integration

**User Story:** As a researcher, I want to train the LSM model on real conversational data from HuggingFace, so that the model can learn from diverse, high-quality dialogue patterns.

#### Acceptance Criteria

1. WHEN the system downloads datasets THEN it SHALL retrieve all six CSV files from the HuggingFace smollm-corpus cosmopedia-v2 dataset
2. WHEN processing downloaded data THEN the system SHALL automatically separate data into train and test sets based on conversation boundaries
3. WHEN splitting data THEN the system SHALL ensure complete conversations remain intact in either train or test sets, never split across sets
4. IF a conversation spans multiple entries THEN the system SHALL group all related entries together for splitting

### Requirement 2: Advanced Tokenization and Embedding System

**User Story:** As a machine learning engineer, I want to replace the custom tokenizer with a real tokenizer and implement embeddings optimized for sinusoidal patterns, so that the model can better learn from the natural structure of language.

#### Acceptance Criteria

1. WHEN tokenizing text THEN the system SHALL use a standard tokenizer instead of the current custom implementation
2. WHEN creating embeddings THEN the system SHALL fit embeddings to maximize sinusoidality of naturally occurring patterns in the training data
3. WHEN training embeddings THEN the system SHALL optimize for patterns that enhance learnability by the reservoir/CNN models
4. IF embeddings are generated THEN they SHALL be designed to work optimally with the sine-activated LSM architecture

### Requirement 3: Enhanced CNN Architecture Support

**User Story:** As a deep learning researcher, I want to extend the CNN models to support both 2D and 3D architectures with different loss functions, so that I can experiment with more sophisticated pattern recognition approaches.

#### Acceptance Criteria

1. WHEN implementing CNN models THEN the system SHALL support both 2D and 3D CNN architectures
2. WHEN using 3D CNNs THEN the system SHALL accept additional dimensional input for system message integration
3. WHEN calculating loss THEN the system SHALL use cosine similarity loss for response-level inference instead of token-level loss
4. IF the model processes system messages THEN it SHALL use 3D CNN architecture with appropriate embedding modifiers

### Requirement 4: Response-Level Inference Pipeline

**User Story:** As a conversational AI developer, I want the system to generate complete responses rather than individual tokens, so that I can produce more coherent and contextually appropriate outputs.

#### Acceptance Criteria

1. WHEN generating responses THEN the system SHALL produce entire messages instead of iterative token generation
2. WHEN processing token sequences THEN a secondary model SHALL accept embeddings from the reservoir/CNN pipeline and predict complete responses
3. WHEN implementing response inference THEN the system SHALL determine whether to reuse the existing reservoir or create a separate one
4. IF token embeddings are processed THEN they SHALL be fed sequentially to the response-level model for final output generation

### Requirement 5: System Message Integration

**User Story:** As a chatbot developer, I want to incorporate system messages that influence model behavior, so that I can control the tone, style, and constraints of generated responses.

#### Acceptance Criteria

1. WHEN processing system messages THEN the system SHALL provide separate reservoir output to 3D CNN models
2. WHEN reading system prompts THEN a dedicated model SHALL produce embedding modifiers that influence response generation
3. WHEN training with system messages THEN the embedding modifier model SHALL be trainable through backpropagation
4. IF system messages are present THEN they SHALL be processed separately from user messages and integrated at the CNN level

### Requirement 6: Message Annotation and Pipeline Architecture

**User Story:** As a system architect, I want to add message annotations and create a modular pipeline, so that the system can be easily deployed and experimented with across different environments.

#### Acceptance Criteria

1. WHEN processing messages THEN the system SHALL add appropriate annotations (e.g., |start|, |end|, |system|)
2. WHEN setting up the pipeline THEN it SHALL be designed for easy cloning and deployment to Google Colab
3. WHEN experimenting with architectures THEN the system SHALL support easy swapping of different model components
4. IF multiple reservoir/CNN combinations are used THEN a final ANN SHALL synthesize outputs into the final response

### Requirement 7: Training Data Optimization

**User Story:** As a data scientist, I want the training process to be optimized for conversational patterns, so that the model learns more effectively from dialogue structure.

#### Acceptance Criteria

1. WHEN splitting training data THEN the system SHALL split by complete conversations rather than by individual windows or tokens
2. WHEN processing conversations THEN the system SHALL maintain conversation context and flow
3. WHEN training THEN the system SHALL optimize for the natural sinusoidal patterns found in conversational embeddings
4. IF conversations have multiple turns THEN they SHALL be processed as coherent sequences during training