#!/usr/bin/env python3
"""
Response Inference Model Demo.

This script demonstrates the usage of the ResponseInferenceModel for
secondary processing of token embedding sequences to predict complete responses.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.inference.response_inference_model import (
    ResponseInferenceModel,
    TrainingConfig,
    create_response_inference_model,
    create_transformer_response_model,
    create_lstm_response_model
)
from lsm.data.tokenization import StandardTokenizerWrapper


def create_sample_data():
    """Create sample embedding sequences and responses for demonstration."""
    print("Creating sample training data...")
    
    # Create sample embedding sequences (simulating CNN outputs)
    embedding_sequences = []
    responses = []
    
    # Sample 1: Question about weather
    seq1 = np.random.randn(24, 256) * 0.5 + np.array([0.2, -0.1] * 128).reshape(1, -1)
    embedding_sequences.append(seq1)
    responses.append("The weather is sunny and warm today.")
    
    # Sample 2: Greeting
    seq2 = np.random.randn(16, 256) * 0.3 + np.array([0.5, 0.3] * 128).reshape(1, -1)
    embedding_sequences.append(seq2)
    responses.append("Hello! How can I help you today?")
    
    # Sample 3: Technical question
    seq3 = np.random.randn(32, 256) * 0.7 + np.array([-0.2, 0.4] * 128).reshape(1, -1)
    embedding_sequences.append(seq3)
    responses.append("The neural network processes information through multiple layers.")
    
    # Sample 4: Simple response
    seq4 = np.random.randn(12, 256) * 0.4 + np.array([0.1, 0.1] * 128).reshape(1, -1)
    embedding_sequences.append(seq4)
    responses.append("Yes, that's correct.")
    
    # Sample 5: Complex explanation
    seq5 = np.random.randn(40, 256) * 0.6 + np.array([0.3, -0.3] * 128).reshape(1, -1)
    embedding_sequences.append(seq5)
    responses.append("Machine learning algorithms learn patterns from data to make predictions.")
    
    print(f"Created {len(embedding_sequences)} sample embedding sequences")
    return embedding_sequences, responses


def demonstrate_basic_usage():
    """Demonstrate basic ResponseInferenceModel usage."""
    print("\n" + "="*60)
    print("BASIC RESPONSE INFERENCE MODEL DEMO")
    print("="*60)
    
    # Create sample data
    embeddings, responses = create_sample_data()
    
    # Create a basic response inference model
    print("\nCreating ResponseInferenceModel...")
    model = create_response_inference_model(
        input_embedding_dim=256,
        max_sequence_length=64,
        vocab_size=1000,
        architecture="transformer"
    )
    
    print(f"Model architecture: {model.architecture.value}")
    print(f"Input embedding dimension: {model.input_embedding_dim}")
    print(f"Max sequence length: {model.max_sequence_length}")
    print(f"Vocabulary size: {model.vocab_size}")
    
    # Create the model
    print("\nBuilding neural network model...")
    keras_model = model.create_model()
    print(f"Model created with {keras_model.count_params():,} parameters")
    
    # Make predictions before training (random initialization)
    print("\nMaking predictions with untrained model...")
    test_embedding = embeddings[0]
    
    result = model.predict_response(test_embedding)
    print(f"Predicted response: {result.predicted_response}")
    print(f"Confidence score: {result.confidence_score:.3f}")
    print(f"Prediction time: {result.prediction_time:.3f}s")
    
    # Train the model
    print("\nTraining the model...")
    training_config = TrainingConfig(
        batch_size=2,
        epochs=5,  # Small number for demo
        learning_rate=0.001,
        early_stopping_patience=3
    )
    
    # Split data for training and validation
    train_embeddings = embeddings[:4]
    train_responses = responses[:4]
    val_embeddings = embeddings[4:]
    val_responses = responses[4:]
    
    training_result = model.train_on_response_pairs(
        train_embeddings,
        train_responses,
        training_config,
        validation_data=(val_embeddings, val_responses)
    )
    
    print(f"Training completed!")
    print(f"Final training loss: {training_result['metrics']['final_loss']:.4f}")
    if 'final_val_loss' in training_result['metrics']:
        print(f"Final validation loss: {training_result['metrics']['final_val_loss']:.4f}")
    
    # Make predictions after training
    print("\nMaking predictions with trained model...")
    for i, (embedding, true_response) in enumerate(zip(embeddings, responses)):
        result = model.predict_response(embedding)
        print(f"\nSample {i+1}:")
        print(f"  True response: {true_response}")
        print(f"  Predicted response: {result.predicted_response}")
        print(f"  Confidence: {result.confidence_score:.3f}")
    
    # Evaluate the model
    print("\nEvaluating model performance...")
    eval_metrics = model.evaluate_on_test_data(embeddings, responses)
    
    print("Evaluation metrics:")
    for metric, value in eval_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Show prediction statistics
    print("\nPrediction statistics:")
    stats = model.get_prediction_statistics()
    for stat, value in stats.items():
        if isinstance(value, float):
            print(f"  {stat}: {value:.4f}")
        else:
            print(f"  {stat}: {value}")


def demonstrate_different_architectures():
    """Demonstrate different model architectures."""
    print("\n" + "="*60)
    print("DIFFERENT ARCHITECTURES DEMO")
    print("="*60)
    
    # Create sample data
    embeddings, responses = create_sample_data()
    test_embedding = embeddings[0]
    
    architectures = [
        ("Transformer", "transformer"),
        ("LSTM", "lstm"),
        ("GRU", "gru"),
        ("1D CNN", "conv1d"),
        ("Hybrid", "hybrid")
    ]
    
    for arch_name, arch_type in architectures:
        print(f"\n--- {arch_name} Architecture ---")
        
        try:
            # Create model
            model = create_response_inference_model(
                input_embedding_dim=256,
                max_sequence_length=32,  # Smaller for demo
                vocab_size=500,
                architecture=arch_type
            )
            
            # Build model
            keras_model = model.create_model()
            print(f"Parameters: {keras_model.count_params():,}")
            
            # Make a prediction
            result = model.predict_response(test_embedding)
            print(f"Sample prediction: {result.predicted_response[:50]}...")
            print(f"Confidence: {result.confidence_score:.3f}")
            
        except Exception as e:
            print(f"Error with {arch_name}: {e}")


def demonstrate_specialized_models():
    """Demonstrate specialized model creation functions."""
    print("\n" + "="*60)
    print("SPECIALIZED MODELS DEMO")
    print("="*60)
    
    # Create sample data
    embeddings, responses = create_sample_data()
    test_embedding = embeddings[0]
    
    print("\n--- Transformer Model (Specialized) ---")
    transformer_model = create_transformer_response_model(
        input_embedding_dim=256,
        max_sequence_length=64,
        vocab_size=1000,
        num_heads=8,
        num_layers=4
    )
    
    keras_model = transformer_model.create_model()
    print(f"Transformer parameters: {keras_model.count_params():,}")
    
    result = transformer_model.predict_response(test_embedding)
    print(f"Transformer prediction: {result.predicted_response}")
    
    print("\n--- LSTM Model (Specialized) ---")
    lstm_model = create_lstm_response_model(
        input_embedding_dim=256,
        max_sequence_length=64,
        vocab_size=1000,
        lstm_units=128,
        num_layers=2
    )
    
    keras_model = lstm_model.create_model()
    print(f"LSTM parameters: {keras_model.count_params():,}")
    
    result = lstm_model.predict_response(test_embedding)
    print(f"LSTM prediction: {result.predicted_response}")


def demonstrate_training_configurations():
    """Demonstrate different training configurations."""
    print("\n" + "="*60)
    print("TRAINING CONFIGURATIONS DEMO")
    print("="*60)
    
    # Create sample data
    embeddings, responses = create_sample_data()
    
    # Create model
    model = create_response_inference_model(
        input_embedding_dim=256,
        max_sequence_length=32,
        vocab_size=500,
        architecture="lstm"  # Faster for demo
    )
    
    # Different training configurations
    configs = [
        ("Fast Training", TrainingConfig(
            batch_size=4,
            epochs=3,
            learning_rate=0.01,
            early_stopping_patience=2
        )),
        ("Careful Training", TrainingConfig(
            batch_size=2,
            epochs=5,
            learning_rate=0.001,
            early_stopping_patience=3
        )),
        ("Custom Loss", TrainingConfig(
            batch_size=2,
            epochs=3,
            learning_rate=0.001,
            loss_type="cosine_similarity",
            early_stopping_patience=2
        ))
    ]
    
    for config_name, config in configs:
        print(f"\n--- {config_name} ---")
        print(f"Batch size: {config.batch_size}")
        print(f"Epochs: {config.epochs}")
        print(f"Learning rate: {config.learning_rate}")
        print(f"Loss type: {config.loss_type}")
        
        try:
            # Train model
            result = model.train_on_response_pairs(
                embeddings[:4],
                responses[:4],
                config
            )
            
            print(f"Final loss: {result['metrics']['final_loss']:.4f}")
            
            # Test prediction
            pred_result = model.predict_response(embeddings[0])
            print(f"Sample prediction: {pred_result.predicted_response[:40]}...")
            
        except Exception as e:
            print(f"Training failed: {e}")


def demonstrate_model_persistence():
    """Demonstrate model saving and loading."""
    print("\n" + "="*60)
    print("MODEL PERSISTENCE DEMO")
    print("="*60)
    
    # Create and train a model
    embeddings, responses = create_sample_data()
    
    print("Creating and training model...")
    model = create_response_inference_model(
        input_embedding_dim=256,
        max_sequence_length=32,
        vocab_size=500,
        architecture="lstm"
    )
    
    # Quick training
    config = TrainingConfig(batch_size=2, epochs=2, early_stopping_patience=1)
    model.train_on_response_pairs(embeddings, responses, config)
    
    # Make a prediction
    original_result = model.predict_response(embeddings[0])
    print(f"Original prediction: {original_result.predicted_response}")
    
    # Save model
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "demo_model.keras")
        
        print(f"Saving model to {model_path}...")
        model.save_model(model_path)
        
        # Create new model and load
        print("Loading model into new instance...")
        new_model = ResponseInferenceModel(
            input_embedding_dim=256,
            max_sequence_length=32,
            vocab_size=500,
            architecture="lstm"
        )
        
        new_model.load_model(model_path)
        
        # Make prediction with loaded model
        loaded_result = new_model.predict_response(embeddings[0])
        print(f"Loaded prediction: {loaded_result.predicted_response}")
        
        # Compare predictions (should be very similar)
        print(f"Predictions match: {original_result.predicted_response == loaded_result.predicted_response}")


def main():
    """Run all demonstrations."""
    print("Response Inference Model Demonstration")
    print("=====================================")
    
    try:
        demonstrate_basic_usage()
        demonstrate_different_architectures()
        demonstrate_specialized_models()
        demonstrate_training_configurations()
        demonstrate_model_persistence()
        
        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()