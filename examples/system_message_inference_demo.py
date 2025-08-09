#!/usr/bin/env python3
"""
System Message Inference Demo

This script demonstrates the system message support in the enhanced LSM inference system.
It shows how to use system messages to influence response generation using both 2D and 3D CNN
architectures with the SystemMessageProcessor and CNN3DProcessor integration.
"""

import os
import sys
import numpy as np
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.inference.inference import EnhancedLSMInference
from lsm.data.tokenization import StandardTokenizerWrapper, SinusoidalEmbedder
from lsm.core.system_message_processor import SystemMessageProcessor, SystemMessageContext
from lsm.core.cnn_3d_processor import CNN3DProcessor, SystemContext


def demo_system_message_processing():
    """Demonstrate system message processing capabilities."""
    print("🔧 System Message Processing Demo")
    print("=" * 50)
    
    # Initialize tokenizer and system processor
    tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2')
    system_processor = SystemMessageProcessor(tokenizer=tokenizer)
    
    # Test different types of system messages
    test_messages = [
        {
            "message": "You are a helpful assistant. Be concise and friendly.",
            "description": "Basic persona instruction"
        },
        {
            "message": "Act as a professional translator specializing in technical documents.",
            "description": "Role-based instruction"
        },
        {
            "message": "Your task is to summarize the following text in exactly 3 sentences.",
            "description": "Task-specific instruction"
        },
        {
            "message": "Do not use any technical jargon. Keep responses simple and accessible.",
            "description": "Constraint-based instruction"
        },
        {
            "message": "Context: This is a customer service conversation. The customer is frustrated.",
            "description": "Context-aware instruction"
        }
    ]
    
    for i, test_case in enumerate(test_messages, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Message: '{test_case['message']}'")
        
        try:
            start_time = time.time()
            context = system_processor.process_system_message(test_case['message'])
            processing_time = time.time() - start_time
            
            print(f"   ✅ Processed in {processing_time:.3f}s")
            print(f"   Format: {context.parsed_content.get('format', 'unknown')}")
            print(f"   Components: {list(context.parsed_content.get('components', {}).keys())}")
            print(f"   Embeddings shape: {context.embeddings.shape}")
            
        except Exception as e:
            print(f"   ❌ Processing failed: {e}")


def demo_system_aware_inference():
    """Demonstrate system-aware inference capabilities."""
    print("\n🤖 System-Aware Inference Demo")
    print("=" * 50)
    
    try:
        # Initialize enhanced inference (without actual model loading for demo)
        inference = EnhancedLSMInference(
            model_path="demo_model",
            use_response_level=True,
            lazy_load=True
        )
        
        # Set up minimal components for demo
        tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2')
        embedder = SinusoidalEmbedder(vocab_size=tokenizer.get_vocab_size(), embedding_dim=128)
        system_processor = SystemMessageProcessor(tokenizer=tokenizer)
        
        inference.standard_tokenizer = tokenizer
        inference.sinusoidal_embedder = embedder
        inference.system_message_processor = system_processor
        inference._enhanced_components_loaded = True
        inference._model_loaded = True
        
        # Test cases with different system messages
        test_cases = [
            {
                "input": "What is machine learning?",
                "system": "You are a teacher explaining to elementary school students.",
                "expected_style": "Simple, educational"
            },
            {
                "input": "What is machine learning?",
                "system": "You are a technical expert writing for other professionals.",
                "expected_style": "Technical, detailed"
            },
            {
                "input": "How do I cook pasta?",
                "system": "You are a professional chef. Be precise about techniques.",
                "expected_style": "Professional, precise"
            },
            {
                "input": "How do I cook pasta?",
                "system": "You are helping a complete beginner. Use simple language.",
                "expected_style": "Beginner-friendly, simple"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{i}. Input: '{test_case['input']}'")
            print(f"   System: '{test_case['system']}'")
            print(f"   Expected style: {test_case['expected_style']}")
            
            try:
                # Process system message
                start_time = time.time()
                system_context = inference._process_system_message(test_case['system'])
                processing_time = time.time() - start_time
                
                print(f"   ✅ System message processed in {processing_time:.3f}s")
                print(f"   System format: {system_context.parsed_content.get('format', 'unknown')}")
                
                # Note: Actual response generation would require a trained model
                print(f"   📝 Response generation would use system context to influence output")
                
            except Exception as e:
                print(f"   ❌ Processing failed: {e}")
                
    except Exception as e:
        print(f"❌ Demo setup failed: {e}")


def demo_system_message_formats():
    """Demonstrate different system message format recognition."""
    print("\n📋 System Message Format Recognition Demo")
    print("=" * 50)
    
    try:
        tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2')
        system_processor = SystemMessageProcessor(tokenizer=tokenizer)
        
        format_examples = {
            "persona": [
                "You are a helpful assistant.",
                "As a professional writer, you should...",
                "Playing the role of a detective, analyze..."
            ],
            "instruction": [
                "Your task is to translate the following text.",
                "Please summarize the main points.",
                "Act as a code reviewer and check..."
            ],
            "constraint": [
                "Do not use any profanity.",
                "Never reveal personal information.",
                "Always provide sources for your claims."
            ],
            "context": [
                "Context: This is a medical consultation.",
                "Background: The user is a beginner programmer.",
                "Given that this is a formal business setting..."
            ]
        }
        
        for format_type, examples in format_examples.items():
            print(f"\n{format_type.upper()} Format Examples:")
            for example in examples:
                try:
                    context = system_processor.process_system_message(example)
                    detected_format = context.parsed_content.get('format', 'unknown')
                    match_indicator = "✅" if detected_format == format_type else "❓"
                    print(f"  {match_indicator} '{example}' -> {detected_format}")
                except Exception as e:
                    print(f"  ❌ '{example}' -> ERROR: {e}")
                    
    except Exception as e:
        print(f"❌ Format demo failed: {e}")


def demo_system_context_creation():
    """Demonstrate system context creation for 3D CNN processing."""
    print("\n🧠 System Context Creation Demo")
    print("=" * 50)
    
    try:
        tokenizer = StandardTokenizerWrapper(tokenizer_name='gpt2')
        system_processor = SystemMessageProcessor(tokenizer=tokenizer)
        
        system_message = "You are a creative writer. Use vivid imagery and emotional language."
        
        print(f"System Message: '{system_message}'")
        
        # Process system message
        context = system_processor.process_system_message(system_message)
        
        # Create SystemContext for 3D CNN
        system_ctx = SystemContext(
            message=context.original_message,
            embeddings=context.embeddings,
            influence_strength=1.0,
            processing_mode="3d_cnn"
        )
        
        print(f"✅ System context created successfully")
        print(f"   Original message length: {len(system_ctx.message)} characters")
        print(f"   Embeddings shape: {system_ctx.embeddings.shape}")
        print(f"   Influence strength: {system_ctx.influence_strength}")
        print(f"   Processing mode: {system_ctx.processing_mode}")
        print(f"   Parsed format: {context.parsed_content.get('format', 'unknown')}")
        
        # Show embedding statistics
        print(f"   Embedding statistics:")
        print(f"     Mean: {np.mean(system_ctx.embeddings):.4f}")
        print(f"     Std: {np.std(system_ctx.embeddings):.4f}")
        print(f"     Min: {np.min(system_ctx.embeddings):.4f}")
        print(f"     Max: {np.max(system_ctx.embeddings):.4f}")
        
    except Exception as e:
        print(f"❌ System context creation failed: {e}")


def main():
    """Run all system message demos."""
    print("🚀 System Message Support Demo")
    print("=" * 60)
    print("This demo showcases the system message support in LSM inference")
    print("including processing, format recognition, and context creation.")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_system_message_processing()
        demo_system_message_formats()
        demo_system_context_creation()
        demo_system_aware_inference()
        
        print("\n🎉 All demos completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ System message processing and validation")
        print("✅ Format recognition (persona, instruction, constraint, context)")
        print("✅ System context creation for 3D CNN processing")
        print("✅ Integration with enhanced inference system")
        print("✅ Embedding generation and analysis")
        
        print("\nNext Steps:")
        print("• Train a model to test actual response generation")
        print("• Experiment with different system message formats")
        print("• Test 3D CNN processing with real reservoir outputs")
        print("• Evaluate system message influence on response quality")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()