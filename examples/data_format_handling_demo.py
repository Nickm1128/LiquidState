#!/usr/bin/env python3
"""
Demo of LSM Convenience API Data Format Handling

This example demonstrates the new data format handling and preprocessing
capabilities of the LSM convenience API, showing how to work with various
conversation formats automatically.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lsm.convenience import (
    DataFormatHandler, ConversationFormat, ConversationData,
    validate_conversation_data, detect_conversation_format,
    convert_conversation_format, get_conversation_statistics,
    preprocess_conversation_data
)


def demo_basic_usage():
    """Demonstrate basic data format handling."""
    print("=== Basic Data Format Handling ===")
    
    # Different input formats
    formats_demo = {
        "Simple List": ["Hello", "Hi there", "How are you?", "I'm doing well!"],
        
        "Chat Markup": "User: Hello\nAssistant: Hi there\nUser: How are you?\nAssistant: I'm doing well!",
        
        "OpenAI Format": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"}
        ],
        
        "Structured Dict": {
            "messages": ["Hello", "Hi there", "How are you?", "I'm doing well!"],
            "system": "Be helpful and friendly",
            "roles": ["User", "Assistant", "User", "Assistant"]
        }
    }
    
    handler = DataFormatHandler()
    
    for format_name, data in formats_demo.items():
        print(f"\n--- {format_name} ---")
        
        # Detect format
        detected = detect_conversation_format(data)
        print(f"Detected format: {detected}")
        
        # Process data
        processed = handler.process_conversation_data(data, return_format="structured")
        conv = processed[0]
        
        print(f"Messages: {len(conv.messages)}")
        print(f"System message: {conv.system_message or 'None'}")
        print(f"Roles: {conv.roles or 'None'}")
        print(f"First message: {conv.messages[0]}")


def demo_format_conversion():
    """Demonstrate format conversion capabilities."""
    print("\n\n=== Format Conversion ===")
    
    # Start with a structured conversation
    original_data = {
        "messages": ["Hello, how can I help you?", "I need help with Python", "Sure! What specifically?"],
        "system": "You are a helpful programming assistant",
        "roles": ["Assistant", "User", "Assistant"]
    }
    
    print("Original data (structured dict):")
    print(f"  Messages: {len(original_data['messages'])}")
    print(f"  System: {original_data['system']}")
    
    # Convert to different formats
    conversions = [
        ("simple_list", "Simple list of messages"),
        ("chat_markup", "Chat markup format"),
        ("openai_chat", "OpenAI chat format")
    ]
    
    for target_format, description in conversions:
        print(f"\n--- Converting to {description} ---")
        converted = convert_conversation_format(original_data, target_format=target_format)
        
        if target_format == "simple_list":
            print(f"Result: {len(converted)} messages")
            print(f"First: {converted[0]}")
        elif target_format == "chat_markup":
            print(f"Result:\n{converted}")
        elif target_format == "openai_chat":
            print(f"Result: {len(converted)} message objects")
            print(f"First: {converted[0]}")


def demo_preprocessing():
    """Demonstrate conversation preprocessing."""
    print("\n\n=== Conversation Preprocessing ===")
    
    # Create messy conversation data
    messy_data = [
        "Hello there!",
        "",  # Empty message
        "   ",  # Whitespace only
        "This is a very long message that goes on and on and on and repeats itself over and over again. " * 20,  # Very long
        "Hi",  # Very short
        "How are you doing today? I hope you're having a great time!",
        "I'm doing well, thank you for asking! How about you?",
        "   Great!   Thanks   for   asking!   ",  # Messy whitespace
    ]
    
    print("Original data:")
    print(f"  Total messages: {len(messy_data)}")
    print(f"  Empty messages: {sum(1 for msg in messy_data if not msg.strip())}")
    print(f"  Long messages (>100 chars): {sum(1 for msg in messy_data if len(msg) > 100)}")
    
    # Apply preprocessing
    cleaned = preprocess_conversation_data(
        messy_data,
        min_message_length=3,
        max_message_length=100,
        normalize_whitespace=True,
        return_format="simple_list"
    )
    
    print(f"\nAfter preprocessing:")
    print(f"  Total messages: {len(cleaned)}")
    print(f"  All messages 3-100 chars: {all(3 <= len(msg) <= 100 for msg in cleaned)}")
    print(f"  Sample cleaned message: '{cleaned[-1]}'")


def demo_statistics():
    """Demonstrate conversation statistics."""
    print("\n\n=== Conversation Statistics ===")
    
    # Create diverse conversation data
    conversations = [
        {
            "messages": ["Hello", "Hi", "How are you?", "Good!"],
            "system": "Be brief"
        },
        {
            "messages": ["Tell me about machine learning", "ML is a subset of AI...", "That's interesting!"],
            "system": "Be educational"
        },
        ["Quick chat", "Sure", "Thanks"],  # No system message
        "User: Long conversation here\nAssistant: I see\nUser: Yes\nAssistant: Understood\nUser: Great"
    ]
    
    # Get statistics
    stats = get_conversation_statistics(conversations)
    
    print("Dataset Statistics:")
    print(f"  Total conversations: {stats.get('total_conversations', 'N/A')}")
    print(f"  Total messages: {stats.get('total_messages', 'N/A')}")
    print(f"  Average conversation length: {stats.get('avg_conversation_length', 'N/A'):.1f}")
    print(f"  Average message length: {stats.get('avg_message_length', 'N/A'):.1f}")
    print(f"  Conversations with system messages: {stats.get('conversations_with_system_messages', 'N/A')}")
    print(f"  System message coverage: {stats.get('system_message_ratio', 0)*100:.1f}%")


def demo_advanced_features():
    """Demonstrate advanced features."""
    print("\n\n=== Advanced Features ===")
    
    # Create handler with custom preprocessing
    handler = DataFormatHandler(preprocessor_config={
        "min_message_length": 5,
        "max_message_length": 200,
        "min_conversation_length": 2,
        "max_conversation_length": 20,
        "normalize_whitespace": True
    })
    
    # Complex conversation data
    complex_data = [
        {
            "messages": ["Hi", "Hello there", "How's your day?", "Pretty good, thanks!"],
            "system": "Be conversational",
            "metadata": {"topic": "greeting", "mood": "friendly"},
            "conversation_id": "conv_001"
        },
        {
            "messages": ["What's 2+2?", "That's 4", "Correct!"],
            "system": "Be accurate",
            "metadata": {"topic": "math", "difficulty": "easy"},
            "conversation_id": "conv_002"
        }
    ]
    
    print("Processing complex structured data...")
    
    # Process with full structure preservation
    processed = handler.process_conversation_data(complex_data, return_format="structured")
    
    for i, conv in enumerate(processed):
        print(f"\nConversation {i+1}:")
        print(f"  ID: {conv.conversation_id}")
        print(f"  Messages: {len(conv.messages)}")
        print(f"  System: {conv.system_message}")
        print(f"  Metadata: {conv.metadata}")
        print(f"  First message: {conv.messages[0]}")
    
    # Validate the data
    validation = handler.validate_conversation_data(complex_data)
    print(f"\nValidation result: {'✓ Valid' if validation['is_valid'] else '✗ Invalid'}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")


def main():
    """Run all demos."""
    print("LSM Convenience API - Data Format Handling Demo")
    print("=" * 50)
    
    try:
        demo_basic_usage()
        demo_format_conversion()
        demo_preprocessing()
        demo_statistics()
        demo_advanced_features()
        
        print("\n" + "=" * 50)
        print("🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Automatic format detection")
        print("• Support for multiple conversation formats")
        print("• Format conversion between different standards")
        print("• Intelligent preprocessing and cleaning")
        print("• Comprehensive data validation")
        print("• Statistical analysis of conversation data")
        print("• Structured data with metadata support")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())