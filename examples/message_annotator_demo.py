#!/usr/bin/env python3
"""
MessageAnnotator Demo Script

This script demonstrates the capabilities of the MessageAnnotator class
for processing conversational data with annotations, flow markers, and metadata.

Usage:
    python examples/message_annotator_demo.py
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lsm.data.message_annotator import (
    MessageAnnotator,
    MessageType,
    AnnotationError
)


def demonstrate_basic_annotation():
    """Demonstrate basic message annotation functionality."""
    print("=" * 60)
    print("BASIC MESSAGE ANNOTATION DEMO")
    print("=" * 60)
    
    annotator = MessageAnnotator()
    
    # Example messages of different types
    messages = [
        ("You are a helpful AI assistant.", "system"),
        ("Hello! How can I help you today?", "assistant"),
        ("I need help with Python programming.", "user"),
        ("I'd be happy to help you with Python!", "assistant")
    ]
    
    print("Original messages:")
    for i, (content, msg_type) in enumerate(messages):
        print(f"  {i+1}. [{msg_type.upper()}] {content}")
    
    print("\nAnnotated messages:")
    annotated_messages = []
    for i, (content, msg_type) in enumerate(messages):
        annotated = annotator.annotate_message(content, msg_type)
        annotated_messages.append(annotated)
        print(f"  {i+1}. {annotated}")
    
    return annotated_messages


def demonstrate_conversation_flow():
    """Demonstrate conversation flow creation and management."""
    print("\n" + "=" * 60)
    print("CONVERSATION FLOW DEMO")
    print("=" * 60)
    
    annotator = MessageAnnotator()
    
    # Create a sample conversation
    conversation_messages = [
        "Hello there!",
        "Hi! How are you doing today?",
        "I'm doing great, thanks for asking!",
        "That's wonderful to hear!"
    ]
    
    print("Original conversation:")
    for i, msg in enumerate(conversation_messages):
        print(f"  Turn {i}: {msg}")
    
    # Add conversation markers
    marked_conversation = annotator.add_conversation_markers(
        conversation_messages, 
        conversation_id="demo_conversation_001"
    )
    
    print("\nConversation with flow markers:")
    for i, marked_msg in enumerate(marked_conversation):
        print(f"  Turn {i}: {marked_msg}")
    
    return marked_conversation


def demonstrate_annotated_message_objects():
    """Demonstrate creating and working with AnnotatedMessage objects."""
    print("\n" + "=" * 60)
    print("ANNOTATED MESSAGE OBJECTS DEMO")
    print("=" * 60)
    
    annotator = MessageAnnotator()
    
    # Create annotated message objects
    messages_data = [
        {
            "content": "You are a helpful assistant specialized in Python programming.",
            "type": "system",
            "metadata": {"role": "system_prompt", "domain": "programming"}
        },
        {
            "content": "Hello! I'm here to help with your Python questions.",
            "type": "assistant",
            "metadata": {"greeting": True, "capabilities": ["python", "debugging"]}
        },
        {
            "content": "Can you help me understand list comprehensions?",
            "type": "user",
            "metadata": {"topic": "list_comprehensions", "difficulty": "beginner"}
        }
    ]
    
    annotated_messages = []
    
    print("Creating AnnotatedMessage objects:")
    for i, msg_data in enumerate(messages_data):
        annotated_msg = annotator.create_annotated_message_object(
            content=msg_data["content"],
            message_type=msg_data["type"],
            conversation_id="demo_conv_002",
            turn_index=i,
            metadata=msg_data["metadata"]
        )
        annotated_messages.append(annotated_msg)
        
        print(f"\nMessage {i+1}:")
        print(f"  Type: {annotated_msg.message_type.value}")
        print(f"  Content: {annotated_msg.content}")
        print(f"  Conversation ID: {annotated_msg.conversation_id}")
        print(f"  Turn Index: {annotated_msg.turn_index}")
        print(f"  Annotations: {annotated_msg.annotations}")
        print(f"  Metadata: {annotated_msg.metadata}")
    
    return annotated_messages


def demonstrate_conversation_flow_creation(annotated_messages):
    """Demonstrate creating and validating ConversationFlow objects."""
    print("\n" + "=" * 60)
    print("CONVERSATION FLOW CREATION DEMO")
    print("=" * 60)
    
    annotator = MessageAnnotator()
    
    # Create conversation flow
    conversation_metadata = {
        "topic": "python_programming",
        "difficulty": "beginner",
        "estimated_duration": "10_minutes"
    }
    
    conversation_flow = annotator.create_conversation_flow(
        messages=annotated_messages,
        conversation_id="demo_conv_002",
        metadata=conversation_metadata
    )
    
    print("ConversationFlow created:")
    print(f"  Conversation ID: {conversation_flow.conversation_id}")
    print(f"  Number of messages: {len(conversation_flow.messages)}")
    print(f"  Metadata: {conversation_flow.metadata}")
    print(f"  Flow markers: {conversation_flow.flow_markers}")
    
    # Extract system messages
    system_messages = annotator.extract_system_messages(conversation_flow)
    print(f"\nSystem messages found: {len(system_messages)}")
    for i, sys_msg in enumerate(system_messages):
        print(f"  System message {i+1}: {sys_msg.content[:50]}...")
    
    # Validate the conversation flow
    is_valid = annotator.validate_conversation_flow(conversation_flow)
    print(f"\nConversation flow validation: {'PASSED' if is_valid else 'FAILED'}")
    
    return conversation_flow


def demonstrate_annotation_parsing():
    """Demonstrate parsing annotated messages."""
    print("\n" + "=" * 60)
    print("ANNOTATION PARSING DEMO")
    print("=" * 60)
    
    annotator = MessageAnnotator()
    
    # Sample annotated messages
    annotated_samples = [
        "|user|Hello, can you help me?",
        "|system|You are a helpful assistant.",
        "|turn:2||assistant|Of course! What do you need help with?",
        "|conversation:conv_123||start||turn:0||user|Starting a new conversation",
        "|turn:5||user|This is complex|end|"
    ]
    
    print("Parsing annotated messages:")
    for i, annotated in enumerate(annotated_samples):
        print(f"\nSample {i+1}: {annotated}")
        
        try:
            parsed = annotator.parse_annotated_message(annotated)
            print(f"  Content: '{parsed['content']}'")
            print(f"  Type: {parsed['type']}")
            print(f"  Annotations: {parsed['annotations']}")
            print(f"  Metadata: {parsed['metadata']}")
        except AnnotationError as e:
            print(f"  ERROR: {e}")


def demonstrate_custom_markers():
    """Demonstrate using custom annotation markers."""
    print("\n" + "=" * 60)
    print("CUSTOM MARKERS DEMO")
    print("=" * 60)
    
    # Create annotator with custom markers
    custom_markers = {
        'priority': '|priority|',
        'urgent': '|urgent|',
        'context': '|context|'
    }
    
    annotator = MessageAnnotator(custom_markers=custom_markers)
    
    print("Available markers:")
    markers = annotator.get_supported_markers()
    for name, marker in markers.items():
        print(f"  {name}: {marker}")
    
    # Add another custom marker
    annotator.add_custom_marker('special', '|special|')
    print(f"\nAdded custom marker: special -> |special|")
    
    # Use custom markers in annotation
    message = "This is an urgent message that needs immediate attention."
    metadata = {"priority": "high", "context": "customer_support"}
    
    annotated = annotator.annotate_message(message, "user", metadata)
    print(f"\nAnnotated with custom metadata:")
    print(f"  {annotated}")


def demonstrate_error_handling():
    """Demonstrate error handling capabilities."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING DEMO")
    print("=" * 60)
    
    annotator = MessageAnnotator()
    
    # Test various error conditions
    error_tests = [
        {
            "name": "Invalid message type",
            "action": lambda: annotator.annotate_message("test", "invalid_type"),
            "expected_error": AnnotationError
        },
        {
            "name": "Invalid message type in object creation",
            "action": lambda: annotator.create_annotated_message_object("test", "bad_type"),
            "expected_error": AnnotationError
        }
    ]
    
    for test in error_tests:
        print(f"\nTesting: {test['name']}")
        try:
            test["action"]()
            print("  ERROR: Expected exception was not raised!")
        except test["expected_error"] as e:
            print(f"  SUCCESS: Caught expected error - {e}")
        except Exception as e:
            print(f"  UNEXPECTED: Caught unexpected error - {e}")


def main():
    """Run all demonstration functions."""
    print("MessageAnnotator Demonstration")
    print("This demo shows the capabilities of the message annotation system.")
    
    try:
        # Run demonstrations
        annotated_messages = demonstrate_basic_annotation()
        marked_conversation = demonstrate_conversation_flow()
        annotated_objects = demonstrate_annotated_message_objects()
        conversation_flow = demonstrate_conversation_flow_creation(annotated_objects)
        demonstrate_annotation_parsing()
        demonstrate_custom_markers()
        demonstrate_error_handling()
        
        print("\n" + "=" * 60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("The MessageAnnotator provides comprehensive functionality for:")
        print("  • Annotating messages with type markers")
        print("  • Adding conversation flow markers")
        print("  • Creating structured message objects")
        print("  • Managing conversation flows")
        print("  • Parsing annotated messages")
        print("  • Custom marker support")
        print("  • Robust error handling")
        
    except Exception as e:
        print(f"\nDEMO FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())