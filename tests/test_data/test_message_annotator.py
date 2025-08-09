"""
Unit tests for MessageAnnotator class.

Tests the message annotation system including:
- Message annotation with different types
- Conversation flow markers
- Annotation parsing and extraction
- Metadata handling
- Error handling and validation
"""

import pytest
from unittest.mock import patch, MagicMock

from src.lsm.data.message_annotator import (
    MessageAnnotator,
    AnnotatedMessage,
    ConversationFlow,
    MessageType,
    AnnotationError
)


class TestMessageAnnotator:
    """Test suite for MessageAnnotator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.annotator = MessageAnnotator()
    
    def test_init_default(self):
        """Test MessageAnnotator initialization with default markers."""
        annotator = MessageAnnotator()
        
        # Check that default markers are loaded
        assert '|start|' in annotator.markers.values()
        assert '|end|' in annotator.markers.values()
        assert '|system|' in annotator.markers.values()
        assert '|user|' in annotator.markers.values()
        assert '|assistant|' in annotator.markers.values()
    
    def test_init_custom_markers(self):
        """Test MessageAnnotator initialization with custom markers."""
        custom_markers = {'custom': '|custom|', 'special': '|special|'}
        annotator = MessageAnnotator(custom_markers=custom_markers)
        
        # Check that custom markers are added
        assert annotator.markers['custom'] == '|custom|'
        assert annotator.markers['special'] == '|special|'
        
        # Check that default markers are still present
        assert annotator.markers['start'] == '|start|'
    
    def test_annotate_message_user(self):
        """Test annotating a user message."""
        message = "Hello, how are you?"
        result = self.annotator.annotate_message(message, "user")
        
        assert result == "|user|Hello, how are you?"
    
    def test_annotate_message_assistant(self):
        """Test annotating an assistant message."""
        message = "I'm doing well, thank you!"
        result = self.annotator.annotate_message(message, "assistant")
        
        assert result == "|assistant|I'm doing well, thank you!"
    
    def test_annotate_message_system(self):
        """Test annotating a system message."""
        message = "You are a helpful assistant."
        result = self.annotator.annotate_message(message, "system")
        
        assert result == "|system|You are a helpful assistant."
    
    def test_annotate_message_with_metadata(self):
        """Test annotating a message with metadata."""
        message = "Test message"
        metadata = {"timestamp": "2023-01-01", "user_id": "123"}
        result = self.annotator.annotate_message(message, "user", metadata)
        
        assert "|timestamp:2023-01-01|" in result
        assert "|user_id:123|" in result
        assert "|user|Test message" in result
    
    def test_annotate_message_invalid_type(self):
        """Test annotating with invalid message type."""
        with pytest.raises(AnnotationError):
            self.annotator.annotate_message("Test", "invalid_type")
    
    def test_annotate_message_strips_whitespace(self):
        """Test that message annotation strips whitespace."""
        message = "  Test message  "
        result = self.annotator.annotate_message(message, "user")
        
        assert result == "|user|Test message"
    
    def test_add_conversation_markers_basic(self):
        """Test adding conversation markers to a basic conversation."""
        conversation = ["Hello", "Hi there", "How are you?"]
        result = self.annotator.add_conversation_markers(conversation)
        
        assert len(result) == 3
        assert "|start|" in result[0]
        assert "|turn:0|" in result[0]
        assert "|turn:1|" in result[1]
        assert "|turn:2|" in result[2]
        assert "|end|" in result[2]
    
    def test_add_conversation_markers_with_id(self):
        """Test adding conversation markers with conversation ID."""
        conversation = ["Hello", "Hi"]
        conversation_id = "conv_123"
        result = self.annotator.add_conversation_markers(conversation, conversation_id)
        
        assert f"|conversation:{conversation_id}|" in result[0]
        assert "|start|" in result[0]
        assert "|end|" in result[1]
    
    def test_add_conversation_markers_empty(self):
        """Test adding conversation markers to empty conversation."""
        result = self.annotator.add_conversation_markers([])
        assert result == []
    
    def test_add_conversation_markers_single_message(self):
        """Test adding conversation markers to single message."""
        conversation = ["Only message"]
        result = self.annotator.add_conversation_markers(conversation)
        
        assert len(result) == 1
        assert "|start|" in result[0]
        assert "|end|" in result[0]
        assert "|turn:0|" in result[0]
    
    def test_parse_annotated_message_user(self):
        """Test parsing a user message."""
        annotated = "|user|Hello there!"
        result = self.annotator.parse_annotated_message(annotated)
        
        assert result['content'] == "Hello there!"
        assert result['type'] == "user"
        assert 'user' not in result['annotations']  # Type is extracted separately
    
    def test_parse_annotated_message_with_turn(self):
        """Test parsing message with turn annotation."""
        annotated = "|turn:5||user|Hello there!"
        result = self.annotator.parse_annotated_message(annotated)
        
        assert result['content'] == "Hello there!"
        assert result['type'] == "user"
        assert result['annotations']['turn'] == "5"
    
    def test_parse_annotated_message_with_conversation_id(self):
        """Test parsing message with conversation ID."""
        annotated = "|conversation:conv_123||user|Hello!"
        result = self.annotator.parse_annotated_message(annotated)
        
        assert result['content'] == "Hello!"
        assert result['type'] == "user"
        assert result['annotations']['conversation_id'] == "conv_123"
    
    def test_parse_annotated_message_complex(self):
        """Test parsing complex annotated message."""
        annotated = "|start||turn:0||user|Hello there!"
        result = self.annotator.parse_annotated_message(annotated)
        
        assert result['content'] == "Hello there!"
        assert result['type'] == "user"
        assert result['annotations']['turn'] == "0"
        assert result['annotations']['start'] is True
    
    def test_parse_annotated_message_no_annotations(self):
        """Test parsing message without annotations."""
        result = self.annotator.parse_annotated_message("Plain message")
        
        assert result['content'] == "Plain message"
        assert result['type'] == "unknown"
        assert len(result['annotations']) == 0
    
    def test_create_annotated_message_object(self):
        """Test creating AnnotatedMessage object."""
        content = "Test message"
        message_type = "user"
        conversation_id = "conv_123"
        turn_index = 5
        metadata = {"key": "value"}
        
        result = self.annotator.create_annotated_message_object(
            content, message_type, conversation_id, turn_index, metadata
        )
        
        assert isinstance(result, AnnotatedMessage)
        assert result.content == "Test message"
        assert result.message_type == MessageType.USER
        assert result.conversation_id == "conv_123"
        assert result.turn_index == 5
        assert result.metadata == {"key": "value"}
        assert result.annotations['turn'] == "5"
        assert result.annotations['conversation_id'] == "conv_123"
    
    def test_create_annotated_message_object_invalid_type(self):
        """Test creating AnnotatedMessage with invalid type."""
        with pytest.raises(AnnotationError):
            self.annotator.create_annotated_message_object(
                "Test", "invalid_type"
            )
    
    def test_create_annotated_message_object_minimal(self):
        """Test creating AnnotatedMessage with minimal parameters."""
        result = self.annotator.create_annotated_message_object("Test", "user")
        
        assert result.content == "Test"
        assert result.message_type == MessageType.USER
        assert result.conversation_id is None
        assert result.turn_index is None
        assert len(result.metadata) == 0
    
    def test_create_conversation_flow(self):
        """Test creating ConversationFlow object."""
        messages = [
            self.annotator.create_annotated_message_object("Hello", "user", turn_index=0),
            self.annotator.create_annotated_message_object("Hi", "assistant", turn_index=1)
        ]
        conversation_id = "conv_123"
        metadata = {"topic": "greeting"}
        
        result = self.annotator.create_conversation_flow(
            messages, conversation_id, metadata
        )
        
        assert isinstance(result, ConversationFlow)
        assert result.conversation_id == "conv_123"
        assert len(result.messages) == 2
        assert result.metadata == {"topic": "greeting"}
        assert result.flow_markers['start'] == 0
        assert result.flow_markers['end'] == 1
    
    def test_create_conversation_flow_with_system_messages(self):
        """Test creating ConversationFlow with system messages."""
        messages = [
            self.annotator.create_annotated_message_object("System prompt", "system"),
            self.annotator.create_annotated_message_object("Hello", "user"),
            self.annotator.create_annotated_message_object("Hi", "assistant")
        ]
        
        result = self.annotator.create_conversation_flow(messages, "conv_123")
        
        assert 'system_messages' in result.flow_markers
        assert result.flow_markers['system_messages'] == [0]
    
    def test_extract_system_messages(self):
        """Test extracting system messages from conversation flow."""
        messages = [
            self.annotator.create_annotated_message_object("System prompt", "system"),
            self.annotator.create_annotated_message_object("Hello", "user"),
            self.annotator.create_annotated_message_object("Another system msg", "system"),
            self.annotator.create_annotated_message_object("Hi", "assistant")
        ]
        
        flow = self.annotator.create_conversation_flow(messages, "conv_123")
        system_messages = self.annotator.extract_system_messages(flow)
        
        assert len(system_messages) == 2
        assert system_messages[0].content == "System prompt"
        assert system_messages[1].content == "Another system msg"
        assert all(msg.message_type == MessageType.SYSTEM for msg in system_messages)
    
    def test_extract_system_messages_none(self):
        """Test extracting system messages when none exist."""
        messages = [
            self.annotator.create_annotated_message_object("Hello", "user"),
            self.annotator.create_annotated_message_object("Hi", "assistant")
        ]
        
        flow = self.annotator.create_conversation_flow(messages, "conv_123")
        system_messages = self.annotator.extract_system_messages(flow)
        
        assert len(system_messages) == 0
    
    def test_validate_conversation_flow_valid(self):
        """Test validating a valid conversation flow."""
        messages = [
            self.annotator.create_annotated_message_object("Hello", "user", turn_index=0),
            self.annotator.create_annotated_message_object("Hi", "assistant", turn_index=1)
        ]
        
        flow = self.annotator.create_conversation_flow(messages, "conv_123")
        result = self.annotator.validate_conversation_flow(flow)
        
        assert result is True
    
    def test_validate_conversation_flow_no_id(self):
        """Test validating conversation flow without ID."""
        messages = [self.annotator.create_annotated_message_object("Hello", "user")]
        flow = ConversationFlow("", messages, {}, {})
        
        result = self.annotator.validate_conversation_flow(flow)
        assert result is False
    
    def test_validate_conversation_flow_no_messages(self):
        """Test validating conversation flow without messages."""
        flow = ConversationFlow("conv_123", [], {}, {})
        
        result = self.annotator.validate_conversation_flow(flow)
        assert result is False
    
    def test_validate_conversation_flow_invalid_markers(self):
        """Test validating conversation flow with invalid markers."""
        messages = [self.annotator.create_annotated_message_object("Hello", "user")]
        flow = ConversationFlow("conv_123", messages, {}, {'start': 10})  # Invalid index
        
        result = self.annotator.validate_conversation_flow(flow)
        assert result is False
    
    def test_validate_conversation_flow_turn_mismatch(self):
        """Test validating conversation flow with turn index mismatch."""
        messages = [
            self.annotator.create_annotated_message_object("Hello", "user", turn_index=5)  # Wrong index
        ]
        
        flow = self.annotator.create_conversation_flow(messages, "conv_123")
        result = self.annotator.validate_conversation_flow(flow)
        
        assert result is False
    
    def test_get_supported_markers(self):
        """Test getting supported markers."""
        markers = self.annotator.get_supported_markers()
        
        assert isinstance(markers, dict)
        assert 'start' in markers
        assert 'end' in markers
        assert 'system' in markers
        assert markers['start'] == '|start|'
    
    def test_add_custom_marker(self):
        """Test adding custom marker."""
        self.annotator.add_custom_marker('custom', '|custom|')
        
        markers = self.annotator.get_supported_markers()
        assert 'custom' in markers
        assert markers['custom'] == '|custom|'
    
    def test_metadata_annotations_creation(self):
        """Test internal metadata annotation creation."""
        metadata = {"key1": "value1", "key2": 123, "key3": 45.6}
        result = self.annotator._create_metadata_annotations(metadata)
        
        assert "|key1:value1|" in result
        assert "|key2:123|" in result
        assert "|key3:45.6|" in result
    
    def test_metadata_extraction_from_annotations(self):
        """Test internal metadata extraction from annotations."""
        annotations = {
            "key1:value1": True,
            "key2:123": True,
            "turn": "5",
            "conversation_id": "conv_123",
            "regular_annotation": True
        }
        
        result = self.annotator._extract_metadata_from_annotations(annotations)
        
        assert result["key1"] == "value1"
        assert result["key2"] == "123"
        assert result["regular_annotation"] is True
        assert "turn" not in result
        assert "conversation_id" not in result


class TestMessageAnnotatorErrorHandling:
    """Test error handling in MessageAnnotator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.annotator = MessageAnnotator()
    
    def test_annotate_message_exception_handling(self):
        """Test exception handling in annotate_message."""
        # This should raise AnnotationError for invalid type
        with pytest.raises(AnnotationError) as exc_info:
            self.annotator.annotate_message("test", "invalid")
        
        assert "Invalid message type" in str(exc_info.value)
    
    def test_add_conversation_markers_exception_handling(self):
        """Test exception handling in add_conversation_markers."""
        # Mock an exception in the processing by making markers access fail
        with patch.object(self.annotator, 'markers', new_callable=lambda: property(lambda self: (_ for _ in ()).throw(Exception("Test error")))):
            with pytest.raises(AnnotationError) as exc_info:
                self.annotator.add_conversation_markers(["test"])
            
            assert "Failed to add conversation markers" in str(exc_info.value)
    
    def test_parse_annotated_message_exception_handling(self):
        """Test exception handling in parse_annotated_message."""
        # Create a scenario that will cause an exception by patching a method used internally
        with patch.object(self.annotator, '_extract_metadata_from_annotations', side_effect=Exception("Processing error")):
            with pytest.raises(AnnotationError) as exc_info:
                self.annotator.parse_annotated_message("|user|test message")
            
            assert "Failed to parse annotated message" in str(exc_info.value)
    
    def test_create_annotated_message_object_exception_handling(self):
        """Test exception handling in create_annotated_message_object."""
        # Test with invalid message type that causes ValueError
        with pytest.raises(AnnotationError) as exc_info:
            self.annotator.create_annotated_message_object("test", "completely_invalid")
        
        assert "Invalid message type" in str(exc_info.value)
    
    def test_create_conversation_flow_exception_handling(self):
        """Test exception handling in create_conversation_flow."""
        # Create invalid messages that might cause issues
        invalid_messages = [None]  # This should cause an exception
        
        with pytest.raises(AnnotationError) as exc_info:
            self.annotator.create_conversation_flow(invalid_messages, "conv_123")
        
        assert "Failed to create ConversationFlow" in str(exc_info.value)
    
    def test_validate_conversation_flow_exception_handling(self):
        """Test exception handling in validate_conversation_flow."""
        # Create a mock that raises an exception when conversation_id is accessed
        mock_flow = MagicMock()
        # Configure the mock to raise an exception when conversation_id is accessed
        type(mock_flow).conversation_id = property(lambda self: (_ for _ in ()).throw(Exception("Test error")))
        
        with pytest.raises(AnnotationError) as exc_info:
            self.annotator.validate_conversation_flow(mock_flow)
        
        assert "Failed to validate ConversationFlow" in str(exc_info.value)


class TestMessageAnnotatorIntegration:
    """Integration tests for MessageAnnotator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.annotator = MessageAnnotator()
    
    def test_full_conversation_processing_workflow(self):
        """Test complete workflow from raw messages to validated flow."""
        # Raw conversation data
        raw_messages = [
            ("You are a helpful assistant.", "system"),
            ("Hello, how can I help you today?", "assistant"),
            ("I need help with Python programming.", "user"),
            ("I'd be happy to help with Python!", "assistant")
        ]
        
        # Step 1: Create annotated messages
        annotated_messages = []
        for i, (content, msg_type) in enumerate(raw_messages):
            annotated_msg = self.annotator.create_annotated_message_object(
                content, msg_type, "conv_integration_test", i
            )
            annotated_messages.append(annotated_msg)
        
        # Step 2: Create conversation flow
        conversation_flow = self.annotator.create_conversation_flow(
            annotated_messages, 
            "conv_integration_test",
            {"topic": "programming_help", "language": "python"}
        )
        
        # Step 3: Validate the flow
        is_valid = self.annotator.validate_conversation_flow(conversation_flow)
        assert is_valid
        
        # Step 4: Extract system messages
        system_messages = self.annotator.extract_system_messages(conversation_flow)
        assert len(system_messages) == 1
        assert system_messages[0].content == "You are a helpful assistant."
        
        # Step 5: Verify flow structure
        assert conversation_flow.conversation_id == "conv_integration_test"
        assert len(conversation_flow.messages) == 4
        assert conversation_flow.metadata["topic"] == "programming_help"
        assert 'system_messages' in conversation_flow.flow_markers
        assert conversation_flow.flow_markers['system_messages'] == [0]
    
    def test_annotation_and_parsing_roundtrip(self):
        """Test that annotation and parsing are inverse operations."""
        original_message = "This is a test message with some content."
        message_type = "user"
        metadata = {"timestamp": "2023-01-01", "priority": "high"}
        
        # Annotate the message
        annotated = self.annotator.annotate_message(original_message, message_type, metadata)
        
        # Parse the annotated message
        parsed = self.annotator.parse_annotated_message(annotated)
        
        # Verify roundtrip integrity
        assert parsed['content'] == original_message
        assert parsed['type'] == message_type
        assert 'timestamp:2023-01-01' in str(parsed['annotations']) or 'timestamp' in parsed['metadata']
        assert 'priority:high' in str(parsed['annotations']) or 'priority' in parsed['metadata']
    
    def test_conversation_markers_and_parsing_integration(self):
        """Test integration between conversation markers and parsing."""
        conversation = [
            "Hello there!",
            "Hi, how are you?",
            "I'm doing well, thanks!"
        ]
        
        # Add conversation markers
        marked_conversation = self.annotator.add_conversation_markers(
            conversation, "test_conv_456"
        )
        
        # Parse each marked message
        parsed_messages = []
        for marked_msg in marked_conversation:
            parsed = self.annotator.parse_annotated_message(marked_msg)
            parsed_messages.append(parsed)
        
        # Verify structure
        assert len(parsed_messages) == 3
        
        # First message should have start and conversation markers
        first_parsed = parsed_messages[0]
        assert 'start' in first_parsed['annotations']
        assert first_parsed['annotations']['conversation_id'] == "test_conv_456"
        assert first_parsed['annotations']['turn'] == "0"
        
        # Last message should have end marker
        last_parsed = parsed_messages[-1]
        assert 'end' in last_parsed['annotations']
        assert last_parsed['annotations']['turn'] == "2"
        
        # Middle message should have turn marker
        middle_parsed = parsed_messages[1]
        assert middle_parsed['annotations']['turn'] == "1"