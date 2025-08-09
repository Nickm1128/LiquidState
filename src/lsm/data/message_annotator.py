"""
Message Annotation System for LSM Training Pipeline Enhancement

This module implements the MessageAnnotator class that provides annotation capabilities
for messages in conversational datasets. It supports adding conversation flow markers,
system message annotations, and metadata handling as specified in requirements 6.1 and 6.2.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..utils.lsm_exceptions import LSMError
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class MessageType(Enum):
    """Enumeration of supported message types for annotation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    START = "start"
    END = "end"


class AnnotationError(LSMError):
    """Exception raised for errors in message annotation processing."""
    pass


@dataclass
class AnnotatedMessage:
    """Data structure representing an annotated message."""
    content: str
    message_type: MessageType
    annotations: Dict[str, str]
    metadata: Dict[str, Any]
    conversation_id: Optional[str] = None
    turn_index: Optional[int] = None


@dataclass
class ConversationFlow:
    """Data structure representing conversation flow with annotations."""
    conversation_id: str
    messages: List[AnnotatedMessage]
    metadata: Dict[str, Any]
    flow_markers: Dict[str, int]  # Maps marker types to positions


class MessageAnnotator:
    """
    Message annotation system for processing conversational data.
    
    Provides functionality to:
    - Add conversation flow markers (|start|, |end|, |system|, etc.)
    - Parse and extract annotations from messages
    - Handle conversation metadata and flow tracking
    - Support system message integration
    """
    
    # Standard annotation markers
    ANNOTATION_MARKERS = {
        'start': '|start|',
        'end': '|end|',
        'system': '|system|',
        'user': '|user|',
        'assistant': '|assistant|',
        'turn': '|turn|',
        'conversation': '|conversation|'
    }
    
    # Regex patterns for parsing annotations
    ANNOTATION_PATTERN = re.compile(r'\|([^|]+)\|')
    MARKER_CONTENT_PATTERN = re.compile(r'\|([^|]+)\|(.*?)(?=\|[^|]+\||$)', re.DOTALL)
    
    def __init__(self, custom_markers: Optional[Dict[str, str]] = None):
        """
        Initialize the MessageAnnotator.
        
        Args:
            custom_markers: Optional dictionary of custom annotation markers
        """
        self.markers = self.ANNOTATION_MARKERS.copy()
        if custom_markers:
            self.markers.update(custom_markers)
        
        logger.info(f"MessageAnnotator initialized with {len(self.markers)} markers")
    
    def annotate_message(self, message: str, message_type: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add annotations to a message based on its type.
        
        Args:
            message: The message content to annotate
            message_type: Type of message ('user', 'assistant', 'system', etc.)
            metadata: Optional metadata to include in annotations
            
        Returns:
            Annotated message string
            
        Raises:
            AnnotationError: If message_type is invalid or annotation fails
        """
        try:
            # Validate message type
            if message_type not in [mt.value for mt in MessageType]:
                raise AnnotationError(f"Invalid message type: {message_type}")
            
            # Clean the message content
            cleaned_message = message.strip()
            
            # Build annotation based on message type
            if message_type == MessageType.SYSTEM.value:
                annotated = f"{self.markers['system']}{cleaned_message}"
            elif message_type == MessageType.USER.value:
                annotated = f"{self.markers['user']}{cleaned_message}"
            elif message_type == MessageType.ASSISTANT.value:
                annotated = f"{self.markers['assistant']}{cleaned_message}"
            elif message_type == MessageType.START.value:
                annotated = f"{self.markers['start']}{cleaned_message}"
            elif message_type == MessageType.END.value:
                annotated = f"{self.markers['end']}{cleaned_message}"
            else:
                # Default annotation
                annotated = f"|{message_type}|{cleaned_message}"
            
            # Add metadata annotations if provided
            if metadata:
                metadata_annotations = self._create_metadata_annotations(metadata)
                annotated = f"{metadata_annotations}{annotated}"
            
            logger.debug(f"Annotated message of type '{message_type}': {len(annotated)} characters")
            return annotated
            
        except Exception as e:
            raise AnnotationError(f"Failed to annotate message: {str(e)}") from e
    
    def add_conversation_markers(self, conversation: List[str], 
                               conversation_id: Optional[str] = None) -> List[str]:
        """
        Add conversation flow markers to a list of messages.
        
        Args:
            conversation: List of message strings
            conversation_id: Optional conversation identifier
            
        Returns:
            List of messages with conversation markers added
            
        Raises:
            AnnotationError: If conversation processing fails
        """
        try:
            if not conversation:
                return []
            
            marked_conversation = []
            
            # Process each message with turn markers
            for i, message in enumerate(conversation):
                marked_message = ""
                
                # Add conversation ID marker to first message
                if i == 0 and conversation_id:
                    marked_message += f"|conversation:{conversation_id}|"
                
                # Add start marker to first message
                if i == 0:
                    marked_message += self.markers['start']
                
                # Add turn marker to all messages with proper separation
                marked_message += f"|turn:{i}|"
                
                # Add the actual message content
                marked_message += message
                
                # Add end marker to last message
                if i == len(conversation) - 1:
                    marked_message += self.markers['end']
                
                marked_conversation.append(marked_message)
            
            logger.info(f"Added conversation markers to {len(conversation)} messages")
            return marked_conversation
            
        except Exception as e:
            raise AnnotationError(f"Failed to add conversation markers: {str(e)}") from e
    
    def parse_annotated_message(self, annotated_message: str) -> Dict[str, str]:
        """
        Parse an annotated message and extract its components.
        
        Args:
            annotated_message: Message string with annotations
            
        Returns:
            Dictionary containing parsed components:
            - 'content': The main message content
            - 'type': The message type
            - 'annotations': Dictionary of found annotations
            - 'metadata': Extracted metadata
            
        Raises:
            AnnotationError: If parsing fails
        """
        try:
            result = {
                'content': '',
                'type': 'unknown',
                'annotations': {},
                'metadata': {}
            }
            
            remaining_content = annotated_message
            
            # Process the string sequentially to handle adjacent markers correctly
            while '|' in remaining_content:
                # Find the next marker
                start_idx = remaining_content.find('|')
                if start_idx == -1:
                    break
                
                end_idx = remaining_content.find('|', start_idx + 1)
                if end_idx == -1:
                    break
                
                # Extract the annotation
                annotation = remaining_content[start_idx + 1:end_idx]
                marker = remaining_content[start_idx:end_idx + 1]
                
                # Process the annotation
                if annotation in ['user', 'assistant', 'system']:
                    result['type'] = annotation
                elif annotation == 'start' or annotation == 'end':
                    result['annotations'][annotation] = True
                elif ':' in annotation:
                    # Handle key:value annotations
                    key, value = annotation.split(':', 1)
                    if key == 'conversation':
                        result['annotations']['conversation_id'] = value
                    elif key == 'turn':
                        result['annotations']['turn'] = value
                    else:
                        result['annotations'][key] = value
                else:
                    result['annotations'][annotation] = True
                
                # Remove the processed marker
                remaining_content = remaining_content.replace(marker, '', 1)
            
            # Clean up the remaining content
            result['content'] = remaining_content.strip()
            
            # Extract metadata from special annotations
            result['metadata'] = self._extract_metadata_from_annotations(result['annotations'])
            
            logger.debug(f"Parsed annotated message: type='{result['type']}', "
                        f"annotations={len(result['annotations'])}")
            
            return result
            
        except Exception as e:
            raise AnnotationError(f"Failed to parse annotated message: {str(e)}") from e
    
    def create_annotated_message_object(self, content: str, message_type: str,
                                      conversation_id: Optional[str] = None,
                                      turn_index: Optional[int] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> AnnotatedMessage:
        """
        Create an AnnotatedMessage object with proper structure.
        
        Args:
            content: Message content
            message_type: Type of message
            conversation_id: Optional conversation identifier
            turn_index: Optional turn index in conversation
            metadata: Optional metadata dictionary
            
        Returns:
            AnnotatedMessage object
            
        Raises:
            AnnotationError: If object creation fails
        """
        try:
            # Validate message type
            msg_type = MessageType(message_type)
            
            # Create annotations dictionary
            annotations = {}
            if turn_index is not None:
                annotations['turn'] = str(turn_index)
            if conversation_id:
                annotations['conversation_id'] = conversation_id
            
            # Create metadata dictionary
            meta = metadata or {}
            
            annotated_msg = AnnotatedMessage(
                content=content.strip(),
                message_type=msg_type,
                annotations=annotations,
                metadata=meta,
                conversation_id=conversation_id,
                turn_index=turn_index
            )
            
            logger.debug(f"Created AnnotatedMessage: type={msg_type.value}, "
                        f"conversation_id={conversation_id}")
            
            return annotated_msg
            
        except ValueError as e:
            raise AnnotationError(f"Invalid message type '{message_type}': {str(e)}") from e
        except Exception as e:
            raise AnnotationError(f"Failed to create AnnotatedMessage: {str(e)}") from e
    
    def create_conversation_flow(self, messages: List[AnnotatedMessage],
                               conversation_id: str,
                               metadata: Optional[Dict[str, Any]] = None) -> ConversationFlow:
        """
        Create a ConversationFlow object from annotated messages.
        
        Args:
            messages: List of AnnotatedMessage objects
            conversation_id: Conversation identifier
            metadata: Optional conversation metadata
            
        Returns:
            ConversationFlow object
            
        Raises:
            AnnotationError: If flow creation fails
        """
        try:
            # Create flow markers mapping
            flow_markers = {}
            
            for i, msg in enumerate(messages):
                if msg.message_type == MessageType.START:
                    flow_markers['start'] = i
                elif msg.message_type == MessageType.END:
                    flow_markers['end'] = i
                elif msg.message_type == MessageType.SYSTEM:
                    flow_markers.setdefault('system_messages', []).append(i)
            
            # Set default markers if not found
            if 'start' not in flow_markers and messages:
                flow_markers['start'] = 0
            if 'end' not in flow_markers and messages:
                flow_markers['end'] = len(messages) - 1
            
            conversation_flow = ConversationFlow(
                conversation_id=conversation_id,
                messages=messages,
                metadata=metadata or {},
                flow_markers=flow_markers
            )
            
            logger.info(f"Created ConversationFlow: id={conversation_id}, "
                       f"messages={len(messages)}, markers={len(flow_markers)}")
            
            return conversation_flow
            
        except Exception as e:
            raise AnnotationError(f"Failed to create ConversationFlow: {str(e)}") from e
    
    def extract_system_messages(self, conversation_flow: ConversationFlow) -> List[AnnotatedMessage]:
        """
        Extract system messages from a conversation flow.
        
        Args:
            conversation_flow: ConversationFlow object
            
        Returns:
            List of system messages
        """
        system_messages = []
        
        for msg in conversation_flow.messages:
            if msg.message_type == MessageType.SYSTEM:
                system_messages.append(msg)
        
        logger.debug(f"Extracted {len(system_messages)} system messages from conversation")
        return system_messages
    
    def validate_conversation_flow(self, conversation_flow: ConversationFlow) -> bool:
        """
        Validate the structure and integrity of a conversation flow.
        
        Args:
            conversation_flow: ConversationFlow object to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            AnnotationError: If validation encounters errors
        """
        try:
            # Check basic structure
            if not conversation_flow.conversation_id:
                logger.warning("ConversationFlow missing conversation_id")
                return False
            
            if conversation_flow.messages is None or not conversation_flow.messages:
                logger.warning("ConversationFlow has no messages")
                return False
            
            # Validate flow markers
            if 'start' in conversation_flow.flow_markers:
                start_idx = conversation_flow.flow_markers['start']
                if start_idx < 0 or start_idx >= len(conversation_flow.messages):
                    logger.warning(f"Invalid start marker index: {start_idx}")
                    return False
            
            if 'end' in conversation_flow.flow_markers:
                end_idx = conversation_flow.flow_markers['end']
                if end_idx < 0 or end_idx >= len(conversation_flow.messages):
                    logger.warning(f"Invalid end marker index: {end_idx}")
                    return False
            
            # Validate message sequence
            for i, msg in enumerate(conversation_flow.messages):
                if msg is None:
                    logger.warning(f"None message found at position {i}")
                    return False
                if msg.turn_index is not None and msg.turn_index != i:
                    logger.warning(f"Message turn_index mismatch at position {i}")
                    return False
            
            logger.debug(f"ConversationFlow validation passed: {conversation_flow.conversation_id}")
            return True
            
        except Exception as e:
            raise AnnotationError(f"Failed to validate ConversationFlow: {str(e)}") from e
    
    def _create_metadata_annotations(self, metadata: Dict[str, Any]) -> str:
        """Create annotation string from metadata dictionary."""
        annotations = []
        for key, value in metadata.items():
            if isinstance(value, (str, int, float)):
                annotations.append(f"|{key}:{value}|")
        return ''.join(annotations)
    
    def _extract_metadata_from_annotations(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from annotations dictionary."""
        metadata = {}
        
        for key, value in annotations.items():
            if ':' in str(key):
                meta_key, meta_value = str(key).split(':', 1)
                metadata[meta_key] = meta_value
            elif key not in ['turn', 'conversation_id']:
                metadata[key] = value
        
        return metadata
    
    def get_supported_markers(self) -> Dict[str, str]:
        """
        Get the dictionary of supported annotation markers.
        
        Returns:
            Dictionary mapping marker names to marker strings
        """
        return self.markers.copy()
    
    def add_custom_marker(self, name: str, marker: str) -> None:
        """
        Add a custom annotation marker.
        
        Args:
            name: Name of the marker
            marker: Marker string (should include | delimiters)
        """
        self.markers[name] = marker
        logger.info(f"Added custom marker: {name} -> {marker}")