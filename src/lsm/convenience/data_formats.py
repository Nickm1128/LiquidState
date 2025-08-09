"""
Data format handling and preprocessing for the LSM convenience API.

This module provides automatic conversion between different conversation formats,
support for structured conversation data with system messages, and comprehensive
data validation and preprocessing utilities.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum

from ..utils.lsm_exceptions import (
    DataValidationError, InvalidInputError, DataLoadError
)
from ..utils.lsm_logging import get_logger
from .config import ConvenienceValidationError

logger = get_logger(__name__)


class ConversationFormat(Enum):
    """Supported conversation formats."""
    SIMPLE_LIST = "simple_list"  # ["msg1", "msg2", ...]
    STRUCTURED_DICT = "structured_dict"  # {"messages": [...], "system": "..."}
    CHAT_MARKUP = "chat_markup"  # "User: hello\nAssistant: hi"
    HUGGINGFACE_CHAT = "huggingface_chat"  # [{"role": "user", "content": "..."}]
    OPENAI_CHAT = "openai_chat"  # [{"role": "user", "content": "..."}]
    PANDAS_DATAFRAME = "pandas_dataframe"  # DataFrame with conversation columns
    JSON_FILE = "json_file"  # JSON file with conversations
    CSV_FILE = "csv_file"  # CSV file with conversation data


@dataclass
class ConversationData:
    """
    Structured representation of conversation data.
    
    Attributes:
        messages: List of message strings in conversation order
        system_message: Optional system message/context
        metadata: Additional metadata about the conversation
        conversation_id: Unique identifier for the conversation
        roles: Optional list of roles corresponding to messages
    """
    messages: List[str]
    system_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    conversation_id: Optional[str] = None
    roles: Optional[List[str]] = None
    
    def __post_init__(self):
        """Validate conversation data after initialization."""
        if not self.messages:
            raise ConvenienceValidationError(
                "Conversation must have at least one message",
                suggestion="Provide a non-empty list of messages"
            )
        
        # Ensure all messages are strings
        self.messages = [str(msg).strip() for msg in self.messages if str(msg).strip()]
        
        if not self.messages:
            raise ConvenienceValidationError(
                "All messages are empty after cleaning",
                suggestion="Provide messages with actual content"
            )
        
        # Validate roles if provided
        if self.roles is not None:
            if len(self.roles) != len(self.messages):
                raise ConvenienceValidationError(
                    f"Number of roles ({len(self.roles)}) must match number of messages ({len(self.messages)})",
                    suggestion="Ensure each message has a corresponding role"
                )
    
    def to_simple_list(self) -> List[str]:
        """Convert to simple list format."""
        return self.messages.copy()
    
    def to_chat_markup(self, default_roles: Optional[List[str]] = None) -> str:
        """
        Convert to chat markup format.
        
        Args:
            default_roles: Default roles to use if none specified
            
        Returns:
            Chat markup string
        """
        if self.roles:
            roles = self.roles
        elif default_roles and len(default_roles) >= len(self.messages):
            roles = default_roles[:len(self.messages)]
        else:
            # Alternate between User and Assistant
            roles = ["User" if i % 2 == 0 else "Assistant" for i in range(len(self.messages))]
        
        lines = []
        if self.system_message:
            lines.append(f"System: {self.system_message}")
        
        for role, message in zip(roles, self.messages):
            lines.append(f"{role}: {message}")
        
        return "\n".join(lines)
    
    def to_openai_format(self) -> List[Dict[str, str]]:
        """
        Convert to OpenAI chat format.
        
        Returns:
            List of message dictionaries
        """
        messages = []
        
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        
        if self.roles:
            # Use provided roles, mapping to OpenAI format
            role_mapping = {
                "User": "user",
                "Assistant": "assistant",
                "System": "system",
                "user": "user",
                "assistant": "assistant",
                "system": "system"
            }
            
            for role, message in zip(self.roles, self.messages):
                openai_role = role_mapping.get(role, "user")
                messages.append({"role": openai_role, "content": message})
        else:
            # Alternate between user and assistant
            for i, message in enumerate(self.messages):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": message})
        
        return messages


class ConversationFormatDetector:
    """Automatically detect conversation data formats."""
    
    @staticmethod
    def detect_format(data: Any) -> ConversationFormat:
        """
        Automatically detect the format of conversation data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Detected conversation format
        """
        if isinstance(data, str):
            # Check if it's a file path
            if ConversationFormatDetector._is_file_path(data):
                if data.lower().endswith('.json'):
                    return ConversationFormat.JSON_FILE
                elif data.lower().endswith('.csv'):
                    return ConversationFormat.CSV_FILE
            
            # Check for chat markup patterns
            if ConversationFormatDetector._is_chat_markup(data):
                return ConversationFormat.CHAT_MARKUP
            
            # Default to simple string (will be converted to simple list)
            return ConversationFormat.SIMPLE_LIST
        
        elif isinstance(data, list):
            if not data:
                return ConversationFormat.SIMPLE_LIST
            
            # Check first element to determine list type
            first_item = data[0]
            
            if isinstance(first_item, str):
                return ConversationFormat.SIMPLE_LIST
            
            elif isinstance(first_item, dict):
                # Check for OpenAI/HuggingFace chat format
                if "role" in first_item and "content" in first_item:
                    return ConversationFormat.OPENAI_CHAT
                elif "messages" in first_item:
                    return ConversationFormat.STRUCTURED_DICT
                else:
                    return ConversationFormat.STRUCTURED_DICT
        
        elif isinstance(data, dict):
            if "messages" in data:
                return ConversationFormat.STRUCTURED_DICT
            else:
                # Treat as single structured conversation
                return ConversationFormat.STRUCTURED_DICT
        
        elif isinstance(data, pd.DataFrame):
            return ConversationFormat.PANDAS_DATAFRAME
        
        else:
            # Default fallback
            return ConversationFormat.SIMPLE_LIST
    
    @staticmethod
    def _is_file_path(data: str) -> bool:
        """Check if string is a file path."""
        try:
            path = Path(data)
            return path.suffix in ['.json', '.csv', '.txt'] or path.exists()
        except:
            return False
    
    @staticmethod
    def _is_chat_markup(data: str) -> bool:
        """Check if string contains chat markup patterns."""
        # Look for patterns like "User:", "Assistant:", etc.
        chat_patterns = [
            r'\b(User|Assistant|System|Human|AI|Bot):\s*',
            r'\b(user|assistant|system|human|ai|bot):\s*',
            r'<\|.*?\|>',  # Special tokens
        ]
        
        for pattern in chat_patterns:
            if re.search(pattern, data):
                return True
        
        return False


class ConversationParser:
    """Parse different conversation formats into structured data."""
    
    def __init__(self):
        """Initialize the conversation parser."""
        self.detector = ConversationFormatDetector()
    
    def parse(self, data: Any, format_hint: Optional[ConversationFormat] = None) -> List[ConversationData]:
        """
        Parse conversation data into structured format.
        
        Args:
            data: Input conversation data
            format_hint: Optional hint about the data format
            
        Returns:
            List of ConversationData objects
        """
        try:
            # Detect format if not provided
            if format_hint is None:
                format_hint = self.detector.detect_format(data)
            
            logger.debug(f"Parsing conversation data with format: {format_hint}")
            
            # Route to appropriate parser
            if format_hint == ConversationFormat.SIMPLE_LIST:
                return self._parse_simple_list(data)
            elif format_hint == ConversationFormat.STRUCTURED_DICT:
                return self._parse_structured_dict(data)
            elif format_hint == ConversationFormat.CHAT_MARKUP:
                return self._parse_chat_markup(data)
            elif format_hint == ConversationFormat.OPENAI_CHAT:
                return self._parse_openai_chat(data)
            elif format_hint == ConversationFormat.HUGGINGFACE_CHAT:
                return self._parse_huggingface_chat(data)
            elif format_hint == ConversationFormat.PANDAS_DATAFRAME:
                return self._parse_pandas_dataframe(data)
            elif format_hint == ConversationFormat.JSON_FILE:
                return self._parse_json_file(data)
            elif format_hint == ConversationFormat.CSV_FILE:
                return self._parse_csv_file(data)
            else:
                raise ConvenienceValidationError(
                    f"Unsupported conversation format: {format_hint}",
                    suggestion="Use one of the supported formats or provide format_hint"
                )
        
        except Exception as e:
            if isinstance(e, ConvenienceValidationError):
                raise
            raise ConvenienceValidationError(
                f"Failed to parse conversation data: {e}",
                suggestion="Check data format and ensure it matches expected structure"
            )
    
    def _parse_simple_list(self, data: Any) -> List[ConversationData]:
        """Parse simple list format."""
        if isinstance(data, str):
            # Single string - treat as one conversation
            return [ConversationData(messages=[data])]
        
        elif isinstance(data, list):
            if not data:
                raise ConvenienceValidationError(
                    "Empty conversation list",
                    suggestion="Provide at least one conversation"
                )
            
            # Check if it's a list of strings (single conversation) or list of lists (multiple conversations)
            if all(isinstance(item, str) for item in data):
                # Single conversation with multiple messages
                return [ConversationData(messages=data)]
            else:
                # Multiple conversations
                conversations = []
                for i, item in enumerate(data):
                    if isinstance(item, str):
                        conversations.append(ConversationData(messages=[item], conversation_id=str(i)))
                    elif isinstance(item, list):
                        conversations.append(ConversationData(messages=item, conversation_id=str(i)))
                    else:
                        logger.warning(f"Skipping unsupported item type at index {i}: {type(item)}")
                
                return conversations
        
        else:
            raise ConvenienceValidationError(
                f"Simple list format expects string or list, got {type(data).__name__}",
                suggestion="Provide data as a string or list of strings"
            )
    
    def _parse_structured_dict(self, data: Any) -> List[ConversationData]:
        """Parse structured dictionary format."""
        if isinstance(data, dict):
            # Single conversation
            return [self._parse_single_structured_dict(data)]
        
        elif isinstance(data, list):
            # Multiple conversations
            conversations = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    conv_data = self._parse_single_structured_dict(item)
                    conv_data.conversation_id = conv_data.conversation_id or str(i)
                    conversations.append(conv_data)
                else:
                    logger.warning(f"Skipping non-dict item at index {i}")
            
            return conversations
        
        else:
            raise ConvenienceValidationError(
                f"Structured dict format expects dict or list of dicts, got {type(data).__name__}",
                suggestion="Provide data as {'messages': [...], 'system': '...'} or list of such dicts"
            )
    
    def _parse_single_structured_dict(self, data: Dict[str, Any]) -> ConversationData:
        """Parse a single structured dictionary."""
        messages = []
        system_message = None
        metadata = {}
        conversation_id = None
        roles = None
        
        # Extract messages
        if "messages" in data:
            messages_data = data["messages"]
            if isinstance(messages_data, list):
                messages = [str(msg) for msg in messages_data]
            else:
                messages = [str(messages_data)]
        elif "text" in data:
            messages = [str(data["text"])]
        elif "content" in data:
            messages = [str(data["content"])]
        else:
            # Try to find any string values as messages
            for key, value in data.items():
                if isinstance(value, str) and len(value.strip()) > 0:
                    messages.append(value.strip())
        
        # Extract system message
        system_keys = ["system", "system_message", "system_prompt", "context"]
        for key in system_keys:
            if key in data and data[key]:
                system_message = str(data[key])
                break
        
        # Extract roles
        if "roles" in data:
            roles_data = data["roles"]
            if isinstance(roles_data, list):
                roles = [str(role) for role in roles_data]
        
        # Extract conversation ID
        id_keys = ["id", "conversation_id", "conv_id", "chat_id"]
        for key in id_keys:
            if key in data:
                conversation_id = str(data[key])
                break
        
        # Extract metadata
        metadata_keys = ["metadata", "meta", "info", "extra"]
        for key in metadata_keys:
            if key in data and isinstance(data[key], dict):
                metadata.update(data[key])
        
        # Add any remaining fields as metadata
        excluded_keys = {"messages", "text", "content", "system", "system_message", 
                        "system_prompt", "context", "roles", "id", "conversation_id", 
                        "conv_id", "chat_id", "metadata", "meta", "info", "extra"}
        
        for key, value in data.items():
            if key not in excluded_keys:
                metadata[key] = value
        
        return ConversationData(
            messages=messages,
            system_message=system_message,
            metadata=metadata,
            conversation_id=conversation_id,
            roles=roles
        )
    
    def _parse_chat_markup(self, data: str) -> List[ConversationData]:
        """Parse chat markup format."""
        lines = data.strip().split('\n')
        conversations = []
        current_messages = []
        current_roles = []
        system_message = None
        
        # Pattern to match role prefixes
        role_pattern = r'^(User|Assistant|System|Human|AI|Bot|user|assistant|system|human|ai|bot):\s*(.*)$'
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = re.match(role_pattern, line)
            if match:
                role, message = match.groups()
                role = role.strip()
                message = message.strip()
                
                if role.lower() == 'system':
                    system_message = message
                else:
                    current_messages.append(message)
                    current_roles.append(role)
            else:
                # Line without role prefix - append to last message if exists
                if current_messages:
                    current_messages[-1] += " " + line
        
        if current_messages:
            conversations.append(ConversationData(
                messages=current_messages,
                system_message=system_message,
                roles=current_roles,
                conversation_id="0"
            ))
        
        return conversations
    
    def _parse_openai_chat(self, data: List[Dict[str, str]]) -> List[ConversationData]:
        """Parse OpenAI chat format."""
        messages = []
        roles = []
        system_message = None
        
        for item in data:
            if not isinstance(item, dict) or "role" not in item or "content" not in item:
                continue
            
            role = item["role"].strip()
            content = item["content"].strip()
            
            if role == "system":
                system_message = content
            else:
                messages.append(content)
                roles.append(role)
        
        if messages:
            return [ConversationData(
                messages=messages,
                system_message=system_message,
                roles=roles,
                conversation_id="0"
            )]
        else:
            return []
    
    def _parse_huggingface_chat(self, data: List[Dict[str, str]]) -> List[ConversationData]:
        """Parse HuggingFace chat format (same as OpenAI for now)."""
        return self._parse_openai_chat(data)
    
    def _parse_pandas_dataframe(self, data: pd.DataFrame) -> List[ConversationData]:
        """Parse pandas DataFrame format."""
        conversations = []
        
        # Try to identify conversation columns
        text_columns = []
        role_columns = []
        system_columns = []
        
        for col in data.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['text', 'message', 'content', 'utterance']):
                text_columns.append(col)
            elif any(keyword in col_lower for keyword in ['role', 'speaker', 'user', 'agent']):
                role_columns.append(col)
            elif any(keyword in col_lower for keyword in ['system', 'context', 'prompt']):
                system_columns.append(col)
        
        # Handle different DataFrame structures
        if 'User' in data.columns and 'Assistant' in data.columns:
            # User-Assistant pair format
            for idx, row in data.iterrows():
                messages = []
                roles = []
                
                if pd.notna(row['User']) and str(row['User']).strip():
                    messages.append(str(row['User']).strip())
                    roles.append('User')
                
                if pd.notna(row['Assistant']) and str(row['Assistant']).strip():
                    messages.append(str(row['Assistant']).strip())
                    roles.append('Assistant')
                
                if messages:
                    conversations.append(ConversationData(
                        messages=messages,
                        roles=roles,
                        conversation_id=str(idx)
                    ))
        
        elif text_columns:
            # Generic text column format
            primary_text_col = text_columns[0]
            primary_role_col = role_columns[0] if role_columns else None
            primary_system_col = system_columns[0] if system_columns else None
            
            # Group by conversation if there's an ID column
            id_columns = [col for col in data.columns if 'id' in col.lower() or 'conv' in col.lower()]
            
            if id_columns:
                # Group by conversation ID
                id_col = id_columns[0]
                for conv_id, group in data.groupby(id_col):
                    messages = []
                    roles = []
                    system_message = None
                    
                    for _, row in group.iterrows():
                        if pd.notna(row[primary_text_col]):
                            text = str(row[primary_text_col]).strip()
                            if text:
                                messages.append(text)
                                
                                if primary_role_col and pd.notna(row[primary_role_col]):
                                    roles.append(str(row[primary_role_col]))
                                
                                if primary_system_col and pd.notna(row[primary_system_col]) and not system_message:
                                    system_message = str(row[primary_system_col]).strip()
                    
                    if messages:
                        conversations.append(ConversationData(
                            messages=messages,
                            system_message=system_message,
                            roles=roles if roles else None,
                            conversation_id=str(conv_id)
                        ))
            else:
                # Treat each row as a separate conversation
                for idx, row in data.iterrows():
                    if pd.notna(row[primary_text_col]):
                        text = str(row[primary_text_col]).strip()
                        if text:
                            role = None
                            if primary_role_col and pd.notna(row[primary_role_col]):
                                role = str(row[primary_role_col])
                            
                            system_message = None
                            if primary_system_col and pd.notna(row[primary_system_col]):
                                system_message = str(row[primary_system_col]).strip()
                            
                            conversations.append(ConversationData(
                                messages=[text],
                                system_message=system_message,
                                roles=[role] if role else None,
                                conversation_id=str(idx)
                            ))
        
        else:
            raise ConvenienceValidationError(
                "Could not identify conversation columns in DataFrame",
                suggestion="Ensure DataFrame has columns like 'text', 'message', 'User', 'Assistant', etc."
            )
        
        return conversations
    
    def _parse_json_file(self, file_path: str) -> List[ConversationData]:
        """Parse JSON file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Recursively parse the loaded JSON data
            return self.parse(data)
        
        except Exception as e:
            raise ConvenienceValidationError(
                f"Failed to parse JSON file {file_path}: {e}",
                suggestion="Ensure the file exists and contains valid JSON"
            )
    
    def _parse_csv_file(self, file_path: str) -> List[ConversationData]:
        """Parse CSV file format."""
        try:
            df = pd.read_csv(file_path)
            return self._parse_pandas_dataframe(df)
        
        except Exception as e:
            raise ConvenienceValidationError(
                f"Failed to parse CSV file {file_path}: {e}",
                suggestion="Ensure the file exists and is a valid CSV"
            )


class ConversationPreprocessor:
    """Preprocess conversation data for training."""
    
    def __init__(self, 
                 min_message_length: int = 1,
                 max_message_length: int = 1000,
                 min_conversation_length: int = 1,
                 max_conversation_length: int = 100,
                 remove_empty_messages: bool = True,
                 normalize_whitespace: bool = True,
                 filter_languages: Optional[List[str]] = None):
        """
        Initialize the conversation preprocessor.
        
        Args:
            min_message_length: Minimum message length in characters
            max_message_length: Maximum message length in characters
            min_conversation_length: Minimum number of messages per conversation
            max_conversation_length: Maximum number of messages per conversation
            remove_empty_messages: Whether to remove empty messages
            normalize_whitespace: Whether to normalize whitespace
            filter_languages: Optional list of languages to keep (ISO codes)
        """
        self.min_message_length = min_message_length
        self.max_message_length = max_message_length
        self.min_conversation_length = min_conversation_length
        self.max_conversation_length = max_conversation_length
        self.remove_empty_messages = remove_empty_messages
        self.normalize_whitespace = normalize_whitespace
        self.filter_languages = filter_languages
    
    def preprocess(self, conversations: List[ConversationData]) -> List[ConversationData]:
        """
        Preprocess a list of conversations.
        
        Args:
            conversations: List of ConversationData objects
            
        Returns:
            List of preprocessed ConversationData objects
        """
        logger.info(f"Preprocessing {len(conversations)} conversations")
        
        processed_conversations = []
        
        for i, conv in enumerate(conversations):
            try:
                processed_conv = self._preprocess_single_conversation(conv)
                if processed_conv is not None:
                    processed_conversations.append(processed_conv)
            except Exception as e:
                logger.warning(f"Failed to preprocess conversation {i}: {e}")
        
        logger.info(f"Preprocessing complete: {len(processed_conversations)} conversations remaining")
        
        return processed_conversations
    
    def _preprocess_single_conversation(self, conv: ConversationData) -> Optional[ConversationData]:
        """
        Preprocess a single conversation.
        
        Args:
            conv: ConversationData object
            
        Returns:
            Preprocessed ConversationData or None if filtered out
        """
        # Clean messages
        cleaned_messages = []
        cleaned_roles = []
        
        for i, message in enumerate(conv.messages):
            # Normalize whitespace
            if self.normalize_whitespace:
                message = re.sub(r'\s+', ' ', message.strip())
            
            # Check message length
            if len(message) < self.min_message_length:
                if not self.remove_empty_messages:
                    continue
            elif len(message) > self.max_message_length:
                # Truncate long messages
                message = message[:self.max_message_length].strip()
            
            # Skip empty messages if configured
            if self.remove_empty_messages and not message.strip():
                continue
            
            cleaned_messages.append(message)
            
            # Keep corresponding role if available
            if conv.roles and i < len(conv.roles):
                cleaned_roles.append(conv.roles[i])
        
        # Check conversation length
        if len(cleaned_messages) < self.min_conversation_length:
            return None
        
        if len(cleaned_messages) > self.max_conversation_length:
            # Truncate conversation
            cleaned_messages = cleaned_messages[:self.max_conversation_length]
            if cleaned_roles:
                cleaned_roles = cleaned_roles[:self.max_conversation_length]
        
        # Clean system message
        system_message = conv.system_message
        if system_message:
            if self.normalize_whitespace:
                system_message = re.sub(r'\s+', ' ', system_message.strip())
            
            if len(system_message) > self.max_message_length:
                system_message = system_message[:self.max_message_length].strip()
        
        # Create processed conversation
        return ConversationData(
            messages=cleaned_messages,
            system_message=system_message,
            metadata=conv.metadata.copy(),
            conversation_id=conv.conversation_id,
            roles=cleaned_roles if cleaned_roles else None
        )


class DataFormatHandler:
    """
    Main handler for conversation data format processing.
    
    This class provides a unified interface for parsing, validating, and preprocessing
    conversation data in various formats.
    """
    
    def __init__(self, 
                 preprocessor_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data format handler.
        
        Args:
            preprocessor_config: Configuration for the preprocessor
        """
        self.parser = ConversationParser()
        
        # Initialize preprocessor with config
        preprocessor_config = preprocessor_config or {}
        self.preprocessor = ConversationPreprocessor(**preprocessor_config)
        
        logger.info("Initialized DataFormatHandler")
    
    def process_conversation_data(self, 
                                data: Any,
                                format_hint: Optional[ConversationFormat] = None,
                                preprocess: bool = True,
                                return_format: str = "simple_list") -> Union[List[str], List[ConversationData]]:
        """
        Process conversation data from any supported format.
        
        Args:
            data: Input conversation data
            format_hint: Optional hint about the data format
            preprocess: Whether to apply preprocessing
            return_format: Format to return ("simple_list" or "structured")
            
        Returns:
            Processed conversation data in requested format
        """
        try:
            # Parse the data
            conversations = self.parser.parse(data, format_hint)
            
            if not conversations:
                raise ConvenienceValidationError(
                    "No valid conversations found in data",
                    suggestion="Check data format and ensure it contains conversation content"
                )
            
            # Preprocess if requested
            if preprocess:
                conversations = self.preprocessor.preprocess(conversations)
                
                if not conversations:
                    raise ConvenienceValidationError(
                        "No conversations remaining after preprocessing",
                        suggestion="Relax preprocessing constraints or check data quality"
                    )
            
            # Return in requested format
            if return_format == "simple_list":
                # Flatten all messages into a single list
                all_messages = []
                for conv in conversations:
                    all_messages.extend(conv.messages)
                return all_messages
            
            elif return_format == "structured":
                return conversations
            
            else:
                raise ConvenienceValidationError(
                    f"Unsupported return format: {return_format}",
                    suggestion="Use 'simple_list' or 'structured'",
                    valid_options=["simple_list", "structured"]
                )
        
        except Exception as e:
            if isinstance(e, ConvenienceValidationError):
                raise
            raise ConvenienceValidationError(
                f"Failed to process conversation data: {e}",
                suggestion="Check data format and preprocessing configuration"
            )
    
    def validate_conversation_data(self, data: Any) -> Dict[str, Any]:
        """
        Validate conversation data and return detailed information.
        
        Args:
            data: Input conversation data
            
        Returns:
            Dictionary with validation results and statistics
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
            "detected_format": None,
            "sample_conversations": []
        }
        
        try:
            # Detect format
            detected_format = self.parser.detector.detect_format(data)
            validation_results["detected_format"] = detected_format.value
            
            # Parse conversations
            conversations = self.parser.parse(data, detected_format)
            
            # Calculate statistics
            if conversations:
                message_lengths = []
                conversation_lengths = []
                has_system_messages = 0
                has_roles = 0
                
                for conv in conversations:
                    conversation_lengths.append(len(conv.messages))
                    message_lengths.extend([len(msg) for msg in conv.messages])
                    
                    if conv.system_message:
                        has_system_messages += 1
                    
                    if conv.roles:
                        has_roles += 1
                
                validation_results["statistics"] = {
                    "total_conversations": len(conversations),
                    "total_messages": len(message_lengths),
                    "avg_conversation_length": np.mean(conversation_lengths) if conversation_lengths else 0,
                    "avg_message_length": np.mean(message_lengths) if message_lengths else 0,
                    "min_message_length": min(message_lengths) if message_lengths else 0,
                    "max_message_length": max(message_lengths) if message_lengths else 0,
                    "conversations_with_system_messages": has_system_messages,
                    "conversations_with_roles": has_roles,
                    "system_message_ratio": has_system_messages / len(conversations) if conversations else 0,
                    "role_coverage_ratio": has_roles / len(conversations) if conversations else 0
                }
                
                # Add sample conversations (first 3)
                for i, conv in enumerate(conversations[:3]):
                    sample = {
                        "conversation_id": conv.conversation_id or str(i),
                        "message_count": len(conv.messages),
                        "has_system_message": bool(conv.system_message),
                        "has_roles": bool(conv.roles),
                        "first_message": conv.messages[0][:100] + "..." if len(conv.messages[0]) > 100 else conv.messages[0]
                    }
                    validation_results["sample_conversations"].append(sample)
            
            else:
                validation_results["errors"].append("No valid conversations found")
                validation_results["is_valid"] = False
        
        except Exception as e:
            validation_results["errors"].append(str(e))
            validation_results["is_valid"] = False
        
        return validation_results
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported conversation formats.
        
        Returns:
            List of supported format names
        """
        return [fmt.value for fmt in ConversationFormat]
    
    def convert_format(self, 
                      data: Any,
                      source_format: Optional[ConversationFormat] = None,
                      target_format: str = "openai_chat") -> Any:
        """
        Convert conversation data from one format to another.
        
        Args:
            data: Input conversation data
            source_format: Source format (auto-detected if None)
            target_format: Target format name
            
        Returns:
            Converted conversation data
        """
        # Parse to structured format
        conversations = self.parser.parse(data, source_format)
        
        if not conversations:
            raise ConvenienceValidationError(
                "No conversations to convert",
                suggestion="Ensure input data contains valid conversations"
            )
        
        # Convert to target format
        if target_format == "simple_list":
            result = []
            for conv in conversations:
                result.extend(conv.messages)
            return result
        
        elif target_format == "chat_markup":
            if len(conversations) == 1:
                return conversations[0].to_chat_markup()
            else:
                return [conv.to_chat_markup() for conv in conversations]
        
        elif target_format == "openai_chat":
            if len(conversations) == 1:
                return conversations[0].to_openai_format()
            else:
                return [conv.to_openai_format() for conv in conversations]
        
        elif target_format == "structured":
            return conversations
        
        else:
            raise ConvenienceValidationError(
                f"Unsupported target format: {target_format}",
                suggestion="Use 'simple_list', 'chat_markup', 'openai_chat', or 'structured'"
            )