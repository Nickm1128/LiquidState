#!/usr/bin/env python3
"""
Standalone System Message Processor for LSM Training Pipeline Enhancement.

This module provides a dedicated processor for handling system messages with
proper tokenization, validation, and embedding generation for broader system
message handling across the LSM architecture.
"""

import re
import json
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from dataclasses import dataclass

from ..data.tokenization import StandardTokenizerWrapper
from ..utils.lsm_exceptions import LSMError, InvalidInputError
from ..utils.lsm_logging import get_logger

logger = get_logger(__name__)


class SystemMessageError(LSMError):
    """Raised when system message processing fails."""
    
    def __init__(self, operation: str, reason: str, details: Optional[Dict[str, Any]] = None):
        error_details = {"operation": operation, "reason": reason}
        if details:
            error_details.update(details)
        
        message = f"System message processing failed during {operation}: {reason}"
        super().__init__(message, error_details)
        self.operation = operation


@dataclass
class SystemMessageContext:
    """Container for processed system message context and embeddings."""
    original_message: str
    parsed_content: Dict[str, Any]
    token_ids: List[int]
    embeddings: np.ndarray
    metadata: Dict[str, Any]
    validation_status: bool = True
    processing_time: float = 0.0


@dataclass
class SystemMessageConfig:
    """Configuration for system message processing."""
    max_length: int = 512
    embedding_dim: int = 256
    add_special_tokens: bool = True
    validate_format: bool = True
    supported_formats: List[str] = None
    default_influence_strength: float = 1.0
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["instruction", "persona", "constraint", "context"]


class SystemMessageProcessor:
    """
    Standalone processor for system messages with tokenization and validation.
    
    This class handles parsing, validation, tokenization, and embedding generation
    for system messages, providing a consistent interface for system message
    processing across different components of the LSM architecture.
    """
    
    # Supported system message formats and their patterns
    # Order matters - more specific patterns should come first
    MESSAGE_FORMATS = {
        "persona": r"^(You are a|As a|Playing the role of|Persona:)",
        "instruction": r"^(Your task is|Please|Instruction:|Act as)",
        "constraint": r"^(Do not|Never|Always|Must|Should|Constraint:)",
        "context": r"^(Context:|Background:|Given that|In this scenario)",
        "custom": r"^(System:|Custom:)"
    }
    
    # Special tokens for system message processing
    SYSTEM_TOKENS = {
        "start": "<|system_start|>",
        "end": "<|system_end|>",
        "separator": "<|sep|>",
        "instruction": "<|instruction|>",
        "persona": "<|persona|>",
        "constraint": "<|constraint|>",
        "context": "<|context|>"
    }
    
    def __init__(self, 
                 tokenizer: StandardTokenizerWrapper,
                 config: Optional[SystemMessageConfig] = None):
        """
        Initialize SystemMessageProcessor.
        
        Args:
            tokenizer: StandardTokenizerWrapper instance for tokenization
            config: Optional configuration for system message processing
            
        Raises:
            SystemMessageError: If initialization fails
        """
        try:
            if tokenizer is None:
                raise ValueError("Tokenizer cannot be None")
            
            self.tokenizer = tokenizer
            self.config = config or SystemMessageConfig()
            
            # Initialize embedding dimension based on tokenizer if not specified
            if hasattr(tokenizer, 'get_vocab_size'):
                self.vocab_size = tokenizer.get_vocab_size()
            else:
                self.vocab_size = 50000  # Default fallback
            
            # Processing statistics
            self._processed_count = 0
            self._validation_failures = 0
            self._format_distribution = {fmt: 0 for fmt in self.MESSAGE_FORMATS.keys()}
            self._format_distribution["unknown"] = 0  # Add unknown format
            
            logger.info(f"Initialized SystemMessageProcessor with vocab_size={self.vocab_size}, "
                       f"max_length={self.config.max_length}")
            
        except Exception as e:
            raise SystemMessageError(
                "initialization",
                f"Failed to initialize SystemMessageProcessor: {str(e)}",
                {"config": config.__dict__ if config else None}
            )
    
    def parse_system_message(self, message: str) -> Dict[str, Any]:
        """
        Parse system message and extract structured information.
        
        Args:
            message: Raw system message text
            
        Returns:
            Dictionary with parsed message components
            
        Raises:
            SystemMessageError: If parsing fails
        """
        try:
            if not message or not isinstance(message, str):
                raise ValueError("Message must be a non-empty string")
            
            # Clean and normalize message
            cleaned_message = self._clean_message(message)
            
            # Detect message format
            message_format = self._detect_format(cleaned_message)
            
            # Extract components based on format
            components = self._extract_components(cleaned_message, message_format)
            
            # Parse special instructions or constraints
            special_instructions = self._extract_special_instructions(cleaned_message)
            
            parsed_result = {
                "format": message_format,
                "content": cleaned_message,
                "components": components,
                "special_instructions": special_instructions,
                "length": len(cleaned_message),
                "word_count": len(cleaned_message.split()),
                "has_constraints": bool(special_instructions.get("constraints")),
                "has_persona": bool(components.get("persona")),
                "complexity_score": self._calculate_complexity_score(cleaned_message)
            }
            
            # Update format distribution statistics
            self._format_distribution[message_format] += 1
            
            return parsed_result
            
        except Exception as e:
            raise SystemMessageError(
                "parsing",
                f"Failed to parse system message: {str(e)}",
                {"message_preview": message[:100] if message else None}
            )
    
    def validate_system_message_format(self, message: str) -> Tuple[bool, List[str]]:
        """
        Validate system message format and content.
        
        Args:
            message: System message to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        try:
            # Basic validation
            if not message or not isinstance(message, str):
                errors.append("Message must be a non-empty string")
                return False, errors
            
            # Length validation
            if len(message) > self.config.max_length * 4:  # Rough character limit
                errors.append(f"Message too long: {len(message)} characters")
            
            if len(message.strip()) < 10:
                errors.append("Message too short: minimum 10 characters required")
            
            # Format validation
            if self.config.validate_format:
                message_format = self._detect_format(message)
                if message_format == "unknown":
                    errors.append("Message format not recognized")
            
            # Content validation
            content_errors = self._validate_content(message)
            errors.extend(content_errors)
            
            # Special character validation
            if self._has_invalid_characters(message):
                errors.append("Message contains invalid or potentially harmful characters")
            
            is_valid = len(errors) == 0
            
            if not is_valid:
                self._validation_failures += 1
            
            return is_valid, errors
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
            return False, errors
    
    def create_system_context_embeddings(self, 
                                       message: str,
                                       influence_strength: float = None) -> np.ndarray:
        """
        Create system context embeddings using StandardTokenizerWrapper.
        
        Args:
            message: System message text
            influence_strength: Optional influence strength multiplier
            
        Returns:
            System context embeddings array
            
        Raises:
            SystemMessageError: If embedding creation fails
        """
        try:
            if influence_strength is None:
                influence_strength = self.config.default_influence_strength
            
            # Tokenize the message
            token_ids = self.tokenizer.encode_single(
                message, 
                add_special_tokens=self.config.add_special_tokens
            )
            
            # Truncate if necessary
            if len(token_ids) > self.config.max_length:
                token_ids = token_ids[:self.config.max_length]
                logger.warning(f"System message truncated to {self.config.max_length} tokens")
            
            # Create embeddings using a simple approach
            # In practice, this could use more sophisticated embedding methods
            embeddings = self._create_embeddings_from_tokens(token_ids)
            
            # Apply influence strength
            embeddings = embeddings * influence_strength
            
            # Normalize embeddings
            norm = np.linalg.norm(embeddings)
            if norm > 0:
                embeddings = embeddings / norm
            
            return embeddings
            
        except Exception as e:
            raise SystemMessageError(
                "embedding_creation",
                f"Failed to create system context embeddings: {str(e)}",
                {"message_length": len(message) if message else 0}
            )
    
    def process_system_message(self, 
                             message: str,
                             validate: bool = True,
                             create_embeddings: bool = True) -> SystemMessageContext:
        """
        Complete system message processing pipeline.
        
        Args:
            message: System message to process
            validate: Whether to validate message format
            create_embeddings: Whether to create embeddings
            
        Returns:
            SystemMessageContext with all processed information
            
        Raises:
            SystemMessageError: If processing fails
        """
        import time
        start_time = time.time()
        
        try:
            # Validation
            validation_status = True
            validation_errors = []
            
            if validate:
                validation_status, validation_errors = self.validate_system_message_format(message)
                if not validation_status and self.config.validate_format:
                    raise ValueError(f"Message validation failed: {'; '.join(validation_errors)}")
            
            # Parse message
            parsed_content = self.parse_system_message(message)
            
            # Tokenize
            token_ids = self.tokenizer.encode_single(
                message,
                add_special_tokens=self.config.add_special_tokens
            )
            
            # Create embeddings if requested
            embeddings = None
            if create_embeddings:
                embeddings = self.create_system_context_embeddings(message)
            
            # Create metadata
            metadata = {
                "validation_errors": validation_errors,
                "token_count": len(token_ids),
                "embedding_dim": embeddings.shape[0] if embeddings is not None else 0,
                "format": parsed_content.get("format", "unknown"),
                "complexity_score": parsed_content.get("complexity_score", 0.0),
                "processor_version": "1.0.0"
            }
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self._processed_count += 1
            
            return SystemMessageContext(
                original_message=message,
                parsed_content=parsed_content,
                token_ids=token_ids,
                embeddings=embeddings,
                metadata=metadata,
                validation_status=validation_status,
                processing_time=processing_time
            )
            
        except Exception as e:
            raise SystemMessageError(
                "complete_processing",
                f"Failed to process system message: {str(e)}",
                {"message_preview": message[:100] if message else None}
            )
    
    def batch_process_system_messages(self, 
                                    messages: List[str],
                                    validate: bool = True,
                                    create_embeddings: bool = True) -> List[SystemMessageContext]:
        """
        Process multiple system messages in batch.
        
        Args:
            messages: List of system messages to process
            validate: Whether to validate message formats
            create_embeddings: Whether to create embeddings
            
        Returns:
            List of SystemMessageContext objects
        """
        results = []
        failed_indices = []
        
        for i, message in enumerate(messages):
            try:
                context = self.process_system_message(
                    message, 
                    validate=validate, 
                    create_embeddings=create_embeddings
                )
                results.append(context)
            except SystemMessageError as e:
                logger.warning(f"Failed to process message {i}: {e.message}")
                failed_indices.append(i)
                # Add placeholder context for failed messages
                results.append(SystemMessageContext(
                    original_message=message,
                    parsed_content={"format": "error", "error": str(e)},
                    token_ids=[],
                    embeddings=np.zeros(self.config.embedding_dim) if create_embeddings else None,
                    metadata={"error": str(e), "failed": True},
                    validation_status=False
                ))
        
        if failed_indices:
            logger.warning(f"Failed to process {len(failed_indices)} out of {len(messages)} messages")
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics and metrics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "total_processed": self._processed_count,
            "validation_failures": self._validation_failures,
            "validation_success_rate": (
                (self._processed_count - self._validation_failures) / max(self._processed_count, 1)
            ),
            "format_distribution": self._format_distribution.copy(),
            "config": {
                "max_length": self.config.max_length,
                "embedding_dim": self.config.embedding_dim,
                "validate_format": self.config.validate_format
            }
        }
    
    # Private helper methods
    
    def _clean_message(self, message: str) -> str:
        """Clean and normalize system message."""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', message.strip())
        
        # Remove potentially harmful characters
        cleaned = re.sub(r'[^\w\s\.,!?;:()\-\'"@#$%&*+=<>/\\|`~\[\]{}]', '', cleaned)
        
        return cleaned
    
    def _detect_format(self, message: str) -> str:
        """Detect system message format based on content patterns."""
        message_lower = message.lower()
        
        # Special handling for "You are" patterns
        if message_lower.startswith("you are "):
            # Check if it's "You are a/an [role]" (persona) or just "You are [adjective]" (instruction)
            if re.match(r"^you are (a|an)\s+\w+", message_lower):
                return "persona"
            else:
                return "persona"  # Default "You are" to persona
        
        # Check other patterns in order
        for format_name, pattern in self.MESSAGE_FORMATS.items():
            if re.search(pattern, message_lower, re.IGNORECASE):
                return format_name
        
        return "unknown"
    
    def _extract_components(self, message: str, message_format: str) -> Dict[str, Any]:
        """Extract structured components from system message."""
        components = {}
        
        if message_format == "persona":
            # Extract persona information
            persona_match = re.search(r'(you are|as a|playing the role of)\s+([^.!?]+)', 
                                    message.lower())
            if persona_match:
                components["persona"] = persona_match.group(2).strip()
        
        elif message_format == "instruction":
            # Extract main instruction
            instruction_patterns = [
                r'(your task is|please|instruction:)\s*([^.!?]+)',
                r'^([^.!?]+)(?:\.|!|\?|$)'
            ]
            for pattern in instruction_patterns:
                match = re.search(pattern, message.lower())
                if match:
                    # Use the last group (highest index) instead of -1
                    components["instruction"] = match.group(match.lastindex).strip()
                    break
        
        elif message_format == "constraint":
            # Extract constraints
            constraint_patterns = [
                r'(do not|never|always|must|should)\s+([^.!?]+)',
                r'(constraint:)\s*([^.!?]+)'
            ]
            constraints = []
            for pattern in constraint_patterns:
                matches = re.finditer(pattern, message.lower())
                for match in matches:
                    constraints.append(match.group(match.lastindex).strip())
            if constraints:
                components["constraints"] = constraints
        
        return components
    
    def _extract_special_instructions(self, message: str) -> Dict[str, Any]:
        """Extract special instructions and constraints."""
        special = {}
        
        # Extract constraints
        constraint_keywords = ["do not", "never", "always", "must", "should", "avoid", "ensure"]
        constraints = []
        
        for keyword in constraint_keywords:
            pattern = rf'\b{keyword}\s+([^.!?]+)'
            matches = re.finditer(pattern, message.lower())
            for match in matches:
                constraints.append(f"{keyword} {match.group(1).strip()}")
        
        if constraints:
            special["constraints"] = constraints
        
        # Extract tone/style instructions
        tone_keywords = ["tone", "style", "manner", "approach"]
        for keyword in tone_keywords:
            pattern = rf'{keyword}[:\s]+([^.!?]+)'
            match = re.search(pattern, message.lower())
            if match:
                special["tone"] = match.group(1).strip()
                break
        
        return special
    
    def _calculate_complexity_score(self, message: str) -> float:
        """Calculate complexity score for system message."""
        # Simple complexity scoring based on various factors
        score = 0.0
        
        # Length factor
        score += min(len(message) / 1000, 1.0) * 0.3
        
        # Word count factor
        word_count = len(message.split())
        score += min(word_count / 100, 1.0) * 0.2
        
        # Sentence complexity
        sentences = re.split(r'[.!?]+', message)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        score += min(avg_sentence_length / 20, 1.0) * 0.2
        
        # Special character density
        special_chars = len(re.findall(r'[^\w\s]', message))
        score += min(special_chars / len(message), 0.3) * 0.1
        
        # Constraint complexity
        constraint_keywords = ["do not", "never", "always", "must", "should", "avoid", "ensure"]
        constraint_count = sum(1 for keyword in constraint_keywords if keyword in message.lower())
        score += min(constraint_count / 5, 1.0) * 0.2
        
        return min(score, 1.0)
    
    def _validate_content(self, message: str) -> List[str]:
        """Validate message content for potential issues."""
        errors = []
        
        # Check for potentially harmful content patterns
        harmful_patterns = [
            r'(ignore|forget|disregard)\s+(previous|all|these)\s+(instructions|rules)',
            r'(tell me|show me|give me)\s+(password|secret|private)',
            r'(execute|run|eval)\s+(code|script|command)',
            r'ignore.*previous.*instructions',
            r'tell.*password'
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, message.lower()):
                errors.append(f"Potentially harmful content detected: {pattern}")
        
        # Check for excessive repetition
        words = message.lower().split()
        if len(words) > 10:
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            max_freq = max(word_freq.values())
            if max_freq > len(words) * 0.3:  # More than 30% repetition
                errors.append("Excessive word repetition detected")
        
        return errors
    
    def _has_invalid_characters(self, message: str) -> bool:
        """Check for invalid or potentially harmful characters."""
        # Define allowed character ranges
        allowed_pattern = r'^[\w\s\.,!?;:()\-\'"@#$%&*+=<>/\\|`~\[\]{}]*$'
        return not re.match(allowed_pattern, message)
    
    def _create_embeddings_from_tokens(self, token_ids: List[int]) -> np.ndarray:
        """
        Create embeddings from token IDs using a simple approach.
        
        This is a placeholder implementation. In practice, this could use
        more sophisticated embedding methods or integrate with SinusoidalEmbedder.
        """
        # Simple hash-based embedding creation
        embeddings = np.zeros(self.config.embedding_dim, dtype=np.float32)
        
        for i, token_id in enumerate(token_ids):
            # Create position-aware embedding
            position_weight = 1.0 / (1.0 + i * 0.1)  # Decay with position
            
            # Simple sinusoidal embedding based on token ID
            for j in range(self.config.embedding_dim):
                freq = (j + 1) / self.config.embedding_dim
                phase = (token_id % 1000) / 1000.0 * 2 * np.pi
                
                if j % 2 == 0:
                    embeddings[j] += np.sin(freq * phase) * position_weight
                else:
                    embeddings[j] += np.cos(freq * phase) * position_weight
        
        # Normalize
        norm = np.linalg.norm(embeddings)
        if norm > 0:
            embeddings = embeddings / norm
        
        return embeddings


# Convenience functions for easy usage

def create_system_message_processor(tokenizer_name: str = "gpt2",
                                  max_length: int = 512,
                                  embedding_dim: int = 256) -> SystemMessageProcessor:
    """
    Create a SystemMessageProcessor with default configuration.
    
    Args:
        tokenizer_name: Name of tokenizer to use
        max_length: Maximum sequence length
        embedding_dim: Embedding dimension
        
    Returns:
        Configured SystemMessageProcessor instance
    """
    tokenizer = StandardTokenizerWrapper(tokenizer_name, max_length)
    config = SystemMessageConfig(
        max_length=max_length,
        embedding_dim=embedding_dim
    )
    
    return SystemMessageProcessor(tokenizer, config)


def process_system_message_simple(message: str,
                                 tokenizer_name: str = "gpt2") -> SystemMessageContext:
    """
    Simple system message processing with default settings.
    
    Args:
        message: System message to process
        tokenizer_name: Tokenizer to use
        
    Returns:
        Processed SystemMessageContext
    """
    processor = create_system_message_processor(tokenizer_name)
    return processor.process_system_message(message)