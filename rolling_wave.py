import numpy as np
from typing import Optional

class RollingWaveBuffer:
    """
    Rolling wave buffer that maintains a 2D temporal pattern of LSM outputs.
    Each new wave is time-shifted and stored in the next row.
    """
    
    def __init__(self, window_size: int):
        """
        Initialize the rolling wave buffer.
        
        Args:
            window_size: Size of the temporal window (creates window_size x window_size buffer)
        """
        self.window_size = window_size
        self.buffer = np.zeros((window_size, window_size), dtype=np.float32)
        self.current_row = 0
        self.timestep = 0
        self.is_full = False
        
    def reset(self):
        """Reset the buffer to initial state."""
        self.buffer.fill(0.0)
        self.current_row = 0
        self.timestep = 0
        self.is_full = False
        
    def append_wave(self, wave: np.ndarray) -> None:
        """
        Append a new wave to the buffer with time-shifting.
        
        Args:
            wave: 1D numpy array representing the current timestep's wave
        """
        if len(wave.shape) != 1:
            raise ValueError("Wave must be a 1D array")
        
        # Create time-shifted wave
        shifted_wave = self._create_shifted_wave(wave)
        
        # Store in current row
        self.buffer[self.current_row] = shifted_wave
        
        # Update pointers
        self.current_row = (self.current_row + 1) % self.window_size
        self.timestep += 1
        
        # Mark as full once we've filled all rows at least once
        if self.timestep >= self.window_size:
            self.is_full = True
    
    def _create_shifted_wave(self, wave: np.ndarray) -> np.ndarray:
        """
        Create time-shifted wave by prepending zeros and truncating.
        
        Args:
            wave: Original wave to be shifted
            
        Returns:
            Shifted and truncated wave of size window_size
        """
        shift_amount = self.timestep % self.window_size
        
        # Create shifted wave with zero padding
        if shift_amount > 0:
            zeros = np.zeros(shift_amount, dtype=np.float32)
            shifted = np.concatenate([zeros, wave])
        else:
            shifted = wave.copy()
        
        # Truncate to window_size
        if len(shifted) > self.window_size:
            shifted = shifted[:self.window_size]
        elif len(shifted) < self.window_size:
            # Pad with zeros if needed
            padding = np.zeros(self.window_size - len(shifted), dtype=np.float32)
            shifted = np.concatenate([shifted, padding])
        
        return shifted
    
    def get_buffer(self) -> np.ndarray:
        """
        Get the current buffer state.
        
        Returns:
            2D array of shape (window_size, window_size)
        """
        if not self.is_full:
            # If buffer isn't full, return only the filled rows
            filled_rows = min(self.timestep, self.window_size)
            result = np.zeros((self.window_size, self.window_size), dtype=np.float32)
            
            # Copy filled rows in correct order
            for i in range(filled_rows):
                row_idx = (self.current_row - filled_rows + i) % self.window_size
                result[i] = self.buffer[row_idx]
            
            return result
        else:
            # If buffer is full, reorder to maintain temporal sequence
            ordered_buffer = np.zeros_like(self.buffer)
            for i in range(self.window_size):
                source_idx = (self.current_row + i) % self.window_size
                ordered_buffer[i] = self.buffer[source_idx]
            return ordered_buffer
    
    def get_buffer_3d(self) -> np.ndarray:
        """
        Get buffer as 3D array suitable for CNN input.
        
        Returns:
            3D array of shape (window_size, window_size, 1)
        """
        buffer_2d = self.get_buffer()
        return np.expand_dims(buffer_2d, axis=-1)
    
    def get_current_pattern(self) -> np.ndarray:
        """
        Get the current 2D pattern for visualization or analysis.
        
        Returns:
            Current buffer as 2D array
        """
        return self.get_buffer()
    
    def is_ready(self) -> bool:
        """
        Check if buffer has received enough waves to be meaningful.
        
        Returns:
            True if at least one full cycle has been completed
        """
        return self.timestep >= self.window_size
    
    def get_temporal_stats(self) -> dict:
        """
        Get statistics about the temporal patterns in the buffer.
        
        Returns:
            Dictionary with statistics
        """
        buffer = self.get_buffer()
        
        stats = {
            'mean': float(np.mean(buffer)),
            'std': float(np.std(buffer)),
            'min': float(np.min(buffer)),
            'max': float(np.max(buffer)),
            'energy': float(np.sum(buffer ** 2)),
            'sparsity': float(np.mean(buffer == 0)),
            'filled_ratio': min(self.timestep / self.window_size, 1.0)
        }
        
        return stats

class MultiChannelRollingWaveBuffer:
    """
    Extended version supporting multiple channels for different reservoir layer outputs.
    """
    
    def __init__(self, window_size: int, num_channels: int = 1):
        """
        Initialize multi-channel rolling wave buffer.
        
        Args:
            window_size: Size of temporal window
            num_channels: Number of channels (e.g., one per reservoir layer)
        """
        self.window_size = window_size
        self.num_channels = num_channels
        self.buffers = [RollingWaveBuffer(window_size) for _ in range(num_channels)]
        
    def reset(self):
        """Reset all channel buffers."""
        for buffer in self.buffers:
            buffer.reset()
    
    def append_waves(self, waves: list) -> None:
        """
        Append waves to multiple channels.
        
        Args:
            waves: List of 1D arrays, one for each channel
        """
        if len(waves) != self.num_channels:
            raise ValueError(f"Expected {self.num_channels} waves, got {len(waves)}")
        
        for i, wave in enumerate(waves):
            self.buffers[i].append_wave(wave)
    
    def get_buffer_3d(self) -> np.ndarray:
        """
        Get combined buffer as 3D array for CNN input.
        
        Returns:
            3D array of shape (window_size, window_size, num_channels)
        """
        channel_buffers = []
        for buffer in self.buffers:
            channel_buffers.append(buffer.get_buffer())
        
        # Stack along channel dimension
        return np.stack(channel_buffers, axis=-1)
    
    def is_ready(self) -> bool:
        """Check if all channels are ready."""
        return all(buffer.is_ready() for buffer in self.buffers)
    
    def get_combined_stats(self) -> dict:
        """Get statistics across all channels."""
        all_stats = [buffer.get_temporal_stats() for buffer in self.buffers]
        
        combined_stats = {}
        for key in all_stats[0].keys():
            values = [stats[key] for stats in all_stats]
            combined_stats[f'{key}_mean'] = np.mean(values)
            combined_stats[f'{key}_std'] = np.std(values)
            combined_stats[f'{key}_min'] = np.min(values)
            combined_stats[f'{key}_max'] = np.max(values)
        
        return combined_stats

if __name__ == "__main__":
    # Test rolling wave buffer
    print("Testing RollingWaveBuffer...")
    
    # Test basic functionality
    buffer = RollingWaveBuffer(window_size=5)
    
    # Test with different wave patterns
    test_waves = [
        np.array([1.0, 0.5, 0.2]),
        np.array([0.8, 1.0, 0.3, 0.1]),
        np.array([0.2, 0.9, 0.7]),
        np.array([1.0, 0.1, 0.6, 0.4, 0.2]),
        np.array([0.5, 0.8, 0.3])
    ]
    
    print("Appending waves:")
    for i, wave in enumerate(test_waves):
        print(f"Wave {i}: {wave}")
        buffer.append_wave(wave)
        print(f"Buffer state:\n{buffer.get_buffer()}")
        print(f"Is ready: {buffer.is_ready()}")
        print(f"Stats: {buffer.get_temporal_stats()}")
        print()
    
    # Test 3D output
    buffer_3d = buffer.get_buffer_3d()
    print(f"3D buffer shape: {buffer_3d.shape}")
    
    # Test multi-channel buffer
    print("\nTesting MultiChannelRollingWaveBuffer...")
    multi_buffer = MultiChannelRollingWaveBuffer(window_size=3, num_channels=2)
    
    test_multichannel_waves = [
        [np.array([1.0, 0.5]), np.array([0.2, 0.8])],
        [np.array([0.8, 0.3]), np.array([0.9, 0.1])],
        [np.array([0.6, 0.7]), np.array([0.4, 0.5])]
    ]
    
    for i, waves in enumerate(test_multichannel_waves):
        multi_buffer.append_waves(waves)
        print(f"Step {i}, Multi-channel buffer shape: {multi_buffer.get_buffer_3d().shape}")
    
    print("RollingWaveBuffer tests completed successfully!")
