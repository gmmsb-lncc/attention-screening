"""
SplitIndices: Immutable data class for train/validation/test split indices.

This module provides a type-safe, immutable container for storing dataset split
indices. It ensures data integrity through validation and provides save/load
functionality for reproducibility.

Design Principles:
- Single Responsibility: Only stores and validates split indices
- Immutability: Created once, never modified
- Type Safety: Enforces int32 dtype for indices
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path
import numpy as np
import json
from datetime import datetime


@dataclass(frozen=True)
class SplitIndices:
    """
    Immutable container for train/validation/test split indices.
    
    This class stores dataset split indices with validation to ensure:
    - No overlap between train/val/test sets
    - All indices are unique within each set
    - All sets (except validation) are non-empty
    - Indices are in int32 format for compatibility
    
    Attributes:
        train_idx: Training set indices (np.ndarray, dtype=int32)
        val_idx: Validation set indices (np.ndarray, dtype=int32)
        test_idx: Test set indices (np.ndarray, dtype=int32)
        metadata: Additional information (dict)
    
    Example:
        >>> train_idx = np.array([0, 1, 2, 3], dtype=np.int32)
        >>> val_idx = np.array([4, 5], dtype=np.int32)
        >>> test_idx = np.array([6, 7], dtype=np.int32)
        >>> splits = SplitIndices(train_idx, val_idx, test_idx)
        >>> splits.save('splits.npz')
        >>> loaded = SplitIndices.load('splits.npz')
    """
    
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """
        Validate and prepare indices after initialization.
        
        Performs:
        1. Type conversion to int32
        2. Validation of uniqueness
        3. Validation of non-overlap
        4. Validation of non-empty sets
        5. Make arrays read-only (immutability)
        6. Add default metadata
        
        Raises:
            ValueError: If validation fails
        """
        # Convert to int32 and make read-only
        train_idx = np.asarray(self.train_idx, dtype=np.int32)
        val_idx = np.asarray(self.val_idx, dtype=np.int32)
        test_idx = np.asarray(self.test_idx, dtype=np.int32)
        
        # Validate non-empty (except validation which can be empty)
        if len(train_idx) == 0:
            raise ValueError("Train set cannot be empty")
        if len(test_idx) == 0:
            raise ValueError("Test set cannot be empty")
        
        # Validate uniqueness within each set
        if len(train_idx) != len(np.unique(train_idx)):
            raise ValueError("Train indices must be unique")
        if len(val_idx) > 0 and len(val_idx) != len(np.unique(val_idx)):
            raise ValueError("Validation indices must be unique")
        if len(test_idx) != len(np.unique(test_idx)):
            raise ValueError("Test indices must be unique")
        
        # Validate no overlap between sets
        train_set = set(train_idx.tolist())
        val_set = set(val_idx.tolist())
        test_set = set(test_idx.tolist())
        
        if train_set & val_set:
            raise ValueError(f"Train and validation sets have overlap: {train_set & val_set}")
        if train_set & test_set:
            raise ValueError(f"Train and test sets have overlap: {train_set & test_set}")
        if val_set & test_set:
            raise ValueError(f"Validation and test sets have overlap: {val_set & test_set}")
        
        # Make arrays read-only (immutability)
        train_idx.flags.writeable = False
        val_idx.flags.writeable = False
        test_idx.flags.writeable = False
        
        # Update arrays using object.__setattr__ (frozen dataclass workaround)
        object.__setattr__(self, 'train_idx', train_idx)
        object.__setattr__(self, 'val_idx', val_idx)
        object.__setattr__(self, 'test_idx', test_idx)
        
        # Add default metadata if not provided
        if not self.metadata:
            metadata = {}
        else:
            metadata = dict(self.metadata)  # Copy to avoid mutation
        
        # Add size information
        metadata.setdefault('train_size', len(train_idx))
        metadata.setdefault('val_size', len(val_idx))
        metadata.setdefault('test_size', len(test_idx))
        metadata.setdefault('total_size', len(train_idx) + len(val_idx) + len(test_idx))
        metadata.setdefault('created_at', datetime.now().isoformat())
        
        # Update metadata using object.__setattr__
        object.__setattr__(self, 'metadata', metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format.
        
        Returns:
            Dictionary with train_idx, val_idx, test_idx, and metadata
        
        Example:
            >>> splits_dict = splits.to_dict()
            >>> splits_dict.keys()
            dict_keys(['train_idx', 'val_idx', 'test_idx', 'metadata'])
        """
        return {
            'train_idx': self.train_idx,
            'val_idx': self.val_idx,
            'test_idx': self.test_idx,
            'metadata': self.metadata.copy()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SplitIndices':
        """
        Create SplitIndices from dictionary.
        
        Args:
            data: Dictionary with train_idx, val_idx, test_idx, and metadata
        
        Returns:
            New SplitIndices instance
        
        Example:
            >>> splits_dict = {'train_idx': [...], 'val_idx': [...], ...}
            >>> splits = SplitIndices.from_dict(splits_dict)
        """
        return cls(
            train_idx=data['train_idx'],
            val_idx=data['val_idx'],
            test_idx=data['test_idx'],
            metadata=data.get('metadata', {})
        )
    
    def save(self, filepath: str) -> None:
        """
        Save split indices to .npz file.
        
        Saves indices and metadata in compressed NumPy format.
        Metadata is stored as JSON string in the .npz file.
        
        Args:
            filepath: Path where to save (should end with .npz)
        
        Example:
            >>> splits.save('results/splits.npz')
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert metadata to JSON string for storage
        metadata_json = json.dumps(self.metadata)
        
        # Save to compressed format
        np.savez_compressed(
            filepath,
            train_idx=self.train_idx,
            val_idx=self.val_idx,
            test_idx=self.test_idx,
            metadata=np.array([metadata_json], dtype=object)
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'SplitIndices':
        """
        Load split indices from .npz file.
        
        Args:
            filepath: Path to .npz file
        
        Returns:
            Loaded SplitIndices instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
        
        Example:
            >>> splits = SplitIndices.load('results/splits.npz')
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Split file not found: {filepath}")
        
        # Load from .npz file
        data = np.load(filepath, allow_pickle=True)
        
        # Extract metadata
        metadata_json = str(data['metadata'][0])
        metadata = json.loads(metadata_json)
        
        return cls(
            train_idx=data['train_idx'],
            val_idx=data['val_idx'],
            test_idx=data['test_idx'],
            metadata=metadata
        )
    
    def get_split_proportions(self) -> Dict[str, float]:
        """
        Calculate proportion of each split.
        
        Returns:
            Dictionary with train/val/test proportions (0-1)
        
        Example:
            >>> splits.get_split_proportions()
            {'train': 0.7, 'val': 0.1, 'test': 0.2}
        """
        total = self.metadata['total_size']
        return {
            'train': self.metadata['train_size'] / total,
            'val': self.metadata['val_size'] / total,
            'test': self.metadata['test_size'] / total
        }
    
    def __repr__(self) -> str:
        """String representation with split sizes."""
        props = self.get_split_proportions()
        return (
            f"SplitIndices("
            f"train={len(self.train_idx)} ({props['train']:.1%}), "
            f"val={len(self.val_idx)} ({props['val']:.1%}), "
            f"test={len(self.test_idx)} ({props['test']:.1%})"
            f")"
        )
    
    def __str__(self) -> str:
        """User-friendly string representation."""
        return self.__repr__()
