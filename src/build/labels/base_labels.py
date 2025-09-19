"""
Base interface for label generation.

Provides abstract base class for different types of label builders
in the protein-ligand interaction prediction pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path

from build.core import BaseBuilder, BuildConfig


class BaseLabels(BaseBuilder):
    """Abstract base class for label generation."""
    
    def __init__(self, config: BuildConfig):
        """Initialize label builder."""
        super().__init__(config)
        self.labels: Optional[np.ndarray] = None
        self.label_mapping: Dict[str, int] = {}
        self.statistics: Dict[str, Any] = {}
    
    def _validate_config(self) -> None:
        """Valida configuração base para geração de labels."""
        super()._validate_config()
        # Validações específicas para labels podem ser adicionadas aqui
    
    def build(self, **kwargs) -> Dict[str, Any]:
        """Build method that delegates to generate_labels."""
        labels = self.generate_labels(**kwargs)
        self.labels = labels
        return {
            'labels': labels,
            'statistics': self.get_label_statistics(),
            'label_mapping': self.label_mapping
        }
    
    @abstractmethod
    def generate_labels(self, **kwargs) -> np.ndarray:
        """
        Generate labels for the dataset.
        
        Returns:
            Generated labels as numpy array
        """
        pass
    
    @abstractmethod
    def validate_labels(self) -> bool:
        """
        Validate generated labels.
        
        Returns:
            True if labels are valid, False otherwise
        """
        pass
    
    def get_label_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the generated labels.
        
        Returns:
            Dictionary with label statistics
        """
        if self.labels is None:
            return {}
        
        stats = {
            'total_samples': len(self.labels),
            'unique_labels': len(np.unique(self.labels)),
            'label_distribution': {}
        }
        
        # Calculate distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        for label, count in zip(unique, counts):
            stats['label_distribution'][str(label)] = {
                'count': int(count),
                'percentage': float(count / len(self.labels) * 100)
            }
        
        return stats
    
    def save_labels(self, output_path: Union[str, Path]) -> bool:
        """
        Save generated labels to file.
        
        Args:
            output_path: Path to save labels
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.labels is None:
                self.logger.error("No labels generated to save")
                return False
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy array
            np.save(str(output_path.with_suffix('.npy')), self.labels)
            
            # Save statistics
            stats_path = output_path.with_suffix('.json')
            self.save_json(self.get_label_statistics(), stats_path)
            
            self.logger.info(f"Labels saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving labels: {e}")
            return False
    
    def load_labels(self, input_path: Union[str, Path]) -> bool:
        """
        Load labels from file.
        
        Args:
            input_path: Path to load labels from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            input_path = Path(input_path)
            
            if input_path.with_suffix('.npy').exists():
                self.labels = np.load(str(input_path.with_suffix('.npy')))
                self.logger.info(f"Labels loaded from {input_path}")
                return True
            else:
                self.logger.error(f"Labels file not found: {input_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading labels: {e}")
            return False
    
    def get_class_weights(self) -> Dict[int, float]:
        """
        Calculate class weights for imbalanced datasets.
        
        Returns:
            Dictionary mapping class labels to weights
        """
        if self.labels is None:
            return {}
        
        try:
            from sklearn.utils.class_weight import compute_class_weight
            
            classes = np.unique(self.labels)
            weights = compute_class_weight(
                'balanced',
                classes=classes,
                y=self.labels
            )
            
            return dict(zip(classes, weights))
            
        except ImportError:
            self.logger.warning("sklearn not available for class weight computation")
            # Simple inverse frequency weighting
            unique, counts = np.unique(self.labels, return_counts=True)
            total = len(self.labels)
            weights = total / (len(unique) * counts)
            return dict(zip(unique, weights))
    
    def split_labels(self, indices: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
        """
        Split labels according to provided indices.
        
        Args:
            indices: Dictionary with 'train', 'val', 'test' keys and index lists
            
        Returns:
            Dictionary with split labels
        """
        if self.labels is None:
            return {}
        
        splits = {}
        for split_name, split_indices in indices.items():
            if len(split_indices) > 0:
                splits[split_name] = self.labels[split_indices]
        
        return splits
