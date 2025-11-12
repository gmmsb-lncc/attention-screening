"""
Binary labels generation for classification tasks.

Converts interaction labels into binary classification labels
based on configurable thresholds.
"""

import os
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pathlib import Path

from src.build.core import BuildConfig
from src.build.labels.base_labels import BaseLabels


class BinaryLabels(BaseLabels):
    """Generates binary labels from interaction data."""
    
    def __init__(self, 
                 config: BuildConfig, 
                 interaction_labels_path: Optional[Union[str, Path]] = None):
        """
        Initialize binary labels generator.
        
        Args:
            config: Build configuration
            interaction_labels_path: Path to interaction labels file
        """
        # Definir atributos ANTES de chamar super().__init__()
        # para garantir que estejam disponíveis em _validate_config()
        self.threshold: float = config.get('binary_threshold', 1000.0)  # 1000 nM default
        self.interaction_labels_path = interaction_labels_path
        self.interaction_data: Optional[np.ndarray] = None
        
        super().__init__(config)
    
    def _validate_config(self) -> None:
        """Valida configuração específica para labels binários."""
        super()._validate_config()
        
        # Validar threshold
        if self.threshold <= 0:
            raise ValueError("Threshold deve ser um valor positivo")
        
        # Verificar se arquivo de labels de interação existe
        if self.interaction_labels_path and not Path(self.interaction_labels_path).exists():
            self.logger.warning(f"Arquivo de labels de interação não encontrado: {self.interaction_labels_path}")
    
    def build(self) -> Dict[str, Any]:
        """Constrói labels binários."""
        self.logger.info("Construindo labels binários...")
        
        binary_labels = self.generate_labels()
        
        # Salvar se output_dir especificado
        if hasattr(self, 'output_dir') and self.output_dir:
            output_file = Path(self.output_dir) / "binary_labels.npy"
            np.save(output_file, binary_labels)
            self.logger.info(f"Labels binários salvos em: {output_file}")
        
        return {
            'binary_labels': binary_labels,
            'threshold': self.threshold,
            'total_pairs': len(binary_labels),
            'active_pairs': np.sum(binary_labels == 1),
            'inactive_pairs': np.sum(binary_labels == 0)
        }
        
    def generate_labels(self, 
                       interaction_data: Optional[np.ndarray] = None,
                       threshold: Optional[float] = None,
                       value_column_index: int = 3) -> np.ndarray:
        """
        Generate binary labels from interaction data.
        
        Args:
            interaction_data: Interaction data array, if None loads from file
            threshold: Threshold for binary classification (default: 1000 nM)
            value_column_index: Index of the standard_value column
            
        Returns:
            Generated binary labels (1 for active, 0 for inactive)
        """
        try:
            # Use provided threshold or default
            threshold = threshold if threshold is not None else self.threshold
            
            # Load interaction data if not provided
            if interaction_data is None:
                if self.interaction_labels_path is None:
                    raise ValueError("No interaction data provided and no path specified")
                
                interaction_data = np.load(self.interaction_labels_path, allow_pickle=True)
            
            self.interaction_data = interaction_data
            
            self.logger.info(f"Generating binary labels with threshold {threshold} nM")
            
            # Extract standard values (column index 3 by default)
            standard_values = interaction_data[:, value_column_index]
            
            # Convert to float, handling potential errors
            numeric_values = []
            valid_indices = []
            
            for i, val in enumerate(standard_values):
                try:
                    numeric_val = float(val)
                    numeric_values.append(numeric_val)
                    valid_indices.append(i)
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid value at index {i}: {val}")
                    continue
            
            if len(numeric_values) == 0:
                raise ValueError("No valid numeric values found in interaction data")
            
            numeric_values = np.array(numeric_values)
            
            # Apply threshold rule: <= threshold = active (1), > threshold = inactive (0)
            binary_labels_valid = np.where(numeric_values <= threshold, 1, 0)
            
            # Create full binary labels array with -1 for invalid entries
            full_binary_labels = np.full(len(standard_values), -1, dtype=int)
            full_binary_labels[valid_indices] = binary_labels_valid
            
            self.labels = full_binary_labels
            
            # Store statistics
            self.statistics = {
                'threshold_nm': threshold,
                'total_samples': len(self.labels),
                'valid_samples': len(numeric_values),
                'invalid_samples': len(self.labels) - len(numeric_values),
                'active_count': np.sum(binary_labels_valid == 1),
                'inactive_count': np.sum(binary_labels_valid == 0),
                'value_range': {
                    'min': float(np.min(numeric_values)),
                    'max': float(np.max(numeric_values)),
                    'mean': float(np.mean(numeric_values)),
                    'median': float(np.median(numeric_values))
                }
            }
            
            self.logger.info(f"Generated binary labels: {self.statistics['active_count']} active, "
                           f"{self.statistics['inactive_count']} inactive "
                           f"({self.statistics['invalid_samples']} invalid)")
            
            return self.labels
            
        except Exception as e:
            self.logger.error(f"Error generating binary labels: {e}")
            raise
    
    def validate_labels(self) -> bool:
        """
        Validate generated binary labels.
        
        Returns:
            True if labels are valid, False otherwise
        """
        if self.labels is None:
            self.logger.error("No labels to validate")
            return False
        
        try:
            # Check if we have data
            if len(self.labels) == 0:
                self.logger.error("Empty labels array")
                return False
            
            # Check for valid binary values (0, 1, -1 for invalid)
            unique_values = np.unique(self.labels)
            valid_values = set([-1, 0, 1])
            
            if not set(unique_values).issubset(valid_values):
                self.logger.error(f"Invalid label values found: {unique_values}")
                return False
            
            # Check if we have at least some valid labels
            valid_labels = self.labels[self.labels != -1]
            if len(valid_labels) == 0:
                self.logger.error("No valid binary labels found")
                return False
            
            # Check for class imbalance
            active_count = np.sum(valid_labels == 1)
            inactive_count = np.sum(valid_labels == 0)
            
            if active_count == 0 or inactive_count == 0:
                self.logger.warning("Severe class imbalance detected")
            
            self.logger.info(f"Validation passed: {len(valid_labels)} valid binary labels")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating binary labels: {e}")
            return False
    
    def get_valid_labels(self) -> np.ndarray:
        """
        Get only the valid binary labels (excluding -1 values).
        
        Returns:
            Array of valid binary labels
        """
        if self.labels is None:
            return np.array([])
        
        return self.labels[self.labels != -1]
    
    def get_valid_indices(self) -> np.ndarray:
        """
        Get indices of valid binary labels.
        
        Returns:
            Array of indices for valid labels
        """
        if self.labels is None:
            return np.array([])
        
        return np.where(self.labels != -1)[0]
    
    def apply_different_threshold(self, new_threshold: float) -> np.ndarray:
        """
        Apply a different threshold to existing interaction data.
        
        Args:
            new_threshold: New threshold value in nM
            
        Returns:
            New binary labels with different threshold
        """
        if self.interaction_data is None:
            raise ValueError("No interaction data available. Generate labels first.")
        
        return self.generate_labels(
            interaction_data=self.interaction_data,
            threshold=new_threshold
        )
    
    def get_class_balance_info(self) -> Dict[str, Any]:
        """
        Get detailed class balance information.
        
        Returns:
            Dictionary with class balance statistics
        """
        if self.labels is None:
            return {}
        
        valid_labels = self.get_valid_labels()
        if len(valid_labels) == 0:
            return {}
        
        active_count = np.sum(valid_labels == 1)
        inactive_count = np.sum(valid_labels == 0)
        total_valid = len(valid_labels)
        
        return {
            'active_count': int(active_count),
            'inactive_count': int(inactive_count),
            'total_valid': int(total_valid),
            'active_percentage': float(active_count / total_valid * 100),
            'inactive_percentage': float(inactive_count / total_valid * 100),
            'balance_ratio': float(min(active_count, inactive_count) / max(active_count, inactive_count)),
            'is_balanced': abs(active_count - inactive_count) / total_valid < 0.2  # Within 20%
        }
    
    def create_stratified_splits(self, 
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               random_state: int = 42) -> Dict[str, np.ndarray]:
        """
        Create stratified splits maintaining class balance.
        
        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set  
            test_ratio: Ratio for test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with split indices
        """
        if self.labels is None:
            return {}
        
        try:
            from sklearn.model_selection import train_test_split
            
            # Get valid indices and labels
            valid_indices = self.get_valid_indices()
            valid_labels = self.get_valid_labels()
            
            if len(valid_indices) == 0:
                return {}
            
            # Create train/temp split
            train_indices, temp_indices, _, temp_labels = train_test_split(
                valid_indices, valid_labels,
                test_size=(val_ratio + test_ratio),
                stratify=valid_labels,
                random_state=random_state
            )
            
            # Create val/test split from temp
            val_test_ratio = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                test_size=(1 - val_test_ratio),
                stratify=temp_labels,
                random_state=random_state
            )
            
            return {
                'train': train_indices,
                'val': val_indices,
                'test': test_indices
            }
            
        except ImportError:
            self.logger.warning("sklearn not available, using simple random splits")
            return self._create_simple_splits(train_ratio, val_ratio, test_ratio, random_state)
    
    def _create_simple_splits(self, 
                            train_ratio: float,
                            val_ratio: float, 
                            test_ratio: float,
                            random_state: int) -> Dict[str, np.ndarray]:
        """Simple random splits without stratification."""
        valid_indices = self.get_valid_indices()
        
        if len(valid_indices) == 0:
            return {}
        
        np.random.seed(random_state)
        shuffled_indices = np.random.permutation(valid_indices)
        
        n_train = int(len(shuffled_indices) * train_ratio)
        n_val = int(len(shuffled_indices) * val_ratio)
        
        return {
            'train': shuffled_indices[:n_train],
            'val': shuffled_indices[n_train:n_train + n_val],
            'test': shuffled_indices[n_train + n_val:]
        }
