"""
Interaction labels generation for protein-ligand interactions.

Converts interaction data from TSV files into structured labels
using Spark for efficient processing.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from ..core import BuildConfig
from ..utils import SparkManager
from .base_labels import BaseLabels


class InteractionLabels(BaseLabels):
    """Generates interaction labels from TSV data using Spark."""
    
    def __init__(self, config: BuildConfig, tsv_path: Union[str, Path]):
        """
        Initialize interaction labels generator.
        
        Args:
            config: Build configuration
            tsv_path: Path to input TSV file
        """
        super().__init__(config)
        self.tsv_path = Path(tsv_path)
        self.spark_manager = SparkManager(config)
        self.interaction_data: Optional[np.ndarray] = None
        self.raw_labels: Optional[np.ndarray] = None
    
    def _validate_config(self) -> None:
        """Validate interaction labels configuration."""
        # Implementação base vazia - validação específica pode ser adicionada aqui
        pass
    
    def build(self) -> np.ndarray:
        """
        Build interaction labels from TSV data.
        
        Returns:
            Generated interaction labels
        """
        return self.generate_labels()
        
    def generate_labels(self, 
                       molregno_col: str = "molregno",
                       kinase_col: str = "target_kinase", 
                       type_col: str = "standard_type",
                       value_col: str = "standard_value") -> np.ndarray:
        """
        Generate interaction labels from TSV data.
        
        Args:
            molregno_col: Column name for molecule registry number
            kinase_col: Column name for target kinase
            type_col: Column name for standard type
            value_col: Column name for standard value
            
        Returns:
            Generated interaction labels
        """
        try:
            self.logger.info(f"Generating interaction labels from {self.tsv_path}")
            
            # Initialize Spark
            spark = self.spark_manager.get_session("InteractionLabelsProcessing")
            
            # Read TSV file
            df = spark.read.csv(
                str(self.tsv_path), 
                sep='\t', 
                header=True, 
                inferSchema=True
            )
            
            # Select relevant columns
            df_selected = df.select(molregno_col, kinase_col, type_col, value_col)
            
            # Collect data
            labels_data = df_selected.collect()
            
            # Convert to numpy array
            self.interaction_data = np.array(labels_data)
            self.raw_labels = self.interaction_data
            
            # Store structured labels (molregno, kinase, type, value)
            self.labels = self.interaction_data
            
            self.logger.info(f"Generated {len(self.labels)} interaction labels")
            return self.labels
            
        except Exception as e:
            self.logger.error(f"Error generating interaction labels: {e}")
            raise
        finally:
            self.spark_manager.stop()
    
    def validate_labels(self) -> bool:
        """
        Validate generated interaction labels.
        
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
            
            # Check if we have 4 columns (molregno, kinase, type, value)
            if len(self.labels.shape) != 2 or self.labels.shape[1] != 4:
                self.logger.error("Labels should have 4 columns")
                return False
            
            # Check for missing values in standard_value column
            standard_values = self.labels[:, 3]
            valid_values = 0
            for val in standard_values:
                try:
                    float(val)
                    valid_values += 1
                except (ValueError, TypeError):
                    continue
            
            if valid_values == 0:
                self.logger.error("No valid standard values found")
                return False
            
            self.logger.info(f"Validation passed: {valid_values}/{len(standard_values)} valid values")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating labels: {e}")
            return False
    
    def get_unique_kinases(self) -> List[str]:
        """
        Get list of unique kinases in the dataset.
        
        Returns:
            List of unique kinase names
        """
        if self.labels is None:
            return []
        
        try:
            kinases = self.labels[:, 1]  # kinase column
            unique_kinases = list(set(str(k) for k in kinases if k is not None))
            return sorted(unique_kinases)
        except Exception as e:
            self.logger.error(f"Error getting unique kinases: {e}")
            return []
    
    def get_unique_compounds(self) -> List[str]:
        """
        Get list of unique compounds in the dataset.
        
        Returns:
            List of unique molregno values
        """
        if self.labels is None:
            return []
        
        try:
            compounds = self.labels[:, 0]  # molregno column
            unique_compounds = list(set(str(c) for c in compounds if c is not None))
            return sorted(unique_compounds)
        except Exception as e:
            self.logger.error(f"Error getting unique compounds: {e}")
            return []
    
    def filter_by_kinase(self, kinase_name: str) -> np.ndarray:
        """
        Filter interactions for specific kinase.
        
        Args:
            kinase_name: Name of kinase to filter by
            
        Returns:
            Filtered interaction data
        """
        if self.labels is None:
            return np.array([])
        
        try:
            kinase_mask = self.labels[:, 1] == kinase_name
            return self.labels[kinase_mask]
        except Exception as e:
            self.logger.error(f"Error filtering by kinase: {e}")
            return np.array([])
    
    def filter_by_standard_type(self, standard_type: str) -> np.ndarray:
        """
        Filter interactions by standard type (e.g., 'IC50', 'Ki').
        
        Args:
            standard_type: Type to filter by
            
        Returns:
            Filtered interaction data
        """
        if self.labels is None:
            return np.array([])
        
        try:
            type_mask = self.labels[:, 2] == standard_type
            return self.labels[type_mask]
        except Exception as e:
            self.logger.error(f"Error filtering by standard type: {e}")
            return np.array([])
    
    def get_interaction_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about interactions.
        
        Returns:
            Dictionary with interaction statistics
        """
        stats = super().get_label_statistics()
        
        if self.labels is None:
            return stats
        
        try:
            # Add interaction-specific stats
            stats.update({
                'unique_kinases': len(self.get_unique_kinases()),
                'unique_compounds': len(self.get_unique_compounds()),
                'kinase_distribution': {},
                'standard_type_distribution': {},
                'value_statistics': {}
            })
            
            # Kinase distribution
            kinases, kinase_counts = np.unique(self.labels[:, 1], return_counts=True)
            for kinase, count in zip(kinases, kinase_counts):
                stats['kinase_distribution'][str(kinase)] = int(count)
            
            # Standard type distribution
            types, type_counts = np.unique(self.labels[:, 2], return_counts=True)
            for stype, count in zip(types, type_counts):
                stats['standard_type_distribution'][str(stype)] = int(count)
            
            # Value statistics (for numeric values)
            try:
                numeric_values = []
                for val in self.labels[:, 3]:
                    try:
                        numeric_values.append(float(val))
                    except (ValueError, TypeError):
                        continue
                
                if numeric_values:
                    numeric_values = np.array(numeric_values)
                    stats['value_statistics'] = {
                        'mean': float(np.mean(numeric_values)),
                        'median': float(np.median(numeric_values)),
                        'std': float(np.std(numeric_values)),
                        'min': float(np.min(numeric_values)),
                        'max': float(np.max(numeric_values)),
                        'valid_count': len(numeric_values)
                    }
            except Exception as e:
                self.logger.warning(f"Could not compute value statistics: {e}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error computing interaction statistics: {e}")
            return stats
