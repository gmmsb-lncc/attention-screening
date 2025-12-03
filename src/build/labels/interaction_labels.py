"""
Interaction labels generation for protein-ligand interactions.

Converts interaction data from TSV files into structured labels
using Spark for efficient processing.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from pathlib import Path

from src.build.core import BuildConfig
from src.build.utils import SparkManager
from src.build.labels.base_labels import BaseLabels


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
                       value_col: str = "standard_value",
                       pchembl_col: str = "pchembl_value") -> np.ndarray:
        """
        Generate interaction labels from TSV data.
        
        Args:
            molregno_col: Column name for molecule registry number
            kinase_col: Column name for target kinase
            type_col: Column name for standard type
            value_col: Column name for standard value (Ki/IC50/Kd in nM)
            pchembl_col: Column name for pChEMBL value (-log10(M))
            
        Returns:
            Generated interaction labels with columns:
            [molregno, kinase, type, standard_value, pchembl_value]
            
        Note:
            pchembl_value is preferred for regression as it provides
            a logarithmic scale. If pchembl_value is missing, it will
            be calculated as: 9 - log10(standard_value_nM)
        """
        try:
            self.logger.info(f"Generating interaction labels from {self.tsv_path}")
            
            # Initialize Spark session
            self.spark_manager.start()
            spark = self.spark_manager.get_session()
            
            # Read TSV file
            df = spark.read.csv(
                str(self.tsv_path), 
                sep='\t', 
                header=True, 
                inferSchema=True
            )
            
            # Check if pchembl_value column exists
            available_columns = df.columns
            has_pchembl = pchembl_col in available_columns
            
            if has_pchembl:
                # Select with pchembl_value as 5th column
                df_selected = df.select(molregno_col, kinase_col, type_col, value_col, pchembl_col)
                self.logger.info(f"Including pchembl_value column for regression")
            else:
                # Fallback: only 4 columns (pchembl will be calculated later)
                df_selected = df.select(molregno_col, kinase_col, type_col, value_col)
                self.logger.warning(f"pchembl_value column not found, will be calculated from standard_value")
            
            # Collect data
            labels_data = df_selected.collect()
            
            # Convert to numpy array
            self.interaction_data = np.array(labels_data, dtype=object)
            self.raw_labels = self.interaction_data
            
            # Fill missing pchembl_value by calculating from standard_value
            # pchembl_value = 9 - log10(standard_value_nM)
            if self.interaction_data.shape[1] == 5:
                self._fill_missing_pchembl_values()
            elif self.interaction_data.shape[1] == 4:
                # Add pchembl column calculated from standard_value
                self._add_pchembl_column()
            
            # Store structured labels
            self.labels = self.interaction_data
            
            self.logger.info(f"Generated {len(self.labels)} interaction labels with {self.labels.shape[1]} columns")
            return self.labels
            
        except Exception as e:
            self.logger.error(f"Error generating interaction labels: {e}")
            raise
        finally:
            self.spark_manager.stop()
    
    def _calculate_pchembl_from_nm(self, value_nm) -> Optional[float]:
        """
        Calculate pChEMBL value from standard_value in nM.
        
        Formula: pChEMBL = 9 - log10(nM) = -log10(M)
        
        Args:
            value_nm: Activity value in nM (nanomolar)
            
        Returns:
            pChEMBL value or None if calculation fails
        """
        try:
            if value_nm is None:
                return None
            
            value_nm_float = float(value_nm)
            
            if value_nm_float <= 0:
                self.logger.debug(f"Invalid value for pChEMBL calculation: {value_nm}")
                return None
            
            # pChEMBL = 9 - log10(nM) = -log10(nM * 1e-9) = -log10(M)
            pchembl = 9 - np.log10(value_nm_float)
            
            # Sanity check: pChEMBL values typically range from ~3 to ~12
            if pchembl < 0 or pchembl > 15:
                self.logger.debug(f"Unusual pChEMBL value {pchembl:.2f} from {value_nm} nM")
            
            return round(pchembl, 2)
            
        except (ValueError, TypeError) as e:
            self.logger.debug(f"Could not calculate pChEMBL from {value_nm}: {e}")
            return None
    
    def _fill_missing_pchembl_values(self) -> None:
        """
        Fill missing pChEMBL values by calculating from standard_value.
        
        IMPORTANT: pchembl_value is ALWAYS required for this pipeline.
        Either the value exists in pchembl_value column OR it exists in
        standard_value column (and will be converted).
        
        pchembl_value is preferred because:
        - It normalizes the range of values (typically 3-12)
        - It's a logarithmic scale which facilitates ML calculations
        - It's directly comparable across different assay types
        
        For rows where pchembl_value (column 4) is None or NaN,
        calculate it from standard_value (column 3).
        """
        if self.interaction_data is None or self.interaction_data.shape[1] < 5:
            return
        
        filled_count = 0
        already_present = 0
        failed_count = 0
        
        for i in range(len(self.interaction_data)):
            pchembl_val = self.interaction_data[i, 4]
            
            # Check if pchembl is missing
            is_missing = (
                pchembl_val is None or 
                (isinstance(pchembl_val, float) and np.isnan(pchembl_val)) or
                pchembl_val == '' or
                str(pchembl_val).lower() == 'nan' or
                str(pchembl_val).lower() == 'none'
            )
            
            if is_missing:
                # Calculate from standard_value (MUST exist if pchembl is missing)
                standard_val = self.interaction_data[i, 3]
                calculated_pchembl = self._calculate_pchembl_from_nm(standard_val)
                
                if calculated_pchembl is not None:
                    self.interaction_data[i, 4] = calculated_pchembl
                    filled_count += 1
                else:
                    # This should not happen - log error but continue
                    failed_count += 1
                    self.logger.error(
                        f"Row {i}: Neither pchembl_value nor valid standard_value found! "
                        f"standard_value={standard_val}"
                    )
            else:
                already_present += 1
        
        self.logger.info(
            f"pChEMBL values: {already_present} already present, "
            f"{filled_count} calculated from standard_value, "
            f"{failed_count} could not be calculated"
        )
    
    def _add_pchembl_column(self) -> None:
        """
        Add pChEMBL column calculated from standard_value for 4-column data.
        
        Transforms [molregno, kinase, type, standard_value] into
        [molregno, kinase, type, standard_value, pchembl_value]
        """
        if self.interaction_data is None or self.interaction_data.shape[1] != 4:
            return
        
        n_samples = len(self.interaction_data)
        pchembl_column = np.empty(n_samples, dtype=object)
        
        successful = 0
        failed = 0
        
        for i in range(n_samples):
            standard_val = self.interaction_data[i, 3]
            pchembl = self._calculate_pchembl_from_nm(standard_val)
            pchembl_column[i] = pchembl
            
            if pchembl is not None:
                successful += 1
            else:
                failed += 1
        
        # Add the new column
        self.interaction_data = np.column_stack([self.interaction_data, pchembl_column])
        
        self.logger.info(
            f"Added pChEMBL column: {successful} calculated successfully, "
            f"{failed} could not be calculated"
        )
    
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
            
            # Check if we have 4 or 5 columns (molregno, kinase, type, value, [pchembl])
            if len(self.labels.shape) != 2 or self.labels.shape[1] not in (4, 5):
                self.logger.error(f"Labels should have 4 or 5 columns, got {self.labels.shape[1]}")
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
